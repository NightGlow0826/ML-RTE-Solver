#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : spectrum.py
@Author  : Gan Yuyang
@Time    : 2024/7/25 下午2:38
"""
import logging
import os
import time
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Mie import Aerogel_Sample
from RTE_Truth_Model import RTE
from Mie import f_n
from scipy.optimize import minimize
import torch
# from InverseNN import Simple_Inverse_NN
from PartitalNN import PartialNN
# import logging

from scipy.optimize import curve_fit


def dynamic_weighting(err_array):
    # err_array[err_array > 0.1] *= 10
    # err_array *= np.exp(err_array * 100)

    return err_array


# logging.basicConfig(level=logging.DEBUG)
# Now we need to find a material that could make T to fit a spectrum
class Material:
    # The external material class
    def __init__(self, rho0, m0=None, func_n=None, func_k=None, lambda_nm=None):
        self.rho0 = rho0
        if m0 is not None:
            self.m = m0
        else:
            if (func_n is None and func_k is None) or lambda_nm is None:
                raise ValueError('Please specify the refractive index function')
            self.m = func_n(lambda_nm) + 1j * func_k(lambda_nm)


# @lru_cache
def spectrum(wavelst_nm: np.ndarray, sep_wl_nm, smooth=50, eps=1e-3):
    # We need to optimize params and finally make T has a huge change on the sep_wl_nm
    # maybe we could use sigmoid-like func to ...
    target_T = 1 / (1 + np.exp(-1 / smooth * (wavelst_nm - sep_wl_nm)))
    target_T = np.clip(target_T, eps, 1 - eps)
    return target_T


def beta_spectrum(wavelst_nm: np.ndarray, target_T, thickness_mm, to_quasi_T=False, use_log_model=True) -> (
np.ndarray, np.ndarray):
    # raise NotImplementedError
    # use the previous inverse RTE model to get the beta spectrum
    optical_depth = np.ones_like(target_T)
    target_beta = optical_depth / (thickness_mm * 1e-3)
    # model = Simple_Inverse_NN()
    model = PartialNN()
    if use_log_model:
        model.load_state_dict(
            torch.load('models/PartialNN_Inverse/log_xmax_model_fine1/epoch_1900.pth', map_location='cpu'))
        # model.load_state_dict(torch.load('models/PartialNN_Inverse/log_xmax_model1/epoch_980.pth'))
    else:
        model.load_state_dict(torch.load('models/PartialNN/epoch_200.pth'))

    model.eval()

    # target_T = torch.from_numpy(target_T).to(torch.float32).to('cuda')
    target_T = torch.from_numpy(target_T).to(torch.float32).reshape(-1, 1)
    logging.debug(target_T)

    if use_log_model:
        # target_xmax = np.exp(model(target_T).detach().cpu().numpy())
        target_xmax = np.exp(model(target_T).detach().numpy())
    else:
        target_xmax = model(target_T).detach().cpu().numpy()

    logging.debug(target_xmax)
    target_beta = target_xmax / (thickness_mm * 1e-3)
    quasi_T = np.exp(-target_xmax)

    if not to_quasi_T:
        return wavelst_nm, target_beta
    else:
        return wavelst_nm, quasi_T


# def opt_beta_spectrum(wavelst_nm: np.ndarray, target_T, thickness_mm, to_quasi_T=False) -> (np.ndarray, np.ndarray):
#     target_R = 1 - target_T
#     rte_obj = RTE(1, 1, 0.9, 5., 21, 21, 'hg', 0.0)
#     xmaxs = []
#     for T in (target_T):
#         R = 1 - T
#         rte_obj.build()
#         rte_obj.opt_inverse_problem(T, R, )
#         xmaxs.append(rte_obj.xmax)
#         print(f'T: {T}, R: {R}, xmax: {rte_obj.xmax}')
#     xmaxs = np.array(xmaxs)
#     betas = xmaxs / (thickness_mm * 1e-3)
#
#     if not to_quasi_T:
#         # _, betas = opt_beta_spectrum(wavelst_nm, spectrum(wavelst_nm, 500, smooth=50), 10)
#         # np.save(f'data/betas.npy', betas)
#         # betas = np.load(f'data/betas.npy')
#         return wavelst_nm, betas
#
#     else:
#         logging.info('beta spectrum returning e^-beta t')
#         optical_depth = np.exp(-xmaxs)
#         return wavelst_nm, optical_depth
#

# params need to be optimized: rho, r, thickness
# Since we need to select material (which has an fixed m func), we need to optimize rho, r, thickness to see whether it could converge

def quasi_T_func(wavelst_nm, thickness_mm, prop_vec, err=False, show=True):
    rho, r_nm = prop_vec
    quasi_T_lst = np.zeros_like(wavelst_nm)
    # target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=50)
    for i, wavel_nm in enumerate(wavelst_nm):
        m = f_n(wavel_nm / 1e9) - 1j

        # print(m)
        material = Material(rho0=2.4e3, m0=m, func_n=None, func_k=None, lambda_nm=wavel_nm)

        aero = Aerogel_Sample(thickness_mm=thickness_mm, density=rho, optical_mean_r_nm=r_nm, saxs_mean_r_nm=r_nm,
                              wavelength_nm=wavel_nm, rho0=material.rho0,
                              m0=material.m)
        aero.build()
        aero.to_opt_set()

        quasi_T_lst[i] = np.exp(-aero.xmax)

    if err is False:
        if show:
            plt.plot(wavelst_nm, quasi_T_lst, '.-', label='Optimized_quasi_T')
            plt.plot(wavelst_nm, target_quasi_T, '.-', label='Target_quasi_T')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('T')
            plt.title('quasi_T')
            plt.legend()
            plt.show()
        else:
            ...
        return quasi_T_lst
    else:
        err = quasi_T_lst.flatten() - target_quasi_T.flatten()
        # print((err**2).mean())
        # err = dynamic_weighting(err)
        return (err ** 2).mean()


def T_func(wavelst_nm, thickness_mm, prop_vec, err=False, verbose=False, show=True,):
    rho, r_nm = prop_vec
    T_lst = np.zeros_like(wavelst_nm)
    # target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=50)
    from PartitalNN import PartialNN
    optical_depth = np.ones_like(wavelst_nm)

    model = PartialNN()
    model.load_state_dict(
        torch.load('models/PartialNN/log_xmax_model_fine1/epoch_220.pth', map_location='cpu'))
    # model.load_state_dict(torch.load('models/PartialNN_Inverse/log_xmax_model1/epoch_980.pth'))
    for i, wavel_nm in enumerate(wavelst_nm):
        m = f_n(wavel_nm / 1e9) - 1j

        # print(m)
        material = Material(rho0=2.4e3, m0=m, func_n=None, func_k=None, lambda_nm=wavel_nm)

        aero = Aerogel_Sample(thickness_mm=thickness_mm, density=rho, optical_mean_r_nm=r_nm, saxs_mean_r_nm=r_nm,
                              wavelength_nm=wavel_nm, rho0=material.rho0,
                              m0=material.m)
        aero.build()
        aero.to_opt_set()
        optical_depth[i] = aero.xmax

    optical_depth = torch.from_numpy(optical_depth).to(torch.float32).reshape(-1, 1)
    T_lst = model(torch.log(optical_depth)).detach().numpy()

        # rte_obj =
        # (aero.omega, 1, 0.9, aero.xmax, 21, 21, 'hg', aero.g)
        # rte_obj.build()
        # T, R = rte_obj.hemi_props()
        #
        # T_lst[i] = T
        # if verbose:
        # print(f'r: {aero.optical_mean_r}, rho: {aero.density}')
        # print(f'Wavelength: {wavel_nm} nm, beta: {aero.beta}, omega: {aero.omega}, g: {aero.g}, xmax: {aero.xmax}')
        # print(f'T: {T}, R: {R}')
    if err is False:
        if show:
            plt.plot(wavelst_nm, T_lst, label='Optimized_T')
            plt.plot(wavelst_nm, target_T, label='Target_T')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('T')
            plt.title('T')
            plt.legend()
            plt.show()
        return T_lst
    else:
        err = T_lst.flatten() - target_T.flatten()
        err = dynamic_weighting(err)
        return (err ** 2).mean()


# def beta_func(wavelst_nm, thickness_mm, prop_vec, err=False, verbose=False):
#     rho, r_nm = prop_vec
#     beta_lst = np.zeros_like(wavelst_nm)
#     # target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=50)
#
#     for i, wavel_nm in enumerate(wavelst_nm):
#         m = f_n(wavel_nm / 1e9) - 1j
#
#         # print(m)
#         material = Material(rho0=2.4e3, m0=m, func_n=None, func_k=None, lambda_nm=wavel_nm)
#
#         aero = Aerogel_Sample(thickness_mm=thickness_mm, density=rho, optical_mean_r_nm=r_nm, saxs_mean_r_nm=r_nm,
#                               wavelength_nm=wavel_nm, rho0=material.rho0,
#                               m0=material.m)
#         aero.build()
#         aero.to_opt_set()
#         beta_lst[i] = aero.beta
#     if err == False:
#         plt.plot(wavelst_nm, beta_lst, label='Optimized')
#         plt.plot(wavelst_nm, target_beta, label='Target')
#         plt.xlabel('Wavelength (nm)')
#         plt.ylabel('Beta')
#         plt.legend()
#         plt.show()
#         return beta_lst
#     else:
#         return ((beta_lst - Target_beta) ** 2).mean()
#

def target_func_T(prop_vec, thickness_mm):
    return T_func(wavelst_nm, thickness_mm, prop_vec, err=True)


# def target_func_beta(prop_vec, thickness_mm):
#     return beta_func(wavelst_nm, thickness_mm, prop_vec, err=True)


def target_func_quasi_T(prop_vec, thickness_mm):
    return quasi_T_func(wavelst_nm, thickness_mm, prop_vec, err=True)


global_step = 0


def print_step(x):
    global global_step
    global_step += 1
    # print(f'step: {global_step}')
    logging.info(f'step: {global_step}')




def real_spectrum(wavelst_nm: np.ndarray, thick_rho_r=(4.75, 144, 3.05), opt_prop_only=False):
    thick, rho, r = thick_rho_r
    aero = Aerogel_Sample(thickness_mm=thick, density=rho, optical_mean_r_nm=r, wavelength_nm=10, m0=1 - 2j)
    rte = RTE(1, 1, 0.9, 5.26, 21, 21, 'hg', 0.0)
    T = np.zeros([3, len(wavelst_nm)])
    opt_props = np.zeros_like(T)
    for (i, wl) in tqdm(enumerate(wavelst_nm), total=len(wavelst_nm)):
        aero.wavelength_nm = wl
        aero.build()
        xmax, omega, g = aero.to_opt_set()
        opt_props[0, i] = xmax
        opt_props[1, i] = omega
        opt_props[2, i] = g
        rte.xmax = xmax
        rte.omega = omega
        rte.g = g
        rte.build()
        # T, R = rte.hemi_props()

        T[0, i] = rte.hemi_props()[0]
        T[1, i] = np.exp(-xmax)
        T[2, i] = T[0, i] - np.exp(-xmax)

    if opt_prop_only:
        return opt_props
    else:
        return T
# def real_sample_spectrum(wavelst_nm, interp_func):
#     T = np.zeros([3, len(wavelst_nm)])
#     opt_props = np.zeros_like(T)
#
#         # T, R = rte.hemi_props()
#     T_total =
#     T[0, i] = rte.hemi_props()[0]
#     T[1, i] = np.exp(-xmax)
#     T[2, i] = T[0, i] - np.exp(-xmax)




# prop: rho, r
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.CRITICAL)
    thickness_mm = 3

    # logging.basicConfig(level=logging.ERROR)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # sep_wl_nm = 800
    # l_ext = 200
    # r_ext = 300
    # minor_extend = 50
    # assert sep_wl_nm > max(l_ext, r_ext)
    # wavelst_nm = np.linspace(sep_wl_nm - l_ext, sep_wl_nm + r_ext, 30)
    # wavelst_nm = np.hstack([
    #                     wavelst_nm,
    #                     np.linspace(sep_wl_nm - minor_extend, sep_wl_nm + minor_extend, 50),
    #                     np.linspace(sep_wl_nm + r_ext - minor_extend, sep_wl_nm + r_ext, 1),
    # ])
    # logging.debug(wavelst_nm)
    # wavelst_nm.sort()

    # Tset = real_spectrum(wavelst_nm, (5.25, 293, 3.5))
    # plt.plot(wavelst_nm, Tset[2])
    # plt.show()
    # quit()
    #
    #################################
    #################################
    # wavelst_nm = np.linspace(200, 1000, 50)
    # datas = np.zeros([(4+1)*3, len(wavelst_nm)])
    # thick_rho_r_set = [(4.75, 144, 3.05), (5.25, 293, 3.50)]
    # # target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=100)
    # fig, axes = plt.subplots(1, 3)
    # for (i, thick_rho_r) in enumerate(thick_rho_r_set):
    #     target_T_set = real_spectrum(wavelst_nm, thick_rho_r) # total, direct, diffuse
    #
    #     # Optimize Quasi_T
    #     _, target_quasi_T = beta_spectrum(wavelst_nm, target_T_set[0], thickness_mm, to_quasi_T=True)
    #     target_quasi_T = target_quasi_T.flatten()
    #     # print(target_quasi_T)
    #     res = minimize(target_func_quasi_T,
    #                    x0=np.array([200, 5.]), args=(3,), bounds=[(20, 900), (0.1, 50)],
    #                      options={'maxiter': 20}, callback=print_step, method='Nelder-Mead')
    #     logging.info(tuple(res.x))
    #     logging.info(res.fun)
    #
    #
    #     opt_T_by_quasi_T = T_func(wavelst_nm, 3, res.x, err=False, show=False).flatten()
    #     axes[i].plot(wavelst_nm, opt_T_by_quasi_T, '.-', label='ML_T_Total', alpha=0.3)
    #     axes[i].plot(wavelst_nm, opt_T_by_quasi_T - target_T_set[1], '.-', label='ML_T_Diffuse', alpha=0.3)
    #     axes[i].plot(wavelst_nm, target_T_set[0], '--', label='Target_T_Total', alpha=0.3)
    #     axes[i].plot(wavelst_nm, target_T_set[2], '--', label='Target_T_Diffuse', alpha=0.3)
    #     datas[i*5] = wavelst_nm
    #     datas[i*5 + 1] = opt_T_by_quasi_T.flatten()
    #     datas[i*5 + 2] = opt_T_by_quasi_T.flatten() - target_T_set[1]
    #     datas[i*5 + 3] = target_T_set[0]
    #     datas[i*5 + 4] = target_T_set[2]
    #
    #     axes[i].set_xlabel('Wavelength (nm)')
    #     axes[i].set_ylabel('T')
    #     axes[i].set_title(
    #         f'Source: Thickness: {thick_rho_r[0]:.2f} mm, $\\rho$: {thick_rho_r[1]:.1f} $\\mathrm{{kg/m^3}}$, $r$: {thick_rho_r[2]:.2f} nm \n'
    #         f'Optim: Thickness: 3 mm, $\\rho$: {res.x[0]:.1f} $\\mathrm{{kg/m^3}}$, $r$: {res.x[1]:.2f} nm')
    #
    #     axes[i].legend()
    # np.save('plot_data/real_comparison.npy', datas)
    #     # opt_quasi_T = quasi_T_func(wavelst_nm, 3, res.x, err=False)
    # fig.suptitle('Transmittance Comparison')
    # fig.tight_layout()
    # plt.show()
#######################################################
#######################################################
    from interpolator import f1_total, f1_dif, f3_total, f3_dif
    wavelst_nm = np.linspace(245, 990, 100)
    wavelst_nm_1 = np.linspace(234, 300, 20)
    wavelst_nm = np.hstack([wavelst_nm, wavelst_nm_1])
    wavelst_nm.sort()
    datas = np.zeros([(4 + 1) * 3, len(wavelst_nm)])
    # thick_rho_r_set = [(4.75, 144, 3.05), (5.25, 293, 3.50)]
    # target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=100)
    fig, axes = plt.subplots(1, 3)
    for (i, thick_rho_r) in enumerate([1, 3]):
        # target_T_set = real_spectrum(wavelst_nm, thick_rho_r)  # total, direct, diffuse
        if thick_rho_r == 1:
           T_total = f1_total(wavelst_nm)
           T_diffuse = f1_dif(wavelst_nm)
           T_direct = T_total - T_diffuse
        else:
            T_total = f3_total(wavelst_nm)
            T_diffuse = f3_dif(wavelst_nm)
            T_direct = T_total - T_diffuse

        # Optimize Quasi_T
        _, target_quasi_T = beta_spectrum(wavelst_nm,T_total, thickness_mm, to_quasi_T=True)
        target_quasi_T = target_quasi_T.flatten()
        # print(target_quasi_T)
        res = minimize(target_func_quasi_T,
                       x0=np.array([200, 5.]), args=(3,), bounds=[(20, 900), (0.1, 50)],
                       options={'maxiter': 20}, callback=print_step, method='Nelder-Mead')
        logging.info(tuple(res.x))
        logging.info(res.fun)

        opt_T_by_quasi_T = T_func(wavelst_nm, 3, res.x, err=False, show=False).flatten()
        axes[i].plot(wavelst_nm, opt_T_by_quasi_T, '.-', label='ML_T_Total', alpha=0.3)
        axes[i].plot(wavelst_nm, opt_T_by_quasi_T - T_direct, '.-', label='ML_T_Diffuse', alpha=0.3)
        axes[i].plot(wavelst_nm,T_total, '--', label='Target_T_Total', alpha=0.3)
        axes[i].plot(wavelst_nm, T_diffuse, '--', label='Target_T_Diffuse', alpha=0.3)
        datas[i * 5] = wavelst_nm
        datas[i * 5 + 1] = opt_T_by_quasi_T
        datas[i * 5 + 2] = opt_T_by_quasi_T - T_direct
        datas[i * 5 + 3] =T_total
        datas[i * 5 + 4] = T_diffuse

        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('T')
        axes[i].set_title(
            f'rho {res.x[0]:.1f} $\\mathrm{{kg/m^3}}$, $r$: {res.x[1]:.2f} nm')

        axes[i].legend()
        total_mse = ((opt_T_by_quasi_T.flatten() - T_total.flatten()) ** 2).mean()
        diffuse_mse = ((opt_T_by_quasi_T.flatten() -T_direct.flatten() - T_diffuse.flatten()) ** 2).mean()
        print(f'Total MSE: {total_mse}, Diffuse MSE: {diffuse_mse}')
    np.save('plot_data/real_comparison_1.npy', datas)


    # opt_quasi_T = quasi_T_func(wavelst_nm, 3, res.x, err=False)
    fig.suptitle('Transmittance Comparison')
    fig.tight_layout()
    plt.show()

    ###################################
    ###################################

    # To better show the generalization of the model, we need to show the optiization to some artificial constructed T

    # fig, axes = plt.subplots(1, 1)
    # axes = [axes]
    # datas = np.zeros([(1 + 3) * 4, 40])  # 40 for len wl, 1 for wl, 3 for 3Ts
    # #
    # # for (i, sep_wl_nm) in enumerate([500, 700]):
    # for (i, sep_wl_nm) in enumerate([500, ]):
    #     l_ext = 300
    #     r_ext = 300
    #     wavelst_nm = np.linspace(sep_wl_nm - l_ext, sep_wl_nm + r_ext, 30)
    #     wavelst_nm = np.hstack([
    #         wavelst_nm,
    #         np.linspace(sep_wl_nm - 50, sep_wl_nm + 50, 10), ])
    #     print(len(wavelst_nm))
    #     wavelst_nm.sort()
    #     target_T = spectrum(wavelst_nm, sep_wl_nm, smooth=100)
    #
    #     _, target_quasi_T = beta_spectrum(wavelst_nm, target_T, thickness_mm, to_quasi_T=True)
    #
    #     t1 = time.perf_counter()
    #     res = minimize(target_func_quasi_T,
    #                    x0=np.array([500, 6.3]), args=(3,), bounds=[(100, 1200), (0.1, 50)],
    #                    options={'maxiter': 20}, callback=print_step, tol=5e-2, method='Nelder-Mead')
    #
    #     t2 = time.perf_counter()
    #     print(f'Quasi_T Optimization time: {t2 - t1:.2f}s')
    #     opt_T_by_quasi_T = T_func(wavelst_nm, 3, res.x, err=False, show=False)
    #     print(res.x)
    #
    #     # quit()
    #     t1 = time.perf_counter()
    #     res_2 = minimize(target_func_T,
    #                      x0=np.array([500, 6.3]), args=(3,), bounds=[(100, 1200), (0.1, 50)],
    #
    #                      options={'maxiter': 20}, callback=print_step, tol=5e-2, method='Nelder-Mead')
    #     t2 = time.perf_counter()
    #     print(f'Conventional Optimization time: {t2 - t1:.2f}s')
    #     opt_T = T_func(wavelst_nm, 3, res_2.x, err=False, show=False)
    #     # opt_T = T_func(wavelst_nm, 3, res_2.x, err=False, show=False)
    #
    #     axes[i].plot(wavelst_nm, target_T, '-', label='Target T')
    #     axes[i].plot(wavelst_nm, opt_T_by_quasi_T, '.-', label='ML  $T_{Total}$', alpha=0.8)
    #     axes[i].plot(wavelst_nm, opt_T, '--', label='RTE $T_{total}$', alpha=0.8)
    #     axes[i].legend()
    #     axes[i].set_xlabel('Wavelength (nm)')
    #     axes[i].set_ylabel('T')
    #     axes[i].set_title(f'Sep_wl: {sep_wl_nm} nm, ')
    #     opt_T_by_quasi_T = opt_T_by_quasi_T.flatten()
    #     opt_T = opt_T.flatten()
    #     target_T = target_T.flatten()
    #     mse_ML_target = ((opt_T_by_quasi_T - target_T) ** 2).mean()
    #
    #     mse_ML_opt = ((opt_T_by_quasi_T - opt_T) ** 2).mean()
    #     print(f'ML_target: {mse_ML_target}, ML_opt: {mse_ML_opt}')

        # datas[i*4] = wavelst_nm
        # datas[i*4+1] = target_T
        # datas[i*4+2] = opt_T_by_quasi_T
        # datas[i*4+3] = opt_T
    # np.save('plot_data/artificial_comparison ', datas)

    # fig.suptitle('Transmittance Comparison')
    # fig.tight_layout()
    # plt.show()
    # quit()
    # optimize T directly
