
import matplotlib.pyplot as plt

from RTE_Truth_Model import RTE
import numpy as np
# import cupy as np
from tqdm import *

import numba
from joblib import Parallel, delayed


def gen_x(num=10000):
    # Return a shape of [num, 2]
    """
    [[xmax, omega],
     [xmax, omega],
     ...
     ]]
    """
    X = np.random.uniform(0, 1, size=(num, 2))
    X[:, 0] *= 5
    X[:0][X[: 0] == 0] = 1e-6
    # plt.scatter(a[0], a[1])
    # plt.show()
    return X



def gen_y(X, phase_types=['isotropic', 'rayleigh', 'hg']):
    data_dct = {}
    # Y = np.zeros((len(phase_types), len(X), 2))
    for (t_idx, type) in enumerate(phase_types):
        assert type in ['isotropic', 'rayleigh', 'hg', 'legendre']
        Y = np.zeros((len(X), 2))
        print(f'Generating data for {type}')
        rte = RTE(0, 1, 0.9, 1, 21, 21, phase_type=type, g=0.5)
        for (i, (xmax, omega)) in tqdm(enumerate(X), total=len(X)):
            rte.xmax: float = xmax
            rte.omega: float = omega
            rte.build()
            T, R = rte.hemi_props()
            Y[i] = [T, R]
        data_dct[type] = Y
    return data_dct


def gen_bound(num=5000):
    # this function is to generate the data with boundary conditions on omega == 1,
    # and xmax == 0, 5
    X = np.random.uniform(0, 1, size=(num, 2))
    X[:, 0] *= 5
    X[:, 1] = np.random.choice([0, 1], size=num)
    return X


def gen_fine_pair(num=300, phase='iso'):
    # The data has labels as "opt (optical props) and  "spe" (spectral props)
    # rte_obj = RTE(1, 1, 0.99, 5., 21, 29, phase, 0.5)
    space_log = np.logspace(np.log10(1e-3), np.log10(1e3), num)
    space_lin_0 = np.linspace(1, 1e2, num)
    space_lin = np.linspace(1e2, 1e3, num)
    depth_space = np.concatenate([space_log, space_lin_0, space_lin])

    omega_space = np.linspace(0, 1-1e-2, 100)
    xmax_omega_pair = np.zeros((len(depth_space)*len(omega_space), 2))
    T_R_pair = np.zeros((len(depth_space)*len(omega_space), 2))

    # Conventional Code
    # for (i, xmax) in tqdm(enumerate(depth_space), total=len(depth_space)):
    #     for (j, omega) in enumerate(omega_space):
    #         rte_obj.xmax = xmax
    #         rte_obj.omega = omega
    #         rte_obj.build()
    #         T, R = rte_obj.hemi_props()
    #         xmax_omega_pair[i*len(omega_space)+j] = [xmax, omega]
    #         T_R_pair[i*len(omega_space)+j] = [T, R]

    # Parallel Code
    def process_inner_loop(j, omega, xmax):
        local_rte_obj = RTE(1, 1, 0.9, 5., 21, 21, phase, 0.5)
        local_rte_obj.xmax = xmax
        local_rte_obj.omega = omega
        local_rte_obj.build()
        T, R = local_rte_obj.hemi_props()
        return (j, omega, T, R)

    for i, xmax in tqdm(enumerate(depth_space), total=len(depth_space)):
        results = Parallel(n_jobs=-1)(
            delayed(process_inner_loop)(j, omega, xmax) for j, omega in enumerate(omega_space)
        )

        for j, omega, T, R in results:
            xmax_omega_pair[i * len(omega_space) + j] = [xmax, omega]
            T_R_pair[i * len(omega_space) + j] = [T, R]
    np.savez(f'data/forward_fine_data/{phase}_{num}', opt=xmax_omega_pair, spe= T_R_pair)
    # np.save(f'data/forward_fine_data/{phase}', xmax_T_pair)
    # return xmax_T_pair

def gen_extra(num=300, phase='iso'):
    # The data has labels as "opt (optical props) and  "spe" (spectral props)
    rte_obj = RTE(1, 1, 0.9, 5., 21, 21, phase, 0.5)

    space_log = np.logspace(np.log10(1e-1), np.log10(1e1), num)
    depth_space = np.concatenate([space_log])

    omega_space = np.linspace(0, 1-1e-2, 100)
    xmax_omega_pair = np.zeros((len(depth_space)*len(omega_space), 2))
    T_R_pair = np.zeros((len(depth_space)*len(omega_space), 2))

    def process_inner_loop(j, omega, xmax, rte_obj):
        local_rte_obj = RTE(1, 1, 0.99, 5., 21, 21 phase, 0.5)
        local_rte_obj.xmax = xmax
        local_rte_obj.omega = omega
        local_rte_obj.build()
        T, R = local_rte_obj.hemi_props()
        return (j, omega, T, R)

    for i, xmax in tqdm(enumerate(depth_space), total=len(depth_space)):
        results = Parallel(n_jobs=-1)(
            delayed(process_inner_loop)(j, omega, xmax, rte_obj) for j, omega in enumerate(omega_space)
        )

        for j, omega, T, R in results:
            xmax_omega_pair[i * len(omega_space) + j] = [xmax, omega]
            T_R_pair[i * len(omega_space) + j] = [T, R]
    np.savez(f'data/forward_fine_data/{phase}_{num}_extra', opt=xmax_omega_pair, spe= T_R_pair)


def gen_fine_pair_2(num=300, phase='iso'):
    # The data has labels as "opt (optical props) and  "spe" (spectral props)
    # rte_obj = RTE(1, 1, 0.99, 5., 21, 21, phase, 0.5)
    # space_log = np.logspace(np.log10(1e-3), np.log10(1e3), num)
    space_log = np.power(10, np.random.uniform(-3, 3, num))
    space_lin_0 = np.random.uniform(1, 1e1, num)
    space_lin = np.random.uniform(1e-1, 1, num)
    depth_space = np.concatenate([space_log, space_lin_0, space_lin])

    omega_space = np.linspace(0, 1-1e-2, 100)
    xmax_omega_pair = np.zeros((len(depth_space)*len(omega_space), 2))
    T_R_pair = np.zeros((len(depth_space)*len(omega_space), 2))

    # Conventional Code
    # for (i, xmax) in tqdm(enumerate(depth_space), total=len(depth_space)):
    #     for (j, omega) in enumerate(omega_space):
    #         rte_obj.xmax = xmax
    #         rte_obj.omega = omega
    #         rte_obj.build()
    #         T, R = rte_obj.hemi_props()
    #         xmax_omega_pair[i*len(omega_space)+j] = [xmax, omega]
    #         T_R_pair[i*len(omega_space)+j] = [T, R]

    # Parallel Code
    def process_inner_loop(j, omega, xmax):
        local_rte_obj = RTE(1, 1, 0.9, 5., 21, 21, phase, 0.5)
        local_rte_obj.xmax = xmax
        local_rte_obj.omega = omega
        local_rte_obj.build()
        T, R = local_rte_obj.hemi_props()
        return (j, omega, T, R)

    for i, xmax in tqdm(enumerate(depth_space), total=len(depth_space)):
        results = Parallel(n_jobs=-1)(
            delayed(process_inner_loop)(j, omega, xmax) for j, omega in enumerate(omega_space)
        )

        for j, omega, T, R in results:
            xmax_omega_pair[i * len(omega_space) + j] = [xmax, omega]
            T_R_pair[i * len(omega_space) + j] = [T, R]
    if phase == 'iso':
        np.savez(f'data/forward_fine_data/{phase}_{num}_unif', X=xmax_omega_pair, iso= T_R_pair)
    elif phase == 'ray':
        np.savez(f'data/forward_fine_data/{phase}_{num}_unif', X=xmax_omega_pair, ray= T_R_pair)
    elif phase == 'hg':
        np.savez(f'data/forward_fine_data/{phase}_{num}_unif', X=xmax_omega_pair, hg= T_R_pair)

def process_first_half(i, phase_type='iso'):
    """Process the first half of the pairs."""
    p = np.random.uniform(0, 1)
    if p > 0.6:
        xmax = np.power(10, np.random.uniform(-3, 3))
    else:
        # xmax = np.power(10, np.random.uniform(-1, 1))
        xmax = np.random.uniform(0, 5)
    omega = np.random.uniform(0, 1 - 1e-2)
    opt_pair_item = [xmax, omega]

   

    rte.build()
    T, R = rte.hemi_props()
    spe_pair_item = [T, R]

    return (i, opt_pair_item, spe_pair_item)


def process_second_half(i, phase_type='iso'):
    """Process the second half of the pairs."""
    p = np.random.uniform(0, 1)
    if p > 0.7:
        xmax = np.power(10, np.random.uniform(-3, 3))
    else:
        xmax = np.random.uniform(0, 5)
    omega = np.random.uniform(1-1e-2, 1)
    opt_pair_item = [xmax, omega]

    rte = RTE(omega, 1, 0.99, xmax, 21, 21, phase_type=phase_type, g=0.5)
    rte.build()
    T, R = rte.hemi_props()
    spe_pair_item = [T, R]

    return (i, opt_pair_item, spe_pair_item)


def gen_uniform(num=300, phase='iso'):
    # Initialize arrays
    opt_pair = np.zeros((num * 2, 2))
    spe_pair = np.zeros((num * 2, 2))

    # Parallel processing for the first half
    first_half_results = Parallel(n_jobs=-1)(
        delayed(process_first_half)(i, phase) for i in range(num)
    )

    # Parallel processing for the second half
    second_half_results = Parallel(n_jobs=-1)(
        delayed(process_second_half)(i, phase) for i in range(num, num * 2)
    )

    # Populate the opt_pair and spe_pair arrays
    for i, opt_item, spe_item in first_half_results:
        opt_pair[i] = opt_item
        spe_pair[i] = spe_item

    for i, opt_item, spe_item in second_half_results:
        opt_pair[i] = opt_item
        spe_pair[i] = spe_item

    if phase == 'iso':
        np.savez(f'data/forward_unif_data/{phase}_{num}.npz', X=opt_pair, iso=spe_pair)
    elif phase == 'ray':
        np.savez(f'data/forward_unif_data/{phase}_{num}.npz', X=opt_pair, ray=spe_pair)
    elif phase == 'hg':
        np.savez(f'data/forward_unif_data/{phase}_{num}.npz', X=opt_pair, hg=spe_pair)


    return opt_pair, spe_pair
if __name__ == '__main__':
    ...
    # Generate X and X_b
    # X = gen_x(2000)
    # X_b = gen_bound(1000)

    # phase_types = ['iso', 'ray', 'hg']
    # gen_extra(num=300, phase='ray')

    # gen_fine_pair_2(num=1000, phase='hg')
    # a = np.load('data/forward_fine_data/iso_1000_unif.npz')
    # X = a['opt']
    # iso = a['iso']
    # np.savez('data/forward_fine_data/iso_1000_unif.npz', X=X, iso=iso)

    # f = np.load('data/forward_fine_data/iso_300_extra.npz')
    # X = f['opt']
    # isotropic = f['spe']
    # f = np.load('data/forward_fine_data/ray_300_extra.npz')
    # rayleigh = f['spe']
    # f =  np.load('data/forward_fine_data/hg_300_extra.npz')
    # hg = f['spe']
    # np.savez('data/forward_fine_data/set_finer_extra.npz', iso=isotropic, ray=rayleigh, hg=hg, X=X)

    gen_uniform(10000, phase='hg')




    # data_dct = gen``_y(X)
    # data_dct_b = gen_y(X_b)
    # #
    # np.savez(r'data/RTE_DATA.npz', X=X, **data_dct)
    # np.savez(r'data/RTE_DATA_BOUND.npz',X=X_b, **data_dct_b)

    # f = np.load(r'data/RTE_DATA.npz')
    # # print(f.files)
    # # print(f['X'])
    # idx = 40
    # xm, omg = f['X'][idx]
    # y = f['isotropic'][idx]
    #
    # rte = RTE(omg, 1, 0.9, xm, 21, 21, phase_type='isotropic', g=0.5)
    # rte.build()
    # print(y)
    # print(rte.hemi_props())
    #
