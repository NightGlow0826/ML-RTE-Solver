
import PyMieScatt as ps
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from RTE_Truth_Model import RTE
from data.refractive import f_n, f_k

from PartitalNN import PartialNN
from NN import Simple_NN
# m = 4.7255 + 4.3990j
# x = 0.6283
# lmd = 0.25 * 1e-6
# diam = 15 * 1e-9
# print(np.pi *diam / lmd)
# res = ps.Mie_ab(m, x)
# res2 = ps.MieQ(m, wavelength=lmd, diameter=diam, asDict=True)
# print(res2)

def MyMieQ(m, wavelength, diameter, nMedium=1.0, asDict=False, asCrossSection=False, x0=0.01):
    #  http://pymiescatt.readthedocs.io/en/latest/forward.html#MieQ
    nMedium = nMedium.real
    m /= nMedium
    wavelength /= nMedium
    x = np.pi * diameter / wavelength
    if x == 0:
        return 0, 0, 0, 1.5, 0, 0, 0
    elif x <= x0:
        return ps.RayleighMieQ(m, wavelength, diameter, nMedium, asDict)
    elif x > x0:
        nmax = np.round(2 + x + 4 * (x ** (1 / 3)))
        n = np.arange(1, nmax + 1)
        n1 = 2 * n + 1
        n2 = n * (n + 2) / (n + 1)
        n3 = n1 / (n * (n + 1))
        x2 = x ** 2

        an, bn = ps.Mie_ab(m, x)

        qext = (2 / x2) * np.sum(n1 * (an.real + bn.real))
        qsca = (2 / x2) * np.sum(n1 * (an.real ** 2 + an.imag ** 2 + bn.real ** 2 + bn.imag ** 2))
        qabs = qext - qsca

        g1 = [an.real[1:int(nmax)],
              an.imag[1:int(nmax)],
              bn.real[1:int(nmax)],
              bn.imag[1:int(nmax)]]
        g1 = [np.append(x, 0.0) for x in g1]
        g = (4 / (qsca * x2)) * np.sum(
            (n2 * (an.real * g1[0] + an.imag * g1[1] + bn.real * g1[2] + bn.imag * g1[3])) + (
                    n3 * (an.real * bn.real + an.imag * bn.imag)))

        qpr = qext - qsca * g
        qback = (1 / x2) * (np.abs(np.sum(n1 * ((-1) ** n) * (an - bn))) ** 2)
        qratio = qback / qsca
        if asCrossSection:
            css = np.pi * (diameter / 2) ** 2
            cext = css * qext
            csca = css * qsca
            cabs = css * qabs
            cpr = css * qpr
            cback = css * qback
            cratio = css * qratio
            if asDict:
                return dict(Cext=cext, Csca=csca, Cabs=cabs, g=g, Cpr=cpr, Cback=cback, Cratio=cratio)
            else:
                return cext, csca, cabs, g, cpr, cback, cratio
        else:
            if asDict:
                return dict(Qext=qext, Qsca=qsca, Qabs=qabs, g=g, Qpr=qpr, Qback=qback, Qratio=qratio)
            else:
                return qext, qsca, qabs, g, qpr, qback, qratio


class Aerogel_Sample:
    def __init__(self, thickness_mm, density,
                 optical_mean_r_nm, saxs_mean_r_nm=None, visible_trans=None, visible_haze=None,
                 wavelength_nm=None,
                 n_e_cm3=None, rho_e_cm=None, rho0=2.65e3,
                 m0: complex = None):
        # When changing parameters, only the listed are valid. Do not change the parameters in build() func
        self.thickness_mm = thickness_mm
        self.density = density
        self.si_density = rho0
        self.optical_mean_r_nm = optical_mean_r_nm
        self.saxs_mean_r_nm = saxs_mean_r_nm
        self.visible_trans = visible_trans
        self.visible_haze = visible_haze
        self.wavelength_nm = wavelength_nm
        self.n_e_cm3 = n_e_cm3
        self.rho_e_cm = rho_e_cm
        self.m0 = m0

    def build(self):
        self.thickness = self.thickness_mm * 1e-3
        self.density = self.density
        self.optical_mean_r = self.optical_mean_r_nm * 1e-9
        # self.saxs_mean_r = self.saxs_mean_r_nm * 1e-9
        self.wavelength = self.wavelength_nm * 1e-9

        # self.n_e = self.n_e_cm3 * 1e6
        # self.rho_e = self.rho_e_cm * 1e-2  # convert Ohm cm to Ohm m
        self.x0 = np.pi * self.optical_mean_r / self.wavelength

        self.N = (3 * self.density) / (4 * np.pi * self.si_density * self.optical_mean_r ** 3)

    def build_m(self):

        if self.m0 is not None and self.m0.imag >= 0:
            n = self.m0.real
            k = self.m0.imag
            # print('Using given m0')
        elif self.m0.imag < 0:
            n = f_n(self.wavelength)
            k = 0
            # print('Suppressed k')
        else:
            n = f_n(self.wavelength)
            k = f_k(self.wavelength)
            # raise NotImplementedError
        self.m = n + 1j * k
        # return n + 1j * k_n(self.wavelength)
        #     k = f
        return n + 1j * k

    def get_mie(self, ITO=False):
        # m = self.build_m_old()
        self.x0 = np.pi * self.optical_mean_r / self.wavelength
        # print(f'x0: {self.x0}')
        if ITO:
            m = self.build_m_old()
        else:
            m = self.build_m()
        # res = ps.MieQ(m=m, wavelength=self.wavelength, diameter=self.optical_mean_r, asDict=True)
        res = MyMieQ(m=m, wavelength=self.wavelength,
                     diameter=self.optical_mean_r * 2, asDict=True, x0=1e-12, )
        # qsca = res['Qsca']
        # qabs = res['Qabs']

        return res

    def to_opt_set(self, ITO=False, verbose=False):
        res = self.get_mie(ITO=ITO, )
        # print(res)
        self.qsca = res['Qsca']
        self.qabs = res['Qabs']
        self.g = res['g']
        self.qext = res['Qext']
        self.sca = self.N * np.pi * self.qsca * self.optical_mean_r ** 2
        self.abs = self.N * np.pi * self.qabs * self.optical_mean_r ** 2
        self.beta = self.sca + self.abs
        self.xmax = self.beta * self.thickness
        self.omega = self.sca / self.beta

        if verbose:
            print(f'N: {self.N}')
            print(f'n: {self.m.real}, k: {self.m.imag}')
            print(f'qsca: {self.qsca}, qabs: {self.qabs}, g: {self.g}, qext: {self.qext}')
            print(f'sca: {self.sca}, abs: {self.abs}, beta: {self.beta}')
            print(f'xmax: {self.xmax}, omega: {self.omega}')
        return self.xmax, self.omega, self.g


def test_props():
    waves_nm = np.logspace(np.log10(.03), np.log10(.6), 10) * 1e3
    # for sample a

    # aero.build()
    T_totals, Rs = [], []
    T_diffuses = []
    exts = []
    Hs = []
    for wave in waves_nm:
        # print(f'wave: {wave} nm')
        aero.wavelength_nm = wave
        aero.build()
        aero.build_m()
        aero.to_opt_set()
        res = aero.get_mie(ITO=False)
        # print(res['Qabs'])
        # print(res['Qsca'])

        rte_obj = RTE(aero.omega, 1, 0.99, aero.xmax, 41, 21, phase_type='hg', g=aero.g)
        rte_obj.build()
        T, R = rte_obj.hemi_props()
        T_direct = np.exp(-aero.beta * aero.thickness)
        T_diffuse = T - T_direct
        H = rte_obj.hemi_props(type='haze')
        # print(f'T: {T}, R: {R}')
        T_totals.append(T)
        # Rs.append(R)
        T_diffuses.append(T_diffuse)
        Hs.append(H)

    plt.plot(waves_nm, T_totals, label='T')
    # plt.plot(waves_nm, Rs, label='R')
    plt.plot(waves_nm, T_diffuses, label='T_diffuse')
    plt.plot(waves_nm, Hs, label='Haze')
    plt.title(f'm = {aero.m.real} + {aero.m.imag}j')
    plt.xlabel('wavelength (nm)')
    plt.legend()
    plt.show()


def traj():
    res = MyMieQ(m=aero.m, wavelength=aero.wavelength, diameter=aero.optical_mean_r * 2, asDict=True, x0=1e-12)
    print(res['Qsca'])
    # ns = np.linspace(1.1, 2.5, 15)
    # rte_obj = RTE(0, 1, 0.99, 0, 21, 21, phase_type='hg', g=0)
    #
    # for (j, wl) in enumerate(wls_nm):
    #     print(j)
    #     for (i, n) in enumerate(ns):
    #         aero.wavelength_nm = wl
    #         aero.m0 = n + 1j * 0
    #         aero.build()
    #         aero.build_m()
    #         aero.to_opt_set()
    #
    #         rte_obj.omega = aero.omega
    #         rte_obj.xmax = aero.xmax
    #         rte_obj.g = aero.g
    #
    #         rte_obj.build()
    #         T, R = rte_obj.hemi_props()
    #         Ts[i, j] = T
    # plt.imshow(Ts, origin='lower', extent=(wls_nm[0], wls_nm[-1], ns[0], ns[-1]), aspect='auto')
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('n')
    # plt.title(f'T, r = {aero.optical_mean_r_nm} nm, thickness = {aero.thickness_mm} mm, density = {aero.density}')
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    # The following code is uesd to test the Mie scattering model with the given ITO model
    # aero.optical_mean_r_nm = 15
    # wls = np.logspace(np.log10(0.25), np.log10(40), 3) * 1e3
    # for wl in wls:
    #     aero.wavelength_nm = wl
    #     aero.build()
    #     aero.build_m_old()
    #     print(aero.get_mie(ITO=True)['Qsca'])
    # quit()
    aero = Aerogel_Sample(thickness_mm=10, density=13.877,
                          optical_mean_r_nm=6.428571428571428, saxs_mean_r_nm=2.99,
                          wavelength_nm=300,
                          m0=1.5 - 1j)
    # Sample C in Zhao's Paper
    aero.optical_mean_r_nm = 3.50
    aero.density = 293
    aero.thickness_mm = 5.26

    aero_1 = Aerogel_Sample(thickness_mm=3, density=13.877,
                            optical_mean_r_nm=6.428571428571428,
                            wavelength_nm=300,
                            m0=1.5 - 1j)

    wl_nm_lst = [300, 450, 600, 750, 900]

    """Following: traj for spectrum from the samples"""
    # err_map = np.zeros((n_rho:=500, n_r:=500))

    from interpolator import f1_total, f3_total
    # target_T = f3_total(np.array(wl_nm_lst)).reshape(len(wl_nm_lst), -1)
    # print(target_T)
    # model = Simple_NN(num_features=1, out_features=1)
    # model.load_state_dict(torch.load('models/PartialNN_Inverse/log_xmax_model_Resnet_1/epoch_440.pth'))

    # model.load_state_dict(
    #     torch.load('models/PartialNN_Inverse/log_xmax_model_fine1/epoch_1600.pth', map_location='cpu'))
    # target_xmax = np.exp(model(torch.tensor(target_T, dtype=torch.float)).detach().numpy()).reshape(-1)
    # print(target_xmax)

    # quit()
    # for (idx, wl_nm) in enumerate(wl_nm_lst):
    #
    #     aero.wavelength_nm = wl_nm
    #     aero.build()
    #     aero.to_opt_set()
    #
    #     # print(target_xmax[idx], aero.xmax)
    #
    #     aero_1.wavelength_nm = wl_nm
    #     # for (j, rho) in tqdm(enumerate(np.linspace(200, 400, n_rho)), total=n_rho):
    #     for (j, rho) in enumerate(np.linspace(200, 400, n_rho)):
    #         for (i, r) in enumerate(np.linspace(3.5, 5, n_r)):
    #             aero_1.optical_mean_r_nm = r
    #             aero_1.density = rho
    #             aero_1.build()
    #             aero_1.to_opt_set()
    #             err = np.log(abs(aero_1.xmax - aero.xmax))
    #             # err = np.log(abs(aero_1.xmax - target_xmax[idx]))
    #             err_map[j, i] += err
    # np.save(f'plot_data/r_rho_real_error_{n_rho}.npy', err_map)
    #
    # plt.imshow(err_map, aspect='auto', origin='lower')
    # plt.show()
    """Following: traj for spectrum from artificial samples"""
    from spectrum import spectrum

    target_T = spectrum(np.array(wl_nm_lst), sep_wl_nm=500, smooth=100)
    target_T = torch.from_numpy(target_T).to(torch.float32).reshape(-1, 1)

    model = PartialNN()
    model2 = PartialNN()

    model.load_state_dict(
        torch.load('models/PartialNN_Inverse/log_xmax_model_fine1/epoch_1900.pth', map_location='cpu'))
    model2.load_state_dict(torch.load('models/PartialNN/'))
    target_xmax = np.exp(model(target_T).detach().numpy()).reshape(-1)
    print(target_T.reshape(-1))
    print(target_xmax)
    err_map = np.zeros((n_rho := 100, n_r := 100))
    for (idx, wl_nm) in tqdm(enumerate(wl_nm_lst)):
        aero_1.wavelength_nm = wl_nm
        # for (j, rho) in tqdm(enumerate(np.linspace(200, 500, n_rho)), total=n_rho):
        for (j, rho) in enumerate(np.linspace(200, 500, n_rho)):
            for (i, r) in enumerate(np.linspace(7, 10, n_r)):
                aero_1.optical_mean_r_nm = r
                aero_1.density = rho
                aero_1.build()
                aero_1.to_opt_set()
                err = np.log(abs(aero_1.xmax - target_xmax[idx]))
                err_exp = np.log(abs(np.exp(-aero_1.xmax) - np.exp(-target_xmax[idx])))
                err_map[j, i] += err_exp
    # np.save(f'plot_data/r_rho_artificial_error_{n_rho}.npy', err_map)
    plt.imshow(err_map, aspect='auto', origin='lower', extent=([200, 500, 7, 10]))
    plt.colorbar()
    plt.show()
