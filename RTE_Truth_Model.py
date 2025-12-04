import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import dataclass
from random import random
import timeit
import scipy.linalg as spla
import scipy.sparse as sp
import scipy.special
import numpy.linalg as npla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
# import stepLR
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from functools import lru_cache
from typing import Literal
# from NN import Simple_NN
import torch
from tqdm import tqdm

# matplotlib.use('TkAgg')


# import scipy as sp


# @dataclass
class RTE:
    def __init__(self, omega: float, phi: float, mu0: float, xmax: float,
                 grid_size: int, nquad: int,
                 phase_type,
                 g=None):
        self.omega: float = omega
        self.phi: float = phi
        self.mu0: float = mu0
        self.xmax: float = xmax
        self.grid_size: int = grid_size
        self.nquad: int = nquad

        self.phase_type = phase_type
        assert phase_type in ['iso', 'ray', 'hg', 'legendre']
        self.g = g

    def build(self):
        # since  figured out the suitable grid size in "truncation.py", we overide the grid size here, just ignore the initialization of the class, params will be overriden here.
        self.nquad = 21
        if self.omega <= 0.2:
            self.grid_size = 51
        elif 0.2 < self.omega <= 0.8:
            self.grid_size = 37
        else:
            self.grid_size = 21
        self.grid = np.linspace(0, self.xmax, self.grid_size)
        self.dx = self.xmax / (self.grid_size - 1)
        self.dx_inv = 1 / self.dx
        self.nodes, self.weights = scipy.special.roots_legendre(self.nquad)


    def get_finite_diff_central(self, grid=None):
        if grid is None:
            grid = self.grid
        else:
            ...
        dx = grid[1:] - grid[:-1]
        n = self.grid_size
        low_diag = np.hstack([-1. / (dx[1:] + dx[:-1]), -1. / dx[-1]])
        up_diag = np.hstack([1. / dx[0], 1. / (dx[1:] + dx[:-1])])
        main_diag = np.zeros(n)
        main_diag[0] = -1. / dx[0]
        main_diag[-1] = 1 / dx[-1]
        return sp.diags([main_diag, up_diag, low_diag], [0, 1, -1])

    # @lru_cache(10)
    def get_discrete_mat(self, omega=None) -> np.ndarray:
        """
        combines the integral part and the linear part
        :param n: nodes num of the G-L
        :param omega: albedo
        :return: operator
        """
        if omega is None:
            omega = self.omega
        else:
            ...
        nodes, weights = scipy.special.roots_legendre(self.nquad)
        A = np.repeat(weights.reshape(1, -1), self.nquad, axis=0) * omega / 2
        A -= np.eye(self.nquad)
        return A

    def phase_mat(self, fit_A=False):
        from temp import phase_func
        mat = np.zeros((self.nquad, self.nquad))
        mat_tensor = np.zeros(
            (self.nquad * self.grid_size, self.nquad * self.grid_size))
        for (i, mu1) in enumerate(self.nodes):
            for (j, mu2) in enumerate(self.nodes):
                phase = phase_func(
                    mu1, mu2, phase_type=self.phase_type, g=self.g)
                mat[i, j] = phase
                mat_tensor[self.grid_size * i:self.grid_size * (i + 1),
                           self.grid_size * j:self.grid_size * (j + 1)] = phase
        if not fit_A:
            return mat
        else:
            return mat_tensor

    # @lru_cache(10)
    def tensor_product_mat(self, sep=False, use_old=False):
        B = self.get_finite_diff_central().toarray()
        A = self.get_discrete_mat()
        B_tensor = spla.block_diag(*[B * mu_ for mu_ in self.nodes])

        n = self.nquad
        m = self.grid_size
        # now use coo format to create mat
        num_elem = self.grid_size * self.nquad ** 2
        I = np.zeros(num_elem, dtype=int)
        J = np.zeros(num_elem, dtype=int)
        V = np.zeros(num_elem)
        cnt = 0

        for i in range(n):
            for j in range(n):
                for k in range(m):
                    I[cnt] = i * m + k
                    J[cnt] = j * m + k
                    V[cnt] = A[i, j]
                    cnt += 1

        A_tensor = sp.coo_matrix((V, (I, J))).toarray()
        if not use_old:
            if self.phase_type == 'iso':
                A_tensor = (A_tensor + np.eye(self.nquad * self.grid_size)) - np.eye(
                    self.nquad * self.grid_size)
            else:
                A_tensor = (A_tensor + np.eye(self.nquad * self.grid_size)) * self.phase_mat(fit_A=True) - np.eye(
                self.nquad * self.grid_size)
        if sep:
            return B_tensor, A_tensor + np.eye(self.nquad * self.grid_size), np.eye(self.nquad * self.grid_size)
        else:
            return B_tensor - A_tensor

    def build_std_mat(self):
        # Used for precomputing, so make dx == 1
        B = self.get_finite_diff_central(np.arange(self.grid_size)).toarray()
        A = self.get_discrete_mat(omega=1)
        D = np.eye(self.nquad)
        A += D
        m, n = self.grid_size, self.nquad
        num_elem = self.grid_size * self.nquad ** 2
        I = np.zeros(num_elem, dtype=int)
        J = np.zeros(num_elem, dtype=int)
        V = np.zeros(num_elem)
        V2 = np.zeros(num_elem)

        cnt = 0
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    I[cnt] = i * m + k
                    J[cnt] = j * m + k
                    V[cnt] = A[i, j]
                    V2[cnt] = D[i, j]
                    cnt += 1
        self.std_A_tensor = sp.coo_matrix((V, (I, J))).toarray()
        self.std_B_tensor = spla.block_diag(*[B * mu_ for mu_ in self.nodes])
        self.std_D_tensor = sp.coo_matrix((V2, (I, J))).toarray()

        rhs, dirichlet = self.bound()
        bound_indice = dirichlet.nonzero()[0]
        self.std_A_tensor[bound_indice, :] = 0
        self.std_A_tensor *= self.phase_mat(fit_A=True)
        self.std_B_tensor[bound_indice, :] = 0
        self.std_D_tensor[bound_indice, :] = 0
        self.std_D_tensor[bound_indice, bound_indice] = 1

    # @lru_cache(10)
    def bound(self):
        m = self.grid_size
        n = self.nquad
        bc_vec = np.zeros(m * n)
        dirichlet_flag = np.zeros(m * n, dtype=int)
        for (j, node) in enumerate(self.nodes):
            if node >= self.mu0:
                bc_vec[j * m] = self.phi
                dirichlet_flag[j * m] = True
            elif node > 0:
                dirichlet_flag[j * m] = True
            else:
                dirichlet_flag[(j + 1) * m - 1] = True  # strange indice
            pass
        return bc_vec, dirichlet_flag

    def rte_solver(self, adjoint=False, reshape=True):
        lhs = self.tensor_product_mat()
        rhs, dirichlet = self.bound()
        # print(dirichlet)
        # intensity = np.zeros(len(dirichlet))

        # full_indice = np.arange(len(dirichlet))
        bound_indice = dirichlet.nonzero()[0]

        # unbound_indice = full_indice[~np.isin(full_indice, bound_indice)]

        # intensity[bound_indice] = rhs[bound_indice]
        # # print(npla.inv(lhs[np.ix_(bound_indice, bound_indice)]))
        # intensity[unbound_indice] = npla.inv(
        #     lhs[np.ix_(unbound_indice, unbound_indice)]) @ rhs[unbound_indice]
        # # ignore dirichlet bc in inv

        lhs[bound_indice, :] = 0
        # lhs[np.ix_(bound_indice, bound_indice)] = 1
        lhs[bound_indice, bound_indice] = 1
        # print(rhs)

        intensity = npla.inv(lhs) @ rhs
        # print(lhs)

        if reshape:
            intensity = intensity.reshape(self.nquad, self.grid_size).T
        else:
            ...

        if adjoint:
            return lhs, intensity
        else:
            return intensity

    def hemi_props(self, type='T'):
        I = self.rte_solver()
        tmp = np.zeros(self.nquad)
        tmp[self.nodes > self.mu0] = 1
        denom = np.sum(self.nodes * self.weights * tmp)
        T = sum(self.weights * I[-1, :] * self.nodes) / denom

        tmp = I[0, :]
        tmp[self.nodes > 0] = 0
        R = -sum(self.weights * tmp * self.nodes) / denom
        T_direct = np.exp(-self.xmax)
        if type == 'haze':
            return (T - T_direct) / T
        else:
            return T, R

    def get_I_adjoint(self):

        lhs, intensity = self.rte_solver(reshape=False, adjoint=True)
        # else:
        # lhs = self.std_B_tensor* (self.grid_size - 1) / self.xmax + self.std_D_tensor - self.omega * self.std_A_tensor

        intensity = self.rte_solver(reshape=False, adjoint=False)
        # print(lhs)
        new_rhs = np.hstack([self.std_A_tensor @ intensity.T,
                             self.std_B_tensor @ intensity.T * (self.grid_size - 1) / self.xmax ** 2]).reshape(2, -1).T

        return intensity, npla.inv(lhs) @ new_rhs

    def hemi_props_adjoint(self):
        intensity, dI = self.get_I_adjoint()
        # print(f'xmax: {self.xmax:.5f}, omega: {self.omega:.5f}, intensity: {intensity}')
        # print(intensity)
        intensity = intensity.reshape(self.nquad, self.grid_size).T
        dI = dI.flatten()
        a, b = dI[0::2].reshape(self.nquad, self.grid_size, ).T, dI[1::2].reshape(
            self.nquad, self.grid_size, ).T
        dI = np.array([a, b])

        tmp = np.zeros(self.nquad)
        tmp[self.nodes > self.mu0] = 1.0
        denom = np.sum(self.nodes * self.weights * tmp)
        tmp = np.zeros(self.nquad)
        tmp[self.nodes >= 0] = 1.0
        T = np.sum(self.weights * intensity[-1, :] * self.nodes * tmp) / denom
        dT = (self.weights * dI[:, -1, :] * self.nodes * tmp).transpose(0, 1)
        dT = np.sum(dT, axis=1) / denom
        tmp = np.zeros(self.nquad)
        tmp[self.nodes <= 0] = 1.0
        R = -sum(self.weights * intensity[0, :] * tmp * self.nodes) / denom
        dR = -np.sum((self.weights * dI[:, 0, :] * tmp *
                      self.nodes).transpose(0, 1), axis=1) / denom

        return T, R, dT, dR

    def inverse_problem(self, given_T, given_R, tol=1e-6, max_iter=100, ini_xmax=None, ini_omega=None, show=False):
        max_iter = max_iter
        target = np.array([given_T, given_R])
        lr = 5e-1
        # params = torch.tensor([ini_xmax, ini_omega])

        def loss(cur_state):
            return np.sqrt(np.sum((cur_state - target) ** 2))

        def d_loss(cur_state):
            return 1 / (loss(cur_state) + 1e-6) * (cur_state - target)

        err_lst = np.zeros(max_iter)
        param_hist = np.zeros((max_iter, 2))
        if ini_xmax is None:
            self.xmax = np.random.uniform(0, 5)
        else:
            self.xmax = ini_xmax
        if ini_omega is None:
            self.omega = np.random.uniform(0, 1)
        else:
            self.omega = ini_omega
        self.build()
        self.build_std_mat()
        param_hist[0] = np.array([self.omega, self.xmax])

        err_lst[0] = loss(np.array(self.hemi_props()))

        err_time = 0
        end_iter = 0
        for i in range(1, max_iter):
            end_iter = i
            self.build()
            self.build_std_mat()
            T, R, dT, dR = self.hemi_props_adjoint()
            next_state = np.array([T, R])
            l = loss(next_state)
            err_lst[i] = l
            if err_lst[i] < tol or lr < 1e-6:
                print(f'Converged, iter: {i}')
                break
            else:
                ...

            if err_lst[i] > err_lst[i - 1] and i >= 20:
                # if err_time >= 2:
                lr *= 0.8
                # print(f'iter: {i}, lr: {lr:.6f}, err: {err_lst[i]:.6f}')
                param_hist[i] = param_hist[i - 2]
                param_hist[i - 1] = param_hist[i - 2]
                # err_time = 0
                # err_time += 1

            else:
                dst = d_loss(next_state)
                grad = dst @ np.array([dT, dR])
                # print(f'iter: {i}, lr: {lr:.6f}, err: {err_lst[i]:.6f}, grad: {grad}, param: {param_hist[i-1]}')
                param_hist[i] = param_hist[i - 1] - lr * grad
                self.omega, self.xmax = param_hist[i]
        err_lst = err_lst[:end_iter]
        param_hist = param_hist[:end_iter]
        if show:
            fig, ax = plt.subplots(1, 2)
            ax[0].semilogy(np.arange(end_iter), err_lst)
            ax[1].plot(param_hist[:, 1], param_hist[:, 0], '.-', )
            ax[0].set_xlabel('iter')
            ax[0].set_ylabel('loss')
            ax[1].set_xlabel('xmax')
            ax[1].set_ylabel('omega')
            plt.show()
        else:
            ...

        return self.xmax, self.omega

    def auto_inverse_problem(self, given_T, given_R, tol=1e-5, max_iter=100,
                             ini_xmax=np.random.uniform(0, 5), ini_omega=np.random.uniform(0, 1)):
        target = np.array([given_T, given_R])

        def loss(cur_state):
            return np.sqrt(np.sum((cur_state - target) ** 2))

        def d_loss(cur_state):
            return 1 / (loss(cur_state) + 1e-6) * (cur_state - target)

        lr = 1e-2

        err_lst = np.zeros(max_iter)
        param_hist = np.zeros((max_iter, 2))

        self.xmax = ini_xmax
        self.omega = ini_omega

        torch_xmax = torch.tensor([self.xmax])
        torch_omega = torch.tensor([self.omega])
        # optimizer = torch.optim.Adagrad([torch_xmax, torch_omega], lr=lr)
        optimizer = torch.optim.Adam([torch_xmax, torch_omega], lr=lr)
        # optimizer = torch.optim.SGD([torch_xmax, torch_omega], lr=lr)
        self.build()
        self.build_std_mat()
        # param_hist[0] = np.array([self.xmax, self.omega])

        param_hist[0] = torch.tensor([self.omega, self.xmax])
        err_lst[0] = loss(np.array(self.hemi_props()))
        switch_opt = 0
        for i in range(1, max_iter):
            self.build()
            self.build_std_mat()
            T, R, dT, dR = self.hemi_props_adjoint()

            next_state = np.array([T, R])
            l = loss(next_state)
            err_lst[i] = l

            # if converged, break

            if err_lst[i] < tol and lr < 1e-6:
                print(f'Converged, iter: {i}')
                break
            else:
                ...

            # if i >= 1000 and not switch_opt:
            #     print('switch to SGD')
            #     optimizer = torch.optim.SGD([torch_xmax, torch_omega], lr=lr)
            #     switch_opt = 1

            dst = d_loss(next_state)
            # print(dst)
            grad = dst @ np.array([dT, dR])
            grad_omega, grad_xmax = grad

            if torch_xmax.grad is not None:
                torch_xmax.grad.zero_()
            if torch_omega.grad is not None:
                torch_omega.grad.zero_()
            torch_xmax.grad = torch.tensor(
                grad_xmax, dtype=torch.float32).unsqueeze(0)
            torch_omega.grad = torch.tensor(
                grad_omega, dtype=torch.float32).unsqueeze(0)
            if i % 20 == 0:
                # print(f'iter: {i}, err: {err_lst[i]:.6f}, grad: {grad}')
                ...
            optimizer.step()
            self.xmax = torch_xmax.item()
            self.omega = torch_omega.item()

            param_hist[i] = np.array([self.xmax, self.omega])

    def opt_inverse_problem(self, given_T, given_R, tol=1e-5, max_iter=100,
                            ini_xmax=np.random.uniform(0, 200), ini_omega=np.random.uniform(0, 1),
                            show=False, force=False):



        def loss(cur_state):
            return np.sqrt(np.sum((cur_state - target) ** 2))

        def call_back(xk):
            param_hist.append(list(xk))
            err_lst.append(loss(np.array(self.hemi_props())))

        def func(x, rteobj: RTE):
            rteobj.omega = x[0]
            rteobj.xmax = x[1]
            rteobj.build()
            return loss(np.array(rteobj.hemi_props()))

        if given_T < 0.2:
            ini_xmax = 1e2
        elif 0.5<given_T<0.8:
            ini_xmax = 1
        elif given_T >0.8:
            ini_xmax = 0.1
        else:
            ...

        x0 = np.array([ini_omega, ini_xmax])
        target = np.array([given_T, given_R])
        param_hist = [[ini_omega, ini_xmax]]
        err_lst = []

        xl = 1e-4
        xh = 200


        minimize(func, x0, args=(self,),
                     bounds=[(0, 1), (xl, xh)], callback=call_back, tol=tol,
                     method='Nelder-Mead')

        if show:
            fig, ax = plt.subplots(1, 2)
            param_hist = np.array(param_hist)

            ax[0].semilogy(err_lst)
            ax[1].plot(param_hist[:, 1], param_hist[:, 0], '.-', )
            ax[0].set_xlabel('iter')
            ax[0].set_ylabel('loss')
            ax[1].set_xlabel('xmax')
            ax[1].set_ylabel('omega')
            plt.show()

    def opt_inverse_problem_logmodel(self, given_T, given_R, tol=1e-5,
                            ini_log_xmax=np.random.uniform(-6, 6), ini_omega=np.random.uniform(0, 1),
                            show=False, force=False):



        def loss(cur_state):
            return np.sqrt(np.sum((cur_state - target) ** 2))


        def func(x, rteobj: RTE):
            rteobj.omega = x[0]
            rteobj.xmax = np.exp(x[1])
            rteobj.build()
            return loss(np.array(rteobj.hemi_props()))


        x0 = np.array([ini_omega, ini_log_xmax])
        target = np.array([given_T, given_R])
        err_lst = []

        giveup = 0
        num_tol = 20
        while giveup < num_tol:
            res = minimize(func, x0, args=(self,),
                     bounds=[(0, 1), (-6, 6)], tol=tol,
                     method='Nelder-Mead')
            if func(res.x, self) <= tol:
                break
            else:
                ini_log_xmax = np.random.uniform(-6, 6)
                x0 = np.array([ini_omega, ini_log_xmax])
                giveup += 1
        if giveup == num_tol:
            print('Fail')
            print('xmax, omega', given_T, given_R)


def draw_T(rte_obj: RTE, type, omega):
    Tts = []
    T_direct_lst = []
    Tfs = []
    # rte_obj.omega = 0.5
    rte_obj.omega = omega
    rte_obj.phase_type = type
    xms = np.power(10, np.linspace(-3, 3, 30))
    for xm in xms:
        rte_obj.xmax = xm
        rte_obj.build()
        T_hemi, R = rte_obj.hemi_props()
        T_direct = np.exp(-xm)

        Tts.append(T_hemi)
        T_direct_lst.append(T_direct)
        Tfs.append(T_hemi - T_direct)
        print(T_hemi - T_direct)

    Tts = np.array(Tts)
    T_direct_lst = np.array(T_direct_lst)
    Tfs = np.array(Tfs)

    plt.semilogx(xms, Tts, '*-', label=f'tol,{type}, omega={omega}')  # total
    # plt.semilogx(xms, Tds, '*-')  # direct
    plt.semilogx(xms, Tfs, '*-', label=f'dfz,{type}, omega={omega}')  # diffuse

    haze = Tfs / Tts
    # plt.semilogx(xms, haze, '*-', label=f'{type}, omega={omega}')


def draw_Ts():
    for type in ['ray', 'hg']:
        for omega in [0.5, 1]:
            print(f'type: {type}, omega: {omega}')
            draw_T(rte_obj, type, omega)
    plt.xlabel('optical depth')
    plt.ylabel('Transmittance')
    # plt.ylabel('Haze')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import time
    import random
    t1 = time.time()
    for _ in range(int(1e2)):
        rte_obj = RTE(0.5, 1, 0.99, 10, 21, 21, phase_type='iso')
        rte_obj.xmax = random.uniform(1, 1000)
        rte_obj.omega = random.uniform(0, .1)
        rte_obj.build()
        T, R = rte_obj.hemi_props()
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1} seconds")

    

        