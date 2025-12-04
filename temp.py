#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : temp.py
@Author  : Gan Yuyang
@Time    : 2024/7/12 上午6:04
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import ellipk, ellipe
from typing import Literal

def compute_phase(mu1, mu2, *args):
    print(1)
    """
    Compute the phase function for radiative transfer.

    Parameters:
    mu1, mu2: float
        Input values for the phase function computation.

    Optional Parameters:
    'phase_type', phase_type: str
        Specify the phase_type of phase function to use ('isotropic', 'rayleigh', 'hg', 'legendre').

    For 'hg' phase phase_type:
    g: float
        Asymmetric factor for Henyey-Greenstein phase.

    For 'legendre' phase phase_type:
    beta: list or numpy array
        Coefficients for Legendre expansion.

    Returns:
    phase: float
        Computed phase value.
    """
    # Set isotropic as default
    if len(args) == 0:
        phase_type = 'isotropic'
    else:
        arg_dict = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
        phase_type = arg_dict.get('phase_type', 'isotropic').lower()

    # Determine which phase function to use
    if phase_type.startswith('iso'):
        # isotropic scattering
        phase = 1
    elif phase_type.startswith('ray'):
        # Rayleigh scattering
        phase = 3 / 4 * (1 + mu1 ** 2 * mu2 ** 2 + 0.5 * (1 - mu1 ** 2) * (1 - mu2 ** 2))
    elif phase_type.startswith('hg'):
        # Henyey-Greenstein
        if 'g' in arg_dict:
            g = arg_dict['g']
        else:
            raise ValueError('Please specify asymmetric factor for H-G')

        a = 1 + g ** 2 - 2 * g * mu1 * mu2
        b = 2 * abs(g) * np.sqrt((1 - mu1 ** 2) * (1 - mu2 ** 2))
        E = ellipe(2 * b / (a + b))
        phase = 2 * (1 - g ** 2) * np.sqrt(a + b) / (np.pi * (a ** 2 - b ** 2)) * E
    elif phase_type.startswith('leg'):
        # Legendre expansion
        if 'beta' in arg_dict:
            beta = arg_dict['beta']
        else:
            raise ValueError('Please specify beta for Legendre expansion')

        pn = len(beta)
        phase = 0
        for i in range(pn):
            lp_mu1 = np.polynomial.legendre.legval(mu1, [0] * i + [1])
            lp_mu2 = np.polynomial.legendre.legval(mu2, [0] * i + [1])
            phase += beta[i] * lp_mu1 * lp_mu2
    else:
        raise ValueError(f'Unknown phase phase_type: {phase_type}')

    return phase


def phase_func(mu1, mu2, phase_type:Literal[None, 'iso', 'ray', 'hg', 'leg'], g=None):
    if phase_type == 'iso'or phase_type == 'isotropic' or phase_type is None:
        phase = 1
    elif phase_type == 'ray' or phase_type == 'rayleigh':
        phase = 3 / 4 * (1 + mu1 ** 2 * mu2 ** 2 + 0.5 * (1 - mu1 ** 2) * (1 - mu2 ** 2))
    elif phase_type == 'hg':
        if g is None:
            raise ValueError('g is required')
        else:
            ...
        a = 1 + g ** 2 - 2 * g * mu1 * mu2
        b = 2 * abs(g) * np.sqrt((1 - mu1 ** 2) * (1 - mu2 ** 2))
        E = ellipe(2 * b / (a + b))
        phase = 2 * (1 - g ** 2) * np.sqrt(a + b) / (np.pi * (a ** 2 - b ** 2)) * E
    elif phase_type == 'leg':
        raise NotImplementedError

    else:
        raise TypeError('Unknown type')

    return phase

def dataset_format_to_fig(X, Y):
    # X is [[xmax, omega], ...], Y is [[T, R], ...]
    x_max = np.max(X[:, 0]) # xmax increases first
    omega_max = np.max(X[:, 1])
    x_size = np.unique(X[:, 0]).size
    omega_size = np.unique(X[:, 1]).size
    T_mat = np.zeros([x_size, omega_size])
    R_mat = np.zeros([x_size, omega_size])
    for (i, (x, o)) in enumerate(X):
        T_mat[i//x_size, i%x_size] = Y[i, 0]
        R_mat[i//x_size, i%x_size] = Y[i, 1]

    return T_mat, R_mat

class Loss_Tracker:
    def __init__(self):
        self.losses = []
        self.train_epoch_losses = []
        self.test_epoch_losses = []







if __name__ == '__main__':
    f = np.load('data/dataset_mesh.npz')
    X = f['X']
    Y = f['isotropic']
    T, R = dataset_format_to_fig(X, Y)
    plt.imshow(T, origin='lower')
    plt.colorbar()
    plt.show()
    ...