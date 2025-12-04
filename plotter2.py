import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import matplotlib
import time
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D

from colormaps import parula

# This file is used to plot the figures needed in the paper.

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from Mie import Aerogel_Sample
from RTE_Truth_Model import RTE
from scipy.optimize import minimize

import matplotlib

matplotlib.use('TKAgg')


def T_R_Heatmap(axes, phase_type='iso'):
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    from NN import  Simple_NN
    # model = ANN()
    model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('model_2/rayleigh/WithPINN/epoch_1940.pth'))

    model.load_state_dict(torch.load(f'forward_models/epoch_380_.pth'))

    T_heatmap = np.zeros([num_omega := 100, num_xmax := 100])
    R_heatmap = np.zeros([num_omega, num_xmax])
    for (j, xmax) in enumerate(xmaxs := np.linspace(0.01, 5, num_xmax)):
        for (i, omega) in enumerate(omegas := np.linspace(0, 1, num_omega)):
            x_inp = np.log(xmax)
            # x_inp = np.log(xmax)
            T_R_vec = model(torch.tensor([[x_inp, omega]]).float()).detach().numpy().flatten()
            T_heatmap[i, j] = T_R_vec[0]
            R_heatmap[i, j] = T_R_vec[1]
    # cmap = plt.get_cmap('viridis')
    # cmap = plt.get_cmap('Spectral')
    # cmap = plt.get_cmap('cool')
    # cmap = plt.get_cmap('gist_rainbow')
    cmap = parula
    from matplotlib.colors import ListedColormap

    # cmap = plt.get_cmap('jet')
    # norm = colors.PowerNorm(gamma=0.5)  # Change gamma to adjust how fast the colormap changes

    axes[0].imshow(T_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    # axes[0].minorticks_on()

    # axes[0].set_xlabel(r'$\beta t$')
    # axes[0].set_ylabel(r'$\omega$')
    # axes[0].set_title('Heatmap of T', fontsize=16)

    axes[1].imshow(R_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    # axes[1].set_xlabel(r'$\beta t$')
    # axes[1].set_ylabel(r'$\omega$')

    # axes[1].set_xlabel(r'$\beta t$')
    # axes[1].set_ylabel(r'$\omega$')
    # axes[1].set_title('Heatmap of R', fontsize=16)
    # axes[1].set_yticklabels([])
    # axes[1].minorticks_on()
    # plt.suptitle(f'{phase_type}', fontsize=20)

    # np.save('plot_data/T_R_heatmap.npy', [T_heatmap, R_heatmap])


def T_R_versus_xmax(ax, use_log_model=False):
    # we restrict omega to 1 (or 0.5), and change betat
    # from NN import ANN, Simple_NN
    # model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('model_2/rayleigh/WithPINN/epoch_1940.pth'))
    # for (i, xmax) in enumerate(xmaxs:=)
    from NN import ANN_Bigger, Simple_NN
    # model = ANN_Bigger()
    model = Simple_NN(num_features=2)
    model2 = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('forward_models/model_ANN_Bigger/hg/SimpleNN_log_model/epoch_100.pth'))
    # model.load_state_dict(torch.load('forward_models/model_Resnet/iso/WithBound_log_model/epoch_380.pth'))
    # model.load_state_dict(torch.load('forward_models/model_Resnet/iso/SimpleNN_log_model/epoch_380.pth'))
    model.load_state_dict(torch.load('forward_models/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))

    xmaxs = torch.logspace(np.log10(1e-3), np.log10(1e3), 100)

    omegas = torch.ones_like(xmaxs) * 0.5
    if use_log_model:
        xmaxs_inp = torch.log(xmaxs)
    else:
        xmaxs_inp = xmaxs.clone()
    inp = torch.stack([xmaxs_inp, omegas], dim=1)
    # print(inp)
    # quit()
    a = model(inp.float()).detach().numpy().T
    b = model2(inp.float()).detach().numpy().T
    T_iso_05_ml, R_iso_05_ml = a[0], a[1]
    T_hg_05_ml, R_hg_05_ml = b[0], b[1]

    omegas = torch.ones_like(xmaxs) * 0.95
    if use_log_model:
        xmaxs_inp = torch.log(xmaxs)
    else:
        xmaxs_inp = xmaxs.clone()
    inp = torch.stack([xmaxs_inp, omegas], dim=1)
    # print(inp)
    # quit()
    a = model(inp.float()).detach().numpy().T
    b = model2(inp.float()).detach().numpy().T
    T_iso_95_ml, R_iso_95_ml = a[0], a[1]
    T_hg_95_ml, R_hg_95_ml = b[0], b[1]
    RTE_T = np.load('plot_data/stored_T.npy')
    # w=0.5
    T_iso_05, T_ray_05, T_hg_05 = RTE_T[0], RTE_T[2], RTE_T[4]
    # w=1
    T_iso_95, T_ray_95, T_hg_95 = RTE_T[1], RTE_T[3], RTE_T[5]
    # T_iso_05, T_ray_05, T_hg_05 = RTE_T[1], RTE_T[3], RTE_T[5]

    ms = 8
    lw = 2

    # Plotting the lines and scatters with harmonious but distinct colors
    xmaxs = xmaxs.numpy()

    ax.semilogx(xmaxs, T_iso_05, '--', label='RTE Iso $T_{total}$', lw=lw, color='#1f77b4')  # Blue line
    ax.semilogx(xmaxs, T_iso_05_ml, 'o', label='ML\u00A0\u00A0\u00A0Iso $T_{total}$', ms=ms,
                color='#4a90e2', markerfacecolor='none', markeredgecolor='#4a90e2')  # Lighter blue hollow points

    # Orange line and hollow points
    ax.semilogx(xmaxs, T_iso_05 - np.exp(-xmaxs), '--', label='RTE Iso $T_{diffuse}$', lw=lw,
                color='#ff7f0e')  # Orange line
    ax.semilogx(xmaxs, T_iso_05_ml - np.exp(-xmaxs), 'o', label='ML\u00A0\u00A0\u00A0Iso\u00A0$T_{diffuse}$', ms=ms,
                color='#ffad66', markerfacecolor='none', markeredgecolor='#ffad66')  # Lighter orange hollow points

    # Green line and hollow points
    ax.semilogx(xmaxs, T_hg_95, '--', label='RTE HG $T_{total}$', lw=lw, color='#660077')  # purple line
    ax.semilogx(xmaxs, T_hg_95_ml, 'o', label='ML\u00A0\u00A0\u00A0HG $T_{total}$', ms=ms,
                color='#66cc66', markerfacecolor='none', markeredgecolor='#c94cbe')  # Lighter purple hollow points

    # Red line and hollow points
    ax.semilogx(xmaxs, T_hg_95 - np.exp(-xmaxs), '--', label='RTE HG $T_{diffuse}$', lw=lw,
                color='#d62728')  # Red line
    ax.semilogx(xmaxs, T_hg_95_ml - np.exp(-xmaxs), 'o', label='ML\u00A0\u00A0\u00A0HG $T_{diffuse}$', ms=ms,
                color='#ff6666', markerfacecolor='none', markeredgecolor='#ff6666')  # Lighter red hollow points
    # ax.semilogx(xmaxs, np.exp(-xmaxs), '--', color='gray', label='$T_{direct}$', lw=2)

    # plt.semilogx(xmaxs, R, label='R')
    # ax.legend()


def T_R_versus_xmax_2(ax, use_log_model=False):
    def samples(*args):
        sampled = []
        for arg in args:
            if len(arg.shape) == 1:
                sampled.append(np.hstack([arg[:20:3], arg[20:80:2], arg[80::3]]))
            else:
                sampled.append(np.hstack([arg[:, :20:3], arg[:, 20:80:2], arg[:, 80::3]]))
        return sampled
    # we restrict omega to 1 (or 0.5), and change betat
    # from NN import ANN, Simple_NN
    # model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('model_2/rayleigh/WithPINN/epoch_1940.pth'))
    # for (i, xmax) in enumerate(xmaxs:=)
    from NN import ANN_Bigger, Simple_NN
    # model = ANN_Bigger()
    model = Simple_NN(num_features=2)
    model2 = Simple_NN(num_features=2)
  
    model.load_state_dict(torch.load('forward_models\epoch_380_.pth'))
    model2.load_state_dict(torch.load('forward_models\model_Resnet_unif_5\hg\WithBound_log_model\epoch_380.pth'))

    xmaxs = torch.logspace(np.log10(1e-3), np.log10(1e3), 100)

    omegas = torch.ones_like(xmaxs) * 0.5
    if use_log_model:
        xmaxs_inp = torch.log(xmaxs)
    else:
        xmaxs_inp = xmaxs.clone()
    inp = torch.stack([xmaxs_inp, omegas], dim=1)
    # print(inp)
    # quit()
    a = model(inp.float()).detach().numpy().T
    b = model2(inp.float()).detach().numpy().T
    a, b = samples(a, b)
    T_iso_05_ml, R_iso_05_ml = a[0], a[1]
    T_hg_05_ml, R_hg_05_ml = b[0], b[1]

    omegas = torch.ones_like(xmaxs) * 0.95
    if use_log_model:
        xmaxs_inp = torch.log(xmaxs)
    else:
        xmaxs_inp = xmaxs.clone()
    inp = torch.stack([xmaxs_inp, omegas], dim=1)
    # print(inp)
    # quit()
    a = model(inp.float()).detach().numpy().T
    b = model2(inp.float()).detach().numpy().T

    xmaxs_ml = xmaxs.numpy()

    a, b, xmaxs_ml = samples(a, b, xmaxs_ml)

    T_iso_95_ml, R_iso_95_ml = a[0], a[1]
    T_hg_95_ml, R_hg_95_ml = b[0], b[1]



    RTE_T = np.load('plot_data/stored_T.npy')
    # w=0.5
    T_iso_05, T_ray_05, T_hg_05 = RTE_T[0], RTE_T[2], RTE_T[4]
    # w=1
    T_iso_95, T_ray_95, T_hg_95 = RTE_T[1], RTE_T[3], RTE_T[5]
    # T_iso_05, T_ray_05, T_hg_05 = RTE_T[1], RTE_T[3], RTE_T[5]

    ms = 8
    lw = 2
    alpha = 0.8
    # Plotting the lines and scatters with harmonious but distinct colors
    xmaxs = xmaxs.numpy()



    ax.semilogx(xmaxs, T_iso_05, '--', label='RTE Iso $T_{total}$', lw=lw, color='tab:blue')
    ax.semilogx(xmaxs_ml, T_iso_05_ml, 'o', label='ML\u00A0\u00A0\u00A0Iso $T_{total}$', ms=ms,
                color='#4a90e2', markerfacecolor='none', markeredgecolor='tab:blue', alpha=alpha)  # Lighter blue hollow points

    # Orange line and hollow points
    ax.semilogx(xmaxs, T_iso_05 - np.exp(-xmaxs), '-', label='RTE Iso $T_{diffuse}$', lw=lw,
                color='tab:blue')
    ax.semilogx(xmaxs_ml, T_iso_05_ml - np.exp(-xmaxs_ml), 'o', label='ML\u00A0\u00A0\u00A0Iso\u00A0$T_{diffuse}$', ms=ms,
                color='#4a90e2', markerfacecolor='none', markeredgecolor='tab:blue', alpha=alpha)




    ax.semilogx(xmaxs, T_hg_95, '--', label='RTE HG $T_{total}$', lw=lw, color='tab:orange')
    ax.semilogx(xmaxs_ml, T_hg_95_ml, 'o', label='ML\u00A0\u00A0\u00A0HG $T_{total}$', ms=ms,
                color='#e8a64e', markerfacecolor='none', markeredgecolor='tab:orange', alpha=alpha)

    # Red line and hollow points
    ax.semilogx(xmaxs, T_hg_95 - np.exp(-xmaxs), '-', label='RTE HG $T_{diffuse}$', lw=lw,
                color='tab:orange')  # Red line
    ax.semilogx(xmaxs_ml, T_hg_95_ml - np.exp(-xmaxs_ml), 'o', label='ML\u00A0\u00A0\u00A0HG $T_{diffuse}$', ms=ms,
                color='#e8a64e', markerfacecolor='none', markeredgecolor='tab:orange', alpha=alpha)  #
    from matplotlib.lines import Line2D

    m1 = Line2D([0], [0], color='k', linestyle='--', lw=2)
    m2 = Line2D([0], [0], color='k', linestyle='-', lw=2)
    m3 = Line2D([0], [0], color='k', marker='o', markersize=10,
                       linestyle='None', fillstyle='none', markeredgewidth=2)
    # legned1 = ax.legend([m1, m2, m3], ['RTE $T_{Total}$', 'RTE $T_{Diffuse}$', 'ML'], loc='center left', fontsize=16)

    m4 = Line2D([0], [0], color='tab:blue', linestyle='-', lw=2)
    m5 = Line2D([0], [0], color='tab:orange', linestyle='-', lw=2)
    # legend2 = ax.legend([m4, m5], ['Iso, w=0.5', 'HG w=0.95'], loc='upper right', fontsize=16)
    # ax.add_artist(legned1)

    # plt.semilogx(xmaxs, R, label='R')
    # ax.legend()

def T_versus_wl(ax):
    # In this fig, we change wl, using the 3rd Aerogel sample
    from PartitalNN import PartialNN
    partial_model = PartialNN()
    partial_model.load_state_dict(
        torch.load('models\partial_inverse.pth'))
    aero = Aerogel_Sample(thickness_mm=5.26, density=293, optical_mean_r_nm=3.50, wavelength_nm=10, m0=1 - 2j)
    wavelst_nm = np.linspace(200, 1000, 100)
    rte = RTE(1, 1, 0.9, 5.26, 21, 21, 'hg', 0.0)
    T_total_rte = np.zeros_like(wavelst_nm)
    T_total_ml = np.zeros_like(wavelst_nm)
    T_diffuse_rte = np.zeros_like(wavelst_nm)
    T_diffuse_ml = np.zeros_like(wavelst_nm)
    datas = np.zeros([5, len(wavelst_nm)])
    datas = np.load('plot_data/real_pred.npy')
    datas = np.hstack([datas[:, :20:2], datas[:, 20::6]])
    # for (i, wl) in tqdm(enumerate(wavelst_nm), total=len(wavelst_nm)):
    #     aero.wavelength_nm = wl
    #     aero.build()
    #     xmax, omega, g = aero.to_opt_set()
    #
    #     # RTE
    #     rte.xmax = xmax
    #     rte.omega = omega
    #     rte.g = g
    #     rte.build()
    #     T_total_rte[i] = rte.hemi_props()[0]
    #     T_diffuse_rte[i] = T_total_rte[i] - np.exp(-xmax)
    #
    #     # Partial NN
    #     x = torch.tensor([[np.log(xmax)]]).float()
    #     T_total_ml[i] = partial_model(x).detach().numpy().flatten()
    #     T_diffuse_ml[i] = T_total_ml[i] - np.exp(-xmax)

    wavelst_nm = datas[0]
    T_total_rte = datas[1]
    T_diffuse_rte = datas[2]
    T_total_ml = datas[3]
    T_diffuse_ml = datas[4]
    lw = 2
    ms = 8
    # Plotting the lines and scatters with harmonious but distinct colors
    ax.plot(wavelst_nm, T_total_rte, '--', label='$T_{total}$ RTE', lw=lw, color='#4a90e2')  # Blue line
    ax.plot(wavelst_nm, T_total_ml, 'o', label='$T_{total}$ ML', ms=ms, color='#1f77b4',
            markerfacecolor='none', markeredgecolor='#1f77b4')  # Hollow blue points

    # Orange line and hollow orange points
    ax.plot(wavelst_nm, T_diffuse_rte, '-', label='$T_{diffuse}$ RTE', lw=lw, color='#ffad66')  # Orange line
    ax.plot(wavelst_nm, T_diffuse_ml, 'o', label='$T_{diffuse}$ ML', ms=ms, color='#ff7f0e',
            markerfacecolor='none', markeredgecolor='#ff7f0e')  # Hollow orange pointsLighter orange points
    # np.save('plot_data/real_pred.npy', datas)

    # ax.set_yticklabels([])

    m1 = Line2D([0], [0], color='k', linestyle='--', lw=2)
    m2 = Line2D([0], [0], color='k', linestyle='-', lw=2)
    m3 = Line2D([0], [0], color='k', marker='o', markersize=10,
                       linestyle='None', fillstyle='none', markeredgewidth=2)
    # ax.legend([m1, m2, m3], ['RTE $T_{Total}$', 'RTE $T_{Diffuse}$', 'ML'], loc='center right', fontsize=16)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('T')
    # plt.title('T ~ Wavelength for Existing Sample', fontsize=16)
    # plt.show()




# def set1():
#     fig = plt.figure(figsize=(12, 8))
#     gs = GridSpec(1, 8, figure=fig)
#     ax1 = fig.add_subplot(gs[0, :5])
#     ax2 = fig.add_subplot(gs[0, 5:8])
#     T_R_versus_xmax(ax1, use_log_model=True)
#     T_versus_wl(ax2)
#     plt.show()


def betat_heatmap(ax, num=20, keep_log=False):
    from InverseNN import Inverse_ANN, Simple_NN

    # model = Inverse_ANN()
    # model.load_state_dict(torch.load('inverse_model/model_ANN/isotropic/SimpleNN/epoch_1900.pth'))
    model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('inverse_model/model_Resnet/iso/SimpleNN/epoch_200.pth'))
    model.load_state_dict(torch.load('inverse_model/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))
    model.eval()
    heatmap = np.zeros([num_R := num, num_T := num])
    # The inverse model takes in the input of [T, R] and output [xmax, omega]
    for (j, R) in tqdm(enumerate(Rs := np.linspace(0, 0.5, num_R)), total=len(Rs)):
        for (i, T) in enumerate(Ts := np.linspace(0.5, 1, num_T)):
            ...
    #         # for (i, T) in enumerate(Ts := np.linspace(0.05, 1, num_T)):
    #         if T + R > 1 or T < 0.05 or R >= 0.5:
    #             heatmap[j, i] = np.nan
    #         else:
    #             x = torch.tensor([[T, R]]).float()
    #             pred = model(x).detach().numpy().flatten()
    #             xmax = pred[0]
    #             if keep_log:
    #                 heatmap[j, i] = xmax
    #             else:
    #                 heatmap[j, i] = np.exp(xmax)
    # cmap = plt.get_cmap('viridis')
    # cmap = plt.get_cmap('jet')
    # cmap = plt.get_cmap('rainbow')
    # np.save('plot_data/betat_versus_TR.npy', heatmap)
    heatmap = np.load('plot_data/betat_versus_TR.npy')
    from colormaps import parula
    cmap = parula
    cmap.set_bad(color='white')
    t = ax.imshow(heatmap, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap, aspect='auto', )
    cbar = plt.colorbar(t, ax=ax)
    cbar.ax.tick_params(labelsize=0)  # 设置刻度标签文字大小为0

    # cbar.ax.set_aspect(10)


def xmax_versus_T(ax):
    from PartitalNN import PartialNN
    from InverseNN import Simple_NN
    # Ts = np.linspace(1e-3, 1 - 1e-3, 20)
    data = np.load('plot_data/betat_versus_T.npy')
    # find where 0.05<T<0.95
    mask = (data[0] > 0.05) & (data[0] < 0.95)
    mask = (data[0] > 0.02)
    Ts = data[0][mask]
    # Ts = data[0]

    # model = PartialNN()
    model = Simple_NN(num_features=1, out_features=1)
    model2 = Simple_NN(num_features=2)

    # model.load_state_dict(torch.load(
    #     f'models/PartialNN_Inverse/log_xmax_model_fine1/epoch_1800.pth'))

    model.load_state_dict(torch.load('models/PartialNN_Inverse/log_xmax_model_Resnet_1/epoch_440.pth'))
    # model2.load_state_dict(torch.load(
    #     f'inverse_model/model_Resnet/iso/SimpleNN/epoch_360.pth'))
    model2.load_state_dict(torch.load('inverse_model/model_Resnet_3/iso/WithBound_log_model/epoch_380.pth'))

    xmaxs_from_opt = data[1][mask]
    xmaxs_from_opt_2 = np.zeros(len(Ts))
    rte_obj = RTE(1, 1, 0.9, 1, 11, 11, phase_type='iso', g=0)
    T_R_2 = np.vstack([Ts, 0.5 * (1 - Ts)])
    # firstly, for w=1, R = 1 - T

    a = model(torch.tensor(Ts).float().view(len(Ts), 1)).detach().numpy()
    # b = model2(torch.tensor(T_R_2.T).float()).detach().numpy()[:, 0]
    # print(b)
    # quit()
    # print(a)
    pred_xmaxs = np.exp(a).reshape(-1)
    # pred_xmaxs_2 = np.exp(b)
    # print(pred_xmaxs_2)
    ax.semilogy(Ts, xmaxs_from_opt, '--', label='OPT Optical Depth', lw=2)
    # plt.semilogy(Ts, xmaxs_from_opt_2, '--', label='OPT R=0.5(1-T)')
    sep = 12
    ratio1 = 1 / 25  # First ratio part
    ratio2 = 1 / 10  # Second ratio part
    amp = 3

    # Defining the split points
    a = int(ratio1 * len(Ts))
    b = int((ratio2) * len(Ts))
    c = int((1 - ratio2) * len(Ts))
    d = int((1 - ratio1) * len(Ts))

    # 5-step sampling
    Ts_ML = np.hstack([
        Ts[:a - sep:int(sep / 3)],  # Step 1
        Ts[a:b - sep:int(sep)],  # Step 2
        Ts[b:c:int(amp * sep)],  # Step 3
        Ts[c + sep:d:int(sep)],  # Step 4
        Ts[d + sep::int(sep / 2)]  # Step 5
    ])

    # Similarly for xmaxs
    xmaxs_from_ML = np.hstack([
        pred_xmaxs[:a - sep:int(sep / 3)],  # Step 1
        pred_xmaxs[a:b - sep:int(sep)],  # Step 2
        pred_xmaxs[b:c:int(amp * sep)],  # Step 3
        pred_xmaxs[c + sep:d:int(sep)],  # Step 4
        pred_xmaxs[d + sep::int(sep / 2)]  # Step 5
    ])

    ax.semilogy(Ts_ML, xmaxs_from_ML, '.', label='ML  Optical Depth', ms=20,
                color='#4a90e2', markerfacecolor='none', markeredgecolor='tab:orange')
    # ax.legend()
def inverse_efficiency(ax):
    data = np.load('benchmark_inverse_results_full.npy', allow_pickle=True).item()
    opt_whole_height = data['nm_mat_mean']
    opt_partial_height = data['ml_opt_mean']
    ml_whole_height = data['ml_mat_mean']
    minor_opt_height = data['nm_opt_mean']

    # Bar labels
    labels = ['Opt Micro', 'ML Micro', 'Opt Optical', 'ML Optical']

    # X locations for the bars
    x = np.arange(len(labels))

    # Width of the bars
    width = 0.5

    # Create the figure and axis

    # Plot the bars
    ax.bar(x[0], opt_whole_height, width, label='Opt Micro', color="#f3c212")
    ax.bar(x[1], opt_partial_height, width, label='Opt Optical ', color='#2dde98')
    ax.bar(x[2], minor_opt_height + ml_whole_height, width, label='ML Micro', color='tab:blue')
    ax.bar(x[3], ml_whole_height, width, label='ML Optical ', color='#ff6c5f')

    # Add labels, title, and legend
    # change scale of y to log
    ax.set_yscale('log')
    ax.set_ylim(top=1e5)
    ax.set_xticks(x)
    ax.set_yticklabels([])
    ax.set_xticklabels([])



def show_forward_eff_2(ax):
    # data = np.load('plot_data/forward_eff_cpu.npy')
    # data_gpu = np.load('plot_data/forward_eff_cuda_1.npy')[1:]
    # data_cpu = np.load('plot_data/forward_eff_cpu_1.npy')[1:]
    # work_load = np.logspace(1, 5, 30)[1:]
    # mean_cpu, std_cpu = np.mean(data_cpu, axis=1), np.std(data_cpu, axis=1)
    # mean_gpu, std_gpu = np.mean(data_gpu, axis=1), np.std(data_gpu, axis=1)
    # error_2sigma_cpu = std_cpu
    # error_2sigma_gpu = std_gpu
    # data_rte = np.load('benchmark_forward_results_with_std.npy', allow_pickle=True).item()
    # work_load_rte = np.array(data_rte['sizes'])
    # mean_rte_1 = np.array(data_rte['rte_times'])
    # error_2sigma_rte_1 = np.array(data_rte['rte_std'])

    data_rte = np.load('benchmark_forward_results_with_std_rte.npy', allow_pickle=True).item()
    work_load_rte = np.array(data_rte['sizes'])
    mean_rte_1 = np.array(data_rte['rte_times'])
    error_2sigma_rte_1 = np.array(data_rte['rte_std'])

    data_ML = np.load('benchmark_forward_results_with_std_ML.npy', allow_pickle=True).item()
    work_load = np.array(data_ML['sizes'])
    mean_cpu = np.array(data_ML['cpu_times'])
    error_2sigma_cpu = np.array(data_ML['cpu_std'])
    mean_gpu = np.array(data_ML['gpu_times'])
    error_2sigma_gpu = np.array(data_ML['gpu_std'])


    ax.semilogy(work_load, mean_cpu, '.-', label='CPU', lw=2, color='tab:orange')
    ax.fill_between(work_load, mean_cpu - error_2sigma_cpu, mean_cpu + error_2sigma_cpu, alpha=0.2, color='tab:orange')

    ax.semilogy(work_load, mean_gpu, '.-', label='GPU', lw=2, color='tab:green')
    ax.fill_between(work_load, mean_gpu - error_2sigma_gpu, mean_gpu + error_2sigma_gpu, alpha=0.2, color='tab:green')

    ax.semilogy(work_load_rte, mean_rte_1, '.-', label='RTE', lw=2, color='tab:blue')
    ax.fill_between(work_load_rte, mean_rte_1 - error_2sigma_rte_1, mean_rte_1 + error_2sigma_rte_1, alpha=0.2, color='tab:blue')
    ax.set_xscale('log')
    print(work_load[3:6])
    print(mean_rte_1[3:6])
    print(mean_gpu[3:6])

    # clear ticks
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # plt.legend()
    # plt.show()


def real_spectrum_fitting(axes):
    assert len(axes) == 2
    datas = np.load('plot_data/real_comparison_1.npy')
    datas = np.hstack([datas[:, :40:2], datas[:, 40::2]])
    # wl, opt_T_total, opt_T_dif, real_T, real_T_dif
    thick_rho_r_set = [(4.75, 144, 3.05), (5.25, 293, 3.50)]
    alpha = 1
    ms = 12
    lw = 2
    for (i_ax, i) in enumerate([0, 1]):
        thick_rho_r = thick_rho_r_set[i]
        wavelst_nm = datas[5 * i]
        opt_T_by_quasi_T = datas[5 * i + 1]
        opt_T_dif = datas[5 * i + 2]
        real_T = datas[5 * i + 3]
        real_T_dif = datas[5 * i + 4]
        axes[i_ax].plot(wavelst_nm, real_T, '--', label='Target $T_{Total}$', alpha=alpha, lw=lw)  # No marker

        axes[i_ax].plot(wavelst_nm, real_T_dif, '--', label='Target $T_{Diffuse}$', alpha=alpha, lw=lw)  # No marker

        axes[i_ax].plot(wavelst_nm, opt_T_by_quasi_T, '.', label='ML $T_{Total}$', alpha=alpha, ms=ms,
                     markerfacecolor='none', markeredgecolor='#248bcc')  # Hollow marker for ML $T_Total$

        axes[i_ax].plot(wavelst_nm, opt_T_dif, '.', label='ML $T_{Diffuse}$', alpha=alpha, ms=ms,
                     markerfacecolor='none', markeredgecolor='#ee8844')  # Hollow marker for ML $T_Diffuse$

        if i == 0:
            m1 = Line2D([0], [0], color='tab:blue', linestyle='--', lw=2)
            m2 = Line2D([0], [0], color='tab:orange', linestyle='--', lw=2)
            m3 = Line2D([0], [0], color='k', marker='o', markersize=10,
                        linestyle='None', fillstyle='none', markeredgewidth=2)
            # legned1 = axes[i_ax].legend([m1, m2, m3], ['Target $T_{Total}$', 'Target $T_{Diffuse}$', 'ML'], loc='center', fontsize=16)
            # axes[i].legend(fontsize=16)
        axes[i_ax].set_yticks(np.linspace(0, 1, 6))
        # if i > 0:
        #     axes[i].set_yticklabels([])
        # opt_quasi_T = quasi_T_func(wavelst_nm, 3, res.x, err=False)
    # fig.suptitle('Optimizing for An Existing Spectrum')
    # fig.tight_layout()


# real_spectrum_fitting()

def artificial_spectrum_fitting(axes):
    datas = np.load('plot_data/artificial_comparison .npy')
    for (i, wl_sep) in enumerate([500, 700]):
        wl = datas[i * 4]
        target_T = datas[i * 4 + 1]
        opt_T_by_quasi_T = datas[i * 4 + 2]
        opt_T = datas[i * 4 + 3]
        axes[i].plot(wl, target_T, '-', label='Target T', lw=2)
        axes[i].plot(wl, opt_T_by_quasi_T, '.-', label='ML  $T_{Total}$', alpha=0.8, lw=2, ms=8)
        axes[i].plot(wl, opt_T, '--', label='RTE $T_{total}$', alpha=0.8, lw=2)
        axes[i].set_yticks(np.linspace(0, 1, 6))
        # axes[i].set_xlabel('Wavelength (nm)')
        # axes[i].set_ylabel('T')
        # axes[i].set_title(f'Sep_wl: {wl_sep} nm', fontsize=16, loc='left')
        if i == 0:
            # axes[i].legend(fontsize=16)
            pass
        # if i > 0:
        #     axes[i].set_yticklabels([])


# def set3():
#     fig, axes = plt.subplots(2, 3)
#     fig = plt.figure()
#     gs = GridSpec(9, 3, figure=fig)
#
#     ax1 = fig.add_subplot(gs[0:3, 0])
#     ax2 = fig.add_subplot(gs[0:3, 1])
#     ax3 = fig.add_subplot(gs[0:3, 2])
#     axes_1 = [ax1, ax2, ax3]
#
#     ax4 = fig.add_subplot(gs[5:, 0])
#     ax5 = fig.add_subplot(gs[5:, 1])
#     ax6 = fig.add_subplot(gs[5:, 2])
#     axes_2 = [ax4, ax5, ax6]
#     real_spectrum_fitting(axes_1)
#     artificial_spectrum_fitting(axes_2)
#     # real_spectrum_fitting(axes[0])
#     # artificial_spectrum_fitting(axes[1])
#     plt.show()


def fwd_set():
    # AllinONe Fig of Forward
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 8, figure=fig)
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 4:7])
    # cax = fig.add_subplot(gs[0, 6])
    axes1 = [ax1, ax2]
    T_R_Heatmap(axes1, phase_type='iso')
    ax3 = fig.add_subplot(gs[1:2, :4])
    ax4 = fig.add_subplot(gs[1:2, 5:])
    T_R_versus_xmax_2(ax3, use_log_model=True)
    T_versus_wl(ax4)
    ax5 = fig.add_subplot(gs[2:, :])
    # show_fwd_eff(ax5)
    show_forward_eff_2(ax5)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    # fig.colorbar(plt.cm.ScalarMappable(cmap=parula), ax=axes1)

    plt.show()



def bwd_set1():
    fig = plt.figure()
    gs = GridSpec(21, 6, figure=fig)

    r = 30
    ax1 = fig.add_subplot(gs[:8, :2])
    ax2 = fig.add_subplot(gs[9:20, :2])
    ax_eff = fig.add_subplot(gs[14:20, 2:])
    betat_heatmap(ax1, num=200, keep_log=False)
    xmax_versus_T(ax2)
    inverse_efficiency(ax_eff)
    ax3 = fig.add_subplot(gs[0:6  , 2:4])
    ax4 = fig.add_subplot(gs[7:13  , 2:4])
    # ax5 = fig.add_subplot(gs[14:20, 2:4])
    ax6 = fig.add_subplot(gs[0:6  , 4:6])
    ax7 = fig.add_subplot(gs[7:13 , 4:6])
    ax3, ax7 = ax7, ax3
    # ax8 = fig.add_subplot(gs[14:20, 4:6])
    real_spectrum_fitting([ax7, ax6, ])
    artificial_spectrum_fitting([ ax4, ax3,])
    # fig.colorbar(plt.cm.ScalarMappable(cmap=parula, norm=None), ax=ax1)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=parula), ax=ax1, )
    # plt.tight_layout()
    for ax in [ax1, ax2, ax3, ax4, ax6, ax7, ax_eff]:
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelleft=False)
    plt.show()

params = {'legend.fontsize': 16,

          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
}
pylab.rcParams.update(params)
plt.rcParams['savefig.directory'] = r'C:\summerintern\Paper\pic2'
# fwd_set()
# bwd_set1()
# T_R_Heatmap()
fig, ax = plt.subplots()
# show_forward_eff_2(ax)
inverse_efficiency(ax)
plt.show()