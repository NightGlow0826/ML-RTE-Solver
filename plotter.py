
import time
import scicomap as scm  # Import the scicomap package
from colormaps import parula

from matplotlib import colors
import viscm
# This file is used to plot the figures needed in the paper.

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from Mie import Aerogel_Sample
from RTE_Truth_Model import RTE
from scipy.optimize import minimize

import matplotlib


### Figures in Forward Process
def T_R_Heatmap(phase_type='iso'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    from NN import  Simple_NN
    # model = ANN()
    model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('model_2/rayleigh/WithPINN/epoch_1940.pth'))

    # model.load_state_dict(torch.load('forward_models/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))
    model.load_state_dict(torch.load(f'forward_models/model_Resnet_2/{phase_type}/WithBound_log_model/epoch_380.pth'))
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
    norm = colors.PowerNorm(gamma=0.5)  # Change gamma to adjust how fast the colormap changes


    axes[0].imshow(T_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    axes[0].minorticks_on()

    axes[0].set_xlabel(r'$\beta t$')
    axes[0].set_ylabel(r'$\omega$')
    axes[0].set_title('Heatmap of T', fontsize=16)

    axes[1].imshow(R_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    axes[1].set_xlabel(r'$\beta t$')
    axes[1].set_ylabel(r'$\omega$')

    axes[1].set_xlabel(r'$\beta t$')
    axes[1].set_ylabel(r'$\omega$')
    axes[1].set_title('Heatmap of R', fontsize=16)
    axes[1].set_yticklabels([])
    axes[1].minorticks_on()
    plt.suptitle(f'{phase_type}', fontsize=20)
    # np.save('plot_data/T_R_heatmap.npy', [T_heatmap, R_heatmap])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes.ravel().tolist())
    # plt.colorbar()
    plt.show()



def T_R_Heatmap_2(phase_type='iso'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    from NN import Simple_NN
    # model = ANN()
    model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('model_2/rayleigh/WithPINN/epoch_1940.pth'))

    # model.load_state_dict(torch.load('forward_models/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))
    model.load_state_dict(torch.load(f'forward_models/model_Resnet_2/{phase_type}/WithBound_log_model/epoch_380.pth'))
    T_heatmap = np.zeros([num_omega := 100, num_xmax := 100])
    R_heatmap = np.zeros([num_omega, num_xmax])
    for (j, xmax) in enumerate(xmaxs := np.logspace(np.log10(1e-3), np.log10(1e3), num_xmax)):
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
    norm = colors.PowerNorm(gamma=0.5)  # Change gamma to adjust how fast the colormap changes


    axes[0].imshow(T_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    axes[0].minorticks_on()

    axes[0].set_xlabel(r'$\beta t$')
    axes[0].set_ylabel(r'$\omega$')
    axes[0].set_title('Heatmap of T', fontsize=16)
    # axes[0].set_xscale('log')
    axes[1].imshow(R_heatmap, origin='lower', extent=(xmaxs[0], xmaxs[-1], omegas[0], omegas[-1]), cmap=cmap,
                   aspect='auto', vmax=1, vmin=0, interpolation='bilinear')
    axes[1].set_xlabel(r'$\beta t$')
    axes[1].set_ylabel(r'$\omega$')

    axes[1].set_xlabel(r'$\beta t$')
    axes[1].set_ylabel(r'$\omega$')
    axes[1].set_title('Heatmap of R', fontsize=16)
    axes[1].set_yticklabels([])
    axes[1].minorticks_on()
    plt.suptitle(f'{phase_type}', fontsize=20)
    # np.save('plot_data/T_R_heatmap.npy', [T_heatmap, R_heatmap])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes.ravel().tolist())
    # plt.colorbar()
    plt.show()


def store_RTE_T_R_versus_xmax():
    xmaxs = torch.logspace(np.log10(1e-3), np.log10(1e3), 100)
    # w = 0.5 and 1, and 3 phasefuncs,
    stored_T = np.zeros([6, len(xmaxs)])
    rte = RTE(1, 1, 0.9, 5.26, 21, 21, 'iso', 0.5)
    for (j, phase_type) in enumerate(['iso', 'ray', 'hg']):
        rte.phase_type = phase_type
        for (i, xmax) in tqdm(enumerate(xmaxs), total=len(xmaxs)):
            rte.xmax = xmax
            rte.omega = 0.5
            rte.build()
            stored_T[2 * j, i] = rte.hemi_props()[0]
            rte.omega = 0.95
            rte.build()
            stored_T[2 * j + 1, i] = rte.hemi_props()[0]

    np.save('plot_data/stored_T.npy', stored_T)


# store_RTE_T_R_versus_xmax()

def T_R_versus_xmax(use_log_model=False):

    from NN import ANN_Bigger, Simple_NN
    # model = ANN_Bigger()
    model = Simple_NN(num_features=2)
    model2 = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('forward_models/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))
    model.load_state_dict(torch.load('forward_models/model_Resnet_unif/iso/WithBound_log_model/epoch_380.pth'))

    # model2.load_state_dict(torch.load('forward_models/model_Resnet_2/hg/WithBound_log_model/epoch_340.pth'))
    model2.load_state_dict(torch.load('forward_models/model_Resnet_unif_2/hg/WithBound_log_model/epoch_340.pth'))

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

    plt.semilogx(xmaxs, T_iso_05, '--', label='RTE Iso $T_{total}$', lw=lw, color='#1f77b4')  # Blue line
    plt.semilogx(xmaxs, T_iso_05_ml, 'o', label='ML\u00A0\u00A0\u00A0Iso $T_{total}$', ms=ms,
                 color='#4a90e2', markerfacecolor='none', markeredgecolor='#4a90e2')  # Lighter blue hollow points

    # Orange line and hollow points
    plt.semilogx(xmaxs, T_iso_05 - np.exp(-xmaxs), '--', label='RTE Iso $T_{diffuse}$', lw=lw,
                 color='#ff7f0e')  # Orange line
    plt.semilogx(xmaxs, T_iso_05_ml - np.exp(-xmaxs), 'o', label='ML\u00A0\u00A0\u00A0Iso\u00A0$T_{diffuse}$', ms=ms,
                 color='#ffad66', markerfacecolor='none', markeredgecolor='#ffad66')  # Lighter orange hollow points

    # Green line and hollow points
    plt.semilogx(xmaxs, T_hg_95, '--', label='RTE HG $T_{total}$', lw=lw, color='#660077')  # purple line
    plt.semilogx(xmaxs, T_hg_95_ml, 'o', label='ML\u00A0\u00A0\u00A0HG $T_{total}$', ms=ms,
                 color='#66cc66', markerfacecolor='none', markeredgecolor='#c94cbe')  # Lighter purple hollow points

    # Red line and hollow points
    plt.semilogx(xmaxs, T_hg_95 - np.exp(-xmaxs), '--', label='RTE HG $T_{diffuse}$', lw=lw,
                 color='#d62728')  # Red line
    plt.semilogx(xmaxs, T_hg_95_ml - np.exp(-xmaxs), 'o', label='ML\u00A0\u00A0\u00A0HG $T_{diffuse}$', ms=ms,
                 color='#ff6666', markerfacecolor='none', markeredgecolor='#ff6666')  # Lighter red hollow points
    plt.semilogx(xmaxs, np.exp(-xmaxs), '--', color='gray', label='$T_{direct}$', lw=2)

    # plt.semilogx(xmaxs, R, label='R')
    plt.legend()
    plt.xlabel('Optical Depth')
    plt.ylabel('T')
    plt.title('T ~ Optical Depth', fontsize=16)
    plt.show()


# T_R_versus_xmax(use_log_model=True)

def rte_forward_efficiency(num=100):
    # Use Cprofile to check the efficiency of the forward model
    rte = RTE(1, 1, 0.9, 5.26, 51, 21, 'hg', 0.5)

    def rte_taker(rte_obj):
        # Takes a initialized RTE object, and compute to time the efficiency
        return rte_obj.hemi_props()

    rte.phase_type = 'hg'
    start = time.perf_counter()
    for i in trange(num):
        # Find the time consumption of the RTE model
        rte.xmax = np.random.uniform(0.01, 5)
        rte.omega = np.random.uniform(0, 1)

        rte.build()
        rte.hemi_props()

    end = time.perf_counter()
    print(f"hg: {(end - start) * 1000:.0f}ms on {num} samples")

    # iso: 17648ms on 100 samples,11742 on full power
    # ray: 18059ms on 100 samples,13577 on full power
    # hg: 17878ms on 100 samples, 14843 on full power


# rte_forward_efficiency()
def pred_real_spectrum():
    from spectrum import real_spectrum
    wavelst_nm = np.linspace(200, 1000, 50)
    T_set = real_spectrum(wavelst_nm=wavelst_nm,
                          thick_rho_r=(4.75, 144, 3.05))
    opt_set = real_spectrum(wavelst_nm=wavelst_nm,
                            thick_rho_r=(4.75, 144, 3.05), opt_prop_only=True)
    xmax = opt_set[0]
    inp = opt_set[:2].T
    inp[:, 0] = np.log(inp[:, 0])

    inp = torch.from_numpy(inp.T)


def ML_forward_efficiency(num_samples=100, device='cuda'):
    from NN import ANN_Bigger
    def ffffffffffffffffffffffffffffffffwd(model, inp):
        return model(inp)

    # device = 'cpu'
    device = device
    length = num_samples
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inp = torch.randn(length, 2).to(device)
    model = ANN_Bigger().to(device)
    model.eval()

    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # Start GPU timing
        ffffffffffffffffffffffffffffffffwd(model, inp)
        end_event.record()  # End GPU timing
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event)
        # print(f"GPU Inference Time: {gpu_time:.3f} ms")
        print(f'{num_samples:6d}\t{gpu_time:.3f}')
    else:
        time1 = time.perf_counter()
        ffffffffffffffffffffffffffffffffwd(model, inp)
        time2 = time.perf_counter()
        # print(f"CPU Inference Time: {(time2 - time1) * 1000:.3f} ms")
        #make the numsample take 6 digits place
        print(f'{num_samples:6d}\t{(time2 - time1) * 1000:.3f}')

    # CPU 492ms, with regular power supply on 1e5 samples, 320ms with full power
    # GPU, 86ms with regular power supply on 1e5 samples, 84ms with full power


# ML_forward_efficiency(1000, device='cpu')
def show_fwd_eff():
    import numpy as np
    import matplotlib.pyplot as plt

    # Data from the table
    num = np.array([1, 5, 31, 177, 1000])
    time1 = np.array([335, 878, 5874, 36942, 204460])
    time2 = np.array([217, 1164, 6139, 34787, 196914])
    time3 = np.array([312, 1105, 6007, 35472, 198102])

    # Calculate the mean and standard deviation
    mean_time_1 = np.mean([time1, time2, time3], axis=0)
    std_time_1 = np.std([time1, time2, time3], axis=0)

    error_2sigma_1 = 2 * std_time_1
    plt.semilogy(num, mean_time_1, label='RTE', lw=2, color='tab:blue')
    plt.fill_between(num, mean_time_1 - error_2sigma_1, mean_time_1 + error_2sigma_1, alpha=0.2, color='tab:blue')

    # Updated data including the new entries
    num_ = np.array([10, 27, 77, 215, 599, 1668, 4641, 12915, 35938, 100000])
    time1_ = np.array(
        [5.389, 1.963, 3.971, 4.873, 6.387, 10.609, 23.367, 59.332, 140.545, 431.299])
    time2_ = np.array(
        [5.291, 2.175, 2.744, 4.942, 6.811, 10.43, 23.579, 54.62, 140.253, 408.365])
    time3_ = np.array(
        [5.864, 3.62, 3.098, 4.094, 6.727, 10.893, 23.278, 54.607, 138.951, 397.432])

    mean_time_1 = np.mean([time1_, time2_, time3_], axis=0)
    std_time_1 = np.std([time1_, time2_, time3_], axis=0)
    error_2sigma_1 = 2 * std_time_1
    plt.semilogy(num_, mean_time_1, label='ML CPU', lw=2, color='tab:orange')

    # Plot the error boundary using fill_between
    plt.fill_between(num_, mean_time_1 - error_2sigma_1, mean_time_1 + error_2sigma_1,
                     color='tab:orange', alpha=0.2)

    # Labeling and legend

    num = np.array([10, 27, 77, 215, 599, 1668, 4641, 12915, 35938, 100000])
    time1 = np.array([0.07, 4.499, 0.236, 0.204, 0.263, 3.121, 1.415, 6.494, 12.988, 33.699])
    time2 = np.array([0.073, 4.464, 0.195, 0.193, 0.264, 1.855, 1.400, 6.824, 12.867, 33.957])
    time3 = np.array([0.074, 3.187, 0.298, 0.196, 0.264, 3.568, 1.403, 6.552, 10.087, 30.634])

    # Calculate the mean and standard deviation for each row
    mean_time = np.mean([time1, time2, time3], axis=0)
    std_time = np.std([time1, time2, time3], axis=0)

    # Calculate the 2-sigma error (2 * standard deviation)
    error_2sigma = 2 * std_time

    # Plot the mean line
    plt.plot(num, mean_time, label='ML GPU', lw=2, color='tab:green')

    # Plot the error boundary using fill_between
    plt.fill_between(num, mean_time - error_2sigma, mean_time + error_2sigma,
                     color='tab:green', alpha=0.2)
    plt.scatter(2500, 1000)
    plt.scatter(25000, 10000)
    # Add labels and legend
    plt.xlabel('Workload Size')
    plt.ylabel('Time (ms)')
    plt.title('Computational Efficiency of the Forward Model', fontsize=16)
    plt.xscale('log')  # Logarithmic scale for num
    plt.legend()

    # Display the plot
    plt.show()


# TODO: MAY ADD A MODEL FOR LOGSPCED XMAX

def T_versus_wl():
    # In this fig, we change wl, using the 3rd Aerogel sample
    from PartitalNN import PartialNN
    from NN import Simple_NN
    simple_model = Simple_NN(num_features=2)
    partial_model = PartialNN()
    partial_model.load_state_dict(
        torch.load('models/PartialNN/log_xmax_model_fine1/epoch_300.pth'))
    simple_model.load_state_dict(torch.load('forward_models/model_Resnet_unif/iso/WithBound_log_model/epoch_380.pth'))
    aero = Aerogel_Sample(thickness_mm=5.26, density=293, optical_mean_r_nm=3.50, wavelength_nm=10, m0=1 - 2j)
    wavelst_nm = np.linspace(200, 1000, 100)
    rte = RTE(1, 1, 0.9, 5.26, 21, 21, 'hg', 0.0)
    datas = np.load('plot_data/real_pred.npy')
    datas = np.hstack([datas[:, :20], datas[:, 20::3]])
    wavelst_nm = datas[0]
    T_total_rte = datas[1]
    T_diffuse_rte = datas[2]
    # T_total_rte = np.zeros_like(wavelst_nm)
    T_total_ml = np.zeros_like(wavelst_nm)
    # T_diffuse_rte = np.zeros_like(wavelst_nm)
    T_diffuse_ml = np.zeros_like(wavelst_nm)
    # datas = np.zeros([5, len(wavelst_nm)])

    for (i, wl) in enumerate(wavelst_nm):
        aero.wavelength_nm = wl
        aero.build()
        aero.build_m()
        # aero.to_opt_set()
        xmax, omega, g = aero.to_opt_set()
        x = torch.tensor([[np.log(xmax), omega]]).float()
        T_R_vec = simple_model(x).detach().numpy().flatten()
        T_total_ml[i] = T_R_vec[0]
        T_diffuse_ml[i] = T_R_vec[0] - np.exp(-xmax)


    # T_total_ml = datas[3]
    # T_diffuse_ml = datas[4]
    lw = 2
    ms = 8
    # Plotting the lines and scatters with harmonious but distinct colors
    plt.plot(wavelst_nm, T_total_rte, '--', label='$T_{total}$ RTE', lw=lw, color='#4a90e2')  # Blue line
    plt.plot(wavelst_nm, T_total_ml, 'o', label='$T_{total}$ ML', ms=ms, color='#1f77b4',
             markerfacecolor='none', markeredgecolor='#1f77b4')  # Hollow blue points

    # Orange line and hollow orange points
    plt.plot(wavelst_nm, T_diffuse_rte, '--', label='$T_{diffuse}$ RTE', lw=lw, color='#ffad66')  # Orange line
    plt.plot(wavelst_nm, T_diffuse_ml, 'o', label='$T_{diffuse}$ ML', ms=ms, color='#ff7f0e',
             markerfacecolor='none', markeredgecolor='#ff7f0e')  # Hollow orange pointsLighter orange points
    # np.save('plot_data/real_pred.npy', datas)

    plt.legend(loc='right')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('T')
    plt.title('T ~ Wavelength for Existing Sample', fontsize=16)
    plt.show()


# T_versus_wl()

##############################
##############################
### Figures in Inverse Proces

# Heatmap of R and T
def betat_heatmap(num=20, keep_log=False):
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
        # for (i, T) in enumerate(Ts := np.linspace(0.05, 1, num_T)):
            if T + R > 1 or T < 0.05 or R >= 0.5:
                heatmap[j, i] = np.nan
            else:
                x = torch.tensor([[T, R]]).float()
                pred = model(x).detach().numpy().flatten()
                xmax = pred[0]
                if keep_log:
                    heatmap[j, i] = xmax
                else:
                    heatmap[j, i] = np.exp(xmax)
    cmap = plt.get_cmap('viridis')
    cmap = parula
    # cmap = plt.get_cmap('jet')
    # cmap = plt.get_cmap('rainbow')

    cmap.set_bad(color='white')
    plt.imshow(heatmap, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap)
    plt.colorbar()
    # plt.xlabel('T')
    # plt.ylabel('R')
    # plt.title('Heatmap of xmax')
    plt.show()

def betat_omega_heatmap(num=20, keep_log=False):
    from InverseNN import Inverse_ANN, Simple_NN

    # model = Inverse_ANN()
    # model.load_state_dict(torch.load('inverse_model/model_ANN/isotropic/SimpleNN/epoch_1900.pth'))
    model = Simple_NN(num_features=2)
    # model.load_state_dict(torch.load('inverse_model/model_Resnet/iso/SimpleNN/epoch_200.pth'))
    model.load_state_dict(torch.load('inverse_model/model_Resnet_2/iso/WithBound_log_model/epoch_380.pth'))
    model.eval()
    heatmap = np.zeros([num_R := num, num_T := num])
    heatmap1 = np.zeros([num_R := num, num_T := num])
    # The inverse model takes in the input of [T, R] and output [xmax, omega]
    for (j, R) in tqdm(enumerate(Rs := np.linspace(0, 0.5, num_R)), total=len(Rs)):
        for (i, T) in enumerate(Ts := np.linspace(0.5, 1, num_T)):
        # for (i, T) in enumerate(Ts := np.linspace(0.05, 1, num_T)):
            if T + R > 1 or T < 0.05 or R >= 0.5:
                heatmap[j, i] = np.nan
                heatmap1[j, i] = np.nan

            else:
                x = torch.tensor([[T, R]]).float()
                pred = model(x).detach().numpy().flatten()
                xmax = pred[0]
                albedo = pred[1]
                if keep_log:
                    heatmap[j, i] = xmax
                    heatmap1[j, i] = albedo

                else:
                    heatmap[j, i] = np.exp(xmax)
                    heatmap1[j, i] = albedo

    cmap = parula
    # cmap = plt.get_cmap('jet')
    # cmap = plt.get_cmap('rainbow')

    cmap.set_bad(color='white')
    # plt.imshow(heatmap, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(heatmap, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap)
    im2 = axes[1].imshow(heatmap1, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap)
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar2 = fig.colorbar(im2, ax=axes[1])
    # axes[0].set_xticklabels('')b
    # plt.xlabel('T')
    # plt.ylabel('R')
    # plt.title('Heatmap of xmax')
    plt.show()


# betat_heatmap()

def betat_heatmap_rte(num=20):
    from joblib import Parallel, delayed

    from InverseNN import Inverse_ANN, Simple_NN
    # model = Inverse_ANN()
    # model.load_state_dict(torch.load('inverse_model/model_ANN/isotropic/SimpleNN/epoch_1900.pth'))

    heatmap = np.zeros([num_R := num, num_T := num])
    # The inverse model takes in the input of [T, R] and output [xmax, omega]
    # for (j, R) in tqdm(enumerate(Rs:=np.linspace(0.02, 1, num_R)), total=num_R):
    #     for (i, T) in enumerate(Ts:=np.linspace(0.02, 1, num_T)):
    #         if T+R > 1:
    #             heatmap[j, i] = np.nan
    #         else:
    #             rte = RTE(T, R, 0.9, 5.26, 21, 21, 'iso', 0.5)
    #             rte.opt_inverse_problem(T, R)
    #             heatmap[j, i] = rte.xmax

    # Initialize heatmap
    heatmap = np.zeros([num_R, num_T])

    # Generate the Rs and Ts arrays
    Rs = np.linspace(0.02, 1, num_R)
    Ts = np.linspace(0.02, 1, num_T)

    # Define the function to compute each entry in the heatmap for a specific (T, R) pair
    def compute_heatmap_entry(i, T, R):
        if T + R > 1 or T < 0.2 or R > 0.3:
            return np.nan  # Return NaN if the condition is met
        else:
            # Create the RTE model and solve the inverse problem
            rte = RTE(0.5, 1, 0.9, 5, 21, 21, 'iso')
            # rte.opt_inverse_problem(T, R, tol=1e-4, force=True)
            rte.opt_inverse_problem_logmodel(T, R, tol=1e-4)
            return rte.xmax  # Return the computed value

    # Parallelize the inner loop
    for j, R in tqdm(enumerate(Rs), total=num_R):
        # Parallel computation for the inner loop (over T values)
        results = Parallel(n_jobs=-1)(delayed(compute_heatmap_entry)(i, T, R) for i, T in enumerate(Ts))

        # Store the results in the heatmap
        heatmap[j, :] = np.log(results)
    np.save('plot_data/heatmap.npy', heatmap)
    cmap = plt.get_cmap('gist_rainbow')
    cmap = parula
    cmap.set_bad(color='white')
    plt.imshow(heatmap, origin='lower', extent=(Ts[0], Ts[-1], Rs[0], Rs[-1]), cmap=cmap)
    plt.colorbar()
    plt.xlabel('T')
    plt.ylabel('R')
    plt.title('Heatmap of xmax')
    plt.show()


# betat_heatmap(num=20)
# betat_heatmap_rte(20)

def store_betat_versus_T(num=20):
    from joblib import Parallel, delayed

    Ts = np.linspace(0.01, 0.99, num)
    rte = RTE(0, 1, 0.99, 1, 41, 41, phase_type='iso', g=0.5)
    xmaxs_opt = np.zeros_like(Ts)

    # for (i, T) in tqdm(enumerate(Ts), total=len(Ts)):
    #     rte.opt_inverse_problem(T, 1 - T, 0.01)
    #     xmaxs_opt[i] = rte.xmax

    def process_task(i, T):
        rte = RTE(0, 1, 0.99, 1, 41, 41, phase_type='iso', g=0.5)
        print(i)
        # rte.opt_inverse_problem(T, 1 - T, 0.01)
        rte.opt_inverse_problem_logmodel(T, 1 - T, 0.01)
        print(i)
        return (i, rte.xmax)

    results = Parallel(n_jobs=5)(delayed(process_task)(i, T) for i, T in tqdm(enumerate(Ts), total=len(Ts)))

    # Extract the results and fill the xmaxs_opt array
    for i, xmax in results:
        xmaxs_opt[i] = xmax

    np.save('plot_data/betat_versus_T.npy', np.vstack([Ts, xmaxs_opt]))

    return True


def betat_versus_T():
    # given w=1, Thus R = 1 - T, with different phase
    from InverseNN import Inverse_ANN
    model = Inverse_ANN()
    model.load_state_dict(torch.load('inverse_model/model_ANN/isotropic/SimpleNN/epoch_1900.pth'))
    model.eval()
    Ts = np.linspace(0.01, 0.99, 20)
    xmaxs = np.zeros_like(Ts)
    xmaxs_opt = np.zeros_like(Ts)
    rte = RTE(0, 1, 0.9, 1, 21, 21, phase_type='isotropic', g=0.5)
    Rs = 1 - Ts
    for (i, T) in tqdm(enumerate(Ts), total=len(Ts)):
        x = torch.tensor([[T, 1 - T]]).float()
        pred = model(x).detach().numpy().flatten()
        xmaxs[i] = pred[0]
        rte.opt_inverse_problem(T, 1 - T, 0.01)
        xmaxs_opt[i] = rte.xmax

    plt.plot(Ts, xmaxs)
    plt.plot(Ts, xmaxs_opt)
    plt.xlabel('T')
    plt.ylabel('xmax')
    plt.title('xmax versus T')
    plt.show()


def betat_versus_T_2():
    # This func use partial nn to predict the xmax
    from PartitalNN import PartialNN
    model = PartialNN()
    model.load_state_dict(
        torch.load('models/PartialNN_Inverse/log_xmax_model_fine1/epoch_1900.pth'))
    model.eval()
    Ts = np.linspace(0.01, 0.99, 20)
    xmaxs = np.zeros_like(Ts)
    xmaxs_opt = np.zeros_like(Ts)
    rte = RTE(0, 1, 0.9, 1, 21, 21, phase_type='isotropic', g=0.5)
    for (i, T) in tqdm(enumerate(Ts), total=len(Ts)):
        x = torch.tensor([[T]]).float()
        pred = model(x).detach().numpy().flatten()
        xmaxs[i] = np.exp(pred[0])

        rte.opt_inverse_problem(T, 1 - T, 0.01)
        xmaxs_opt[i] = rte.xmax
        # xmaxs[i] = pred[0]
    plt.semilogy(Ts, xmaxs)
    plt.semilogy(Ts, xmaxs_opt)
    plt.show()


def real_spectrum_fitting():
    datas = np.load('plot_data/real_comparison.npy')
    datas = np.hstack([datas[:, :20], datas[:, 20::2]])
    # wl, opt_T_total, opt_T_dif, real_T, real_T_dif
    thick_rho_r_set = [(4.75, 144, 3.05), (2.54, 302, 3.50), (5.25, 293, 3.50)]
    fig, axes = plt.subplots(1, 3)
    alpha = 1
    ms = 12
    lw = 2
    for i in range(3):
        thick_rho_r = thick_rho_r_set[i]
        wavelst_nm = datas[5 * i]
        opt_T_by_quasi_T = datas[5 * i + 1]
        opt_T_dif = datas[5 * i + 2]
        real_T = datas[5 * i + 3]
        real_T_dif = datas[5 * i + 4]
        axes[i].plot(wavelst_nm, real_T, '--', label='Target $T_{Total}$', alpha=alpha, lw=lw)  # No marker

        axes[i].plot(wavelst_nm, real_T_dif, '--', label='Target $T_{Diffuse}$', alpha=alpha, lw=lw)  # No marker

        axes[i].plot(wavelst_nm, opt_T_by_quasi_T, '.', label='ML $T_{Total}$', alpha=alpha, ms=ms,
                     markerfacecolor='none', markeredgecolor='#248bcc')  # Hollow marker for ML $T_Total$

        axes[i].plot(wavelst_nm, opt_T_dif, '.', label='ML $T_{Diffuse}$', alpha=alpha, ms=ms,
                     markerfacecolor='none', markeredgecolor='#ee8844')  # Hollow marker for ML $T_Diffuse$
        axes[i].set_xlabel('Wavelength (nm)')
        # axes[i].set_ylabel('T')
        axes[i].set_title(
            f'Thickness: {thick_rho_r[0]:.2f} $\\mathrm{{mm}}$,\n'
            f' $\\rho$: {thick_rho_r[1]:.1f} $\\mathrm{{kg/m^3}}$, '
            f' $r$: {thick_rho_r[2]:.2f} $\\mathrm{{nm}}$',
        loc='left', fontsize=16)
        if i == 2:
            axes[i].legend(fontsize=16)

        if i > 0:
            axes[i].set_yticklabels([])
        # opt_quasi_T = quasi_T_func(wavelst_nm, 3, res.x, err=False)
    # fig.suptitle('Optimizing for An Existing Spectrum')
    fig.tight_layout()
    plt.show()


# real_spectrum_fitting()

def artificial_spectrum_fitting():
    datas = np.load('plot_data/artificial_comparison .npy')
    fig, axes = plt.subplots(1, 3)
    for (i, wl_sep) in enumerate([500, 700, 900]):
        wl = datas[i * 4]
        target_T = datas[i * 4 + 1]
        opt_T_by_quasi_T = datas[i * 4 + 2]
        opt_T = datas[i * 4 + 3]
        axes[i].plot(wl, target_T, '-', label='Target T', lw=2)
        axes[i].plot(wl, opt_T_by_quasi_T, '.-', label='ML  $T_{Total}$', alpha=0.8, lw=2, ms=8)
        axes[i].plot(wl, opt_T, '--', label='RTE $T_{total}$', alpha=0.8, lw=2)
        axes[i].set_xlabel('Wavelength (nm)')
        # axes[i].set_ylabel('T')
        # axes[i].set_title(f'Sep_wl: {wl_sep} nm')
        if i == 2:
            axes[i].legend()

        if i > 0:
            axes[i].set_yticklabels([])
    # fig.suptitle('Optimizing for An Artificial Spectrum')
    fig.tight_layout()
    plt.show()


def xmax_versus_T():
    from PartitalNN import PartialNN
    from InverseNN import Simple_NN
    # Ts = np.linspace(1e-3, 1 - 1e-3, 20)
    data = np.load('plot_data/betat_versus_T.npy')
    # find where 0.05<T<0.95
    mask = (data[0] > 0.05) & (data[0] < 0.95)
    mask = (data[0] > 0.02)
    Ts = data[0][mask]
    # Ts = data[0]

    model = PartialNN()
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
    plt.semilogy(Ts, xmaxs_from_opt, '--', label='OPT R=1-T', lw=2)
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
        Ts[:a - sep:int(sep/3)],  # Step 1
        Ts[a:b-sep:int(sep)],  # Step 2
        Ts[b:c:int(amp * sep)],  # Step 3
        Ts[c+sep:d:int(sep )],  # Step 4
        Ts[d + sep::int(sep/2)]  # Step 5
    ])

    # Similarly for xmaxs
    xmaxs_from_ML = np.hstack([
        pred_xmaxs[:a - sep:int(sep/3)],  # Step 1
        pred_xmaxs[a:b-sep:int(sep)],  # Step 2
        pred_xmaxs[b:c:int(amp * sep)],  # Step 3
        pred_xmaxs[c+sep:d:int(sep )],  # Step 4
        pred_xmaxs[d + sep::int(sep/2)]  # Step 5
    ])

    plt.semilogy(Ts_ML, xmaxs_from_ML, '.', label='ML  R=1-T', ms=20,
                 color='#4a90e2', markerfacecolor='none', markeredgecolor='tab:orange')
    # plt.semilogy(Ts, pred_xmaxs_2, '.-', label='ML R=0.5(1-T)')

    # plt.xlabel('T')
    # plt.ylabel('Optical Depth')
    # plt.legend()
    plt.show()


def inverse_efficiency():
    opt_whole_height = 450 * 1e3
    opt_partial_height = 320 * 1e3
    ml_whole_height = 10
    minor_opt_height = 232  # Part of the third bar (stacked)

    # Bar labels
    labels = ['Opt Micro', 'ML Micro', 'Opt Optical', 'ML Optical']

    # X locations for the bars
    x = np.arange(len(labels))

    # Width of the bars
    width = 0.5

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the bars
    ax.bar(x[0], opt_whole_height, width, label='Opt Micro', color='tab:blue')
    ax.bar(x[1], minor_opt_height + ml_whole_height, width, label='ML-enabled Micro', color='#ff6c5f')
    ax.bar(x[2], opt_partial_height, width, label='Opt Optical ', color='#ffc168')
    ax.bar(x[3], ml_whole_height, width, label='ML Optical ', color='#2dde98')

    # Add labels, title, and legend
    # change scale of y to log
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Efficiency of the Inverse Model', fontsize=16)
    ax.legend()

    # Show the plot
    plt.show()


def count_forward_eff(device='cuda'):
    from NN import Simple_NN
    # model = Simple_NN(num_features=2).to(device)
    time_comsuption = np.zeros([N1:=30, repeat:=20])
    for (i, workload) in tqdm(enumerate(np.logspace(1, 5, N1)), total=N1):
        workload = int(workload)
        for j in range(repeat):
            # print(workload)
            t1 = time.perf_counter()
            model = Simple_NN(num_features=2).to(device)
            input = torch.randn(int(workload), 2).to(device)
            model(input)
            t2 = time.perf_counter()
            time_comsuption[i, j] = (t2 - t1)
            del model
    np.save(f'plot_data/forward_eff_{device}_1.npy', time_comsuption)

def count_forward_eff_rte():
    from joblib import Parallel, delayed
    time_comsuption = np.zeros([N1:=20, repeats:=3])
    def a(N):
        t1 = time.perf_counter()
        for _ in range(N):
            rte = RTE(0.9, 1, 0.9, 1, 21, 21, 'iso', g=0)
            rte.xmax = np.random.rand()
            rte.build()
            rte.hemi_props()
        t2 = time.perf_counter()
        return t2 - t1
    work_load_rte = np.logspace(1, 3.5, 10)

    for (i, workload) in enumerate(work_load_rte):
        workload = int(workload)
        for (j, repeat) in enumerate(range(repeats)):
            time_comsuption[i, j] = a(workload)
        print(workload)

    np.save('plot_data/forward_eff_rte.npy', time_comsuption)


def count_forward_eff_rte_2():
    work_load_rte = np.logspace(1, 3.5, 10)
    time_comsuption = np.zeros([len(work_load_rte), 3])
    for (i, workload) in enumerate(work_load_rte):
        workload = int(workload)
        for j in range(3):
            t1 = time.perf_counter()
            rte = RTE(0.9, 1, 0.9, 1, 21, 21, 'iso', g=0)
            for _ in range(workload):
                rte.xmax = np.random.rand() + 1e-3
                rte.build()
                rte.hemi_props()
            t2 = time.perf_counter()
            time_comsuption[i, j] = t2 - t1
        print(workload)
        print(time_comsuption[i])
    np.save('plot_data/forward_eff_rte_2.npy', time_comsuption)



def show_forward_eff_2():
    # data = np.load('plot_data/forward_eff_cpu.npy')
    data_gpu = np.load('plot_data/forward_eff_cuda_1.npy')[2:]
    data_cpu = np.load('plot_data/forward_eff_cpu_1.npy')[2:]
    work_load = np.logspace(1, 5, 30)[2:]
    mean_cpu, std_cpu = np.mean(data_cpu, axis=1), np.std(data_cpu, axis=1)
    mean_gpu, std_gpu = np.mean(data_gpu, axis=1), np.std(data_gpu, axis=1)
    error_2sigma_cpu = std_cpu
    error_2sigma_gpu = std_gpu
    plt.semilogy(work_load, mean_cpu, label='CPU', lw=2, color='tab:orange')
    plt.fill_between(work_load, mean_cpu - error_2sigma_cpu, mean_cpu + error_2sigma_cpu, alpha=0.2, color='tab:orange')
    plt.semilogy(work_load, mean_gpu, label='GPU', lw=2, color='tab:green')
    plt.fill_between(work_load, mean_gpu - error_2sigma_gpu, mean_gpu + error_2sigma_gpu, alpha=0.2, color='tab:green')
    plt.xscale('log')  # Logarithmic scale for num

    data_rte_1 = np.load('plot_data/forward_eff_rte_2.npy')[:10]
    work_load_rte = np.logspace(1, 3.5, 10)
    mean_rte_1, std_rte_1 = np.mean(data_rte_1, axis=1), np.std(data_rte_1, axis=1)
    error_2sigma_rte_1 = std_rte_1
    plt.semilogy(work_load_rte, mean_rte_1, label='RTE', lw=2, color='tab:blue')
    plt.fill_between(work_load_rte, mean_rte_1 - error_2sigma_rte_1, mean_rte_1 + error_2sigma_rte_1, alpha=0.2, color='tab:blue')



    plt.legend()
    plt.show()



# inverse_efficiency()
if __name__ == '__main__':
    # custom_colors = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#ff8884']
    import brewer2mpl

    bmap = brewer2mpl.get_map('Set1', 'qualitative', 8)
    # colors = bmap.mpl_colors
    # colors = plt.get_cmap('tab10').colors

    custom_colors = ['#248bcc', '#823c3d', '#c89fa5', '#6c94cd']

    # Set the color cycle
    # plt.rc('axes', prop_cycle=plt.cycler(color=custom_colors))
    # plt.rc('axes', prop_cycle=plt.cycler(color=colors))

    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 16,
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'figure.titlesize': 16}
    pylab.rcParams.update(params)
    plt.rcParams['savefig.directory'] = r'D:\summerintern\Paper\SIpic'
    # T_R_Heatmap()
    # ML_forward_efficiency(10)
    # for i in np.logspace(1, 5, 10):
    #     # rte_forward_efficiency(int(i))
    #     # print(i)
    #     ML_forward_efficiency(int(i), device='cuda')
    # T_R_versus_xmax(use_log_model=True)
    # T_versus_wl()

    # real_spectrum_fitting()
    # artificial_spectrum_fitting()


    # T_R_Heatmap_2()
    # store_betat_versus_T(num=1000)
    # data = np.load('plot_data/betat_versus_T.npy')
    # T = data[0]
    # xmax = data[1]
    # plt.semilogy(T, xmax)
    # plt.show()
    # store_RTE_T_R_versus_xmax()
    # T_R_versus_xmax(use_log_model=True)
    # T_versus_wl()
    # real_spectrum_fitting()
    # show_forward_eff_2()
    # inverse_efficiency()
    betat_omega_heatmap(200, keep_log=False)
    # betat_heatmap_rte(20)
    # xmax_versus_T()
    # T_R_Heatmap('hg')
    # count_forward_eff('cpu')
    # count_forward_eff_rte()
    # count_forward_eff_rte_2()
    # show_forward_eff_2()
    # a = np.load('plot_data/forward_eff_rte.npy')
    # print(a)