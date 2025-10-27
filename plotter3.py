
from RTE_code.PythonCode.colormaps import _parula_data, parula
import numpy as np
# import colormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('TKAgg')
import torch
def T_R_Heatmap():

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    from NN import Simple_NN
    # model = ANN()
    model1 = Simple_NN(num_features=2)
    model2 = Simple_NN(num_features=2)
    model1.load_state_dict(torch.load(f'forward_models/model_Resnet_unif_4/iso/WithBound_log_model/epoch_380.pth'))
    model2.load_state_dict(torch.load(f'forward_models/model_Resnet_unif_4/hg/WithBound_log_model/epoch_380.pth'))
    # model2.load_state_dict(torch.load(f'forward_models/model_Resnet_2/iso/WithBound_log_model/epoch_260.pth'))
    datas = np.zeros([N_omega:=100, N_xmax:=100, 4])
    for (j, xmax) in enumerate(np.linspace(0.01, 5, N_xmax)):
        for (i, omega) in enumerate(omegas := np.linspace(0, 1, N_omega)):
            x_inp = np.log(xmax)
            # x_inp = np.log(xmax)
            T_R_vec = model1(torch.tensor([[x_inp, omega]]).float()).detach().numpy().flatten()
            datas[i, j, 0] = T_R_vec[0]
            datas[i, j, 1] = T_R_vec[1]
            T_R_vec = model2(torch.tensor([[x_inp, omega]]).float()).detach().numpy().flatten()
            datas[i, j, 2] = T_R_vec[0]
            datas[i, j, 3] = T_R_vec[1]
    # datas[:, :, 1] = np.clip(datas[:, :, 1], 0.02, np.inf)
    # cmap = colormap.parula_map()
    axes[0, 0].imshow(datas[:, :, 0], cmap=parula, aspect='auto', origin='lower', interpolation='bilinear', extent=([0, 5, 0, 1]), vmin=0, vmax=1)
    axes[0, 1].imshow(datas[:, :, 1], cmap=parula, aspect='auto', origin='lower', interpolation='bilinear', extent=([0, 5, 0, 1]), vmin=0, vmax=1)
    axes[1, 0].imshow(datas[:, :, 2], cmap=parula, aspect='auto', origin='lower', interpolation='bilinear', extent=([0, 5, 0, 1]), vmin=0, vmax=1)
    axes[1, 1].imshow(datas[:, :, 3], cmap=parula, aspect='auto', origin='lower', interpolation='bilinear', extent=([0, 5, 0, 1]), vmin=0, vmax=1)
    plt.show()










def gen_validate_data(N_sample=1000, nquad=21):
    from RTE_Truth_Model import RTE
    from joblib import Parallel, delayed

    for phase in ['iso', 'ray', 'hg']:
        print('Computing', phase)
        data = np.zeros([4, N_sample])
        # rte = RTE(omega=0.5, xmax=1, phase_type=phase, mu0=0.99, nquad=41, grid_size=21, g=0.5, phi=1)
        # for i in range(N_sample):
        #     rte.xmax = np.power(10, np.random.uniform(-3, 3))
        #     rte.omega = np.random.uniform(0, 1-1e-3)
        #     rte.build()
        #     T, R = rte.hemi_props()
        #     data[0, i] = rte.xmax
        #     data[1, i] = rte.omega
        #     data[2, i] = T
        #     data[3, i] = R
        def compute_sample(i, phase):
            rte = RTE(omega=0.5, xmax=1, phase_type=phase, mu0=0.99, nquad=21, grid_size=21, g=0.5, phi=1)
            rte.xmax = np.power(10, np.random.uniform(-3, 3))
            rte.omega = np.random.uniform(0, 1 - 1e-2)
            rte.build()
            T, R = rte.hemi_props()
            return rte.xmax, rte.omega, T, R

        # Parallel execution of the loop using joblib
        results = Parallel(n_jobs=-1)(delayed(compute_sample)(i, phase) for i in range(N_sample))

        # Store results in the data array
        for i, (xmax, omega, T, R) in enumerate(results):
            data[0, i] = xmax
            data[1, i] = omega
            data[2, i] = T
            data[3, i] = R
        np.save(f'plot_data/validate_{phase}.npy', data)

# gen_validate_data()

def validate_models():
    for phase in ['iso', 'hg']:
        data = np.load(f'plot_data/validate_{phase}.npy')
        xmax = data[0]
        omega = data[1]
        X = torch.tensor(np.vstack([np.log(xmax), omega]).T).float()

        T = data[2]
        R = data[3]

        def largest_deviation(arr):
            # Deviation for values less than 0
            below_zero_deviation = np.abs(arr[arr < 0])

            # Deviation for values greater than 1
            above_one_deviation = np.abs(arr[arr > 1] - 1)

            # Combine both deviations
            all_deviations = np.concatenate([below_zero_deviation, above_one_deviation])

            # If there are no deviations, return 0
            if all_deviations.size == 0:
                return 0

            # Return the largest deviation
            return np.max(all_deviations)
        from NN import Simple_NN
        model = Simple_NN(num_features=2)
        for model_type in ['SimpleNN_log_model', 'WithBound_log_model', 'WithPINN_log_model']:
            model.load_state_dict(torch.load(f'forward_models/model_Resnet_unif_4/{phase}/{model_type}/epoch_380.pth'))
            pred = model(X).detach().numpy().T
            T_R = np.vstack([T, R])
            max_err = np.max(np.abs(pred - T_R))
            mean_err = np.mean(np.abs(pred - T_R))
            max_phy_err = largest_deviation(pred)



            PINN_deviation = T - np.exp(-xmax)
            PINN_deviation = np.clip(PINN_deviation, -np.inf, 0)
            mean_PINN_err = np.mean(np.abs(PINN_deviation))
            print(f'{phase}, {model_type}, max_err: {max_err}, mean_err: {mean_err}, max_phy_err: {max_phy_err}', f'mean_PINN_err: {mean_PINN_err}')

def corres_opt_mat():
    fig, axes = plt.subplots(1,2)
    real_map = np.load('plot_data/r_rho_real_error_500.npy')
    arti_map = np.load('plot_data/r_rho_artificial_error_500.npy')

    a = axes[0].imshow(real_map, cmap=parula, aspect='auto', origin='lower', extent=([200, 400, 3.5, 5]), )
    b = axes[1].imshow(arti_map, cmap=parula, aspect='auto', origin='lower', extent=([200, 500, 7, 10]), vmin=-20)
    for ax in axes:
        # close all numbers
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    c1 = fig.colorbar(a, ax=axes[0], orientation='vertical', )
    c2 = fig.colorbar(b, ax=axes[1], orientation='vertical')
    c1.ax.set_yticklabels([])  # 隐藏左图颜色条的刻度数字
    c2.ax.set_yticklabels([])
    plt.show()

# gen_validate_data()
# validate_models()
# T_R_Heatmap()
corres_opt_mat()