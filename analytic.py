
import matplotlib.pyplot as plt
import torch

from RTE_Truth_Model import RTE
import numpy as np
from tqdm import *
from NN import Simple_NN
import matplotlib.colors as mcolors
from matplotlib import cm


def evaluate(gt, pred, subdomain=False):
    gt = np.array(gt)
    pred = np.array(pred)

    if subdomain:
        print('Evaluating using subdomain')
        eps = 1e-6
        mask = (gt[:, 0] < eps) | (gt[:, 0] > 5 - eps) | (gt[:, 1] < eps) | (gt[:, 1] > 1 - eps)
        gt = gt[~mask]
        pred = pred[~mask]
    mean_err = ((gt - pred) ** 2).mean()
    max_err = np.max(np.abs(gt - pred))


    return mean_err, max_err


# omega is from 0-1; beta_t, ie. x is from 0 - 5

omega_mesh = np.linspace(0, 1, 20)
xmax_mesh = np.linspace(1e-6, 5, 20)

O, X = np.meshgrid(xmax_mesh, omega_mesh)



def gen_analytic_mesh():
    rte_obj = RTE(0.5, 1, 0.9, 1, 21, 21, phase_type='hg', g=0.5)
    rte_obj.build()
    tlen = len(omega_mesh) * len(xmax_mesh)

    T_gt = np.zeros([len(omega_mesh), len(xmax_mesh)])
    R_gt = np.zeros([len(omega_mesh), len(xmax_mesh)])
    data_dct_gt = {}
    dataset_dct = {}
    X = np.zeros([tlen, 2])
    for (p_idx, phase_type) in enumerate(['isotropic', 'rayleigh', 'hg']):
        rte_obj.phase_type = phase_type

        Y = np.zeros([tlen, 2])
        for (i, omega) in tqdm(enumerate(omega_mesh), total=len(omega_mesh)):
            for (j, xmax) in enumerate(xmax_mesh):
                if p_idx == 0:
                    X[i * len(xmax_mesh) + j] = [xmax, omega]
                rte_obj.omega = omega
                rte_obj.xmax = xmax
                rte_obj.build()
                T_, R_ = rte_obj.hemi_props()
                T_gt[i, j] = T_
                R_gt[i, j] = R_
                Y[i * len(xmax_mesh) + j] = [T_, R_]
        data_dct_gt[f'T_gt_{phase_type}'] = T_gt
        data_dct_gt[f'R_gt_{phase_type}'] = R_gt
        dataset_dct[phase_type] = Y
    np.savez('data/mesh_bench_new.npz', **data_dct_gt)
    np.savez('data/dataset_mesh.npz', X=X, **dataset_dct) # save the mesh data for easier access


if __name__ == '__main__':

    # gen_analytic_mesh()

    ds_f = np.load('data/dataset_mesh.npz')
    X_set = ds_f['X']
    Y_set = ds_f['isotropic']
    # idx = 100
    # xm, om = X_set[idx]
    # T, R = Y_set[idx]
    # print(T, R)
    # rte_obj = RTE(om, 1, 0.9, xm, 21, 21, phase_type='isotropic', g=0.5)
    # rte_obj.build()
    # print(rte_obj.hemi_props())


    # quit()

    str_loss = ['SimpleNN', 'WithBound', 'WithPINN', 'DualPINN']
    epoch = 1500
    # parentpath = 'gt'
    # parentpath = 'model_2'
    parentpath = 'model_boundsample'

    phase_type = 'rayleigh'
    chosen_losstypes = 4
    fig, axes = plt.subplots(4, chosen_losstypes)
    for loss_mode in range(chosen_losstypes):
    # loss_mode = 3
        model = Simple_NN(2)
        model.load_state_dict(torch.load(f'{parentpath}/{phase_type}/{str_loss[loss_mode]}/epoch_{epoch}.pth'))

        Y_pred = model(torch.from_numpy(X_set).to(torch.float32)).detach().cpu().numpy()
        Y_gt = Y_set
        mean_err, max_err = evaluate(Y_gt, Y_pred, subdomain=True)
        print(f'loss_mode: {str_loss[loss_mode]}, mean_err: {mean_err:.5f}, max_err: {max_err:.5f}')
        from temp import dataset_format_to_fig
        T_pred, R_pred = dataset_format_to_fig(X_set, Y_pred)
        T_gt, R_gt = dataset_format_to_fig(X_set, Y_gt)

        # axes[0, 0].imshow(T_gt, origin='lower', extent=(0, 5, 0, 1), )
        # axes[0, loss_mode].imshow(T_pred, origin='lower', extent=(0, 5, 0, 1),vmax=1)
        axes[0, loss_mode].imshow(T_gt, origin='lower', extent=(0, 5, 0, 1),vmax=1)

        axes[2, loss_mode].imshow(np.abs(T_gt - T_pred), origin='lower', extent=(0, 5, 0, 1), vmax=0.05)
        # axes[1, loss_mode].imshow(R_pred, origin='lower', extent=(0, 5, 0, 1),vmax=1)
        axes[1, loss_mode].imshow(R_gt, origin='lower', extent=(0, 5, 0, 1),vmax=1)

        axes[3, loss_mode].imshow(np.abs(R_gt - R_pred), origin='lower', extent=(0, 5, 0, 1),vmax=0.05)

        axes[0, loss_mode].set_title(f'{str_loss[loss_mode]}')

        if loss_mode == 0:
            axes[0, loss_mode].set_ylabel('T_pred')
            axes[1, loss_mode].set_ylabel('R_pred')
            axes[2, loss_mode].set_ylabel('T_err')
            axes[3, loss_mode].set_ylabel('R_err')

        # add colorbar with norm1 to row 0 and 2
    norm = mcolors.Normalize(vmin=0, vmax=1)
    norm_err = mcolors.Normalize(vmin=0, vmax=0.05)
    plt.colorbar(cm.ScalarMappable(norm=norm), ax=axes[:2, :])
    plt.colorbar(cm.ScalarMappable(norm=norm_err), ax=axes[2:, :])
    for ax in axes.flatten():
        ax.set_aspect('auto')
    plt.show()


    # T = np.zeros([len(omega_mesh), len(xmax_mesh)])
    # R = np.zeros([len(omega_mesh), len(xmax_mesh)])
    # phase = 'isotropic'
    # f = np.load('data/mesh_bench_new.npz')
    # T_gt = f[f'T_gt_{phase}']
    # R_gt = f[f'R_gt_{phase}']
    #
    # # plt.imshow(T_gt, label='T_gt', origin='lower', extent=(0, 5, 0, 1), aspect='auto')
    # # plt.show()
    # # quit()
    #
    # model = Simple_NN(2)
    # parentpath = 'gt'
    # # model_types = ['SimpleNN', 'WithBound', 'WithPINN', 'DualPINN']
    # model_types = ['SimpleNN', 'DualPINN']
    # # model_types = ['SimpleNN', 'WithPINN', 'DualPINN' ]
    # fig, axes = plt.subplots(3, len(model_types))
    # axes = axes.reshape(3, len(model_types))
    # norm = mcolors.Normalize(vmin=0, vmax=0.05)

