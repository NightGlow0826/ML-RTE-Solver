#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : NN.py
@Author  : Gan Yuyang
@Time    : 2024/7/7 上午10:22
"""
import os
import sys
import time
from RTE_Truth_Model import RTE
import matplotlib.pyplot as plt
import numpy as np
# import swats
import torch
from tqdm import *
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class RTE_Dataset(Dataset):
    def __init__(self, path, phase_type=None, extrapath=None, use_log_model=False, use_inverse_model=False):
        super().__init__()
        self.use_inverse_model = use_inverse_model

        f = np.load(path)

        if phase_type is None:
            phase_type = 'Y'  # for the old version, isotropic data is stored in 'Y'
            print('++++++++++++++Warning+++++++++')
        else:
            assert phase_type in ['isotropic', 'rayleigh', 'hg', 'legendre', 'iso', 'ray']

        self.X = torch.from_numpy(f['X']).to(torch.float32).to(device)


        self.Y = torch.from_numpy(f[phase_type]).to(torch.float32).to(device)
        if extrapath is not None:
            f_extra = np.load(extrapath)
            self.X = torch.cat([self.X, torch.from_numpy(f_extra['X']).to(torch.float32).to(device)], dim=0)
            self.Y = torch.cat([self.Y, torch.from_numpy(f_extra[phase_type]).to(torch.float32).to(device)], dim=0)

        if use_log_model:
            self.X[:, 0] = torch.log(self.X[:, 0])
        else:
            ...

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if not self.use_inverse_model:
            return self.X[item], self.Y[item]
        else:
            return self.Y[item], self.X[item]


# class GT_Dataset(Dataset):
#     # This is a dataset using test set to train to see if the model can fit the ground truth
#     def __init__(self, gt_path, phase_type=None):
#         super().__init__()
#         f = np.load(gt_path)
#         # use gt in analytic.py
#         self.X = torch.from_numpy(f['X']).to(torch.float32).to(device)
#         self.Y = torch.from_numpy(f[phase_type]).to(torch.float32).to(device)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, item):
#         return self.X[item], self.Y[item]
#

class Simple_NN(nn.Module):
    def __init__(self, num_features, out_features=2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(num_features, 128)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_features)
        )
    def forward(self, x) -> torch.Tensor:
        h1 = self.block1(x)
        x = self.block2(x) + h1
        x = self.block3(x)
        return x


# class ANN(nn.Module):
#     def __init__(self):
#         super(ANN, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(2, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 2)
#         )
#
#     def forward(self, x):
#         return self.block(x)


class ANN_Bigger(nn.Module):
    def __init__(self):
        super(ANN_Bigger, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.block(x)


def train(model: nn.Module, train_loader, optimizer, epochs, test_loader, save=True, loss_mode=0,
          phase_type='isotropic', parent_path='model_1', use_log_model=False, use_inverse_model=False):
    parent_path = parent_path
    str_loss = {0: 'SimpleNN', 1: 'WithBound', 2: 'WithPINN', 3: 'DualPINN'}
    print(f'Using {str_loss[loss_mode]}')
    if use_log_model:
        os.makedirs(f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model', exist_ok=True)
    else:
        os.makedirs(f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}', exist_ok=True)
    epoch_train_losses = []
    epoch_test_losses = []
    # swith_opti = 0
    for epoch in range(epochs):
        epoch_total_loss = 0.
        epoch_total_bound_loss = 0.
        epoch_total_PINN_loss = 0.
        epoch_total_PINN2_loss = 0.
        epoch_total_samples = 0
        # for (i, (X, Y)) in tqdm(enumerate(train_loader), total=len(train_loader)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1)
        model.train()
        for (i, (X, Y)) in enumerate(train_loader):
            Y_pred = model(X)

            loss_fn = nn.MSELoss(reduction='mean')
            # SAMPLE LOSS
            loss_sample = loss_fn(Y, Y_pred)


            # BOUND LOSS
            bound_mask = (Y_pred < 0) | (Y_pred > 1)
            loss_bound = loss_fn(
                Y_pred[bound_mask], torch.clamp(Y_pred[bound_mask], 0, 1))
            loss_bound = torch.nan_to_num(loss_bound, nan=0.)

            # Part 1 mainly correcting err for insufficient discretization
            PINN_mask = X[:, 1] < 0.2  # w<0.1
            xmax_part = X[:, 0][PINN_mask]
            if use_log_model:
                xmax_orig = torch.exp(xmax_part)
            else:
                xmax_orig = xmax_part
            T_part_analytic = torch.exp(-xmax_orig)
            T_part_pred = Y_pred[:, 0][PINN_mask]
            # loss is where T_part_pred is less than T_part_analytic, analytical is the lower bound
            err = torch.clamp(T_part_pred - T_part_analytic, max=0.)
            loss_PINN = loss_fn(err, torch.zeros_like(err))
            loss_PINN = torch.nan_to_num(loss_PINN, nan=0.)


            # Part 2
            PINN_mask2 = X[:, 1] > 1 - 1e5  # w>0.9
            sum_part_pred: torch.Tensor = Y_pred[:, 0][PINN_mask2] + Y_pred[:, 1][PINN_mask2]
            sum_analytic = sum_part_pred.clamp_(max=1.)
            loss_PINN_2 = loss_fn(sum_part_pred, sum_analytic)

            epoch_total_loss += loss_sample.item() * len(Y)
            epoch_total_bound_loss += loss_bound.item() * len(Y)
            epoch_total_PINN_loss += loss_PINN.item() * len(Y)
            epoch_total_PINN2_loss += loss_PINN_2.item() * len(Y)
            epoch_total_samples += len(Y)



            if loss_mode == 0:
                loss = loss_sample
            elif loss_mode == 1:
                loss = loss_sample + 0.5 * loss_bound
            elif loss_mode == 2:
                loss = loss_sample + 0.5 * loss_bound + 0.5 * loss_PINN
            elif loss_mode == 3:
                loss = loss_sample + 0.5 * loss_bound + 0.5 * loss_PINN + 0.1 * loss_PINN_2
            else:
                raise ValueError(f'Unknown loss mode {loss_mode}')
            # if   swith_opti = 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_mean_loss = epoch_total_loss / epoch_total_samples
        epoch_mean_bound_loss = epoch_total_bound_loss / epoch_total_samples
        epoch_mean_PINN_loss = epoch_total_PINN_loss / epoch_total_samples
        epoch_mean_PINN2_loss = epoch_total_PINN2_loss / epoch_total_samples

        epoch_train_losses.append(epoch_mean_loss)

        eval_loss = eval(model, test_loader, epoch)
        epoch_test_losses.append(eval_loss)
        if epoch % 20 == 0:
            print(f'epoch: {epoch}, train: {epoch_mean_loss}, eval: {eval_loss}, violation: {epoch_mean_bound_loss}, pinnl_1: {epoch_mean_PINN_loss}, pinnl_2: {epoch_mean_PINN2_loss}' )

        if epoch % 20 == 0 and save:
            if use_log_model:
                torch.save(model.state_dict(),
                       f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model/epoch_{epoch}.pth')
            else:
                torch.save(model.state_dict(),
                       f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}/epoch_{epoch}.pth')
            ...

        scheduler.step(eval_loss)
    plt.semilogy(epoch_train_losses, label='train loss')
    plt.semilogy(epoch_test_losses, label='test loss')
    plt.title(f'Phase: {phase_type}, loss_mode: {str_loss[loss_mode]}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return model, optimizer




def train_tb(model: nn.Module, train_loader, optimizer, epochs, test_loader, save=True, loss_mode=0,
          phase_type='isotropic', parent_path='model_1', use_log_model=False, use_inverse_model=False):
    writer = SummaryWriter()
    parent_path = parent_path
    str_loss = {0: 'SimpleNN', 1: 'WithBound', 2: 'WithPINN', 3: 'DualPINN'}
    print(f'Using {str_loss[loss_mode]}')
    if use_log_model:
        os.makedirs(f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model', exist_ok=True)
    else:
        os.makedirs(f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}', exist_ok=True)

    loss_datas = np.zeros([epochs, 5])

    # swith_opti = 0
    for epoch in range(epochs):
        print(epoch)
        epoch_total_loss = 0.
        epoch_total_bound_loss = 0.
        epoch_total_PINN_loss = 0.
        epoch_total_PINN2_loss = 0.
        epoch_total_samples = 0
        epoch_PINN1_samples = 0
        epoch_PINN2_samples = 0
        # for (i, (X, Y)) in tqdm(enumerate(train_loader), total=len(train_loader)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1)
        model.train()
        for (i, (X, Y)) in enumerate(train_loader):
            Y_pred = model(X)

            loss_fn = nn.MSELoss(reduction='mean')
            # SAMPLE LOSS
            loss_sample = loss_fn(Y, Y_pred)


            # BOUND LOSS
            bound_mask = (Y_pred < 0) | (Y_pred > 1)
            loss_bound = loss_fn(
                Y_pred[bound_mask], torch.clamp(Y_pred[bound_mask], 0, 1))
            loss_bound = torch.nan_to_num(loss_bound, nan=0.)

            # Part 1 mainly correcting err for insufficient discretization
            # PINN_mask = X[:, 1] < 0.2  
            # # < 0.2 and >
            # xmax_part = X[:, 0][PINN_mask]
            xmax_part = X[:, 0]
            if use_log_model:
                xmax_orig = torch.exp(xmax_part)
            else:
                xmax_orig = xmax_part
            T_part_analytic = torch.exp(-xmax_orig)
            T_part_pred = Y_pred[:, 0]
            # loss is where T_part_pred is less than T_part_analytic, analytical is the lower bound
            err = torch.clamp(T_part_pred - T_part_analytic, max=0.)
            loss_PINN = loss_fn(err, torch.zeros_like(err))
            loss_PINN = torch.nan_to_num(loss_PINN, nan=0.)


            # Part 2
            PINN_mask2 = X[:, 1] > 1 - 1e3  # w>0.9


            sum_part_pred: torch.Tensor = Y_pred[:, 0][PINN_mask2] + Y_pred[:, 1][PINN_mask2]
            sum_analytic = sum_part_pred.clamp_(max=1.)
            loss_PINN_2 = loss_fn(sum_part_pred, sum_analytic)


            epoch_total_samples += len(Y)
            epoch_PINN1_samples += len(Y)
            epoch_PINN2_samples += len(Y[PINN_mask2])

            epoch_total_loss += loss_sample.item() * len(Y)
            epoch_total_bound_loss += loss_bound.item() * len(Y)
            epoch_total_PINN_loss += loss_PINN.item() * len(Y)
            epoch_total_PINN2_loss += loss_PINN_2.item() * len(Y[PINN_mask2])




            if loss_mode == 0:
                loss = loss_sample
            elif loss_mode == 1:
                loss = loss_sample + 10 * loss_bound
            elif loss_mode == 2:
                loss = loss_sample + 0.5 * loss_bound + 0.5 * loss_PINN
            elif loss_mode == 3:
                loss = loss_sample + 0.5 * loss_bound + 0.5 * loss_PINN + 0.1 * loss_PINN_2
            else:
                raise ValueError(f'Unknown loss mode {loss_mode}')
            # if   swith_opti = 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        eps = 1e-10
        epoch_mean_loss = epoch_total_loss / epoch_total_samples + eps
        epoch_mean_bound_loss = epoch_total_bound_loss / epoch_total_samples + eps
        epoch_mean_PINN_loss = epoch_total_PINN_loss / epoch_PINN1_samples + eps
        epoch_mean_PINN2_loss = epoch_total_PINN2_loss / epoch_PINN2_samples + eps
        eval_loss = eval(model, test_loader, epoch)

        writer.add_scalars('Loss', {'Sample': epoch_mean_loss, 'Bound': epoch_mean_bound_loss, 'PINN': epoch_mean_PINN_loss, 'Energy': epoch_mean_PINN2_loss}, epoch)
        writer.add_scalar('Eval', eval_loss, epoch)
        print(
            f'epoch: {epoch}, train: {epoch_mean_loss}, eval: {eval_loss}, violation: {epoch_mean_bound_loss}, pinnl_1: {epoch_mean_PINN_loss}, pinnl_2: {epoch_mean_PINN2_loss}')
        loss_datas[epoch] = [epoch_mean_loss, epoch_mean_bound_loss, epoch_mean_PINN_loss, epoch_mean_PINN2_loss, eval_loss]
        if epoch % 20 == 0 and epoch > 200 and save:
            if use_log_model:
                torch.save(model.state_dict(),
                       f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model/epoch_{epoch}.pth')
            else:
                torch.save(model.state_dict(),
                       f'forward_models/{parent_path}/{phase_type}/{str_loss[loss_mode]}/epoch_{epoch}.pth')
            ...
        scheduler.step(eval_loss)


    writer.close()
    np.save(f'plot_data/{parent_path}_{phase_type}_{str_loss[loss_mode]}.npy', loss_datas)
    return model, optimizer



def eval(model: nn.Module, test_loader, cur_epoch=None):
    model.eval()
    total_loss = 0.
    total_samples = 0
    for (i, (X, Y)) in enumerate(test_loader):
        Y_pred = model(X)
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(Y, Y_pred)
        total_loss += loss.item() * len(Y)
        total_samples += len(Y)
    mean_loss = total_loss / total_samples
    return mean_loss


def smooth_data(data, smoothing_factor=0.7):
    """Applies exponential moving average smoothing to the data."""
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # First value remains the same
    for i in range(1, len(data)):
        smoothed_data[i] = smoothing_factor * smoothed_data[i - 1] + (1 - smoothing_factor) * data[i]
    return smoothed_data



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # sys.path.append('..')
    print(device)
    model = Simple_NN(num_features=2).to(device)

    phase_type = 'iso'
    use_log_model = True

    # dataset = RTE_Dataset(path=f'data/forward_fine_data/set_finer.npz',
    #                       extrapath=f'data/forward_fine_data/set_finer_extra.npz',
    #                       phase_type=phase_type,  use_log_model=use_log_model)
    # dataset = RTE_Dataset(path=f'data/forward_unif_data/{phase_type}_10000.npz', phase_type=phase_type, use_log_model=use_log_model)


    dataset = RTE_Dataset(path=f'data/forward_fine_data/{phase_type}_1000_unif.npz', phase_type=phase_type, use_log_model=use_log_model)
    generator1 = torch.Generator().manual_seed(42)
    train_ratio = 0.8
    train_set, test_set = random_split(
        dataset,
        (train_ratio, 1 - train_ratio), generator=generator1)
    train_loader = DataLoader(train_set, batch_size=2000, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2000, shuffle=True, )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    train_tb(model, train_loader, optimizer, 400,
          test_loader, save=True, loss_mode=1,
          phase_type=phase_type, parent_path='model_Resnet_unif_5', use_log_model=use_log_model)
