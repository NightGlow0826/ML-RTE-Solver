#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : PartitalNN.py
@Author  : Gan Yuyang
@Time    : 2024/7/31 下午12:22
"""
import logging
import os

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from temp import Loss_Tracker
from NN import Simple_NN
"""
Since the original NN has boundary limits, we could just train another model to predict on the boundary
"""

from RTE_Truth_Model import RTE
from torch import nn as nn
import torch


def construct_bound_database(num=10000):
    from joblib import Parallel, delayed

    rte_obj = RTE(1, 1, 0.9, 5., 21, 21, 'iso', 0.0)
    space_log = np.logspace(np.log10(1e-3), np.log10(1e3), num)
    space_lin_1 = np.linspace(1e2, 1e3, num)
    space_lin_2 = np.linspace(1e0, 1e2, num)
    space = np.concatenate([space_log, space_lin_1, space_lin_2])
    xmax_T_pair = np.zeros((len(space), 2))

    # for (i, xmax) in tqdm(enumerate(space), total=2*num):
    #     rte_obj.xmax = xmax
    #     rte_obj.build()
    #     T, R = rte_obj.hemi_props()
    #     xmax_T_pair[i] = [xmax, T]
    def process_iteration(i, xmax):
        rte_obj = RTE(1, 1, 0.99, 5., 41, 41, 'iso', 0.0)
        rte_obj.xmax = xmax
        rte_obj.build()
        T, R = rte_obj.hemi_props()
        return i, xmax, T

    # Parallelized execution
    results = Parallel(n_jobs=-1)(delayed(process_iteration)(i, xmax)
                                  for i, xmax in tqdm(enumerate(space), total=len(space)))

    # Collecting the results
    for i, xmax, T in results:
        xmax_T_pair[i] = [xmax, T]

    np.save('data/RTE_bound_data_3.npy', xmax_T_pair)
    return xmax_T_pair




# construct_bound_database(int(1e4))


class PartialNN(nn.Module):
    def __init__(self):
        super(PartialNN, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.block(x)
        # h2 = self.block2(x)
        # x = h + h2
        # x = self.block3(x)
        return x


class PartialDataset(Dataset):
    def __init__(self, data, inverse=False):
        super().__init__()
        self.data = data
        self.inverse = inverse
        if self.inverse:
            print('Using inverse Dataset')
        else:
            print('Using Forward Dataset')

    def __getitem__(self, item):
        if not self.inverse:
            # X, Y = self.data[item, 0], self.data[item, 1]  # xmax, T
            X, Y = np.log(self.data[item, 0]), self.data[item, 1]  # xmax, T
        else:
            # X, Y = self.data[item, 1], self.data[item, 0]  # T, xmax
            X, Y = (self.data[item, 1]), np.log(self.data[item, 0])  # T, xmax

        X = torch.from_numpy(np.array([X])).to(torch.float32).to(device)
        Y = torch.from_numpy(np.array([Y])).to(torch.float32).to(device)
        return X, Y

    def __len__(self):
        return len(self.data)


def train(model, train_loader, test_loader, epoch=1000, inverse=False, base_epoch=0, path_prefix='model1'):
    if inverse:
        os.makedirs(f'models/PartialNN_Inverse/{path_prefix}', exist_ok=True)
        if base_epoch != 0:
            model.load_state_dict(torch.load(f'models/PartialNN_Inverse/{path_prefix}/epoch_{base_epoch}.pth'))
        logging.info('Training Inverse Model')
    else:
        if base_epoch != 0:
            model.load_state_dict(torch.load(f'models/PartialNN/{path_prefix}/epoch_{base_epoch}.pth'))
        os.makedirs(f'models/PartialNN/{path_prefix}', exist_ok=True)
        print('Training Forward Model')

    lt = Loss_Tracker()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for e in range(epoch):
        epoch_loss = 0.
        for (x, y) in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        lt.train_epoch_losses.append(epoch_loss)
        model.eval()
        with torch.no_grad():
            test_loss = 0.
            for (x, y) in test_loader:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss += loss.item()
            lt.test_epoch_losses.append(test_loss)
        if e % 20 == 0:
            print(f'Epoch {e + base_epoch}, Train Loss: {epoch_loss}, Test Loss: {test_loss}')
            if not inverse:
                torch.save(model.state_dict(), f'models/PartialNN/{path_prefix}/epoch_{e + base_epoch}.pth')
            else:
                torch.save(model.state_dict(), f'models/PartialNN_Inverse/{path_prefix}/epoch_{e + base_epoch}.pth')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # construct_bound_database(10000)
    # quit()

    # xmax_T_pair = np.load('data/RTE_bound_data.npy')
    # xmax_T_pair = np.load('data/RTE_bound_data_3.npy')
    # # print(xmax_T_pair)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using {device}')
    # inverse = True
    # trainset, testset = random_split(PartialDataset(xmax_T_pair, inverse=inverse), (0.9, 0.1))
    # # print(trainset[100])
    # train_loader = DataLoader(trainset, batch_size=2000, shuffle=True)
    # test_loader = DataLoader(testset, batch_size=2000, shuffle=False)
    #
    # model = PartialNN().to(device)
    # train(model, train_loader, test_loader, 1000, inverse=inverse, base_epoch=0, path_prefix='log_xmax_model_fine_2')
    # quit()

    xmax_T_pair = np.load('data/RTE_bound_data_3.npy')
    # print(xmax_T_pair)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    inverse = True
    trainset, testset = random_split(PartialDataset(xmax_T_pair, inverse=inverse), (0.9, 0.1))
    # print(trainset[100])
    train_loader = DataLoader(trainset, batch_size=2000, shuffle=True)
    test_loader = DataLoader(testset, batch_size=2000, shuffle=False)


    model = Simple_NN(num_features=1, out_features=1).to(device)
    train(model, train_loader, test_loader, 1000, inverse=inverse, base_epoch=0, path_prefix='log_xmax_model_Resnet_1')
    quit()





