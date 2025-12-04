
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

# from temp import Loss_Tracker
from RTE_Truth_Model import RTE
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from NN import eval, Simple_NN
class Simple_Inverse_NN(nn.Module):
    def __init__(self, input_prop = 2, hidden_base=128):
        super(Simple_Inverse_NN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_prop, hidden_base),
            nn.Tanh(),
            nn.Linear(hidden_base, hidden_base * 2),
            nn.Tanh(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(input_prop, hidden_base * 2),
            nn.Tanh(),
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_base * 2, hidden_base),
            nn.Tanh(),
            nn.Linear(hidden_base, 2),
        )
    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(x)
        x = h1 + h2
        x = self.block3(x)
        return x

class Inverse_ANN(nn.Module):
    def __init__(self):
        super(Inverse_ANN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        return self.block1(x)


class Inverse_Dataset(Dataset):
    def __init__(self, path, phase_type=None, extrapath=None, use_log_model=False,
                 subdomain=False):
        super().__init__()
        f = np.load(path)
        self.target = torch.from_numpy(f['X']).to(torch.float32).to(device) # xmax, omega
        self.inp = torch.from_numpy(f[phase_type]).to(torch.float32).to(device)# T, R

        if extrapath is not None:
            f = np.load(extrapath)
            self.target = torch.cat((self.target, torch.from_numpy(f['X']).to(torch.float32).to(device)), dim=0)
            self.inp = torch.cat((self.inp, torch.from_numpy(f[phase_type]).to(torch.float32).to(device)), dim=0)
        else:
            ...

        if use_log_model:
            print('Using log model')
            self.target[:, 0] = torch.log(self.target[:, 0])
        else:
            print('Not using log model')

        if subdomain:
            # choose the T > 0.2 and R < 0.3
            mask = (self.inp[:, 0] > 0.05) & (self.inp[:, 1] < 0.5)
            self.inp = self.inp[mask]
            self.target = self.target[mask]
        else:
            print('Suggest Using Subdomain')
            ...
    def __getitem__(self, item):
        return self.inp[item], self.target[item]

    def __len__(self):
        return len(self.inp)

def train(model: nn.Module, train_loader, optimizer, epochs, test_loader, save=True, loss_mode=0,
          phase_type='iso', parent_path='model_1'):


    lt = Loss_Tracker()
    parent_path = parent_path
    str_loss = {0: 'SimpleNN', 1: 'WithBound'}
    print(f'Using {str_loss[loss_mode]}')
    # os.makedirs(f'inverse_model/{parent_path}/{phase_type}/{str_loss[loss_mode]}', exist_ok=True)

    if use_log_model:
        os.makedirs(f'inverse_model/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model', exist_ok=True)
    else:
        os.makedirs(f'inverse_model/{parent_path}/{phase_type}/{str_loss[loss_mode]}', exist_ok=True)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.1)
    for epoch in range(epochs):
        epoch_total_loss = 0.
        model.train()
        for (i, (X, Y)) in enumerate(train_loader):
            Y_pred = model(X)

            loss_fn = nn.MSELoss(reduction='mean')
            # SAMPLE LOSS
            loss_sample = loss_fn(Y, Y_pred)
            epoch_total_loss += loss_sample.item()

            # BOUND LOSS
            bound_mask = (Y_pred[:, 1] < 0) | (Y_pred[:, 1] > 1)
            loss_bound = loss_fn(
                Y_pred[:, 1][bound_mask], torch.clamp(Y_pred[:, 1][bound_mask], 0, 1))




            if loss_mode == 0:
                loss = loss_sample
            elif loss_mode == 1:
                loss = loss_sample + loss_bound
            else:
                raise ValueError('...')

            # if   swith_opti = 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_mean_loss = epoch_total_loss / len(train_loader)

        lt.train_epoch_losses.append(epoch_mean_loss)
        eval_loss = eval(model, test_loader, epoch)
        lt.test_epoch_losses.append(eval_loss)
        if epoch % 20 == 0:
            print(f'epoch: {epoch}, train: {epoch_mean_loss}, eval: {eval_loss}, violation: {loss_bound}')

        if epoch % 20 == 0 and save:
            if use_log_model:
                torch.save(model.state_dict(),
                       f'inverse_model/{parent_path}/{phase_type}/{str_loss[loss_mode]}_log_model/epoch_{epoch}.pth')
            else:
                torch.save(model.state_dict(),
                           f'inverse_model/{parent_path}/{phase_type}/{str_loss[loss_mode]}/epoch_{epoch}.pth')
            ...

        # scheduler.step(eval_loss)
    plt.semilogy(lt.train_epoch_losses, label='train loss')
    plt.semilogy(lt.test_epoch_losses, label='test loss')
    plt.title(f'Phase: {phase_type}, loss_mode: {str_loss[loss_mode]}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return model, optimizer
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_log_model = True
    phase_type = 'iso'
    subdomain = True
    # trainset, testset = random_split(Inverse_Dataset('data/RTE_DATA.npz', phase_type, use_log_model=use_log_model), [0.9, 0.1])
    extra_path = 'data/forward_fine_data/set_finer_extra.npz'
    trainset, testset = random_split(Inverse_Dataset('data/forward_fine_data/set_finer.npz', extrapath=extra_path,phase_type=phase_type, use_log_model=use_log_model, subdomain=subdomain), [0.9, 0.1])


    trainloader = DataLoader(trainset, batch_size=9999, shuffle=True)
    testloader = DataLoader(testset, batch_size=9999, shuffle=False)
    model = Simple_NN(num_features=2).to(device)
    # model = Simple_Inverse_NN(input_prop=2, hidden_base=128).to(device)
    # model = Inverse_ANN().to(device)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    train(model, trainloader, optimizer, 400, testloader,
          save=True, loss_mode=1,phase_type=phase_type, parent_path='model_Resnet_3')


