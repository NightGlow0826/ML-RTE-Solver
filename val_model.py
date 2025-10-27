
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X.to(torch.float32).to(device)
        self.Y = Y.to(torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

f = np.load('data/dataset_mesh.npz')

X = torch.from_numpy(f['X']).to(torch.float32)
Y = torch.from_numpy(f['isotropic']).to(torch.float32)

loader = DataLoader(Dataset(X, Y), batch_size=5000, shuffle=True)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

ls = []
for epoch in range(1000):
    epoch_loss = 0
    for (i, (x, y)) in enumerate(loader):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(loader)
    ls.append(epoch_loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, iteration {epoch}, loss: {epoch_loss}')
plt.semilogy(ls)
plt.show()

Y_pred = model(X.to(device).to(torch.float32)).detach().cpu().numpy()

from analytic import evaluate
mean, max = evaluate(Y, Y_pred)
print(f'mean: {mean}, max: {max}')


