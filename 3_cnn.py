import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import func.split as my_split


X_train, X_test, y_train, y_test = my_split.split_data(my_split.comb_data(), 0)


# 创建一个RNN网络
class Net(nn.Module):
    def __init__(self, k, m):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, k, 2, 2, 1),
            # nn.MaxPool1d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # nn.Linear(6, 1),
            nn.Linear(k*2, m),
            nn.ReLU(),
            nn.Linear(m, 1),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# mse = 100

# for k in range(3, 10):
#     for m in range(20, 400, 10):
#         model = Net(k, m)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.01)
#         scheduler = ExponentialLR(optimizer, gamma=0.9)

#         # train model
#         for epoch in range(200):
#             inputs = torch.from_numpy(X_train)
#             inputs = inputs.reshape(-1, 1, 2)
#             labels = torch.from_numpy(y_train).view(-1, 1)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             # if (epoch+1) % 10 == 0:
#             #     print('epoch {}, loss {}'.format(epoch+1, loss.item()))

#         # test model
#         model.eval()
#         inputs = torch.from_numpy(X_test)
#         inputs = inputs.reshape(-1, 1, 2)
#         labels = torch.from_numpy(y_test)
#         outputs = model(inputs)
#         outputs = outputs.detach().numpy().reshape(-1)
#         rmse = np.sqrt(mean_squared_error(labels, outputs))
#         print('rmse {}'.format(rmse))
#         if mse > rmse:
#             kk = k
#             mm = m
#             mse = rmse
#             print('k={}, m={}, rmse={}'.format(k, m, rmse))
# print('k={}, m={}, rmse={}'.format(kk, mm, mse))

model = Net(3, 350)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# train model
for epoch in range(200):
    inputs = torch.from_numpy(X_train)
    inputs = inputs.reshape(-1, 1, 2)
    labels = torch.from_numpy(y_train).view(-1, 1)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print('epoch {}, loss {}'.format(epoch+1, loss.item()))

# test model
model.eval()
inputs = torch.from_numpy(X_test)
inputs = inputs.reshape(-1, 1, 2)
labels = torch.from_numpy(y_test)
outputs = model(inputs)
outputs = outputs.detach().numpy().reshape(-1)
rmse = np.sqrt(mean_squared_error(labels, outputs))
print('rmse {}'.format(rmse))
# plot
plt.plot(labels, label='true')
plt.plot(outputs, label='forecast')
plt.legend()
plt.show()
