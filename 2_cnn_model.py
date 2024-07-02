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


# 创建一个RNN网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 3, 2, 2, 1),
            # nn.MaxPool1d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # nn.Linear(6, 1),
            nn.Linear(3*2, 350),
            nn.ReLU(),
            nn.Linear(350, 1),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


sub_data = my_split.comb_data()
MSE = []

for i in range(len(sub_data)):
    X_train, X_test, y_train, y_test = my_split.split_data(sub_data, i)

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # train model
    for epoch in range(100):
        inputs = torch.from_numpy(X_train)
        inputs = inputs.reshape(-1, 1, 2)
        labels = torch.from_numpy(y_train).view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print('epoch {}, loss {}'.format(epoch, loss.item()))

    # test model
    model.eval()
    inputs = torch.from_numpy(X_test)
    inputs = inputs.reshape(-1, 1, 2)
    labels = torch.from_numpy(y_test)
    outputs = model(inputs)
    outputs = outputs.detach().numpy().reshape(-1)
    rmse = np.sqrt(mean_squared_error(labels, outputs))
    MSE.append(rmse)
    print('rmse {}'.format(rmse))


np.save('./data/outcome/cnn_s.npy', np.array(MSE))
