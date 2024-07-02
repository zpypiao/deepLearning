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


sub_data = my_split.comb_data()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(2, 30)
        self.fc = nn.Linear(30, 1)

    def init_hidden(self, seq_len):
        return (torch.zeros(1, seq_len, 30), torch.zeros(1, seq_len, 30))

    def forward(self, x):
        seq_len = x.size(0)
        self.hidden = self.init_hidden(seq_len)
        x, self.hidden = self.lstm(x.view(1, seq_len, -1), self.hidden)
        x = x.view(-1, 30)
        x = self.fc(x)
        x = x.view(seq_len, -1)
        return x


MSE = []

for i in range(len(sub_data)):
    X_train, X_test, y_train, y_test = my_split.split_data(sub_data, i)

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    # train model
    for epoch in range(200):
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if epoch+1 % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    # test model
    model.eval()
    inputs = torch.from_numpy(X_test)
    labels = torch.from_numpy(y_test).view(-1, 1)
    outputs = model(inputs)
    outputs = outputs.detach().numpy()
    rmse = np.sqrt(mean_squared_error(labels, outputs[:, 0]))
    print('rmse {}'.format(rmse))
    MSE.append(rmse)


np.save('./data/outcome/lstm_s.npy', MSE)
