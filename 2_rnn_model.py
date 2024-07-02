import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import func.split as my_split

sub_data = my_split.comb_data()


# 创建一个RNN网络
class Net(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


MSE = []


for i in range(len(sub_data)):
    X_train, X_test, y_train, y_test = my_split.split_data(sub_data, i)

    model = Net(2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    h_state = None

    # train model
    for epoch in range(150):
        inputs = torch.from_numpy(X_train).unsqueeze(0)
        labels = torch.from_numpy(y_train[np.newaxis, :, np.newaxis])
        optimizer.zero_grad()
        outputs, h_state = model(inputs, h_state)
        h_state = h_state.data
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print('epoch {}, loss {}'.format(epoch, loss.item()))
    # test model
    model.eval()
    inputs = torch.from_numpy(X_test).unsqueeze(0)
    labels = y_test
    print(labels)
    outputs, h_state = model(inputs, h_state)
    outputs = outputs.detach().numpy()
    outputs = outputs.squeeze()
    rmse = np.sqrt(mean_squared_error(labels, outputs))
    print('rmse {}'.format(rmse))

    MSE.append(rmse)

np.save('./data/outcome/rnn_s.npy', MSE)
