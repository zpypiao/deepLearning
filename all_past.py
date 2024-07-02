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
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import plot_importance

data = pd.read_excel('./data/post_data_expand.xlsx', skiprows=1)
data = data.dropna()
data = data.values
data = data.astype('float32')

scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data)
# 分割数据与特征
X, y = data[:, :-1], data[:, -1]

# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)
y_train = scaler2.fit_transform(y_train.reshape(-1, 1))
y_test = scaler2.transform(y_test.reshape(-1, 1))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 10, 2, 2, 1),
            nn.MaxPool1d(2, 2),
            # nn.Conv1d(10, 20, 2, 2, 1),
        )
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# train model
for epoch in range(200):
    inputs = torch.from_numpy(X_train)
    inputs = inputs.reshape(-1, 1, 6)
    labels = torch.from_numpy(y_train)
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
inputs = inputs.reshape(-1, 1, 6)
labels = torch.from_numpy(y_test)
outputs = model(inputs)
outputs = outputs.detach().numpy()


cnn_outputs = scaler2.inverse_transform(outputs)


class LST(nn.Module):
    def __init__(self):
        super(LST, self).__init__()
        self.lstm = nn.LSTM(6, 50)
        self.fc = nn.Linear(50, 1)

    def init_hidden(self, seq_len):
        return (torch.zeros(1, seq_len, 50), torch.zeros(1, seq_len, 50))

    def forward(self, x):
        seq_len = x.size(0)
        self.hidden = self.init_hidden(seq_len)
        x, self.hidden = self.lstm(x.view(1, seq_len, -1), self.hidden)
        x = x.view(-1, 50)
        x = self.fc(x)
        x = x.view(seq_len, -1)
        return x


model = LST()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# train model
for epoch in range(200):
    inputs = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch+1 % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


# test model
model.eval()
inputs = torch.from_numpy(X_test)
labels = torch.from_numpy(y_test)
outputs = model(inputs)
outputs = outputs.detach().numpy()
lst_outputs = scaler2.inverse_transform(outputs)

# 创建一个RNN网络


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN, self).__init__()

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


model = RNN(6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
h_state = None

# train model
for epoch in range(200):
    inputs = torch.from_numpy(X_train).unsqueeze(0)
    labels = torch.from_numpy(y_train[np.newaxis, :, np.newaxis])
    optimizer.zero_grad()
    outputs, h_state = model(inputs, h_state)
    h_state = h_state.data
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
# test model
model.eval()
inputs = torch.from_numpy(X_test).unsqueeze(0)
labels = y_test
outputs, h_state = model(inputs, h_state)
outputs = outputs.detach().numpy()
outputs = outputs.squeeze().reshape(-1, 1)
rnn_outputs = scaler2.inverse_transform(outputs)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rf_outputs = scaler2.inverse_transform(preds.reshape(-1, 1))


data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
data_test = xgb.DMatrix(data=X_test)

# 定义xgboost回归器参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}
parameters = list(params.items())

# 训练次数
num = 300

# 训练
model = xgb.train(parameters, data_dmatrix, num)

# 预测
preds = model.predict(data_test)


xgb_outputs = scaler2.inverse_transform(preds.reshape(-1, 1))

X_test = scaler1.inverse_transform(X_test)


y_test = scaler2.inverse_transform(y_test)

std = [0.1, 0.15, 0.08, 0.1, 0.1, 0.05]

yll = []
for each in y_test:
    yl = []
    for i in range(6):
        yl.append(np.random.normal(each, std[i])[0])
    yll.append(yl)

yll = pd.DataFrame(yll)
yll.to_excel('./data/y_test.xlsx', index=False)

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
cnn_outputs = pd.DataFrame(cnn_outputs)
lst_outputs = pd.DataFrame(lst_outputs)
rnn_outputs = pd.DataFrame(rnn_outputs)
rf_outputs = pd.DataFrame(rf_outputs)
xgb_outputs = pd.DataFrame(xgb_outputs)


result = pd.concat([X_test, y_test, cnn_outputs, lst_outputs,
                   rnn_outputs, rf_outputs, xgb_outputs], axis=1)


result.to_excel('./data/result.xlsx', index=False)
