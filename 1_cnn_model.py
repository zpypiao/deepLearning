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
import func.process_data as process_data


# load data
X_train, X_test, Y_train, Y_test = process_data.load_data()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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


losss = []

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# train model
for epoch in range(200):
    inputs = torch.from_numpy(X_train)
    inputs = inputs.reshape(-1, 1, 6)
    labels = torch.from_numpy(Y_train)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    losss.append(loss.item())

# test model
model.eval()
inputs = torch.from_numpy(X_test)
inputs = inputs.reshape(-1, 1, 6)
labels = torch.from_numpy(Y_test)
outputs = model(inputs)
outputs = outputs.detach().numpy().reshape(-1)
mse = mean_squared_error(labels, outputs)
print('mse {}'.format(mse))

# plot
plt.plot(labels, label='true')
plt.plot(outputs, label='forecast')
plt.legend()
plt.show()

np.save('./data/loss/cnn.npy', np.array(losss))
