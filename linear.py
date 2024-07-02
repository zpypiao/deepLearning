import matplotlib.pyplot as plt
import func.process_data as ld
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成数据
X_train, X_test, y_train, y_test = ld.load_data()


# 定义神经网络


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(6, 50),
                                # nn.ReLU(),
                                nn.Linear(50, 20),
                                # nn.ReLU(),
                                nn.Linear(20, 5),
                                nn.Linear(5, 1)
                                )

    def forward(self, x):
        return self.fc(x)


model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练网络
for epoch in range(200):
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).float()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))


# 测试网络
model.eval()
inputs = torch.from_numpy(X_train).float()
outputs = model(inputs)

# 画图显示测试输出与真实值

plt.plot(outputs.detach().numpy(), label='predict')
plt.plot(y_train, label='label')
plt.legend()
plt.show()


# 测试网络
model.eval()
inputs = torch.from_numpy(X_test).float()
outputs = model(inputs)

# 画图显示测试输出与真实值

plt.plot(outputs.detach().numpy(), label='predict')
plt.plot(y_test, label='label')
plt.legend()
plt.show()
