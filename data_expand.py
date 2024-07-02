import pandas as pd
import numpy as np
import random

# 读取数据集
data = pd.read_excel('./data/post_data.xlsx', skiprows=1, sheet_name='Sheet2')
print(data)
data.drop(columns=data.columns[0], axis=1, inplace=True)
data = data.dropna()
data = data.values
print(data)
data = data.astype('float32')

# 从数据集中随机采样数据，并添加扰动，以生成新的数据点
print(data)


def random_sample(data, num):
    new_data = []
    for i in range(num):
        index = random.randint(0, len(data)-1)
        new_data.append(
            data[index] + np.random.normal(0, 0.01, len(data[index])))
    return np.array(new_data)


# 扩张数据集
new_data = random_sample(data, 24)
print(new_data)
data = np.vstack((data, new_data))

# 保存数据集
new_data = pd.DataFrame(data)
new_data.to_excel('./data/post_data_expand.xlsx', index=False)
print("Data expanded and saved.")
