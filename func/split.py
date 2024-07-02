import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def comb_data():
    data = pd.read_excel('./data/post_data_expand.xlsx', skiprows=1)
    # data.drop(columns=data.columns[0], axis=1, inplace=True)
    data = data.dropna()
    data = data.values
    data = data.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    sub_data = []

    for i in range(6):
        for j in range(i+1, 6):
            sub_data.append(np.concatenate((
                data[:, i].reshape(-1, 1), data[:, j].reshape(-1, 1), data[:, -1].reshape(-1, 1)), axis=1))

    return sub_data


def split_data(dat, ind=0):
    data = dat[ind]
    X, y = data[:, :-1], data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    comb_data()
