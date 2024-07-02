import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data():
    data = pd.read_excel('./data/post_data_expand.xlsx', skiprows=1)
    # data.drop(columns=data.columns[0], axis=1, inplace=True)
    data = data.dropna()
    data = data.values
    data = data.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    # 分割数据与特征
    X, y = data[:, :-1], data[:, -1]

    # 分割训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test
