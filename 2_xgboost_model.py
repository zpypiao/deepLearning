# 导入包
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import split as my_split

sub_data = my_split.comb_data()

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


MSE = []

for i in range(len(sub_data)):
    data = sub_data[i]

    # 分割数据与特征
    X, y = data[:, :-1], data[:, -1]

    # 分割训练集与测试集
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    data_test = xgb.DMatrix(data=X_test)

    # 训练
    model = xgb.train(parameters, data_dmatrix, num)

    # 预测
    preds = model.predict(data_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print('rmse {}'.format(rmse))
    MSE.append(rmse)

np.save('./data/outcome/xgboost_s.npy', np.array(MSE))
