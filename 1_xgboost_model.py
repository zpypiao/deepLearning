# 导入包
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
data = pd.read_excel('./data/post_data_expand.xlsx', skiprows=1)
data.drop(columns=data.columns[0], axis=1, inplace=True)
data = data.dropna()
data = data.values
data = data.astype('float32')

# 分割数据与特征
X, y = data[:, :-1], data[:, -1]

# 分割训练集与测试集
data_dmatrix = xgb.DMatrix(data=X, label=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
data_test = xgb.DMatrix(data=X_test)

# 定义xgboost回归器参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
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

# 计算均方误差
rmse = np.sqrt(mean_squared_error(y_test, preds))

# 绘图展示y-test和预测值
plt.plot(y_test, label='y_test')
plt.plot(preds, label='preds')
plt.legend()
plt.show()


# 输出均方误差
print("RMSE: %f" % (rmse))
