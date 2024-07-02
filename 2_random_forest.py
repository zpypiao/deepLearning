import func.process_data as process_data
import func.split as my_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import func.split as my_split

sub_data = my_split.comb_data()

MSE = []

for i in range(len(sub_data)):
    X_train, X_test, y_train, y_test = my_split.split_data(sub_data, i)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    MSE.append(rmse)
    # print('rmse {}'.format(rmse))
    # plt.plot(y_test, label='y_test')
    # plt.plot(preds, label='preds')
    # plt.legend()
    # plt.show()


print('MSE:', MSE)

random_forest_s = np.array(MSE)

np.save('./data/outcome/random_forest_s.npy', random_forest_s)
