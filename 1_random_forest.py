import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import func.process_data as process_data


X_train, X_test, y_train, y_test = process_data.load_data()
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print('rmse {}'.format(rmse))
plt.plot(y_test, label='y_test')
plt.plot(preds, label='preds')
plt.legend()
plt.show()
