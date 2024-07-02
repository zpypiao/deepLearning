import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

current_path = os.path.dirname(__file__)
path = current_path + '/data/'

load_path = path + 'loss/'
files = os.listdir(load_path)

x = range(200)

for file in files:
    file_path = load_path + file
    data = np.load(file_path)
    plt.plot(x, data, label=file.split('.')[0])


plt.legend()
plt.show()
