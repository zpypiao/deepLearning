import numpy as np
import pandas as pd
import sys
import os

current_path = os.path.dirname(__file__)
path = current_path + '/data/'

load_path = path + 'outcome/'
files = os.listdir(load_path)

out = []
header = []

for file in files:
    file_path = load_path + file
    data = np.load(file_path).tolist()
    out.append(data)
    header.append(file.split('.')[0])


out_csv = pd.DataFrame(out).T

out_csv.columns = header

out_csv.to_csv(path + 'outcome.csv', index=True)
