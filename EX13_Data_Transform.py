#coding:utf-8
from sklearn import preprocessing
import numpy as np

ar = np.array([[5000], [58000], [16000]])
min_max_scaler = preprocessing.MinMaxScaler()
minmax_ar = min_max_scaler.fit_transform(ar)
print(minmax_ar)
