# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:47:46 2018

@author: Kaai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer

# %%



df = pd.read_excel('formatted data.xlsx')
index = np.arange(0, len(df))
df.index = index

columns = list(df.columns.values)

X = df[columns[0:6]]
y = df[columns[6]]

# %%

lr = LinearRegression()
cv = KFold(n_splits=10, shuffle=True)

y_actual = []
y_predicted = []

y_actual = []
y_predicted = []


for train_index, test_index in cv.split(X):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    lr.fit(X_train, y_train)
    y_predicted.extend(lr.predict(X_test))
    y_actual.extend(y_test)


# %%

plt.figure(1, figsize=(9, 9))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)

plt.plot(y_actual, y_predicted, color='#c37edd', marker='o',
         linestyle='None', mew=2, markerfacecolor='w', markersize=12, label='Linear Regression')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)


plt.xlabel('Actual', fontsize=22)
plt.ylabel('Predicted', fontsize=22)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
plt.legend()


plt.show()
print('Score:', r2_score(y_actual, y_predicted))

LR.coef_

