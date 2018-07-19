# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:47:46 2018

@author: Kaai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV

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


#sns.distplot(y.values, fit=norm)



# %%
#
#
#def grid_search(ml_model, X, y, parameter_candidates, n_cv=5, scale_data=True):
#    if scale_data is True:
#        scaler = StandardScaler().fit(X)
#        X = pd.DataFrame(scaler.transform(X),
#                         index=X.index.values,
#                         columns=X.columns.values)
#
#        normalizer = Normalizer().fit(X)
#        X = normalizer.transform(X)
#
#    grid = GridSearchCV(estimator=ml_model,
#                                        param_grid=parameter_candidates,
#                                        cv=n_cv,
#                                        n_jobs=2)
#
#    grid.fit(X, y)
#    optimised_svr = grid.best_params_
#    
#    return optimised_svr, grid.cv_results_
#
#
#def display_grid_search(parameter_candidates, grid_results, log_scale=True, save=True, save_name='grid_search'):
#    keys = list(parameter_candidates.keys())
#    xx = parameter_candidates[keys[0]]
#    yy = parameter_candidates[keys[1]]
#    
#    grid_scores = list(grid_results['mean_test_score'])
#    
#    composite_list = [grid_scores[x:x+len(xx)] for x in range(0, len(grid_scores),len(xx))]
#    C = np.array(composite_list)
#    
#    XX, YY = np.meshgrid(xx, yy)
#    
#    plt.figure(1, figsize=(10, 10))
#    plt.pcolor(XX, YY, C, cmap='coolwarm')
#    plt.xlabel(keys[0])
#    plt.ylabel(keys[1])
#    plt.title('Model Score (R^2)')
#    plt.xticks(xx)
#    plt.yticks(yy)
#    plt.clim(0, 1)
#    plt.colorbar()
#    if log_scale is True:
#        plt.loglog()
#    if save is True:
#        plt.savefig('figures\\' + save_name + '.tiff', format='tiff', dpi=1200)
#    plt.show()
#
#
## %%
#parameter_candidates = {}
#parameter_candidates['C'] = np.logspace(-3, 1, num=10)
#parameter_candidates['gamma'] = np.logspace(-1, 3, num=10)
#
#ml_model = SVR(kernel='rbf', C=1e2, gamma=1e-1)
#svr_params, grid_results = grid_search(ml_model, X, y, parameter_candidates, n_cv=5, scale_data=True)
#
#display_grid_search(parameter_candidates, grid_results, log_scale=True, save=True, save_name='grid_search')
#

# %%


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

normalizer = Normalizer()
X_scaled = normalizer.fit_transform(X_scaled)

X_scaled = pd.DataFrame(X_scaled)

## polynomial kernel
svr = SVR(kernel='poly', C=10, degree=3)
## best estimator (rbf kernel)
#svr = SVR(**svr_params)
cv = KFold(n_splits=5, shuffle=True,  random_state=1)

y_actual = []
y_predicted = []

y_actual = []
y_predicted = []

for train_index, test_index in cv.split(X_scaled):
    X_train, y_train = X_scaled.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X_scaled.iloc[test_index], y.iloc[test_index]
    svr.fit(X_train, y_train)
    y_predicted.extend(svr.predict(X_test))
    y_actual.extend(y_test)

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

print('Score = ', r2_score(y_actual, y_predicted))


