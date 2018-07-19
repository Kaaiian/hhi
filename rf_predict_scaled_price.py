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
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, Normalizer

# %%


df = pd.read_excel('formatted data.xlsx')
index = np.arange(0, len(df))
df.index = index

columns = list(df.columns.values)

X = df[columns[0:5]]
y = df[columns[-1]]

# %%

rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', min_samples_split=2)
cv = KFold(n_splits=10, shuffle=True)

y_actual = []
y_predicted = []

for train_index, test_index in cv.split(X):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    rf.fit(X_train, y_train)
    y_predicted.extend(rf.predict(X_test))
    y_actual.extend(y_test)

#y_predicted = df[columns[5]]

# %%

# =============================================================================
# COPY THIS FUNCTION FOR ALL THE DIFFERENT PROGRAMS
# =============================================================================

axis_dict = {}
axis_dict['algorithm'] = 'Random Forest Regression'

def plot(y_actual, y_predicted, axis_dict, color='#c37edd'):

    plt.figure(1, figsize=(9, 9))
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    plt.plot(y_actual, y_predicted, color=color, marker='o',
             linestyle='None', mew=2, markerfacecolor='w', markersize=12, label=axis_dict['algorithm'])
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal Performance')

    plt.xlabel('Actual Price ($)', fontsize=22)
    plt.ylabel('Predicted Price ($)', fontsize=22)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
    plt.legend()

    plt.savefig('figures/'+axis_dict['algorithm']+'.eps')
    plt.savefig('figures/'+axis_dict['algorithm']+'.png')

    plt.show()
    print('Score:', r2_score(y_actual, y_predicted))

plot(y_actual, y_predicted, axis_dict, color='#c37edd')

# %%

rf.fit(X, y)

# %%

## BROKEN SOMEWHERE IN HER FIX LATER WHEN WE CARE ABOUT THIS
def rf_feature_importance(rf, X_train, N='all', std_deviation=False):
    '''Get feature importances for trained random forest object
    
    Parameters
    ----------
    rf : sklearn RandomForest object
        This needs to be a sklearn.ensemble.RandomForestRegressor of RandomForestClassifier object that has been fit to data
    N : integer, optional (default=10)
        The N most important features are displayed with their relative importance scores
    std_deviation : Boolean, optional (default=False)
        Whether or not error bars are plotted with the feature importance. (error can be very large if maximum_features!='all' while training random forest
    Output
    --------
    graphic :
        return plot showing relative feature importance and confidence intervals

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rf = RandomForestRegressor(max_depth=20, random_state=0)
    >>> rf.fit(X_train, y_train)
    >>> rf_feature_importance(rf, N=15)
    '''

    if N=='all':
        N=X_train.shape[1]
    importance_dic = {}
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    indices = indices[0:N]
    df_I = pd.DataFrame(indices, columns=['indices'])
    df_I['importance'] = importances[indices]

#    Print the feature ranking
    print("Feature ranking:")
    
    for f in range(0, N):
        importance_dic[X_train.columns.values[indices[f]]]=importances[indices[f]]
        print(("%d. feature %d (%.3f)" % (f + 1, indices[f], importances[indices[f]])),':', X_train.columns.values[indices[f]])
        if X_train.columns.values[indices[f]] == 'random':
            rnd_index = indices[f]

    important_indices = df_I[df_I['importance'] >= 2*importances[rnd_index]]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    if std_deviation == True:
        plt.bar(range(0, N), importances[indices], color="r", yerr=std[indices], align="center")
    else:
        plt.bar(range(0, N), importances[indices], color="r", align="center")
    plt.xticks(range(0, N), indices, rotation=90)
    plt.xlim([-1, N])
    plt.show()
    return indices, important_indices
