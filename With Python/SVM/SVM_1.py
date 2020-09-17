# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:26:20 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import  load_breast_cancer

cancer = load_breast_cancer()

cancer.describe()

type(cancer)

cancer.keys()

print(cancer)

print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

df_feat

df_feat.info()

df_feat.describe()

cancer['target_names']

df_feat.isnull().any()

df_feat.isnull().sum()


from sklearn.model_selection import train_test_split

X=df_feat
y=cancer['target']
X_train, X_test, y_train, y_test= train_test_split(X, y , test_size=.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

'''
c-control the misclassification rate 
large c value will give low bias and high variance
small c value will give high bias and low variance
'''
from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_

grid.best_estimator_

grid_prediction = grid.predict(X_test)

print(classification_report(y_test, grid_prediction))
print(confusion_matrix(y_test, grid_prediction))
