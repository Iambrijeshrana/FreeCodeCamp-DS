# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:35:18 2019

@author: Brijeshkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

'''
Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will 
not work properly without normalization. For example, the majority of classifiers calculate the distance between two 
points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed 
by this particular feature. Therefore, the range of all features should be normalized so that each feature 
contributes approximately proportionately to the final distance.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

' ***************** Training and Implementation *************************** '

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

'''
parameter, i.e. n_neigbours. This is basically the value for the K. There is no ideal value for K and it is 
selected after testing and evaluation, however to start out, 5 seems to be the most commonly used value for KNN 
algorithm.
'''
y_pred = classifier.predict(X_test)

' ************************** Accuracy Check ****************************** '

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))






