# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:39:19 2020

@author: Brijesh Rana
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('D:/TensorFlow_FILES/Data/cancer_classification.csv')

df.shape()

df.info()

df.describe().transpose()

sns.countplot(x='benign_0__mal_1', data=df)  

df.corr()

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.show()

df.corr()['benign_0__mal_1']

df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

df.corr()['benign_0__mal_1'].sort_values()[:-1].plot(kind='bar')

from sklearn.model_selection import train_test_split

X=df.drop('benign_0__mal_1', axis=1).values
y=df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.42, random_state=101)



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
