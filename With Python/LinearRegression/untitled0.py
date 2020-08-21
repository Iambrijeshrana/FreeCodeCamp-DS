# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:43:35 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  plotly as pt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score
from sklearn import metrics
traindata = pd.read_csv('D:/Personal/Dataset/LR/train.csv')

testdata = pd.read_csv('D:/Personal/Dataset/LR/test.csv')


traindata.describe()

# Missing values 
# To check column name which have missing values
traindata.isnull().any()
# to check no of missing values, column wise
traindata.isnull().sum()

traindata.isnull().sum().sum()

sns.boxplot(traindata['y'])

traindata.shape
# fill missing values by mean
traindata['y'].fillna(traindata['y'].mean(), inplace=True)


sns.relplot(x="x", y="y", data=traindata)

sns.relplot(x=traindata['x'], y=traindata['y'], data=traindata)

sns.regplot(x=traindata['x'], y=traindata['y'], data=traindata)

sns.distplot(traindata['x'], bins=10)

sns.boxplot(traindata['x'], orient='v')

sns.boxplot(traindata['y'], orient='v')

Q1=traindata.quantile(.25)
Q3=traindata.quantile(.75)
IQR = traindata.apply(stats.iqr)
upper = (Q3 + 1.5 * IQR)
lower = (Q1 - 1.5 * IQR)
# No of outliers for each column

# To see no of outliers for each variable
(traindata > (Q3 + 1.5 * IQR)).sum()
(traindata < (Q1 - 1.5 * IQR)).sum()

df=((traindata < (Q1 - 1.5 * IQR)) | (traindata > (Q3 + 1.5 * IQR))).sum()

sns.distplot(traindata['y'])
sns.distplot(traindata['x'])

# See the outliers 
df_out = traindata[((traindata < lower) |(traindata > upper)).any(axis=1)]

# Exclude outliers
newtraindata = traindata[~((traindata < (Q1 - 1.5 * IQR)) |(traindata > (Q3 + 1.5 * IQR))).any(axis=1)]

newtraindata.shape


f, axes = plt.subplots(1, 2)
sns.distplot(newtraindata['y'], ax=axes[0])
sns.distplot(newtraindata['x'], ax=axes[1])

regressor = LinearRegression() 

Xtrain = newtraindata['x'].values.reshape(-1,1)
ytrain = newtraindata['y'].values.reshape(-1,1)

Xtest = testdata['x']
regressor.fit(Xtrain, ytrain)

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

ytest=testdata['y'].values.reshape(-1,1)
Xtest=testdata['x'].values.reshape(-1,1)

ypred = regressor.predict(Xtest)


df=pd.DataFrame({'Actual': ytest.flatten(), 'Predicted': ypred.flatten()})

df

print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, ypred))  
print('Mean Squared Error:', metrics.mean_squared_error(ytest, ypred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print('Median absolute error:',metrics.median_absolute_error(ytest, ypred))

r2=regressor.score(ytest, ypred)

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(ytest, ypred)

from yellowbrick.regressor import ResidualsPlot

# residuals vs. predicted values
visualizer = ResidualsPlot(regressor)
visualizer.score(Xtest, ytest)  # Evaluate the model on the test data
visualizer.show() 


sns.residplot(ytest, ypred)

np.mean(ytest-ypred)


sns.distplot(ytest-ypred)

