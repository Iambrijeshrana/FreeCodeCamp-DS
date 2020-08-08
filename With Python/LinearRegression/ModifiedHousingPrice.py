# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:08:14 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
from pandas_profiling import ProfileReport
from scipy import stats
from sklearn.model_selection import train_test_split # Import train_test_split function
# Import the Housing Price dataset
housingData = pd.read_csv("D:/Personal/Dataset/housingdata.csv")

# Rename the columns (Give the name to columns)
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housingData.columns = housing_colnames

housingData.columns

# Lets start data analysis part -- EDA

# shape of data 
housingData.shape

# corelation between variables - 

#We can see the corelation in table format also and in heatmap also, well i prefer heatmap

# Corelation in table format
housingData.corr()
# Corelation using heatmap
sns.heatmap(housingData.corr(), annot=True, annot_kws={'size': 10})
# - annot=True - to show the corelation values
# - annot_kws - use to avoide the coreltion values overlap

# To see is there any missing values for any columns in the dataset 
# (result will be true or false)
housingData.isnull().any()
housingData.isnull().values.any()

# Total missing values for each feature
housingData.isnull().sum()

# Total number of missing values in the dataset
housingData.isnull().sum().sum()

# Find outliers 

z = np.abs(stats.zscore(housingData))
print(z)

threshold = 3
print(np.where(z > 3))

# z value of an outlier
z[55,1]

housingData.iloc[[55],1]

# Remove outliers
data_clean = housingData[(z<3).all(axis=1)]
data_clean.shape

# Fine and remove outliers using IQR 
#find Q1, Q3, and interquartile range for each column
Q1 = housingData.quantile(q=.25)
Q3 = housingData.quantile(q=.75)
IQR = housingData.apply(stats.iqr)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
data_clean2 = housingData[~((housingData < (Q1-1.5*IQR)) | (housingData > (Q3+1.5*IQR))).any(axis=1)]
#find how many rows are left in the dataframe 
data_clean2.shape

(89,3)

# Check how our target variable is distributed 

# f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(housingData['MEDV'], bins=10)
sns.pairplot(housingData)

#

''
# EDA with sweetviz
sweetviz_report = sv.analyze(housingData)
sweetviz_report.show_html('sweetviz_report.html')
# Since it's a simple linear regression model and we have seen the corelation between rm and 
# medv is very high so lets take rm variable to predict the vakues odf medv


# run the profile report 
profile = housingData.profile_report(title='Pandas Profiling Report') 
   
# save the report as html file 
profile.to_file(output_file="D:/pandas_profiling1.html") 
   
# save the report as json file 
profile.to_file(output_file="D:/pandas_profiling2.json") 

# lets do regression

# Train test split

# Devide the data into train and test dataset
# we have to do the reshape if only one feature is there in the model
X = housingData['RM'].values.reshape(-1,1)
y = housingData['MEDV'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=0)

# Relationship between dependent and Independent variables in training dataset
sns.scatterplot(housingData['RM'], housingData['MEDV'])
sns.set(set(xlabel='RM', ylabel='MEDV'))

plt.relplot(X_test, y_test,  color='gray')
