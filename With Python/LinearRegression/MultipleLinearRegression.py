# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:11:51 2019

@author: brijesh
"""

"In this we will try to predict the price of House based on more than one predictor"

import pandas as pd # Import pandas to import and analyze the data
import numpy as np # Import numpy to perform mathematical funcations on data

import plotly
%matplotlib inline
#import plotly.plotly as py
import matplotlib.pyplot as plt # Import pyplot
import seaborn as sns
from matplotlib import style
from scipy import stats
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score

"************************** Step 1. Collect the data *****************************"

# Import the Housing Price dataset
housingData = pd.read_csv("D:/Personal/Dataset/housingdata.csv")

# Rename the columns (Give the name to columns)
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housingData.columns = housing_colnames

housingData.columns

"************************** Step 2 & 3. Analyze the data & Data Cleaning *****************************"

# Check the shape of data
housingData.shape

# To check the relationship (Correlation) between target variable (MEDV) and dependent variable 

# To see the correlation in tabular format
housingData.corr()

# To plot the correlation of features with target variable using heatmap of seaborn
fig = plt.subplots(figsize = (10,10))
sns.set(font_scale=1.5)
sns.heatmap(housingData.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.show()

''' well correlation between RM and MEDV is very high (.70), so we are going to build the model based on RM'''

# To see the relation between Dependent variable and Independent variable in Graphs 
#Utility function for plotting the relationships b/w features and target variables
def plotFeatures(col_list,title):
    plt.figure(figsize=(10, 14))
    i = 0
    print(len(col_list))
    for col in col_list:
        i+=1
        plt.subplot(7,2,i)
        plt.plot(housingData[col],housingData["MEDV"],marker='.',linestyle='none')
        plt.title(title % (col))   
        plt.tight_layout()
        
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
plotFeatures(colnames,"Relationship bw %s and MEDV")

# Plot Simple histogram to see the distribution of MEDV data in the dataset 
plt.hist(housingData['MEDV'])

# Plot histogram with curve line to see the distribution of MEDV data in the dataset
sns.distplot(housingData['MEDV'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# To see is there any missing values for any columns in the dataset (result will be true or false)
housingData.isnull().values.any()

# Total missing values for each feature
housingData.isnull().sum()

# Total number of missing values in the dataset
housingData.isnull().sum().sum()

# Find outliers in the entire dataset for all the variables
# If z value > 3 or <-3 than it's outlier
z = np.abs(stats.zscore(housingData))
print(z)

threshold = 3
print(np.where(z > 3))

""" Don’t be confused by the results. The first array contains the list of row numbers and 
second array respective column numbers, which mean z[55][1] have a Z-score higher than 3.
"""
# z value of an outlier
z[55,1]

# So the data point — 55th record on column ZN is an outlier, Actual value of Outlier in dataset
housingData.iloc[[55],1]

"************************** Step 4. Train Test (Build the model) *****************************"