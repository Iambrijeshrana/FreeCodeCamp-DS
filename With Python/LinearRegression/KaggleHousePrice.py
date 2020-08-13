# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:41:58 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Import train and test data set 

trainData = pd.read_csv('D:/Personal/Dataset/House-Price/train.csv')
testData = pd.read_csv('D:/Personal/Dataset/House-Price/test.csv')

# Check data

trainData.head()

trainData.shape

# Data preprocessing
# To see is there any missing values for any columns in the dataset (result will be true or false)
trainData.isnull().any()
# Total no of variables where values are missing
trainData.isnull().any().sum()
# Total number of missing values in the dataset
trainData.isnull().sum().sum()

# total missing values for each variable (missing values variable wise)
trainData.isnull().sum()

#Percentage of missing Values for each columns
percent_missing = trainData.isnull().sum() * 100 / len(trainData)
missing_value_df = pd.DataFrame({'column_name': trainData.columns,
                                 'percent_missing': percent_missing})

missing_value_df[percent_missing>50]
# Sort the dataframe values based on missing percentage values
missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)

# For the column PoolIQC, MiscFeature, Alley, Fence missing values are more then 50% so lets drop these columns from the dataset
trainData=trainData.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

trainData.shape

trainData.dtypes

# Now lets devide the dataset into two gropus - 1. categoric and 2. numeric
categoricalTrainData = trainData.select_dtypes(include=['object'])
numericTrainData = trainData.select_dtypes(exclude=['object'])

categoricalTrainData.head(5)

categoricalTrainData.shape

numericTrainData.shape

# Now again check the missing values 
# No of missing values in each column
categoricalTrainData.isnull().sum()

percent = categoricalTrainData.isnull().sum()*100/len(categoricalTrainData)

catePercent = pd.DataFrame({'column_name': categoricalTrainData.columns,
                           'percent_missing': percent})


catePercent.sort_values('percent_missing', ascending=False, inplace=True)

catePercent

# For the column FireplaceQu missing values are 47% so lets fill them with None

categoricalTrainData['FireplaceQu'].fillna('None', inplace=True)

# Now lets fill remainning missing values by Mode
for column in categoricalTrainData.columns:
    categoricalTrainData[column].fillna(categoricalTrainData[column].mode()[0], inplace=True)

'''Now we have cleaned datafor categorica datatype'''    

numericTrainData.isnull().sum()

percentage = numericTrainData.isnull().sum()*100/len(numericTrainData)

numPercent = pd.DataFrame({'Column_name':numericTrainData.columns
                           ,'percent_missing':percentage})

numPercent.sort_values('percent_missing', ascending=False, inplace=True)

numPercent

# Lets fill the missing values by mean

# First check the outliers 

Q1=numericTrainData.quantile(.25)
Q3=numericTrainData.quantile(.75)

IQR = numericTrainData.apply(stats.iqr)

numericTrainData[numericTrainData>(q3+1.5*IQR)]

df=numericTrainData[((numericTrainData < (Q1 - 1.5 * IQR)) |
                     (numericTrainData > (Q3 + 1.5 * IQR))]

data_clean2 = numericTrainData[((numericTrainData < (Q1-1.5*IQR)) | (numericTrainData > (Q3+1.5*IQR))).any(axis=1)]
                                   
plt.figure(figsize=(15,10))
sns.boxplot(numericTrainData.SalePrice, vert=True)                                                          
plt.show()                     
                                                          
                                                          