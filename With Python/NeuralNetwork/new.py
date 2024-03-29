# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:58:47 2020

@author: Brijesh.R
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('D:/Personal/TensorFlow_FILES/Data/kc_house_data.csv')
i=12
abc = 'ddd-'+str(i)

# check missing data 
df.isnull().any()
df.isnull().sum()
df.isnull().sum().sum()

df.describe()

df.describe().transpose()

plt.figure(figsize=(10,5))
sns.distplot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])

df['bedrooms'].value_counts()
df['bedrooms'].value_counts(normalize=True)*100

df.corr()['price'].sort_values()

sns.heatmap(df.corr(), annot=True)

plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='sqft_living', data =df)

plt.figure(figsize=(10,5))
sns.scatterplot(x='bedrooms', y='price', data =df)
sns.boxplot(x='bedrooms', y='price', data =df)


plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data =df)


plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='lat', data =df)
