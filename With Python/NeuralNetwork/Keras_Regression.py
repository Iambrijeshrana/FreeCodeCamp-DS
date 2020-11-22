# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:29:46 2020

@author: Brijesh Rana
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('D:/TensorFlow_FILES/Data/kc_house_data.csv')

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

plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='sqft_living', data =df)

plt.figure(figsize=(10,5))
sns.scatterplot(x='bedrooms', y='price', data =df)
sns.boxplot(x='bedrooms', y='price', data =df)


plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data =df)


plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='lat', data =df)

#
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =df, hue='price')

df.price 

df.sort_values('price',ascending=False)
# To remove outliers we are dropping 1 % of data, 1%=215 house
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =non_top_1_perc,
                edgecolor=None,alpha=.2,palette='RdYlGn',hue='price')


plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =non_top_1_perc,hue='price')


sns.boxplot(x='waterfront', y='price', data=df)

# drop id column
df=df.drop('id',axis=1)

df['date']=pd.to_datetime(df['date'])

df['date']

df['year']=df['date'].apply(lambda date: date.year)

df['month']=df['date'].apply(lambda date: date.month)

df.head()

plt.figure(figsize=(12,8))
sns.boxplot(x='month', y='price', data=df)


df.groupby('month').mean()['price']

df.groupby('month').mean()['price'].plot()

df.groupby('year').mean()['price'].plot()
# drop date column its not required 
df=df.drop('date', axis=1)

df.columns

df.head()

df['zipcode'].value_counts()

# zip code we can drop here rite now, but this approach is not usefel in every case (here domain knowledge require)
# based on zip code we can identify the expensive areas
df=df.drop('zipcode', axis=1)

df['yr_renovated'].value_counts()

df=df.drop('yr_renovated', axis=1)