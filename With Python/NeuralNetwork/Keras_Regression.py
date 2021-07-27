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

# To see basic stats of dataset
df.describe()
# To transpose the matrix 
df.describe().transpose()

# ********** EDA ***********
# Price distribution
plt.figure(figsize=(10,5))
sns.distplot(df['price'])
plt.show()
# no of houses based on bedroom
sns.countplot(df['bedrooms'])

df['bedrooms'].value_counts()
df['bedrooms'].value_counts(normalize=True)*100

# correlation between variables
df.corr()['price'].sort_values()
# variable corelation using heatmap
plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot=True)
plt.show()
# relationship between price and sqft living 
plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='sqft_living', data =df)
# relationship between price and bedrooms 
plt.figure(figsize=(10,5))
sns.scatterplot(x='bedrooms', y='price', data =df)
sns.boxplot(x='bedrooms', y='price', data =df)

# relationship between price and longitude
plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data =df)

# relationship between price and latitude
plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='lat', data =df)

# plot price along with longitude and latitude 
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =df, hue='price')

df.price 

df.sort_values('price',ascending=False)
# To remove outliers we are dropping 1 % of data, 1%=215 house
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

# plot price along with longitude and latitudeafter removing outliers
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =non_top_1_perc,
                edgecolor=None,alpha=.2,palette='RdYlGn',hue='price')


plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data =non_top_1_perc,hue='price')

# price vs waterfront
sns.boxplot(x='waterfront', y='price', data=df)

# drop id column
df=df.drop('id',axis=1)
# convert date column datatype from object to datetime
df['date']=pd.to_datetime(df['date'])

df['date']
# Extract year from date
df['year']=df['date'].apply(lambda date: date.year)
# Extract month from date
df['month']=df['date'].apply(lambda date: date.month)

df.head()

# price month wise
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

df['sqft_basement'].value_counts()

# store predictors into X and target variable into y
X=df.drop('price', axis=1).values
y=df['price'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

X_train.shape

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
'''
 validation data parameter added to check the loss funcation on test data (how model is performing)
 Its helpful to avoid the overfiting
 validation dataset will not impact weight and biases of train data
 Since dataset is quite large so batch sixe parameter added to pass the data in batches
 small batch size gonna take long time to train the model but it avoid overfitting
'''
model.fit(x=X_train, y=y_train, 
          validation_data=(X_test, y_test), batch_size=128, epochs=400)
# model history (history of loss)
model.history.history
# convert into dataframe
# loss - loss on test data
# val_loss - loss on test
losses=pd.DataFrame(model.history.history)

losses
# plot the loss funcation of test and train 
losses.plot()
# to see the loss funcation values
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
# predict price based on x test 
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
mean_squared_error(y_test, prediction)
rmse=np.sqrt(mean_squared_error(y_test, prediction))
df['price'].describe()
# to see how much variance is explained 
explained_variance_score(y_test, prediction)

plt.figure(figsize=(12,8))
plt.scatter(y_test, prediction)
plt.show()

plt.figure(figsize=(12,8))
plt.scatter(y_test, prediction)
plt.plot(y_test,y_test,'r')
plt.show()
# predict price for new house
singlehouse=df.drop('price', axis=1).iloc[0]
singlehouse=scaler.transform(singlehouse.values.reshape(-1,19))
model.predict(singlehouse)
df.head()