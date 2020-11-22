# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:00:06 2020

@author: Brijesh.R
"""
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split


df = pd.read_csv('D:/TensorFlow_FILES/Data/fake_reg.csv')

df

sns.pairplot(df)

# Convert Pandas to Numpy for Keras

# Features
# In pansdas we have to pass numpy array insted of pandas 
X = df[['feature1','feature2']].values

# Label
y = df['price'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape

from sklearn.preprocessing import MinMaxScaler

help(MinMaxScaler)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

help(Sequential)


model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')

model.fit(X_train,y_train,epochs=250)

model.history.history

df=pd.DataFrame(model.history.history)
df

df.plot()

loss = model.history.history['loss']

sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");

model.metrics_names

training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)

training_score

test_score


test_predictions = model.predict(X_test)

pred_df = pd.DataFrame(y_test,columns=['Test Y'])


test_predictions = pd.Series(test_predictions.reshape(300,))

pred_df = pd.concat([pred_df,test_predictions],axis=1)


pred_df.columns = ['Test True Y','Model Predictions']

sns.scatterplot(x='Test True Y',y='Model Predictions',data=pred_df)

pred_df['Error'] = pred_df['Test True Y'] - pred_df['Model Predictions']


sns.distplot(pred_df['Error'],bins=50)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(pred_df['Test True Y'],pred_df['Model Predictions'])

mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions'])

# Essentially the same thing, difference just due to precision
test_score

#RMSE (Root mean square error)
test_score**0.5


# Predicting on brand new data
# [[Feature1, Feature2]]
new_gem = [[998,1000]]

# Don't forget to scale!
scaler.transform(new_gem)


new_gem = scaler.transform(new_gem)

model.predict(new_gem)

# Saving and Loading a Model
from tensorflow.keras.models import load_model

model.save('D:/TensorFlow/my_model.h5')  # creates a HDF5 file 'my_model.h5'

later_model = load_model('D:/TensorFlow/my_model.h5')

later_model.predict(new_gem)
