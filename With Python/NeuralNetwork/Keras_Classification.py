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

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
df.shape

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

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.25, random_state=101)



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
model=Sequential()

X_train.shape

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
# Binary classification
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

'''
 validation data parameter added to check the loss funcation on test data (how model is performing)
 Its helpful to avoid the overfiting
 validation dataset will not impact weight and biases of train data
 Since dataset is quite large so batch sixe parameter added to pass the data in batches
 small batch size gonna take long time to train the model but it avoid overfitting
'''
model.fit(x=X_train, y=y_train, 
          validation_data=(X_test, y_test), batch_size=128, epochs=600)

# model history (history of loss)
model.history.history
# convert into dataframe
# loss - loss on test data
# val_loss - loss on test
losses=pd.DataFrame(model.history.history)

losses
# plot the loss funcation of test and train 
losses.plot()

'''By ploting the loss funcation we understood that its overfiting
  Now let see how we can control the overfiting'''
  
model=Sequential()

X_train.shape

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
# Binary classification
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')
  
from tensorflow.keras.callbacks import EarlyStopping
help(EarlyStopping)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)
model.fit(x=X_train, y=y_train, 
          validation_data=(X_test, y_test), batch_size=128, epochs=600,
          callbacks=[early_stop])
       
model_losses=pd.DataFrame(model.history.history)
model_losses.plot()

# Another way to prevent overfiting is dropout layers

model=Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(.5))
# Binary classification
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x=X_train, y=y_train, 
          validation_data=(X_test, y_test), batch_size=128, epochs=600,
          callbacks=[early_stop])

model_losses=pd.DataFrame(model.history.history)
model_losses.plot()
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
