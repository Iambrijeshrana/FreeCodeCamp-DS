# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:10:22 2019

@author: Brijeshkumar
"""

# import necessary modules  
import pandas  as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 

# load the data set 
data = pd.read_csv('D:/Personal/Dataset/creditcard.csv') 
  
# print info about columns in the dataframe 
print(data.info()) 

data.columns

data.head(4)

# normalise the amount column 
data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1)) 
  
# drop Time and Amount columns as they are not relevant for prediction purpose  
data = data.drop(['Time', 'Amount'], axis = 1) 
data.head(4)  
# as you can see there are 492 fraud transactions. 
data['Class'].value_counts() 

X = data.drop(['Class'], axis=1)
y =data['Class']
  
# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
  
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

'Now train the model without handling the imbalanced class distribution'

# logistic regression object 
lr = LogisticRegression() 
# train the model on train set 
lr.fit(X_train, y_train.ravel()) 

predictions = lr.predict(X_test) 
# print Accuracy and Classification Report
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test,predictions)) 
print(classification_report(y_test, predictions)) 

'''
The accuracy comes out to be 100% but did you notice something strange ?
The recall of the minority class in very less. It proves that the model is more biased towards majority class. 
So, it proves that this is not the best model.Now, we will apply different imbalanced data handling techniques 
and see their accuracy and recall results.
'''




