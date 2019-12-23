# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:48:14 2019

@author: Brijeshkumar
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv("D:/Personal/Dataset/titanic_train.csv")

train.head()

# To see is there any missing values for any columns in the dataset (result will be true or false)
train.isnull().values.any()

# Total missing values for each feature
train.isnull().sum()

# Total number of missing values in the dataset
train.isnull().sum().sum()

train.columns

# To drop NA - train.dropna()
# data = train.dropna()

sns.countplot(x='Survived',data=train)
'''This is a count plot that shows the number of people who survived which is our target variable.
Further, we can plot count plots on the basis of gender and passenger class.'''

sns.countplot(x='Survived',hue='Sex',data=train)
'Here, we see a trend that more females survived than males.'

sns.countplot(x='Survived',hue='Pclass',data=train)
'From the above plot, we can infer that passengers belonging to class 3 died the most.'

' ************************ Data Cleaning ************************* '

'''
We want to fill in the missing age data instead of just dropping the missing age data rows. One 
way to do this is by filling in the mean age of all the passengers (imputation). However, we can 
be smarter about this and check the average age by passenger class. For example:
'''

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train)
?sns.boxplot
'We can see the passengers in the higher classes tend to be older, which makes sense'

avgAge = np.mean(train.Age)

train.Age.isnull

pd.isnull(train.Age) == 'True'

# Fill missing vales for Age with mean
train['Age'].fillna(avgAge, inplace = True)


'To replace the null values with mean/median/mode, type the following code:'
# train.fillna(train.mean())
'To drop all the missing values from data set'
# modifiedDataset = train.dropna()

# Column Cabuin is not required
train.drop('Cabin',axis=1,inplace=True)

train.info()


' ************* Add dummy Variables ************ '

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
# Name and Ticket not required 
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)
train.columns

' ********************* Split the data **************************** '

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

' *************** Evalution ****************** '

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

# draw confusison matrix to see the accuracy of model
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, predictions) 
print ("Confusion Matrix : \n", cm)  

# Total Accuracy
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, predictions)) 


'''
ROC Curve
Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate.
It shows the tradeoff between sensitivity and specificity.
AUC score for the case is 0.86. AUC score 1 represents perfect classifier, and 0.5 represents a worthless 
classifier.
'''
from sklearn import metrics
y_pred_proba = logmodel.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

