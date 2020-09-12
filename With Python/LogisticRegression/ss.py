# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:35:48 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statistics

train = pd.read_csv('D:/Personal/Dataset/titanic_train.csv')


train.head()

pd.set_option('display.max_rows',500)
pd.set_option('display.expand_frame_repr', True)

print(train)

train.info()
train.shape

train.describe

# check missing values for each columns 
train.isnull().any()
# total number of missing values for each columns
train.isnull().sum()

mv= (train.isnull().sum())*100/len(train)

mv=mv.to_frame()
type(mv)

mv.columns


train.Cabin.unique

sns.heatmap(train.isnull())

train = train.drop(['Cabin'], axis=1)
train.columns

sns.countplot(train['Survived'], hue='Sex', data = train)

sns.countplot(train['Survived'], hue='Pclass', data = train, )

sns.countplot(train['Survived'], hue='Age', data = train, )

sns.distplot(train['Age'].dropna(), bins=40)

train.info()

sns.countplot(train['SibSp'])

train.Fare


sns.distplot(train.Fare, bins=40)

train.hist(train['Fare'])

train.Embarked.unique()

sns.countplot(train['Embarked'])


#Cleaning the data 

train['Age'].info()

sns.boxplot(data = train, y='Age', x='Pclass')


np.mean(train)

avg = train.groupby('Pclass')

    
avg.mean()
stclassavg = train.Age[train.Pclass==1]

np.mean(stclassavg)

train.Pclass

train['Age'].fillna(np.mean(train['Age']), inplace=True)

train.isnull().sum()

train['Embarked'].unique()

sns.countplot(train['Embarked'])

embarkedMode = statistics.mode(train['Embarked'])
train['Embarked'].fillna(embarkedMode, inplace=True)

train.isnull().sum()

ss=train.groupby('Embarked')

ss.count()


# create dummy variables 

sex=pd.get_dummies(train['Sex'], drop_first=True)

embark=pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, sex, embark], axis=1)

train .info()

train = train.drop(['Sex', 'Embarked'], axis=1)

y=train['Survived']
X = train.drop(['PassengerId', 'Survived','Ticket', 'Name'], axis=1)

X.columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=101)

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

df = pd.DataFrame({'y_pred': y_pred, 'y_actual': y_test})

from sklearn.metrics import classification_report

report= classification_report(y_test, y_pred)

print(report)

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)

print(cm)
