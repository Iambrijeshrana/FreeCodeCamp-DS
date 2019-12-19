# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:38:06 2019

@author: brijesh
"""

'In this we will try to predict the price of House based on one predictor'

'''-------------------- Data set description --------------------
Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
   3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
   5 - default: has credit in default? (categorical: "no","yes","unknown")
   6 - housing: has housing loan? (categorical: "no","yes","unknown")
   7 - loan: has personal loan? (categorical: "no","yes","unknown")
   # related with the last contact of the current campaign:
   8 - contact: contact communication type (categorical: "cellular","telephone") 
   9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
  11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
   # other attributes:
  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  14 - previous: number of contacts performed before this campaign and for this client (numeric)
  15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
   # social and economic context attributes
  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
  20 - nr.employed: number of employees - quarterly indicator (numeric)

  Output variable (desired target):
  21 - y - has the client subscribed a term deposit? (binary: "yes","no")
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


"************************** Step 1. Collect the data *****************************"

# Import the Housing Price dataset
bankData = pd.read_csv("D:/Personal/Dataset/bank-additional-full.csv", sep=';')

bankData.head()

bankData.columns

"************************** Step 2 & 3. Analyze the data & Data Cleaning *****************************"
# To check the dimensions of data 
bankData.shape 

# To see is there any missing values for any columns in the dataset (result will be true or false)
bankData.isnull().values.any()

# Total missing values for each feature
bankData.isnull().sum()

# Total number of missing values in the dataset
bankData.isnull().sum().sum()


''' The education column of the dataset has many categories and we need to reduce the categories for a better 
modelling. The education column has the following categories:'''

bankData['education'].unique()

'Let us group basic.4y, basic.9y and basic.6y together and call them basic.'


bankData['education']=np.where(bankData['education'] =='basic.9y', 'Basic', bankData['education'])
bankData['education']=np.where(bankData['education'] =='basic.6y', 'Basic', bankData['education'])
bankData['education']=np.where(bankData['education'] =='basic.4y', 'Basic', bankData['education'])

bankData['education'].unique()

# To check the ration target varaible of 0's and 1's in data set
bankData['y'].value_counts()

# To see the y values using Histogram (How our data is spread)
plt.hist(bankData.y)
plt.title('Count Plot')

bankData.groupby('y').mean()

'''
The average age of customers who bought the term deposit is higher than that of the customers who didn’t.
The pdays (days since the customer was last contacted) is understandably lower for the customers who bought it. 
The lower the pdays, the better the memory of the last call and hence the better chances of a sale.
Surprisingly,campaigns (number of contacts or calls made during the current campaign) are lower for customers who 
bought the term deposit.
We can calculate categorical means for other categorical variables such as education and marital status to get a more 
detailed sense of our data.
'''

bankData.groupby('education').mean()
bankData.groupby('education').mean()
bankData.groupby('education').mean()
bankData.groupby('education').mean()

# To check the correlation between variables.
bankData.corr()


# To plot the correlation of features with target variable using heatmap of seaborn
fig = plt.subplots(figsize = (10,10))
sns.set(font_scale=1.5)
sns.heatmap(bankData.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.show()

sns.pairplot(bankData)
plt.bar(bankData.y, bankData.education)