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
plt.bar(bankData.job,bankData.y)

# The best way to visualize the two categorical variables is - 
# 1st using cross tab get the frequancy count for each category
# 2nd plot the output with bars 
pd.crosstab(bankData.job,bankData.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

# Relationship between education and target variable
pd.crosstab(bankData.education,bankData.y).plot(kind='bar')
plt.title('Purchase Frequency for education')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_education')
# So, illetrate people are not doing the subscribe so we can remove the data directly where education category illetrate
'Education seem a strong predictor for the outcome variable.'

# Relation between Marital status and target variable
table=pd.crosstab(bankData.marital,bankData.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')
'The marital status does not seem a strong predictor for the outcome variable.'

# Relation between Month and target variable
pd.crosstab(bankData.month,bankData.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')

# Customers age
bankData.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
'Most of the customers of the bank in this dataset are in the age range of 30–40.'


pd.crosstab(bankData.poutcome, bankData.y).plot(kind='bar')
'Poutcome seems to be a good predictor of the outcome variable.'


'''************************* CREATE DUMMY VARIABLES ************************************
The regression can only use numerical variable as its inputs data. Due to this, the categorical 
variables need to be encoded as dummy variables.
Dummy coding encodes the categorical variables as 0 and 1 respectively if the observation does not or 
does belong to the group.
Basically, the code below select all the variables that are strings, dummy code them thanks to 
get_dummies and then join it to the data frame.

The basic strategy is to convert each category value into a new column and assign a 1 or 0 
(True/False) value to the column. This has the benefit of not weighting a value improperly.
'''
# Add dummy variables in dataset and remove categorical variables
columns=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for column in columns:
 if bankData[column].dtype==object:
   dummyCols='column'+'_'+column
   dummyCols=pd.get_dummies(bankData[column], prefix=column)
   bankData=bankData.join(dummyCols)
   del bankData[column]


bankData.columns

"************************** Step 4. Train Test (Build the model) *****************************"
data = bankData
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))