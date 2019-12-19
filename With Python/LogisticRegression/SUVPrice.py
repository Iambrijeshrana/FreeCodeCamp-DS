# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:17:03 2019

@author: brijesh
"""

"""
(a) binary logistic regression requires the dependent variable to be binary and
(b) ordinal logistic regression requires the dependent variable to be ordinal.
observations should be independent of each other
little or no multi-collinearity among the independent variables
independent variables are linearly related to the log odds
"""

import numpy as np
import pandas as pd
import seaborn as sns
import plotly
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

'1. Collecting the data'
# Read the data
suv_data = pd.read_csv('D:/Personal/Dataset/SUV.csv')

suv_data.head(2)

'2. Analyze the data'
# Chek the data shape
suv_data.shape
# To see Mean median and all of numerical data
suv_data.describe()
# Check data type of all columns
suv_data.info()

# To check missing values is there or not in the entire data set (Result will be boolean)
suv_data.isnull().values.any()
# To count missing values for each columns
suv_data.isnull().sum()
# To count total number missing values in the entire dataset
suv_data.isnull().sum().sum()

# Find outliers in the entire dataset for all the variables
# If z value > 3 or <-3 than it's outlier
'For outlier we need numric data so we have to drop categorical variables from the data to check the outliers'
z = np.abs(stats.zscore(suv_data._drop_labels_or_levels('Gender')))
print(z)

threshold = 3
print(np.where(z > 3))
""" Don’t be confused by the results. The first array contains the list of row numbers and 
second array respective column numbers, which mean z[55][1] have a Z-score higher than 3.
"""

# Now check the relationship between variables means check the correlation between variables
# To see correlation in tabular format
suv_data.corr()

# To see correlation values in one map
plt.matshow(suv_data.corr())

# To plot the correlation of features and target variable with each other using heatmap of seaborn (this is best way)
fig = plt.subplots(figsize = (10,10))
sns.set(font_scale=1.5)
sns.heatmap(suv_data.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.show()

suv_data.columns
# Devide the data into train and test dataset
# input 
X = suv_data.iloc[:, [2, 3]].values
# output 
y = suv_data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test.shape
y_test.shape
# realtion ship with sage 
plt.scatter(X_test['Age'], y_test, color='red')
# relationship with salary
plt.scatter(X_test['EstimatedSalary'], y_test, color='red')

'''
Now, it is very important to perform feature scaling here because Age and Estimated Salary 
values lie in different ranges. If we don’t scale the features then Estimated Salary 
feature will dominate Age feature when the model finds the nearest neighbor to a data point 
in data space.
''''

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
X_train = sc_x.fit_transform(X_train)  
X_test = sc_x.transform(X_test) 
print (X_train[0:10, :]) 

'''
Here once see that Age and Estimated salary features values are sacled and now there in the 
-1 to 1. Hence, each feature will contribute equally in decision making i.e. finalizing the 
hypothesis.

Finally, we are training our Logistic Regression model.
'''

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)


# draw confusison matrix to see the accuracy of model
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print ("Confusion Matrix : \n", cm)  

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred)) 

# Calssification reprot 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

''' Classification Report:

The classification report displays the Precision, Recall , F1  and Support scores for the model.

Precision score means the the level up-to which the prediction made by the model is precise. The precision for 
a 0 is 0.92 and for the 1 is 0.94.

Recall is the amount up-to which the model can predict the outcome. Recall for a 0 is 0.98 and for a 1 is 0.77. 
F1 and Support scores are the amount of data tested for the predictions. 

'''