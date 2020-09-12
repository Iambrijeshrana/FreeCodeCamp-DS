# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:22:40 2019

@author: Brijeshkumar
"""

## Import the package
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score
import seaborn as sns

## Reading the file, latin encoding is used to avoid some errors
data=pd.read_csv('d:/Personal/Dataset/autos.csv',encoding=('latin'),quoting=3)
data.shape
data.columns
data.dtypes

data.info()

data.boxplot()

sns.boxplot(data['price'])

##Selecting variable (We are going to build the model using below variables)
data=data[[
 'seller',
 'price',
 'yearOfRegistration',
 'gearbox',
 'powerPS',
 'kilometer',
 'fuelType',
 'notRepairedDamage']]


data.shape
data.columns
data.dtypes
data['fuelType']

##Selecting cars which cost less than 50,000
data = data[data['price'] < 50000] 

##Creating the list of input and output variables. And 
inputVariables=list(data)
del inputVariables[1]
outputVariables=list(data)[1]
inputData=data[inputVariables]
outputVariables=data['price']

inputData.columns
outputVariables.columns


"********* Dummy coding of categorical variables *************"
"""
The regression can only use numerical variable as its inputs data. Due to this, the categorical 
variables need to be encoded as dummy variables.
Dummy coding encodes the categorical variables as 0 and 1 respectively if the observation does not or 
does belong to the group.
Basically, the code below select all the variables that are strings, dummy code them thanks to 
get_dummies and then join it to the data frame.

The basic strategy is to convert each category value into a new column and assign a 1 or 0 
(True/False) value to the column. This has the benefit of not weighting a value improperly.
"""
# Add dummy variables in dataset and remove categorical variables
for column in inputData.columns:
 if inputData[column].dtype==object:
  dummyCols=pd.get_dummies(inputData[column])
  inputData=inputData.join(dummyCols)
  del inputData[column]

"********* Running the linear regression *************"  
""" Now that data can be used by the scikit-learn module. 
We will just use the LinearRegression function from the module."""

from sklearn.linear_model import LinearRegression
model_1=LinearRegression()
model_1.fit(inputData,data[outputVariables])

#For retrieving the coffieient:
coefficients=pd.DataFrame({'name':list(inputData),'value':model_1.coef_})
#For retrieving the slope:
print(model_1.intercept_)

"And now let’s have a look at the R² and the MSE of the model:"
predictedValue = model_1.predict(inputData)
actualValue = data[outputVariables]
# Evaluate the model based on test data 
print('Mean Absolute Error:', metrics.mean_absolute_error(actualValue, predictedValue))  
print('Mean Squared Error:', metrics.mean_squared_error(actualValue, predictedValue))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(actualValue, predictedValue)))
print('Median absolute error:',metrics.median_absolute_error(actualValue, predictedValue))
r2 = r2_score(actualValue, predictedValue)


# anothr example to add dummy variables in data
import pandas as pd

sample_data = [[1,2,'a'],[3,4,'b'],[5,6,'c'],[7,8,'b']]
df = pd.DataFrame(sample_data, columns=['numeric1','numeric2','categorical'])
dummies = pd.get_dummies(df.categorical)
newdata = df.join(dummies)
newdata

df.dtypes



