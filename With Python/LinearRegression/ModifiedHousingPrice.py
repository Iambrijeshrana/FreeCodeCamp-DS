# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:08:14 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
from pandas_profiling import ProfileReport
from scipy import stats
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score
from sklearn import metrics
# Import the Housing Price dataset
housingData = pd.read_csv("D:/Personal/Dataset/housingdata.csv")

# Rename the columns (Give the name to columns)
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housingData.columns = housing_colnames

housingData.columns

# Lets start data analysis part -- EDA

# shape of data 
housingData.shape

# corelation between variables - 

#We can see the corelation in table format also and in heatmap also, well i prefer heatmap

# Corelation in table format
housingData.corr()
# Corelation using heatmap
sns.heatmap(housingData.corr(), annot=True, annot_kws={'size': 10})
# - annot=True - to show the corelation values
# - annot_kws - use to avoide the coreltion values overlap

# To see is there any missing values for any columns in the dataset 
# (result will be true or false)
housingData.isnull().any()
housingData.isnull().values.any()

# Total missing values for each feature
housingData.isnull().sum()

# Total number of missing values in the dataset
housingData.isnull().sum().sum()

# Find outliers 

z = np.abs(stats.zscore(housingData))
print(z)

threshold = 3
print(np.where(z > 3))

# z value of an outlier
z[55,1]

housingData.iloc[[55],1]

# Remove outliers
data_clean = housingData[(z<3).all(axis=1)]
data_clean.shape


# Fine and remove outliers using IQR 
#find Q1, Q3, and interquartile range for each column
Q1 = housingData.quantile(q=.25)
Q3 = housingData.quantile(q=.75)
IQR = housingData.apply(stats.iqr)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
data_clean2 = housingData[~((housingData < (Q1-1.5*IQR)) | (housingData > (Q3+1.5*IQR))).any(axis=1)]
#find how many rows are left in the dataframe 
data_clean2.shape

(89,3)

# Check how our target variable is distributed 

# f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(housingData['MEDV'], bins=10)
sns.pairplot(housingData)

#

''
# EDA with sweetviz
sweetviz_report = sv.analyze(housingData)
sweetviz_report.show_html('sweetviz_report.html')
# Since it's a simple linear regression model and we have seen the corelation between rm and 
# medv is very high so lets take rm variable to predict the vakues odf medv


# run the profile report 
profile = housingData.profile_report(title='Pandas Profiling Report') 
   
# save the report as html file 
profile.to_file(output_file="D:/pandas_profiling1.html") 
   
# save the report as json file 
profile.to_file(output_file="D:/pandas_profiling2.json") 

# lets do regression

# Train test split

# Devide the data into train and test dataset
# we have to do the reshape if only one feature is there in the model
X = data_clean['RM'].values.reshape(-1,1)
y = data_clean['MEDV'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=0)

# Relationship between dependent and Independent variables in training dataset
sns.scatterplot(housingData['RM'], housingData['MEDV'])
sns.set(set(xlabel='RM', ylabel='MEDV'))

plt.relplot(X_test, y_test,  color='gray')

#training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

# Predict the future values
y_pred = regressor.predict(X_test)


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten())

df.columns = headers
# Store the actual and predicted value in a dataframe
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

df.columns

sns.scatterplot(x=df['Actual'], y=df['Predicted'])

sns.regplot(x=df['Actual'], y=df['Predicted'])

df.plot(x=df['Actual'], y=df['Predicted'], kind='scatter')
plt.show()

df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# check accuracy
# Evaluate the model on test data 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Median absolute error:',metrics.median_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(r2)
# Another way to check the R2
print(regressor.score(X_test, y_test)) 


def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(df['Actual'], df['Predicted'])

" ******************** Linear Regression Assumption *********************** "
sns.scatterplot(y=housingData.MEDV, x=housingData.RM)
sns.scatterplot(y=data_clean.MEDV, x=data_clean.RM)

from yellowbrick.regressor import ResidualsPlot

# residuals vs. predicted values
visualizer = ResidualsPlot(regressor)
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show() 


visualizer = ResidualsPlot(regressor, hist=False)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# Observed vs Predicted
sns.residplot(y_test, y_pred)

np.mean(y_test-y_pred)

# Statistic method to check normality (Using the Anderson-Darling test for normal distribution)
from statsmodels.stats.diagnostic import normal_ad
p_value_thresh = .05
# Performing the test on the residuals
p_value = normal_ad(y_test-y_pred)[1]
print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
# Reporting the normality of the residuals
if p_value < p_value_thresh:
    print('Residuals are not normally distributed')
else:
    print('Residuals are normally distributed')
    
# Plotting the residuals distribution with Histogram to see the how the residuals are spread
plt.subplots(figsize=(12, 6))
plt.title('Distribution of Residuals')
sns.distplot(y_test-y_pred)
plt.show()


# For each feature, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(housingData.values, i) for i in range(housingData.shape[1])]
vif["features"] = housingData.columns
vif.round(1)


from statsmodels.stats.stattools import durbin_watson
print('Assumption 4: No Autocorrelation', '\n')
    
print('\nPerforming Durbin-Watson Test')
print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
print('0 to 2< is positive autocorrelation')
print('>2 to 4 is negative autocorrelation')
print('-------------------------------------')
durbinWatson = durbin_watson(y_test-y_pred)
print('Durbin-Watson:', durbinWatson)
if durbinWatson < 1.5:
    print('Signs of positive autocorrelation', '\n')
    print('Assumption not satisfied')
elif durbinWatson > 2.5:
    print('Signs of negative autocorrelation', '\n')
    print('Assumption not satisfied')
else:
    print('Little to no autocorrelation', '\n')
    print('Assumption satisfied')