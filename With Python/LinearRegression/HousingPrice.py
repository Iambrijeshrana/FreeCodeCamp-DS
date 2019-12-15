# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:24:19 2019

@author: brijesh
"""

"In this we will try to predict the price of House based on one predictor"

'''-------------------- Data set description --------------------
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
'''

import pandas as pd # Import pandas to import and analyze the data
import numpy as np # Import numpy to perform mathematical funcations on data

import plotly
%matplotlib inline
import plotly.plotly as py
import matplotlib.pyplot as plt # Import pyplot
import seaborn as sns
from matplotlib import style
from scipy import stats
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score

"************************** Step 1. Collect the data *****************************"

# Import the Housing Price dataset
housingData = pd.read_csv("D:/Personal/Dataset/housingdata.csv")

# Rename the columns (Give the name to columns)
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housingData.columns = housing_colnames

housingData.columns

"************************** Step 2 & 3. Analyze the data & Data Cleaning *****************************"

# Check the shape of data
housingData.shape

# To check the relationship (Correlation) between target variable (MEDV) and dependent variable 

# To see the correlation in tabular format
housingData.corr()

# To plot the correlation of features with target variable using heatmap of seaborn
fig = plt.subplots(figsize = (10,10))
sns.set(font_scale=1.5)
sns.heatmap(housingData.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})
plt.show()

''' well correlation between RM and MEDV is very high (.70), so we are going to build the model based on RM'''

# To see the relation between Dependent variable and Independent variable in Graphs 
#Utility function for plotting the relationships b/w features and target variables
def plotFeatures(col_list,title):
    plt.figure(figsize=(10, 14))
    i = 0
    print(len(col_list))
    for col in col_list:
        i+=1
        plt.subplot(7,2,i)
        plt.plot(housingData[col],housingData["MEDV"],marker='.',linestyle='none')
        plt.title(title % (col))   
        plt.tight_layout()
        
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
plotFeatures(colnames,"Relationship bw %s and MEDV")

# Plot Simple histogram to see the distribution of MEDV data in the dataset 
plt.hist(housingData['MEDV'])

# Plot histogram with curve line to see the distribution of MEDV data in the dataset
sns.distplot(housingData['MEDV'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# To see is there any missing values for any columns in the dataset (result will be true or false)
housingData.isnull().values.any()

# Total missing values for each feature
housingData.isnull().sum()

# Total number of missing values in the dataset
housingData.isnull().sum().sum()

# Find outliers in the entire dataset for all the variables
# If z value > 3 or <-3 than it's outlier
z = np.abs(stats.zscore(housingData))
print(z)

threshold = 3
print(np.where(z > 3))

""" Don’t be confused by the results. The first array contains the list of row numbers and 
second array respective column numbers, which mean z[55][1] have a Z-score higher than 3.
"""
# z value of an outlier
z[55,1]

# So the data point — 55th record on column ZN is an outlier, Actual value of Outlier in dataset
housingData.iloc[[55],1]

"************************** Step 4. Train Test (Build the model) *****************************"


# Store the Dependent and Independent feature in variables
#X = housingData['RM'].values.reshape(-1,1)
#y = housingData['MEDV'].values.reshape(-1,1)

#X = housingData['RM'].values
#y = housingData['MEDV'].values

# we have to do the reshape if only one feature is there in the model
X = housingData['RM'].values.reshape(-1,1)
y = housingData['MEDV'].values.reshape(-1,1)

X.columns

# Devide the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Relationship between dependent and Independent variables in training dataset
plt.scatter(X_test, y_test,  color='gray')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()


#training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

# Predict the future values
y_pred = regressor.predict(X_test)
# Store the actual and predicted value in a dataframe
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

# Draw the figure for predicted and actual values 
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

'**************************** Step 4. Accuracy ***************************************'

''' ----------------------- Matrix for Regression Model to Evaluate ------------------------

metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
metrics.max_error(y_true, y_pred)	max_error metric calculates the maximum residual error.
metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
metrics.mean_squared_error(y_true, y_pred[, …])	Mean squared error regression loss
metrics.mean_squared_log_error(y_true, y_pred)	Mean squared logarithmic error regression loss
metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
metrics.r2_score(y_true, y_pred[, …])	R^2 (coefficient of determination) regression score function.


STATISTIC	      CRITERION
R-Squared	      Higher the better
Adj R-Squared     Higher the better
AIC	              Lower the better
BIC	              Lower the better
Mallows cp	      Should be close to the number of predictors in model
MAPE              Lower the better
MSE               Lower the better
Min_Max Accuracy- mean(min(actual, predicted)/max(actual, predicted))	Higher the better

MAPE - Mean absolute percentage error
MSE  - Mean squared error
'''

# Evaluate the model on test data 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Median absolute error:',metrics.median_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(r2)


# model evaluation on training set
y_train_predict = regressor.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# calculate the accuracy parameter with statistics formulas from the theory
SS_Residual = sum((y_test-y_pred)**2)
SS_Total = sum((y_test-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(r_squared, adjusted_r_squared)


"""
After you fit the model, unlike with statsmodels, SKLearn does not automatically print the concepts 
or have a method like summary. So we have to print the coefficients separately. While SKLearn isn’t 
as intuitive for printing/finding coefficients, it’s much easier to use for cross-validation and 
plotting models. With a data set this small, these things may not be that necessary, but with most 
things you’ll be working with in the real world, these are essential steps.
"""


import statsmodels.api as sm
"An intercept is not included by default in statsmodels and should be added by the user.(we ahev to add the constant manully)"
" See statsmodels.tools.add_constant."

X_endog = sm.add_constant(X_test)
res = sm.OLS(y_test, X_endog)
res.fit().summary()

# Another method to fit Linear regression Model
import statsmodels.regression.linear_model as smf
model = smf.OLS(y_test, X_endog,data= X_test).fit()
model.summary()


