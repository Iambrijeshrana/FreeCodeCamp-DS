# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:11:51 2019

@author: brijesh
"""

"In this we will try to predict the price of House based on more than one predictor"

import pandas as pd # Import pandas to import and analyze the data
import numpy as np # Import numpy to perform mathematical funcations on data

import plotly
%matplotlib inline
#import plotly.plotly as py
import matplotlib.pyplot as plt # Import pyplot
import seaborn as sns
from matplotlib import style
from scipy import stats # Import stats for scientific Test's
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.linear_model import LinearRegression # Import Linear Regression
from sklearn import metrics # Import metrics to calculate the accuracy parameters 
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

' well correlation between RM and MEDV is very high (.70) and with LSTAT(-.6), so we are both the variables in model'

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


# Plot histogram with curve line to see the distribution of MEDV data in the dataset
sns.distplot(housingData['MEDV'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# To see how our target variable data distributed
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(housingData['MEDV'])

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


X = housingData[['RM','LSTAT']].values
y = housingData['MEDV'].values

# Devide the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Relationship between dependent and Independent variables
plt.scatter(X_test['LSTAT'], y_test,  color='gray')
plt.show()

plt.scatter(X_test['RM'], y_test,  color='gray')
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

df = df.head()
# Draw the figure for Actual and Predicted values 
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

'**************************** Step 4. Accuracy ***************************************'

# Evaluate the model on test data 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Median absolute error:',metrics.median_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(r2)
# Another way to check the R2
print(regressor.score(X_test, y_test)) 

# model evaluation on training set
y_train_predict = regressor.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


import statsmodels.api as sm
"An intercept is not included by default in statsmodels and should be added by the user."
" See statsmodels.tools.add_constant."

X_endog = sm.add_constant(X_test)
res = sm.OLS(y_test, X_endog)
res.fit().summary()

# Another method to fit Linear regression Model
import statsmodels.regression.linear_model as smf
model = smf.OLS(y_test, X_endog,data= X_test).fit()
model.summary()

# calculate the accuracy parameter with statistics formulas from the theory
SS_Residual = sum((y_test-y_pred)**2)
SS_Total = sum((y_test-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(r_squared, adjusted_r_squared)


" ******************** Linear Regression Assumption *********************** "

' LINEAR RELATIONSHIP'

from yellowbrick.regressor import ResidualsPlot

# residuals vs. predicted values
visualizer = ResidualsPlot(regressor)
#visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()      

visualizer = ResidualsPlot(regressor, hist=False)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
# Test R2 is the r2 on test data of model

# Observed vs Predicted
sns.residplot(y_test, y_pred)


' Multivariate normality '
np.mean(y_test-y_pred)
# Here mean is not zero so residuals are not normally distributed
    
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
 
 
# Q Q plot to check normality
stats.probplot(model.resid, plot= plt)
plt.title("Q-Q Plot")
plt.show() 
# Since most of the points are not lying on the line so residuals are distrinuted normally.

' No or little multicollinearity '

corr = housingData.corr()
print(corr)
# IF VALUE is greater than .8 than exisst and those variables are highly dependent

# For each feature, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(housingData.values, i) for i in range(housingData.shape[1])]
vif["features"] = housingData.columns
vif.round(1)

' No autocorrelation of residuals '

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
    
#Check auto corelation with ACF plot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y_test-y_pred)

' Homoscedasticity '    

 # Plotting the residuals
plt.subplots(figsize=(12, 6))
ax = plt.subplot(111)  # To remove spines
plt.scatter(x=df.index, y=df.Actual-df.Predicted, alpha=0.5)
plt.plot(np.repeat(0, df.index.max()), color='darkorange', linestyle='--')
ax.spines['right'].set_visible(False)  # Removing the right spine
ax.spines['top'].set_visible(False)  # Removing the top spine
plt.title('Residuals')
plt.show()  


residulas = y_test-y_pred
fig, ax = plt.subplots(figsize=(12, 6))
_ = ax.scatter(residulas, y_pred)
 
