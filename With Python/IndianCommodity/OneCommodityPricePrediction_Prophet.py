
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:55:58 2019
To predict only one commodity price using facebook prophet model
@author: Brijesh Rana
"""

import pymssql
import pandas as pd
import sqlalchemy
from fbprophet import Prophet 
import pyodbc	
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sqlalchemy import create_engine, MetaData, Table, select
from six.moves import urllib

## instance a python db connection object- same form as psycopg2/python-mysql drivers also
conn = pymssql.connect(server="SERVERNAME", user="Brijesh",password="PASSWORD", port=1433)  # You can lookup the port number inside SQL server. 

#Get all database names from 6 srever
sql="select name FROM sys.databases where name like '%Hiranandani%'"
dbnames = pd.read_sql(sql,conn)
dbnames

## Hey Look, college data
stmt = "SELECT *  FROM [DATABASENAME].[dbo].[TABLENAME]"

# Excute Query here
df = pd.read_sql(stmt,conn)
# to check total number of records in dataframe
df
len(df)
# to see the data
print(df)
# to check toatl numaber or rows and columns in the data 
df.shape
# to check column data types 
df.dtypes
# check data type of Date column from Object to Date
df['Date'] = df['Date'].astype('datetime64[ns]')
df.dtypes
# String datatype in Python is always object
df['Centre'] = df['Centre'].astype('str')
df.dtypes
# To convert price datatype from object to flaot
df['Price per Kg'] = df['Price per Kg'].astype(float)
df.dtypes
# select two columns from dataframe
df[['Date', 'Price per Kg']]
# create another data frame with specific columns 
df2 = pd.DataFrame(df[['Date', 'Price per Kg']])
# print the data
df2
# rename the actual column name  because in prophet the column name should be DS and Y
df2 = df2.rename(columns={'Date': 'ds',
                        'Price per Kg': 'y'})
df2.dtypes
# check the data type for new data frame
df2.columns
# print new data frame
df2
# Plot the data 
ax = df2.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Price of Commoditity')
ax.set_xlabel('Date')
plt.show()
# create prophet model
my_model = Prophet(interval_width=0.95)
# fit the prophet model
my_model.fit(df2)
# select futre date for which we want to predict
future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()
# predict future value
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast.columns
my_model.plot(forecast,
              uncertainty=True)
# plot seasionality, trend based on yearly monthly day wise etc.
my_model.plot_components(forecast)
# check columns name in forecast model
forecast.coloumns
# to see first top 10 rows in forecast model
forecast.head(10)
#to check coulmns data types
forecast.dtypes

# to export the predicted values in csv file
forecast[['ds', 'yhat']].to_csv("D:/JOB/out.csv")
# to combine actual data frame with predicted values 
metric_df = forecast.set_index('ds')[['yhat']].join(df2.set_index('ds').y).reset_index()
# to print the data
metric_df
# to drop NA values
#metric_df.dropna(inplace=True)
#metric_df
# check r2 value
r2_score(metric_df.y, metric_df.yhat)
# check mse
mean_squared_error(metric_df.y, metric_df.yhat)
# check MAE
mean_absolute_error(metric_df.y, metric_df.yhat)

# To insert data frame into MS SQL database without iterate the dataframe
#params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=Brijesh;PWD=PASSWORD")
#engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
#engine.connect() 
#insert.to_sql(name='table_name11',con=engine, index=True, if_exists='append')

# Another method to insert sata frame into ms sql table      
# Insert data frame into table
#insert.to_sql(name="table_name", con=engine, index=False, if_exists='append')

# rename column name asper earlier
metric_df = metric_df.rename(columns={'ds': 'Date',
                        'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})

# add centre column
metric_df['Centre']='Lucknow'
# add region column
metric_df['Region']='North'
# add Country column
metric_df['Country']='India'
# add Commodity column
metric_df['Commodity']='Tur/Arhar Dal'

metric_df.columns


# To insert data frame into MS SQL database without iterate the dataframe
params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=Brijesh;PWD=PASSWORD")
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
engine.connect() 
metric_df.to_sql(name='TempSales_Predictive',con=engine, index=True, if_exists='append')
