# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

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

# Create dataframe for each category
# creare dataframe with unique Centre Names
#centreSQL = "select distinct Centre from [HIRANANDANI_REPORTS].[dbo].[tempsales1]"
#centreDF = pd.read_sql(centreSQL,conn)

# creare dataframe with unique Region Names
#regionSQL = "select distinct Region from [HIRANANDANI_REPORTS].[dbo].[tempsales1]"
#regionDF = pd.read_sql(regionSQL,conn)

# creare dataframe with unique Commodity Names
#commoditySQL = "select distinct Commodity from [HIRANANDANI_REPORTS].[dbo].[tempsales1]"
#commodityDF = pd.read_sql(commoditySQL,conn)

# creare dataframe with unique Commodity Names, Region Names, and Centre Names
combineSQL = "select distinct Centre, Region, Commodity from [DATABADSENAME].[dbo].[TABLENAME]"
combineDF = pd.read_sql(combineSQL,conn)
combineDF 
combineDF = pd.DataFrame(combineDF)
for row in combineDF.itertuples():
    ## Hey Look, college data
    stmt = "SELECT Date,[Price per Kg]  FROM [HIRANANDANI_REPORTS].[dbo].[tempsales1] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
    #"SELECT Date,[Price per Kg]  FROM [HIRANANDANI_REPORTS].[dbo].[tempsales1] where Centre='MYSORE' and Region='SOUTH' and Commodity='Rice'"
    
    # Excute Query here
    stmt
    df = pd.read_sql(stmt,conn)
    if(len(df) >= 2):
        # check data type of Date column from Object to Date
        df['Date'] = df['Date'].astype('datetime64[ns]')
        df.dtypes
        # To convert price datatype from object to flaot
        df['Price per Kg'] = df['Price per Kg'].astype(float)
        df.dtypes
        # rename the actual column name  because in prophet the column name should be DS and Y
        df = df.rename(columns={'Date': 'ds',
                                'Price per Kg': 'y'})
        df.dtypes
        # print new data frame
        df
        # Plot the data 
        #ax = df.set_index('ds').plot(figsize=(12, 8))
        #ax.set_ylabel('Monthly Price of Commoditity')
        #ax.set_xlabel('Date')
        #plt.show()
        ## create prophet model
        my_model = Prophet(interval_width=0.95)
        # fit the prophet model
        my_model.fit(df)
        # select futre date for which we want to predict
        future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
        #future_dates.tail()
        # predict future value
        forecast = my_model.predict(future_dates)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        #forecast.columns
        #my_model.plot(forecast,
        #              uncertainty=True)
        # plot seasionality, trend based on yearly monthly day wise etc.
        #my_model.plot_components(forecast)
        # check columns name in forecast model
        #forecast.coloumns
        # to see first top 10 rows in forecast model
        #forecast.head(10)
        #to check coulmns data types
        #forecast.dtypes
        # to combine actual data frame with predicted values 
        metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
        # to print the data
        #metric_df
        # to drop NA values
        #metric_df.dropna(inplace=True)
        #metric_df
        # check r2 value
        #r2_score(metric_df.y, metric_df.yhat, )
        # check mse
        #mean_squared_error(metric_df.y, metric_df.yhat)
        # check MAE
        #mean_absolute_error(metric_df.y, metric_df.yhat)
        
        # To insert data frame into MS SQL database without iterate the dataframe
        #params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=Brijesh;PWD=PASSWORD")
        #engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
        #engine.connect() 
        #insert.to_sql(name='table_name11',con=engine, index=True, if_exists='append')
        
        # Another method to insert sata frame into ms sql table      
        #engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=Brijesh;PWD=PASSWORD')
        # Insert data frame into table
        #insert.to_sql(name="table_name", con=engine, index=False, if_exists='append')
        
        # rename column name asper earlier
        metric_df = metric_df.rename(columns={'ds': 'Date',
                                'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
        
        # add centre column
        metric_df['Centre']=row.Centre
        # add region column
        metric_df['Region']=row.Region
        # add Country column
        metric_df['Country']='India'
        # add Commodity column
        metric_df['Commodity']=row.Commodity
        
        #metric_df.columns
        
        
        # To insert data frame into MS SQL database without iterate the dataframe
        params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=Brijesh;PWD=PASSSWORD")
        engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
        engine.connect() 
        metric_df.to_sql(name='TempSales_Predictive1',con=engine, index=False, if_exists='append')
        metric_df.re
    else:
        print("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2")
