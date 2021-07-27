# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:43:26 2021

@author: Brijesh Rana
"""

import cvlib as cv
import cv2
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
import os
import pandas as pd
import numpy as np
dest_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'

def detect_common_objects(image, confidence=0.5, nms_thresh=0.3, model='yolov3', enable_gpu=False):

    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes
    global dest_dir

    if model == 'yolov3-tiny':
        config_file_name = 'yolov3-tiny.cfg'
        cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
        weights_file_name = 'yolov3-tiny.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)


    else:
        config_file_name = 'yolov3.cfg'
        cfg_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg'
        weights_file_name = 'yolov3.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)    

    config_file_abs_path = dest_dir + os.path.sep + config_file_name
    weights_file_abs_path = dest_dir + os.path.sep + weights_file_name 
    
im=cv2.imread("D:/abc.jpg")

bbox, label, conf =cv.cvimagedetect_common_objects(im)

output_image = cv.draw_bbox(im, bbox, label, conf) 

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")
     # -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:14:48 2019

@author: Brijesh Rana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:26 2019

@author: Brijesh Rana
"""
import logging
from datetime import datetime
import os
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

class MainClass:
    # Create log file
    LOG_FILENAME = datetime.now().strftime('D:/log/mylogfile_%H_%M_%S_%d_%m_%Y.log')
    for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Forecastiong Job Started...')
    ##instance a python db connection object- same form as psycopg2/python-mysql drivers also
    logging.info('calling dbConnection()...')
    conn = DBConnection.dbConnection() 
    country = 'India'
    # creare dataframe with unique Commodity Names, Region Names, and Centre Names
    combineSQL = "select distinct Centre, Region, Commodity from [DATABASENAME].[dbo].[TABLENAME]"
    logging.info("SQL Statement to get unique Commodity Names, Region Names, and Centre Names : "+combineSQL) 
    # read data from database 
    combineDF = pd.read_sql(combineSQL,con=conn)
    # convert output into dataframe 
    combineDF = pd.DataFrame(combineDF)
    # Iterate the dataframe
    # we choose itertuples insted of iterrows here because itertuples is faster than iterrows
    logging.info("Iterating the dataframe to get Commodity, Region, Centre name one by one")
    for row in combineDF.itertuples():
        ## get Target variable (Commodity price) and date based on Commodity, Centre and Region names
        stmt = "SELECT Date,[Price per Kg]  FROM [DATABASENAME].[dbo].[TABLENAME] where Centre='"+row.Centre+"' and Region='"+row.Region+"' and Commodity='"+row.Commodity+"'"
        logging.info("SQL Statement to get Date and Target variable based on parameters : "+stmt)
        # print sql script
        #print(stmt)
        # Excute Query here
        df = pd.read_sql(stmt,con=conn)
        """
        In Facebook Prophet model we can't do prediction if number of observation is less than 2, We need at least 2 data points to do forecasting
        So we have given the conditon here to check whether data points is more than 2 or not
        """
        logging.info("Checking the size of dataframe...")
        if(len(df) >= 2):
            # change the data type of Date column from Object to Date
            df['Date'] = df['Date'].astype('datetime64[ns]')
            # to check data types in dataframe 
            #df.dtypes
            # To convert price datatype from object to flaot
            df['Price per Kg'] = df['Price per Kg'].astype(float)
            #logging.info("Dataframe datatypes after conversion : "+df.dtypes)
            #df.dtypes
            """ rename the actual column name  because in prophet the column name should be DS and Y
                Target variable  name should be y and data variable name should be ds
            """
            df = df.rename(columns={'Date': 'ds',
                                    'Price per Kg': 'y'})
            #logging.info("Dataframe columns name rename : "+df.columns)
            #df.columns
            # Plot the data 
            ax = df.set_index('ds').plot(figsize=(12, 8))
            ax.set_ylabel('Monthly Price of Commoditity')
            ax.set_xlabel('Date')
            plt.show()
            ## create prophet model
            my_model = Prophet(interval_width=0.95, seasonality_prior_scale = 10,weekly_seasonality= True)
            # fit the prophet model
            my_model.fit(df)
            """ select futre date for which we want to predict, we want to do prediction for next 3 years 
            and our data frequancy is month level so we have given periods = 36 and freq=MS 
            """
            future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
            #future_dates.tail()
            # predict future value
            logging.info("Building the Prophet forecast model...")
            forecast = my_model.predict(future_dates)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            #forecast.columns
            my_model.plot(forecast,
                          uncertainty=True)
            logging.info("Prophet forecasting model has been builded")
            # plot seasionality, trend based on yearly monthly day wise etc.
            logging.info("Prophet forecasting model doing forecast for future")
            my_model.plot_components(forecast)
            # check columns name in forecast model
            #forecast.coloumns
            # to combine actual data frame (df) with forecasted dataframe (forecast) to see predicted values and actual values 
            metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
            logging.info("Predicted values and Actual values combined into one dataframe")
            # to print the data
            validateDF = pd.DataFrame(metric_df)
            #metric_df
            # to drop NA values
            validateDF.dropna(inplace=True)
            #metric_df
            # check r2 value
            r2 = r2_score(validateDF.y, validateDF.yhat, )
            logging.info("R2 value of the model is : "+r2.astype('str'))
            # check mse
            mse = mean_squared_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Squared Error of model is : "+mse.astype('str'))
            # check MAE
            mae = mean_absolute_error(validateDF.y, validateDF.yhat)
            logging.info("Mean Absolute Error of model is : "+mae.astype('str'))
            # rename column name as per our database table name
            metric_df = metric_df.rename(columns={'ds': 'Date',
                                    'y': 'Price per Kg', 'yhat' : 'Pridictedprice'})
            logging.info("Dataframe column renamed as per SQL table")
            # add centre column
            metric_df['Centre']=row.Centre
            # add region column
            metric_df['Region']=row.Region
            # add Country column
            metric_df['Country']=country
            # add Commodity column
            metric_df['Commodity']=row.Commodity
            
            #metric_df.columns
            logging.info("Inserting the Prophet result into table...")
            params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
            engine.connect() 
            # To insert data frame into MS SQL database without iterate the dataframe
            metric_df.to_sql(name='PREDICTEDTABLENAME',con=engine, index=False, if_exists='append')
            logging.info("Prophet result inserted into table")
        else:
            logging.info("for "+row.Centre+" "+row.Region+" "+row.Commodity+" number of records are less than 2 so we can't predict")      
     #logging.info("Prediction Job completed")