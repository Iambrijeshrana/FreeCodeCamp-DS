# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:26:00 2019

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

serverName="********"
schemaName="dbo"
databaseName="**********"
userName="Brijesh"
password="***********"

connectionURL = "DRIVER={SQL Server};SERVER="+serverName+";DATABASE="+databaseName+";UID="+userName+";PWD="+password



# reading WDICountry csv file  
wdiCountryDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDICountry.csv")# reading WDICountry csv file  
wdiCountrySeriesDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDICountry-Series.csv")# reading WDICountry csv file  
wdiDataDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDIData.csv")# reading WDICountry csv file  
wdiFootNoteDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDIFootNote.csv")# reading WDICountry csv file  
wdiSeriesDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDISeries.csv")
wdiSeriesDF = pd.read_csv("C:/Users/Brijesh Rana/Desktop/WDI_csv/WDISeries-Time.csv")

params = urllib.parse.quote_plus(connectionURL)
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
engine.connect() 

wdiCountryDF.to_sql(name='WDICountry', con=engine, index=False, if_exists='append')
wdiCountrySeriesDF.to_sql(name='WDICountry_Series', con=engine, index=False, if_exists='append')
wdiDataDF.to_sql(name='WDIData', con=engine, index=False, if_exists='append')
wdiFootNoteDF.to_sql(name='WDIFootNote', con=engine, index=False, if_exists='append')
wdiSeriesDF.to_sql(name='WDISeries', con=engine, index=False, if_exists='append')
wdiSeriesDF.to_sql(name='WDISeries_Time', con=engine, index=False, if_exists='append')
