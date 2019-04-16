# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 00:58:42 2019
This file is used to get the database connection
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

class DBConnection:
   'Common base class for all employees'

   def dbConnection():
       # You can lookup the port number inside SQL server. 
       #conn = pymssql.connect(server="SERVERNAME", user="Brijesh",password="PASSWORD", port=1433) 
       #return conn
       
       params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER=SERVERNAME;DATABASE=DATABASENAME;UID=USERNAME;PWD=PASSWORD")
       engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
       return engine.connect() 
      
