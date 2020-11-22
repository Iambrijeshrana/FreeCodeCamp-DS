# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 13:10:51 2020
This file is used to get the database connection.
@author: Brijesh Rana
"""
import configparser
import pymssql
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, select
from six.moves import urllib

config = configparser.RawConfigParser()
# Property file path
strpath = "E:\Brijesh\FuzzyMatch\Config.properties"
config.read(strpath)
# To read only database credentials
databaseDetails = dict(config.items('DATABASE CONNECTION DETAILS'))
serverName=databaseDetails.get('server_name')
schemaName=databaseDetails.get('scheam_name')
databaseName=databaseDetails.get('database_name')
userName=databaseDetails.get('user_name')
password=databaseDetails.get('password')
databaseName2=databaseDetails.get('database_name2')

connectionURL = "DRIVER={SQL Server};SERVER="+serverName+";DATABASE="+databaseName+";UID="+userName+";PWD="+password

def dbConnection():
 try:
  params = urllib.parse.quote_plus(connectionURL)
  engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
  return engine.connect() 
 except:
  print('Failed to establish the database connection')   

connectionURL2 = "DRIVER={SQL Server};SERVER="+serverName+";DATABASE="+databaseName2+";UID="+userName+";PWD="+password

def dbConnection2():
 try:
  params = urllib.parse.quote_plus(connectionURL2)
  engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
  return engine.connect() 
 except:
  print('Failed to establish the database connection')   
       

# con=dbConnection()
#con.close()

