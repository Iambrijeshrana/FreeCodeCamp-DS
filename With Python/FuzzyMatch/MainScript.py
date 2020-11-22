# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:26:08 2020

@author: Brijesh.R
"""
import pandas as pd
from datetime import datetime
import logging
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 
import FuzzyMatch.InputString as inputData
from itertools import zip_longest 
import FuzzyMatch.Config as config 
import configparser

  
def createLogFile(path, fileName):
 try:
  LOG_FILENAME = datetime.now().strftime(path+''+fileName)
  logger = logging.getLogger()
  fhandler = logging.FileHandler(filename=LOG_FILENAME, mode='w')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fhandler.setFormatter(formatter)
  logger.addHandler(fhandler)
  logger.setLevel(logging.DEBUG)
  fhandler.close()
 except Exception as e:
  logging.info("Error while creating the log file")
  logging.info(e)
    
def fuzzyMatch():
 try:
  inputString=inputData.getInputString()
  choiceString=inputData.getChoiceString()   
  for row in inputString.itertuples():
   matchVar=process.extract(row.var1,choiceString.RXNORM,scorer=fuzz.token_set_ratio, limit=5) 
   matchingVarDF = pd.DataFrame(matchVar, columns =['RXNORM', 'Score', 'Index']) 
   matchingVarDF['Item_Code']=row.ITEM_CODE
   matchingVarDF['Generic_Name']=row.GENERIC_NAME
   matchingVarDF['Drug_Name']=row.DRUG_NAME
   matchingVarDF['Route_Desc']=row.ROUTE_DESC
   matchingVarDF['Form_Desc']=row.FORM_DESC
   matchingVarDF['Var']=row.var1
   matchingVarDF.drop(['Index'], axis = 1) 
   inputData.insertOutputVar1(matchingVarDF)
   logging.info('FuzzyMatch Data inserted for '+row.DRUG_NAME)
 except Exception as e:
  logging.info('Failed to do Fuzzy Match')
  logging.info(e)

try:
 # Property file path
 strpath = "E:\Brijesh\FuzzyMatch\Config.properties"
 config = configparser.RawConfigParser()
 config.read(strpath)
 # To read Log File detail info
 logFileInfo = dict(config.items('LOG FILE DETAILS'))
 logFilePath=logFileInfo.get('path')
 logFileName=logFileInfo.get('file_name')

 createLogFile(logFilePath, logFileName)
 fuzzyMatch()
except Exception as e:
 logging.info('main method failed')
 logging.info(e)   
   