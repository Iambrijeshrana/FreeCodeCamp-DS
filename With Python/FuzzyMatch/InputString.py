# -*- coding: utf-8 -*-

import pandas as pd
from FuzzyMatch.DatabaseConnection import dbConnection, dbConnection2  
import FuzzyMatch.FuzzySQLScripts as sqlScript

def getInputString():
 try:
  conn=dbConnection()
  inputString = pd.read_sql(sqlScript.INPUTSCRIPT,conn)
  return inputString
 except:
  print('Failed to fetch the input string')  
 finally:
  conn.close()
   
def getChoiceString():
 try:
  conn=dbConnection2()
  choiceString = pd.read_sql(sqlScript.CHOICESCRIPT,conn)
  return choiceString
 except:
  print('Failed to fetch the input string')  
 finally:
  conn.close()
  
def insertOutputVar1(dfVar1):
 try:   
  conn=dbConnection()
  dfVar1.to_sql(name='FuzzyMatchWithDrugName', con=conn, index=False, if_exists='append')   
  print('DataInserted') 
 except:
  print('Failed to fetch the input string')  
 finally:
  conn.close()
  
