# -*- coding: utf-8 -*-

INPUTSCRIPT= '''
             SELECT top 200 [ITEM_CODE]
             ,[DRUG_NAME]
             ,[GENERIC_NAME]
             ,[FORM_DESC]
             ,ROUTE_DESC
             ,concat([GENERIC_NAME],' ',[DRUG_NAME],' ',[ROUTE_DESC],' ',[FORM_DESC]) as var1
             FROM ********************************
             where [DRUG_YN]='Y'
             '''
             
CHOICESCRIPT = '''
                SELECT [STR] AS RXNORM 
                FROM *******************
                where [SAB]='RXNORM' and [TTY] in ('SBD', 'SCD', 'BPCK', 'GPCK')
                '''             

