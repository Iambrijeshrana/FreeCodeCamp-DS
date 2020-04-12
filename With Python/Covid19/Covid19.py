# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:15:38 2020

@author: Brijesh.R
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
import mylib

confirmcase = pd.read_csv("D:/Personal/Dataset/Covid19/main/time_series_covid19_confirmed_global_narrow.csv")
confirmcase 
deathcacase = pd.read_csv("D:/Personal/Dataset/Covid19/main/time_series_covid19_deaths_global_narrow.csv")
recovercase = pd.read_csv("D:/Personal/Dataset/Covid19/main/time_series_covid19_recovered_global_narrow.csv")

confirmcase.size
confirmcase.shape
deathcacase.shape
recovercase.shape

confirmcase.columns

indiaconfirmcase = confirmcase.loc[confirmcase['Country/Region']=='India']

indiaconfirmcase.head()
sns.barplot(x='Date', y='Value', data = indiaconfirmcase.head(),orient="v")

indiandatawithoutzero = indiaconfirmcase.loc[indiaconfirmcase['Value']>0]


indiandatawithoutzero.shape
sns.set(rc={'figure.figsize':(10,15)})
sns.countplot(y='Date', x='Value', data = indiandatawithoutzero,orient="h")

grouped = confirmcase.groupby('Country/Region')

print(grouped['Value'].agg(np.sum))

a=grouped['Value'].agg(np.sum)

sns.lineplot(x='Date', y='Value', data = indiaconfirmcase)
