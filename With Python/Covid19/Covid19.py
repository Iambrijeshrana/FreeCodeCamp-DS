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
import matplotlib.pylab as plt

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

indiandatawithoutzero = indiaconfirmcase.loc[indiaconfirmcase['Value']>0]

grouped = confirmcase.groupby('Country/Region')

print(grouped['Value'].agg(np.sum))

a=grouped['Value'].agg(np.sum)
    
indiandatawithoutzero.shape
sns.set(rc={'figure.figsize':(12,12)})
g=sns.barplot(y='Date', x='Value', data = indiandatawithoutzero,orient="h")
for index, row in indiandatawithoutzero.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black')
    
sns.lineplot(x='Date', y='Value', data = indiaconfirmcase)

sns.distplot(indiandatawithoutzero.Value)

# sort indiandatawithoutzero by Date column
indiandatawithoutzero = indiandatawithoutzero.sort_values(['Value']).reset_index(drop=True)


plt.figure(figsize=(16,10))
# plot barh chart with index as x values
ax = sns.barplot(indiandatawithoutzero.index, indiandatawithoutzero.Value, color='red')
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Date", ylabel='Value')
# add proper Dim values as x labels
ax.set_xticklabels(indiandatawithoutzero.Date)
for item in ax.get_xticklabels(): item.set_rotation(90)
for i, v in enumerate(indiandatawithoutzero["Value"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='m', va ='bottom', rotation=90)
plt.tight_layout()
plt.show()

# ---------------------------------------------------



plt.figure(figsize=(16,10))
# plot barh chart with index as x values
ax = sns.barplot(indiandatawithoutzero.index, indiandatawithoutzero.Value, color='red')
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Date", ylabel='Value')
# add proper Dim values as x labels
ax.set_xticklabels(indiandatawithoutzero.Date)
for item in ax.get_xticklabels(): item.set_rotation(90)
for i, v in enumerate(indiandatawithoutzero["Value"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='m', va ='bottom', rotation=90)
plt.tight_layout()
plt.show()
