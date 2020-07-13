# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:21:26 2020

@author: Brijesh.R
"""

import pandas as pd
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class
import numpy as np

irisData = pd.read_csv('D:/Personal/Github/UpdatedMain/FreeCodeCamp-DS/With Python/Visualization/Dataset/iris_csv.csv', 
                          names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

print(irisData.head())

wineReviews = pd.read_csv('D:/Personal/Github/UpdatedMain/FreeCodeCamp-DS/With Python/Visualization/Dataset/winemag-data-130k-v2.csv')

print(wineReviews.head())

'''Matplotlib is the most popular python plotting library
   Matplotlib is specifically good for creating basic graphs like line charts, bar charts, histograms and many more.
'''

AV = AutoViz_Class()

df = AV.AutoViz('D:/Personal/Github/UpdatedMain/FreeCodeCamp-DS/With Python/Visualization/Dataset/winemag-data-130k-v2.csv')

# Sample plot with 4 numbers
plt.plot([1,3,2,4])
plt.title('Sample with 4 numvers')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()


# Sample plot with X and Y values
plt.plot([1,2,3,4], [1,4,9,16])
# To add the points in some shape
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axes([0,6,0,20])
plt.title('Sample with 4 numvers', color='red', fontsize=8)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

# Change Figure size, plot red dots set axes scal X=0-8, Y=0-20, devide the bin size on x and y axis
plt.figure(figsize=(5,7))
plt.plot([1,2,3,6], [1,4,9,16], 'rX')
#plt.axes([0,6,0,20])
plt.xlim([0, 8])
plt.ylim([0, 20])
plt.yticks(np.arange(0, 20, 3))
plt.xticks(np.arange(0, 8, 2))
#plt.annotate('square it',(3,6))
plt.show()


# Bar charts with 4 bar ( bar maily use to see the frequance of category)
x = np.arange(4)
# x = ('Bha', 'Man','Bri', 'Shri')
y = [8,10,4,6]
plt.xticks(x,('Bha', 'Man','Bri', 'Shri'))
plt.bar(x,y,color='r')
plt.show()

# Two sets of 10 random dots plotted
plt.clf()
d = {'Red O' : np.random.rand(10),
     'Grn X' : np.random.rand(10)}
df=pd.DataFrame(d)
df.plot(style=['ro','gx'])
plt.legend(loc='top left')
plt.show()

# Time series - six months of random floats
ts = pd.Series(np.random.randn(180), index=pd.date_range('1/1/2018', periods=180))
df = pd.DataFrame(np.random.randn(180, 3), index=ts.index, columns=list('ABC'))
df.cumsum().plot()
plt.show()

# Random dots in a scatter¶
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sizes = (30 * np.random.rand(N))**2   # 0 to 15 point radii
plt.scatter(x, y, s=sizes, c=colors, alpha=0.5)
plt.show()


# Now, let's do al operations on file 
df = pd.read_csv('D:/Personal/Github/UpdatedMain/FreeCodeCamp-DS/With Python/Visualization/Dataset/Fremont_weather.txt')

'''
#For auto display the data
filePath = 'D:/Personal/Github/UpdatedMain/FreeCodeCamp-DS/With Python/Visualization/Dataset/Fremont_weather.txt'
df = AV.AutoViz(filePath)
'''
p1=plt.bar(df['month'], df['record_high'], color='r')
p2=plt.bar(df['month'], df['record_low'], color='c')
plt.plot(df['month'], df['avg_high'], color='k')
plt.plot(df['month'], df['avg_low'], color='b')
plt.legend()
plt.show()
# In the above graph problem is if record high values is less then it will hide by record low, so we can use clustered bar insted of stacked

# Plot subplots within the same window or figure
fig = plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
ax1.bar(df['month'], df['record_high'], color='r')
ax2.bar(df['month'], df['record_low'], color='c')
#ax2.axhline(0.45)
plt.show()


# Plot stacked bar chart using subplots within the same window or figure
fig = plt.figure()
ax1=fig.add_subplot(111)
ax1.bar(df['month'], df['record_high'], color='r')
ax1.bar(df['month'], df['record_low'], color='c')
#ax2.axhline(0.45)
plt.show()

p1=plt.bar(df['month'], df['record_high'], color='r')
plt.text(df['record_high'],ha = 'center', va='bottom')
p2=plt.bar(df['month'], df['record_low'], color='c')
for r1,r2 in zip(p1,p2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x()+r1.get_width()/2., h1+h2, '%s'% (h1+h2), ha = 'center', va='bottom')
    
   
# Stacked bar graph with displaying values on top of each bars    
def bar_group(classes, values, width=0.8):
    plt.xlabel('Month', weight='semibold')
    plt.ylabel('Value of Record high and low', weight='semibold')
    total_data = len(values)
    classes_num = np.arange(len(classes))
    for i in range(total_data):
        bars = plt.bar(classes_num - width / 2. + i / total_data * width, values[i], 
                width=width / total_data, align="edge", animated=0.4)

        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.xticks(classes_num, classes, rotation=-45, size=11)
    plt.legend(['record_high', 'record_low','avg_high','avg_low'])

fig = plt.figure(figsize=(20,5)) 
plt.show()

# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py   


# Line Graph 

gas = pd.read_csv('D:/Personal/Dataset/gas_prices.csv')


plt.plot(gas.Year, gas.USA)
plt.plot(gas.Year, gas.Canada)
plt.plot(gas.Year, gas['South Korea'])
plt.plot(gas.Year, gas.Australia)
plt.legend()
