# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:17:24 2021

@author: Brijesh Rana
"""

import numpy as np 

# convert list to array

mylist = [1,2,3,4,5]
mylist

arr = np.array(mylist)

arr

# List of lits - convert into matrics 

my_mat = [[1,2,3],[4,5,6],[7,8,9]]

np.array(my_mat)

# arrange funcation - similar to range funcation 
# arrange(start, end, step)

np.arange(0,11,2)

# generate specific arrays - one for 1's and 0's
np.zeros(3)

np.ones((5,5))

np.ones(3)


# generate number of points between two numbers
np.linspace(1,2,4)

# identity matrix
np.eye(4)

# generate array of random numbers 

arr=np.random.rand(8)

np.random.rand(5,5)

a=arr.reshape(2,4)

arr.max()

arr.min()

a.shape

a.dtype


arr=np.arange(1,11)

boolarr = arr>5
boolarr 

arr[arr>5]

arr[boolarr]


arr


arr=arr+arr
arr

arr*100 

## PANDAS 
import pandas as pd
label=['a','b','c']
my_data=[10,20,30]
arr=np.array(my_data)

pd.Series(my_data)

pd.Series(data=my_data, index=label)
pd.Series(arr)

d={'a':10,'b':20}
pd.Series(d)

pd.Series(['a','b','c'],[10,20,30],[20,40,50])

d={'a':10,'b':20,'c':{'d':5,'f':6}}

pd.Series(d)

pd.Series(d['c'])


data_dict = {"a": 1, "b": 2, "c": 3}
data_items = data_dict. items()
data_list = list(data_items)
df = pd. DataFrame(data_list)

data_items = d. items()
data_list = list(data_items)
df = pd. DataFrame(data_list)

df=pd.read_html('https://towardsdatascience.com')


df=pd.read_ex
