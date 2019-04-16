# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:25:33 2019

@author: Brijesh Rana
"""

import numpy as np
import pandas as pd
outliers=[]

dataset= [10,12,12,13,12,11,14,13,15,10,10,10,100,12,14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10,15,12,10,14,13,15,10]

def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(dataset)
print(outlier_datapoints)


import pandas as pd

col_names =  ['A', 'B', 'C']
my_df  = pd.DataFrame(columns = col_names)
my_df

my_df.loc[len(my_df)] = [2, 4, 5]