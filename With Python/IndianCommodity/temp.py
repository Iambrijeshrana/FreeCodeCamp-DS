# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:04:12 2019

@author: Brijesh Rana
"""

 df = pd.DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2]},
                      index=['a', 'b'])
df
for row in df.itertuples():
     print(row.col1, row.col2)