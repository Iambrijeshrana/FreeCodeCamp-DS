# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:37:27 2020

@author: Brijesh.R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/personal/Dataset/Classified data", index_col=0)

df

df.info()
df.describe()

df.plot().box


scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

scale_feature = scaler.transform(df.drop('TARGET CLASS', axis=1))

scale_feature

df_new = pd.DataFrame(scale_feature, columns=df.columns[:-1])

df_new

X_train, X_text, y_train, y_test = train_test_split(df_new, df['TARGET CLASS'], test_size=.3, random_state=101)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_text)

df = pd.DataFrame({'Actual': y_test, 'Pred': y_pred})

c=df[df['Actual'] != df['Pred']]

len(c)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# way to choose correct k value


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_text)
    error.append(np.mean(pred_i != y_test))
    
    
'The next step is to plot the error values against K values. Execute the following script to create the plot:'

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

' From the graph we can see error rate is minimum from 5 to almost 18, so to decide the best valu for K we have to check multiple values and find the minimum error point'
