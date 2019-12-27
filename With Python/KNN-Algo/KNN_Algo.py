# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:35:18 2019

@author: Brijeshkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

'''
Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will 
not work properly without normalization. For example, the majority of classifiers calculate the distance between two 
points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed 
by this particular feature. Therefore, the range of all features should be normalized so that each feature 
contributes approximately proportionately to the final distance.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

' ***************** Training and Implementation *************************** '

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

'''
parameter, i.e. n_neigbours. This is basically the value for the K. There is no ideal value for K and it is 
selected after testing and evaluation, however to start out, 5 seems to be the most commonly used value for KNN 
algorithm.
'''
y_pred = classifier.predict(X_test)

' ************************** Accuracy Check ****************************** '

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
'''
The results show that our KNN algorithm was able to classify all the 30 records in the test set with 97% accuracy, 
which is excellent. Although the algorithm performed very well with this dataset, don't expect the same results 
with all applications. As noted earlier, KNN doesn't always perform as well with high-dimensionality or 
categorical features.
'''

' ******************** Comparing Error Rate with the K Value **************************** '
''' There is no rule to select K value, One way to help you find the best value of K is to plot the 
graph of K value and the corresponding error rate for the dataset.
To do so, let's first calculate the mean of error for all the predicted values where K ranges from 1 and 40. 
'''

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

'The next step is to plot the error values against K values. Execute the following script to create the plot:'

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

' From the graph we can see error rate is minimum from 5 to almost 18, so to decide the best valu for K we have to check multiple values and find the minimum error point'

