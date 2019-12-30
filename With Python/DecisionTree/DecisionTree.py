# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:00:50 2019

@author: Brijeshkumar
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)

"For checking the length & dimensions of our dataframe, we can use len() method & .shape."
print("Dataset Lenght:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

balance_data.head()


# Data slicing (Split the data into test and train)
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

"************ Decision Tree Training ****************"
'''
Now we fit Decision tree algorithm on training data, predicting labels for validation dataset 
and printing the accuracy of the model using various parameters.
'''

'''
DecisionTreeClassifier(): This is the classifier function for DecisionTree. It is the main function for 
implementing the algorithms. Some important parameters are:

1. criterion: It defines the function to measure the quality of a split. Sklearn supports 
“gini” criteria for Gini Index & “entropy” for Information Gain. By default, it takes 
“gini” value.
2. splitter: It defines the strategy to choose the split at each node. Supports “best” 
value to choose the best split & “random” to choose the best random split. By default, 
it takes “best” value.
3. max_features: It defines the no. of features to consider when looking for the best split.
We can input integer, float, string & None value.
    If an integer is inputted then it considers that value as max features at each split.
    If float value is taken then it shows the percentage of features at each split.
    If “auto” or “sqrt” is taken then max_features=sqrt(n_features).
    If “log2” is taken then max_features= log2(n_features).
    If None, then max_features=n_features. By default, it takes “None” value.
4. max_depth: The max_depth parameter denotes maximum depth of the tree. It can take any 
integer value or None. If None, then nodes are expanded until all leaves are pure or until 
all leaves contain less than min_samples_split samples. By default, it takes “None” value.
5. min_samples_split: This tells above the minimum no. of samples reqd. to split an internal
node. If an integer value is taken then consider min_samples_split as the minimum no. 
If float, then it shows percentage. By default, it takes “2” value.
6. min_samples_leaf: The minimum number of samples required to be at a leaf node. If an 
integer value is taken then consider min_samples_leaf as the minimum no. If float, then it 
shows percentage. By default, it takes “1” value.
7. max_leaf_nodes: It defines the maximum number of possible leaf nodes. If None then it 
takes an unlimited number of leaf nodes. By default, it takes “None” value.
8. min_impurity_split: It defines the threshold for early stopping tree growth. A node will 
split if its impurity is above the threshold otherwise it is a leaf.

Let’s build classifiers using criterion as gini index & information gain. We need to fit our classifier using fit(). We will plot our decision tree classifier’s visualization too.
'''

# Decision Tree with Gini Index Criterion
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
model = clf_gini.fit(X_train, y_train)

# Plot the tree
tree.plot_tree(model) 

# To export the tree in graph
''' tHIS CODE WILL NOT WORK HERE '''
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X_train.column

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
# -------------------------------------------------------------

# Decision Tree with Information Gain (Entropy) Criterion
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
model2 = clf_entropy.fit(X_train, y_train)

tree.plot_tree(model2) 

"*************** Prediction *****************"
'''Now, we have modeled 2 classifiers. One classifier with gini index & another one with 
information gain as the criterion. We are ready to predict classes for our test set. We can 
use predict() method. Let’s try to predict target variable for test set’s 1st record.
'''

y_pred = clf_gini.predict(X_test)
y_pred

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

"***************** Calculating Accuracy Score *********************"
'''
The function accuracy_score() will be used to print accuracy of Decision Tree algorithm. 
By accuracy, we mean the ratio of the correctly predicted data points to all the predicted 
data points. Accuracy as a metric helps to understand the effectiveness of our algorithm. 
It takes 4 parameters.

y_true,
y_pred,
normalize,
sample_weight.
Out of these 4, normalize & sample_weight are optional parameters. The parameter y_true  
accepts an array of correct labels and y_pred takes an array of predicted labels that are 
returned by the classifier. It returns accuracy as a float value.

Accuracy for Decision Tree classifier with criterion as gini index and Entropy
'''

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)