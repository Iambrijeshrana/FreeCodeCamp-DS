# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:09:22 2020

@author: Brijeshkumar
"""

import nltk

nltk.download_shell()

messages = [line.rstrip() for line in open('D:/Personal/Dataset/smsspamcollection/SMSSpamCollection')]
print(len(messages))

messages[1]
messages[0]
messages[5573]


for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
    
import pandas as pd    

messages = pd.read_csv('D:/Personal/Dataset/smsspamcollection/SMSSpamCollection', sep='\t'
                       ,names=["label", "message"])

messages

messages.describe()

messages.groupby('label').describe()


messages['length'] = messages['message'].apply(len)
messages.head()


import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

messages['length'].plot(bins=50, kind='hist') 

sns.distplot(messages['length'], bins=250)

sns.distplot(messages['length'], bins=250, hist=False)

?sns.distplot

messages.hist(column='length', by='label', bins=150,figsize=(12,4))


# Text Pre-processing
# First removing punctuation
import string
mess = 'Sample message! Notice: it has punctuation.'

string.punctuation

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)

# Remove stop words
from nltk.corpus import stopwords
stopwords.words('english')

len(stopwords.words('english'))

nopunc.split()

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

clean_mess 


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



# Check to make sure its working
messages['message'].head(5).apply(text_process)
# Steaming will do later


from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

message4 = messages['message'][3]
print(message4)

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])

messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))