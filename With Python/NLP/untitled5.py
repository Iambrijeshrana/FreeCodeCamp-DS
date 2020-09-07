# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:28:01 2020

@author: Brijesh.R
"""


import spacy
import pickle
import random


train_data = pickle.load(open('D:/train_data.pkl', 'rb'))

train_data[0]

nlp = spacy.blank('en')

def train_model(train_data):
    if 'ner'not in nlp.pipe_names:
        ner=nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2    ])