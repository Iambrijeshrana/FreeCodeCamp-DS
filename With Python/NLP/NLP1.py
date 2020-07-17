# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:31:06 2020

@author: Brijesh.R
"""


import  spacy
import en_core_web_sm

doc = 'Tata going to sell india based cars to USA in 3$ million '
nlp=en_core_web_sm.load()

doc=nlp(doc)

for token in doc:
    print(token.text)
    
'''
A Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in 
some language and assigns parts of speech to each word (and other token), 
such as noun, verb, adjective, etc
POS tags have been used for a variety of NLP tasks and are extremely useful 
since they provide linguistic signal on how a word is being used within the 
scope of a phrase, sentence, or document.
Sometimes the POS is very very useful in cases where it distinguishes the word 
sense (the meaning of the word).
'''
    
for token in doc:
    print(token.text, token.pos)    
    
for token in doc:
    print(token.text, token.pos_)       

# .dep - will tell us the depandacy between words    
for token in doc:
    print(token.text, token.pos_, token.dep_)    

nlp.pipeline
nlp.pipe_names

# to get the shape of our word
for token in doc:
    print(token.text, token.pos_, token.shape_)
    
# .is_alpha-> return boolean if the word alphebeat
for token in doc:
    print(token.text, token.pos_, token.is_alpha)
    
# .is_stop -> return useless word (commonly use word) like the, to, in, is etc
for token in doc:
    print(token.text, token.pos_, token.is_stop)  
    
#     