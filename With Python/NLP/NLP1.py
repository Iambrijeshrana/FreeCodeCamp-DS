# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:31:06 2020

@author: Brijesh.R
"""


import  spacy
import en_core_web_sm
from spacy import displacy

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
    
# Tokenizing
# Vocab
# Chenking
# Display

# tokenizing - process of spliting the sentence or graph into parts.
# In a sentece word is token and in a paragraph sentence is a token.


txt = '''Brijesh RanaProcessing raw text intelligently is difficult: most words are rare, 
          and it’s common for words that look completely different to mean 
          almost the same thing. The same words in a different order can mean 
          something completely different.
          brijeshrana.cse@gmail.com
          '''
txt = nlp(txt)
for token in txt:
   print(token)      

for token in txt:
   print(token, end=" | ")
   
len(txt)

# Vocab
# get the list of vocab
vacanlist = list(nlp.vocab.strings)

print(vacanlist)
len(vacanlist)

' doc, laxme, vocab, StringStpre'

doc = nlp('I love you')

# slicing 
txt[0]
txt[5:28]
# with slicing we can not change the values 
txt[0] = 'Hello'

# find out entities

for entity in doc.ents:
    print(entity.text, entity)
    
    
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Part-of-speech tags and dependencies
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)    

# Named entity    
for ent in doc.ents:
 print(ent.text, ent.start_char, ent.end_char, ent.label_)   
 
for ent in doc.ents:
 print(ent.text, ent.start_char, ent.end_char, ent.label_)  


displacy.serve(doc, style="dep")  


doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)
    
# Add named entity    
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("fb is hiring a new vice president of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('Before', ents)
# the model didn't recognise "fb" as an entity :(

fb_ent = Span(doc, 0, 1, label="ORG") # create a Span for the new entity
doc.ents = list(doc.ents) + [fb_ent]

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('After', ents)
# [('fb', 0, 2, 'ORG')] 🎉    

nlp.pipe_names

spacy.explain('advcl')
doc = nlp("Google went to play basketball")

for token in doc:
    print(token.text, "-->", token.dep_)
    
    
for cc in doc.ents:
    print(cc.text, cc.entity)