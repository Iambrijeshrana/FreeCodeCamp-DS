# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:37:04 2020

@author: Brijesh.R
"""

'''
The FuzzyWuzzy library is built on top of difflib library, python-Levenshtein 
is used for speed. So it is one of the best way for string matching in python.
'''


from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 



fuzz.ratio('abc123ab', 'abc124')

fuzz.ratio('ABC12', 'abc124')   
  
fuzz.ratio('How are you', '! how are YOU@') 


fuzz.partial_ratio("howareyou", "howareyoou!") 
# Exclamation mark in second string, 
# still partially words are same so score comes 100
fuzz.partial_ratio("how are you", "you are how")   

fuzz.partial_ratio("how are you", "you are how")  
# score is less because there is a extra 
#token in the middle middle of the string. 


# Token Sort Ratio 
fuzz.token_sort_ratio("how are you", "you are how")   
# This gives 100 as every word is same, irrespective of the position  
  
# Token Set Ratio 
fuzz.token_sort_ratio("how are you", "you are how are you how")
fuzz.token_set_ratio("how are you", "you are how are you how")  
# Score comes 100 in second case because token_set_ratio  
# considers duplicate words as a single word. 

'''
Now suppose if we have list of list of options and we want to find the 
closest match(es), we can use the process module
'''

query = 'hey hi'
choices = ['hey hi what you doing', 'hey are you thr', 'g. for hi'] 

# Get a list of matches ordered by score, default limit to 5 
process.extract(query, choices) 
   
# If we want only the top one 
process.extractOne(query, choices) 
('geeks geeks', 95) 

process.extractOne(query, choices, scorer=fuzz.WRatio)

'''
There is also one more ratio which is used often called WRatio, 
sometimes its better to use WRatio instead of simple ratio as 
WRatio handles lower and upper cases and some other parameters too.
'''

fuzz.WRatio('how are you', 'HOW ARE YOU') 
fuzz.WRatio('how are you', 'HOW #ARE @YOU.!') 
# whereas simple ratio will give for above case 


############# Another example 

s1 = "I love coding"
s2 = "I am loving coding!"
print("FuzzyWuzzy Ratio: ", fuzz.ratio(s1, s2))
print("FuzzyWuzzy PartialRatio: ", fuzz.partial_ratio(s1, s2))
print("FuzzyWuzzy TokenSortRatio: ", fuzz.token_sort_ratio(s1, s2)) 
print("FuzzyWuzzy TokenSetRatio: ", fuzz.token_set_ratio(s1, s2)) 
print("FuzzyWuzzy WRatio: ", fuzz.WRatio(s1, s2),'\n\n')
  