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
