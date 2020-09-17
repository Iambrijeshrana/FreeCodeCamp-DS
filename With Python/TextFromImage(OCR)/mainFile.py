# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:23:31 2020

@author: Brijesh.R
"""

import os 
import sys 
from ImagetoTextbyPytesseract import readAndInsert

# Create empty list to store all the pdf file names available in the directory
all_files = []
# Directory path where I have all the files (images)
dirPath="D:/Java/Images"
test = os.listdir(dirPath)
for item in test:
 # I want to extract the text from pdf images only   
 if item.endswith(".pdf"):
  print(item)
  all_files.append(item)

for i in all_files:
 inputfile=dirPath+"/"+i
 # output file path where I want to store the extracted text from images
 outfile="D:/PDFTOTEXT/"+i.replace('.pdf','')+'.txt'
 # call method to extract the text from image and store the text in content variable
 content = readAndInsert(inputfile)
 # content = os.linesep.join([s for s in content.splitlines() if s])
 # Create txt file to insert the text (extracted from image)
 f2 = open(outfile, "a") 
 # insert the image extracted text
 #f2.writelines(content.strip().replace('\n\n', '\n'))
 f2.writelines(content)
 # close the txt file after inserting all the extracted text from image
 f2.close()

