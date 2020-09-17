# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:55:39 2020

@author: Brijesh.R
"""


from PIL import Image 
import pytesseract 
from pdf2image import convert_from_path 
import pyocr
import pyocr.builders

def readAndInsert(PDF_file): 
 ''' 1st step - Converting PDF to images '''
 # Store all the pages of the PDF in a variable 
 pages = convert_from_path(PDF_file, 500) 
 # Counter to store images of each page of PDF to image 
 image_counter = 1
 
 # Iterate through all the pages stored above 
 for page in pages: 
  ''' Declaring filename for each page of PDF as JPG 
   For each page, filename will be: 
   PDF page 1 -> page_1.jpg 
   PDF page 2 -> page_2.jpg 
   .... 
  # PDF page n -> page_n.jpg 
  '''
  filename = "page_"+str(image_counter)+".jpg"
  # Save the image of the page in system
  page.save(filename, 'JPEG') 
  # Increment the counter to update filename 
  image_counter = image_counter + 1

 ''' 2nd step - Recognizing text from the images using OCR '''
 # Variable to get count of total number of pages 
 filelimit = image_counter-1
 
  
 # This variable we going to use to store all the text data
 filetext=""
 # To store the text data page by page in the list
 fulltext =[]
 
 # pytesseract .exe path (in case pytesseract not set in envirnment variable
 pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
 
 # Iterate from 1 to total number of pages 
 for i in range(1, filelimit + 1): 
  ''' Set filename to recognize text from Again, these files will be: 
  page_1.jpg 
  page_2.jpg 
  .... 
  page_n.jpg '''
  filename = "page_"+str(i)+".jpg" 
  # Recognize the text as string in image using pytesserct 
  # I use psm (Page segmentation modes) 4 to extract the table data
  text = str(pytesseract.image_to_string(Image.open(filename), lang='eng', config='--oem 3 --psm 4')) 
  
  '''
  Store recognized text in variable text, replace every '-\n' to with '', 
  in case half of the word in next line 
  '''
  text = text.replace('-\n', '')
  # Stroe the text in list	
  fulltext.append(text)
  # concat all strings from list and store into filetext variable
  filetext = ' '.join(fulltext)
 # return the text extracted from the image 
 return filetext