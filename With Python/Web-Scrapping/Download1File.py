# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:32:06 2019

@author: Brijesh Rana
"""
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

url = 'http://web.mta.info/developers/turnstile.html'
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

soup.findAll('a')

one_a_tag = soup.findAll('a')[36]
link = one_a_tag['href']

download_url = 'http://web.mta.info/developers/'+ link
urllib.request.urlretrieve(download_url,'./'+link[link.find('/turnstile_')+1:]) 
# Put the thread in sleep mode to avoid the traffic on website
time.sleep(1)