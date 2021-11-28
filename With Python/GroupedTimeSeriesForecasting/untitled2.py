# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:43:47 2021

@author: Brijesh Rana
"""

import turtle
col=('red','yellow','green','cyan','pink','white')
t=turtle.Turtle()
screen=turtle.Screen()
screen.bgcolor('black')
t.speed(25)
for i in range(250):
    t.color(col[i%6])
    t.forward(i*1.5)
    t.left(59)
    t.width(3)