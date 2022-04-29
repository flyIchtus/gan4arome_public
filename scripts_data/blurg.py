#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:12:46 2022

@author: brochetc

see fetch
"""
import os
import glob

os.chdir('/home/mrmn/brochetc/scratch_link/')
varnames=['_rrdecum', '_t2m', '_u', '_v', '_rr']
list0=os.listdir()

toFetch=[]
date=''
for filename in list0:
    if len(filename)>20:
        dateNew=filename[:20]
        if date!=dateNew:
            date=dateNew
            if len(glob.glob(date+'*'))<5:
                print(glob.glob(date+'*'))
            