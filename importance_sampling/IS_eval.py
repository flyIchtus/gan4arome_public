#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:51:51 2022

@author: brochetc

importance sampling evaluation procedures

"""
import numpy  as np
import os
import glob

path='/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done'

def Mean_Dataset(path):
    
    listing=glob.glob(path+'/_sample*')
    mean_array=np.zeros((len(listing),5))
  
    for i,filename in enumerate(listing):
        M=np.load(filename, allow_pickle=True)
        if i%1000==0: print(i,M.shape,M.mean(axis=(1,2)).shape)
        mean_array[i,:]=M.mean(axis=(1,2))
    return mean_array

def Mean_Sample(crop_size):
    path=os.getcwd()
    listing=glob.glob(path+'/_sample*')
    sum0=np.zeros((crop_size, crop_size,4))
    for i,filename in enumerate(listing):
        M=np.load(filename, allow_pickle=True)
        sum0=sum0+M
    return sum0/len(listing)

def Variance_Sample(crop_size, meanSample):
    path=os.getcwd()
    listing=glob.glob(path+'/*'+'sample*')
    sum0=np.zeros((crop_size, crop_size,4))
    for i,filename in enumerate(listing):
        M=np.load(filename, allow_pickle=True)
        sum0=sum0+(M-meanSample)**2
    return sum0/len(listing)

def Max_Dataset(path, mean=np.zeros(5,)):
    listing=glob.glob(path+'/_sample*')
    max_array=np.zeros((5,))
    for i,filename in enumerate(listing):
        M=np.load(filename, allow_pickle=True)-mean
        if i%1000==0 : print(i,M.max(axis=(1,2)).shape)
        local_min=M.min(axis=(1,2))
        local_max=M.max(axis=(1,2))
        m0=0.5*(abs(local_min)+local_max)+0.5*abs(local_max-abs(local_min))
        assert m0.shape==max_array.shape
        max_array=0.5*(max_array+m0)+0.5*abs(max_array-m0)
    return max_array
