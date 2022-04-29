#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:35:32 2022

@author: brochetc

Metrics to be used in traning and testing of 2D gan4arome

"""
import scipy.stats as sc
import numpy as np
from multiprocessing import Pool

###### TO CREATE : metrics function class, should have name, long_name, identical calling possibilities

class metric2D():
    def __init__(self,long_name,func):
        self.long_name=long_name
        self.func=func
    
    def __call__(self, *args, **kwargs):
        return self.func(*args,**kwargs)
    

#### TO CREATE : criterion class, should have name, long_name, identical calling possibilities




#from time import perf_counter

def wasserstein_wrap(data):
    real_data, fake_data, real_weights, fake_weights=data
    return sc.wasserstein_distance(real_data, fake_data, \
                                   real_weights, fake_weights)

def W1_on_image_samples(real_data, fake_data, num_proc=4,\
                        Crop_Size=64,
                        real_weights=None, fake_weights=None):
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    data is cropped at the center so as to reduce comput. overhead
    
    """
    Side_Size=fake_data.shape[2]
    HALF=Side_Size//2-1
    half=Crop_Size//2
    real_data=real_data[:,:-1,:,:]  #dropping last channel as its orography
    fake_data=fake_data[:,:-1,HALF-half:HALF+half, HALF-half:HALF+half]
    Channel_size=real_data.shape[1]
    Lists=[]
    for ic in range(Channel_size):
        for i_x in range(Crop_Size):
            for j_y in range(Crop_Size):
                Lists.append((real_data[:,ic,i_x,j_y],fake_data[:,ic,i_x,j_y],\
                              real_weights, fake_weights))
    #print(Lists[0:2][0][:2], Lists[0:2][1][:2])
    with Pool(num_proc) as p:
        W_list=p.map(wasserstein_wrap,Lists)
    return np.array(W_list).mean()

def orography_RMSE(fake_batch, test_data):
    orog=test_data[0,-1:,:,:]
    fake_orog=fake_batch[:,-1:,:,:]
    res=np.sqrt(((fake_orog-orog)**2).mean())
    return  res



Orography_RMSE=metric2D('RMS Error on orography synthesis  ', orography_RMSE)