#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:41:23 2022

@author: brochetc

Diverse Wasserstein distances computations
"""

import scipy.stats as sc
import torch
import numpy as np
from multiprocessing import Pool

###############################################################################
################### Wasserstein distances ############################
###############################################################################

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
    return [np.array(W_list).mean()]

def W1_center(real_data, fake_data,Crop_Size=64):
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    data is cropped at the center so as to reduce comput. overhead
    
    
    """
    Side_Size=fake_data.shape[2]
    HALF=Side_Size//2-1
    half=Crop_Size//2
    real_data=real_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    fake_data=fake_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    Channel_size=real_data.shape[1]
    dist=torch.tensor([0.], dtype=torch.float32).cuda()
    for i in range(Crop_Size):
        for j in range(Crop_Size):
            for c in range(Channel_size):
                real,_=torch.sort(real_data[:,c,i,j],dim=0)
                fake,_=torch.sort(fake_data[:,c,i,j],dim=0)
                dist=dist+torch.abs(real-fake).mean()
    return dist*(1e3/(Crop_Size**2*Channel_size))
                

