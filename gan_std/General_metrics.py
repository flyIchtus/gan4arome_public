#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:42:42 2022

@author: brochetc


General metrics
"""

import torch
import numpy as np

############################ General simple metrics ###########################

def intra_map_var(useless_data, data):
    res=torch.mean(torch.var(data, dim=(2,3)), dim=0)
    return res

def inter_map_var(useless_data,data):
    res=torch.mean(torch.var(data, dim=0), dim=(1,2))
    return res

def orography_RMSE(fake_batch, test_data, torch=False):
    orog=test_data[0,-1:,:,:]
    fake_orog=fake_batch[:,-1:,:,:]
    if torch :
        res=torch.sqrt(((fake_orog-orog)**2).mean())
    else :
        res=np.sqrt(((fake_orog-orog)**2).mean())
    return res


