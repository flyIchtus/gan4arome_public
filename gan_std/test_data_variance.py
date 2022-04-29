#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 08:35:15 2022

@author: brochetc

test_data manipulation

"""
import torch
import DataSet_Handler_horovod_v2 as DSH
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

N_times=5
test_samples=1024
data_dir="/scratch/mrmn/brochetc/GAN_2D_10/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/"

Dl_test=DSH.ISData_Loader(data_dir, test_samples)
test_dataloader=Dl_test.loader(hvd.size(), hvd.rank(), {})

def intra_map_var(data):
    res=torch.mean(torch.var(data, dim=(2,3)), dim=0)
    return res

def inter_map_var(data):
    res=torch.mean(torch.var(data, dim=0), dim=(1,2))
    return res



for i in range(N_times):
    real_samples,_,_=next(iter(test_dataloader))
    real_samples.cuda()
    print('inter_map_var ', inter_map_var(real_samples))
    print('intra_map_var ', intra_map_var(real_samples))
