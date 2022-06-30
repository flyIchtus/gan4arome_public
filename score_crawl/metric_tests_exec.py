#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""

import sys
sys.path.append('/home/mrmn/brochetc/gan4arome/metrics4arome/')

import metric_test_snippets as snip
import metrics4arome as metrics
from glob import glob
import numpy as np
from config import getAndmakeDirs

CI=(78,206,55,183)

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_fsteps='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/samples/'
data_dir_f='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/samples/'

original_data_dir='/scratch/mrmn/brochetc/'
output_dir='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/log/'

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

data_dirs_f, log_dirs, list_steps=getAndmakeDirs()
print(len(data_dirs_f), len(log_dirs), len(list_steps))
    
if __name__=="__main__":
    N_samples_fake=16 #16384]
    N_samples_real=16384    
    program={i :(1,N_samples_real) for i in range(1)}  
    distance_metrics_list=["scat_SWD_metric_renorm","scat_SWD_metric"]
    stand_alone_metrics_list=["spectral_compute", "struct_metric"]

    for data_dir_f, log_dir, steps in zip(data_dirs_f, log_dirs, list_steps):
        try:
            
           #parallelEstimation_standAlone(data_dir_f, data_dir, log_dir, steps)
           #logdir0=data_dir
           #sequentialEstimation_realVSfake(data_dir,\
           #                                logdir0, program, add_name='fid')
           #break
           sequentialEstimation_realVSfake(data_dir_f, data_dir,\
                                           log_dir,steps, program, 
                                           add_name='swd_scat_comparison_')
           
        except (FileNotFoundError, IndexError):
            print('File Not found  for {}  !'.format(data_dir_f))