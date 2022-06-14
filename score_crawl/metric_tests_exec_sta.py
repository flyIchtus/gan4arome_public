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
import pickle
from glob import glob
import numpy as np

CI=(78,206,55,183)

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_fsteps='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/samples/'
data_dir_f='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/samples/'

original_data_dir='/scratch/mrmn/brochetc/'
output_dir='/scratch/mrmn/brochetc/GAN_2D/Set_37/resnet_128_wgan-hinge_64_32_1_0.001_0.001/Instance_1/log/'

if __name__=="__main__":
    print(data_dir_f, output_dir)
    N_samples_fake=16 #16384]
    N_samples_real=16384    
    #program={i :(1,N_samples_real) for i in range(1)}

    
    stand_alone_metrics_list=["spectral_compute","struct_metric"]
    
    list_steps=[1500*k for k in range(40)]+[59999]  #list of steps to test
    
    results={} 
    results["header"]=stand_alone_metrics_list
    
    for step in list_steps:
        print('Iteration Step',step)
        #getting first (and only) item of the random real dataset program
        #dataset_r=snip.build_datasets(data_dir, program)[0]
        
        #getting files to analyze from fake dataset
        files=glob(data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')

        for metric in stand_alone_metrics_list:
            #cuda=True if metric!="spectral_dist" else False
            print(metric)
            data=(metric, files, 0, False, 'fake')
            
            if step==0:
                assert hasattr(metrics,metric)
            
            res0=snip.global_dataset_eval(data)
            
            if step==0: results[metric]=[res0]
            else :
                results[metric].append(res0)
                
    for metric in stand_alone_metrics_list:
        results[metric]=np.array(results[metric])
        
    pickle.dump(results, open(output_dir+'stand_alone_metrics_'+str(N_samples_real)+'.p', 'wb'))