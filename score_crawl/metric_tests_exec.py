#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""
import score_crawl.evaluation_frontend as frontend
from score_crawl.configurate import getAndNameDirs, select_Config


original_data_dir='/scratch/mrmn/brochetc/'


root_expe_path='/scratch/mrmn/brochetc/GAN_2D/'

    
if __name__=="__main__":
    
    configuration_set=getAndNameDirs(root_expe_path)
    
    N_samples=16384    
    program={i :(1,N_samples) for i in range(1)}  
    
    distance_metrics_list=["W1_random_NUMPY", "W1_Center_NUMPY"]
    standalone_metrics_list=["spectral_compute", "struct_metric"]
    
    for ind in range(configuration_set.length):
        
        expe_config = select_Config(configuration_set, ind)
         
        try :
            
            mC=frontend.MetricsCalculator(expe_config, 'test_for_score_crawl')
            
            mC.estimation(standalone_metrics_list, program, standalone=True, parallel=True)
           
        except (FileNotFoundError, IndexError) :
            print('File Not found  for {}  ! This can be due to either \
                  inexistent experiment set, missing ReadMe file,\
                  or missing data file. Continuing !'.format(expe_config.data_dir_f))