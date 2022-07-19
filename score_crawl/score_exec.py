#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:04:09 2022

@author: brochetc

score comparison executable

"""

import configurate as config
import score_comparison_tools as sct

root_expe_path = '/scratch/mrmn/brochetc/GAN_2D/'

multi_config = config.getAndNameDirs(root_expe_path)

for i in range(multi_config.length):
    
    expe_config = config.select_Config(multi_config, i)
    
    log_dir = expe_config.log_dir
    print(log_dir)
    
    steps = expe_config.steps
    try :
        sct.big_csv_file(log_dir, steps)
    
        scores = sct.select_best_from_csv(log_dir+'metrics_summary.csv')
        print(scores)
        
    except FileNotFoundError :
        
        print('File not found for {} !'.format(log_dir))