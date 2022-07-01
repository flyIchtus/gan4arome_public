#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:18:43 2022

@author: brochetc

Score analysis and comparison
"""

import pickle


def average_dynamics(step,N_samples,directories):
    """
    compute an average of different metrics dynamics stored in separate directories
    """
    dir0=directories[0]
    
    results=pickle.load(open(dir0+str(step)+'_'+str(N_samples)+'.p','rb'))
    metrics_list=results["header"]
    MEAN_RES={k : v for (k,v) in results.items()}
    
    for direct in directories:

        res=pickle.load(open(dir0+str(step)+'_'+str(N_samples)+'.p','rb'))
        for metric in metrics_list:
            MEAN_RES[metric]+=res[metric]
            
    for metric in metrics_list:
        MEAN_RES[metric]=MEAN_RES[metric]/len(directories)
        
    return MEAN_RES