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
from multiprocessing import Pool




########### standard parameters #####

num_proc=4


def parallelEstimation_realVSfake(data_dir_f, data_dir, log_dir,program, steps, add_name=''):
    results={} 
    results["header"]=distance_metrics_list
    
    for metric in distance_metrics_list :
        print(metric)
        assert hasattr(metrics, metric)
        data_list=[]         
    
        for step in steps:
            
            #getting first (and only) item of the random real dataset program
            dataset_r=snip.build_datasets(data_dir, program)[0]
            
            #getting files to analyze from fake dataset
            files=glob(data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
            
          
            data_list.append((metric, {'real':dataset_r,'fake': files}, step))
        
        with Pool(num_proc) as p :
            res=p.map(snip.eval_distance_metrics, data_list)
        results[metric]=np.array(res)
                        
    pickle.dump(results, open(log_dir+add_name+'distance_metrics_'+str(N_samples_real)+'.p', 'wb'))
    

def sequentialEstimation_realVSfake(data_dir_f, data_dir, log_dir, steps, program, add_name=''):
    
    results={} 
    results["header"]=distance_metrics_list
    
    for metric in distance_metrics_list :
        print(metric)
        assert hasattr(metrics, metric)
    
        for step in steps:
            
            #getting first (and only) item of the random real dataset program
            dataset_r=snip.build_datasets(data_dir, program)[0]
            
            #getting files to analyze from fake dataset
            files=glob(data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
            
          
            data=(metric, {'real':dataset_r,'fake': files}, step)
        
            if step==0: results[metric]=[snip.eval_distance_metrics(data)]
            else :
                
                results[metric].append(snip.eval_distance_metrics(data))
       
        results[metric]=np.array(results[metric])
                        
    pickle.dump(results, open(log_dir+add_name+'distance_metrics_'+str(N_samples_real)+'.p', 'wb'))
    
    
def parallelEstimation_realVSreal(data_dir, log_dir, program, steps, add_name=''):
    results={} 
    results["header"]=distance_metrics_list
    
    for metric in distance_metrics_list :
        print(metric)
        assert hasattr(metrics, metric)
        
        datasets=snip.build_datasets(data_dir, program)
        data_list=[]         
    
        #getting the two random datasets programs
            
        for i in range(len(datasets)):
          
            data_list.append((metric, {'real0':datasets[i][0],'real1': datasets[i][1]}, i))
        
        with Pool(num_proc) as p :
            res=p.map(snip.eval_distance_metrics, data_list)
        results[metric]=np.array(res)
                        
    pickle.dump(results, open(log_dir+add_name+'realVreal_metrics_'+str(N_samples_real)+'.p', 'wb'))
    
def sequentialEstimation_realVSreal(data_dir, log_dir, program, add_name=''):
    results={} 
    results["header"]=distance_metrics_list
    
    for metric in distance_metrics_list :
        print(metric)
        assert hasattr(metrics, metric)
    
        
        #getting first (and only) item of the random real dataset program
        datasets=snip.build_datasets(data_dir, program)
            
        for i in range(len(datasets)):
          
            data=(metric, {'real0':datasets[i][0],'real1': datasets[i][1]}, i)
        
            if i==0: results[metric]=[snip.eval_distance_metrics(data)]
            else :  
                results[metric].append(snip.eval_distance_metrics(data))
       
        results[metric]=np.array(results[metric])
                        
    pickle.dump(results, open(log_dir+add_name+'realVreal_metrics_'+str(N_samples_real)+'.p', 'wb'))
    
    
def parallelEstimation_standAlone(data_dir_f, data_dir, log_dir, steps, program=None,option='fake', add_name=''):
    results={} 
    results["header"]=stand_alone_metrics_list
    print(data_dir_f, data_dir)
    
    if option=='real':
        assert program is not None
        dataset_r=snip.build_datasets(data_dir, program)
    
    for metric in stand_alone_metrics_list :
        
        print(metric)
        assert hasattr(metrics, metric)
        data_list=[]         
        
        for i,step in enumerate(steps):
                 
            #getting files to analyze from fake dataset
            if option=='fake' :
                files=glob(data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
                data_list.append((metric, files,step, False, option))
                
            elif option=="real":
                data_list.append((metric, dataset_r[i], step, False, option))
                

        with Pool(num_proc) as p :
            res=p.map(snip.global_dataset_eval, data_list)
        results[metric]=np.array(res)
                        
    pickle.dump(results, open(log_dir+add_name+'stand_alone_metrics_'+str(N_samples_real)+'.p', 'wb'))
    


