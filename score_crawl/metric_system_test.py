#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""

import sys
import argparse
sys.path.append('/home/mrmn/brochetc/gan4arome/metrics4arome/')

import metric_test_snippets as snip
import metrics4arome as metrics
import pickle
from glob import glob
import numpy as np
from multiprocessing import Pool


def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        print(li)
        li2=li[1:-1].split(',')
        print(li2)
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def getAndmakeDirs():
    
    parser=argparse.ArgumentParser()
    
    
    parser.add_argument('-expe_set', type=int,help='Set of experiments to dig in.')
    parser.add_argument('-lr0', type=str2list, help='Initial learning rates experimented')
    parser.add_argument('-batch_sizes',type=str2list, help='Set of batch sizes experimented')
    parser.add_argument('-instance_num', type=str2list, help='Instances of experiment to dig in')
    
    config=parser.parse_args()
    
    
    names=[]
    list_steps=[]
    for lr in config.lr0 :
        for batch in config.batch_sizes :
            print(batch)
            for instance in config.instance_num:
                names.append('/scratch/mrmn/brochetc/GAN_2D/Set_'+str(config.expe_set)\
                                    +'/resnet_128_wgan-hinge_64_'+str(batch)+\
                                    '_1_'+str(lr)+'_'+str(lr)+'/Instance_'+str(instance))
                if int(batch)<=64:
                    list_steps.append([1500*k for k in range(40)]+[59999])
                else:
                    list_steps.append([1500*k for k in range(22)])
    data_dir_names, log_dir_names=[f+'/samples/' for f in names],[f+'/log/' for f in names]
    
        
    return data_dir_names, log_dir_names, list_steps

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

data_dirs_f, log_dirs, list_steps=getAndmakeDirs()
print(len(data_dirs_f), len(log_dirs), len(list_steps))
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