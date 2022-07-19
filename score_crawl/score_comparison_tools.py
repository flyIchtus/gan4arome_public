#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:18:43 2022

@author: brochetc

Score analysis and comparison
"""

import pickle
from glob import glob
import numpy as np
import pandas as pd

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


metrics_storage_list = {'test_for_score_crawl_standalone_metrics_16384.p' : ['spectral_compute'],
                        'swd_scat_distance_metrics_16384.p' :['scat_SWD_metric_renorm'],
                        'W1_estimates_distance_metrics_16384.p' : ['W1_random_NUMPY', 'W1_Center_NUMPY'],
                        'SWD_karras_mod_distance_metrics_16384.p' : ['SWD_metric_torch']}

metrics_names = {'spectral_compute' : ['RMSE_spectral_u', 'RMSE_spectral_v', 'RMSE_spectral_t2m'],\
                 'scat_SWD_metric_renorm' : ['sparse_swd_glob', 'sparse_swd_u', 'sparse_swd_v', 'sparse_swd_t2m',
                                            'shape_swd_glob', 'shape_swd_u', 'shape_swd_v', 'shape_swd_t2m'],
                 'W1_random_NUMPY' : ['w1_random'],
                 'W1_Center_NUMPY' : ['w1_center'],
                 'SWD' : ['SWD_128', 'SWD_64', 'SWD_32', 'SWD_16', 'SWD_avg']
                 }

data_dir = '/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

print('loading_real_spectrum')
spectral_real = pickle.load(open(data_dir+'real3stand_alone_metrics_66048.p','rb'))['spectral_compute'][0,:,:,0]

def rmse_spectrum(real, fake) :
    
    Shape = fake.shape
    
    rmse=[]
    
    for c in range(Shape[0]) :
        
      rmse.append(np.sqrt(((fake[c,:]-real[c,:])**2).mean(axis=0))) 
    
    return np.array(rmse)

def big_csv_file(log_path, step_list):
        
    with open(log_path+'metrics_summary.csv', 'w') as recept :
        
        recept.write('Step,')
        for i,f in enumerate(metrics_storage_list.keys()) :
            names = {}
            
            sto = metrics_storage_list[f]
            for j,m in enumerate(sto) :
                
                names[m] = metrics_names[m]
                
                for name in names[m] :
                
                    recept.write(name)
                    
                    if name != 'w1_center' :
                        recept.write(',')
                    else :
                        recept.write('\n')

        res = []
        print('Names', names)
        print('loading_metrics')
        for k,f in enumerate(metrics_storage_list.keys()) :
            res.append(pickle.load(open(log_path + f, 'rb')))
                    
        print('metrics_loaded')
        
        for s_index,step in enumerate(step_list) :
            
            print(step)
            recept.write(str(step)+',')
            
            for k,f in enumerate(metrics_storage_list.keys()) :
                    
                print(k,f)                
            
                name_ind = 0
                
                res0 = res[k]
                
                metrics = metrics_storage_list[f]
                
                if 'spectral_compute' in metrics :
                    
                    data = res0['spectral_compute'][s_index]
                    
                    data_real = spectral_real
                    
                    rmse = rmse_spectrum(data_real, data)
                    
                    for i in range(rmse.shape[0]):
                        recept.write(str(rmse_spectrum(data_real, data)[i]))
                        recept.write(',')
                        
                        name_ind += 1
                    
                elif 'scat_SWD_metric_renorm' in metrics :
                    
                    data = res0['scat_SWD_metric_renorm'][s_index]
                    
                    n_ests = 2
                    n_scores = 4 
                    for i in range(n_ests) :
                        for j in range(n_scores):
                            recept.write(str(data[i,j]))
                            recept.write(',')
                            
                            name_ind += 1
                            
                elif 'W1_random_NUMPY' in metrics :
                    
                    data = res0['W1_random_NUMPY'][s_index]*1e3
                    
                    recept.write(str(data))
                    recept.write(',')
                    
                    name_ind += 1
                    
                    data = res0['W1_Center_NUMPY'][s_index]
                    
                    recept.write(str(data))
                    recept.write(',')
                    
                    name_ind += 1
                    
                elif 'SWD_metric_torch' in metrics :
                    
                    data = res0['SWD_metric_torch'][s_index]
                    
                    recept.write(str(data))
                    recept.write('\n')
                    
                    name_ind += 1
                    
                    
        recept.close()
                
def select_best_from_csv(filename, start=0):
    
    metric_bests={}
    dataframe = pd.read_csv(filename)['Step' >=start]
    metrics_list = dataframe.columns
    for metric in metrics_list[metrics_list != 'Step'] :
        metric_bests[metric] = dataframe[metric].min()
    
    return metric_bests
    
    