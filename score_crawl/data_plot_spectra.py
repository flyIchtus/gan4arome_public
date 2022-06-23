#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:17:26 2022

@author: brochetc


data print/plot spectra

"""


import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle


data_dir_real='/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

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
    parser.add_argument('-batch_sizes',type=str2list, help='Set of batch sizes experimented')
    parser.add_argument('-instance_num', type=str2list, help='Instances of experiment to dig in')
    
    config=parser.parse_args()
    
    
    names=[]
    list_steps=[]
    
    for batch in config.batch_sizes :
        print(batch)
        for instance in config.instance_num:
            names.append('/scratch/mrmn/brochetc/GAN_2D/Set_'+str(config.expe_set)\
                                +'/resnet_128_wgan-hinge_64_'+str(batch)+\
                                '_1_0.001_0.001/Instance_'+str(instance))
            if int(batch)<=64:
                list_steps.append([1500*k for k in range(40)]+[59999])
            else:
                list_steps.append([1500*k for k in range(22)])
    data_dir_names, log_dir_names=[f+'/samples/' for f in names],[f+'/log/' for f in names]
    
        
    return data_dir_names, log_dir_names, list_steps


_,log_dirs, list_steps=getAndmakeDirs()

var=['u','v', 't2m']
res_real=pickle.load(open(data_dir_real+'real3stand_alone_metrics_66048.p', 'rb'))
print(res_real.keys())
spec0_real=res_real['spectral_compute'][:,:,:,0]

for logd, list_step in zip(log_dirs, list_steps):
    print(logd)
    try :
        res=pickle.load(open(logd+'vquantiles_stand_alone_metrics_66048.p', 'rb'))
        print(res.keys())
        spec=res['spectral_compute'][:,:,:,0]
        
        spec_diff=np.sqrt(np.mean(((np.log10(spec)-np.log10(spec0_real))**2)[:,:,-15:], axis=-1))
        print(np.min(spec_diff, axis=0).mean(axis=0))
        for i in range(3):
            print(var[i],min(spec_diff[:,i]))
            print(var[i],np.mean(spec_diff[-10:,i]))
            plt.plot(list_step,spec_diff[:,i])
        plt.savefig(logd+'rmse_spectrum.png')
        plt.close()
    except FileNotFoundError:
        print(logd, 'file not found')