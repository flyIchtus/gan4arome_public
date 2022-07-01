#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:17:43 2022

@author: brochetc

metrics computation configuration tools

"""
import argparse

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        
        if ', ' in li :
            li2=li[1:-1].split(', ')
        else :
            
            li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
        
def retrieve_domain_parameters(path, instance_num):
    
    with open(path+'Readme_'+str(instance_num)+'.txt', 'r') as f :
        li=f.readlines()
        for line in li:
            if "crop_indexes" in line :
                CI=[int(c) for c in str2list(line[24:-1])]
                print(CI)
            if "var_names" in line :
                print(line[24:])
                var_fake=[v[1:-1] for v in str2list(line[24:-1])]
        
        f.close()
        try :
            
            var_real_indices=[var_dict[v] for v in var_fake]
        
        except NameError :
            raise (NameError, 'Variable names not found in configuration file')
        
    return CI, var_fake, var_real_indices

def getAndNameDirs(root_expe_path):
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-expe_set', type=int,help='Set of experiments to dig in.')
    parser.add_argument('-lr0', type=str2list, help='Set of initial learning rates')
    parser.add_argument('-batch_sizes',type=str2list, help='Set of batch sizes experimented')
    parser.add_argument('-instance_num', type=str2list, help='Instances of experiment to dig in')
    
    multi_config=parser.parse_args()
    
    names=[]
    short_names=[]
    list_steps=[]
    
    for lr in multi_config.lr0:
        for batch in multi_config.batch_sizes :
            print(batch)
            for instance in multi_config.instance_num:
                
                names.append(root_expe_path+'Set_'+str(multi_config.expe_set)\
                                    +'/resnet_128_wgan-hinge_64_'+str(batch)+\
                                    '_1_'+str(lr)+'_'+str(lr)+'/Instance_'+str(instance))
                
                short_names.append('Instance_{}_Batch_{}_LR_{}'.format(instance, batch,lr))
                
                if int(batch)<=64:
                    list_steps.append([1500*k for k in range(40)]+[59999])
                    
                else:
                    list_steps.append([1500*k for k in range(22)])
                    
    data_dir_names, log_dir_names=[f+'/samples/' for f in names],[f+'/log/' for f in names]
    
    multi_config.data_dir_names=data_dir_names
    multi_config.log_dir_names=log_dir_names
    multi_config.short_names=short_names
    multi_config.list_steps=list_steps
    
    return multi_config

def select_Config(multi_config, index=-1):
    
    config=argparse.Namespace()
    
    if index>=0 :
        
        config.data_dir_f = multi_config.data_dir_names[index]
        config.log_dir = multi_config.log_dir_names[index]
        config.steps = multi_config.list_steps[index]
        
    return config
    