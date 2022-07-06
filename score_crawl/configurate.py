#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:17:43 2022

@author: brochetc

metrics computation configuration tools

"""
import argparse
from score_crawl.evaluation_backend import var_dict

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
    
    with open(path+'ReadMe_'+str(instance_num)+'.txt', 'r') as f :
        li=f.readlines()
        for line in li:
            if "crop_indexes" in line :
                CI=[int(c) for c in str2list(line[15:-1])]
                print(CI)
            if "var_names" in line :
                var_fake=[v[1:-1] for v in str2list(line[12:-1])]
        print('variables', var_fake)
        f.close()
        try :
            var_real_indices=[var_dict[v] for v in var_fake]
        except NameError :
            raise NameError('Variable names not found in configuration file')
        
        try :
            print(CI)
        except UnboundLocalError :
            CI=[78,206,55,183]
        
    return CI, var_fake, var_real_indices

def getAndNameDirs(root_expe_path):
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--expe_set', type=int,help='Set of experiments to dig in.', default = 1)
    parser.add_argument('--lr0', type=str2list, help='Set of initial learning rates', default = [0.001])
    parser.add_argument('--batch_sizes',type=str2list, help='Set of batch sizes experimented', default=[8,16,32])
    parser.add_argument('--instance_num', type=str2list, help='Instances of experiment to dig in', default = [1,2,3,4])
    
    multi_config=parser.parse_args()
    
    names=[]
    short_names=[]
    list_steps=[]
    
    for lr in multi_config.lr0:
        for batch in multi_config.batch_sizes :
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
    
    multi_config.length=len(data_dir_names)
    
    return multi_config

def select_Config(multi_config, index):
    
    
    insts = len(multi_config.instance_num)
        
    config=argparse.Namespace()
    
    config.data_dir_f = multi_config.data_dir_names[index]
    config.log_dir = multi_config.log_dir_names[index]
    config.steps = multi_config.list_steps[index]
    
    
    instance_index = index%insts
    
    config.instance_num=multi_config.instance_num[instance_index]
        
    return config

class Experiment():
    
    def __init__(self, expe_config):
        
        self.data_dir_f = expe_config.data_dir_f
        self.log_dir = expe_config.log_dir
        self.expe_dir = self.log_dir[:-4]
            
        self.steps = expe_config.steps
        
        self.instance_num = expe_config.instance_num
        
        indices = retrieve_domain_parameters(self.expe_dir, self.instance_num)
        
        self.CI, self.var_names , self.VI = indices
        
    def __print__(self) :
    
        print("Fake data directory {}".format(self.data_dir_f))
        print("Log directory {}".format(self.log_dir))
        print("Experiment directory {}".format(self.expe_dir))
        print("Instance num {}".format(self.instance_num))
        print("Step list : ", self.steps)
        print("Crop indices", self.CI)
        print("Var names" , self.var_names)
        print("Var indices", self.VI)