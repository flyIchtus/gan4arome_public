#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:22:30 2022

@author: brochetc


experiment initializer, v2

    Include :
        -> better management of input/output files
        -> management of many experiments launched at once
        -> automatic launching of experiments
"""

import os
from glob import glob
import argparse
from itertools import product
import subprocess

ensemble={'--lr_D': [1e-3],'--lr_G' : [1e-3],'--batch_size':[32],\
          '--n_dis':[1,2,5], '--sn_on_g': [True], '--use_amp':[False],\
          '--lrD_sched':['exp'], '--lrG_sched':['exp'], '--lrD_gamma':[0.9],\
          '--lrG_gamma':[0.9], '--ortho_init': [True]}

script_dir='/home/mrmn/brochetc/gan4arome2/gan_horovod/'



def str2bool(v):
    return v.lower() in ('true')

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

def create_new(keyword):
    previous=len(glob('*'+keyword+'*'))
    INSTANCE_NUM=previous+1
    os.mkdir(keyword+'_'+str(INSTANCE_NUM))

def get_dirs():
    parser=argparse.ArgumentParser()
     # Path
    parser.add_argument('--data_dir', type=str, \
                        default='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/')
    parser.add_argument('--output_dir', type=str, \
                        default='/scratch/mrmn/brochetc/GAN_2D/')
    
    return parser.parse_args()


def get_expe_parameters():

    parser = argparse.ArgumentParser()

    # Model architecture hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', \
                        choices=['resnet', 'coco-gan'])
    parser.add_argument('--train_type', type=str, default='wgan-hinge',\
                        choices=['vanilla','wgan-gp', 'wgan-hinge'])
    parser.add_argument('--version', type=str, default='resnet_128')
    parser.add_argument('--patch_size', type=int, default=128)
    
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--d_channels', type=int, default=3)
    parser.add_argument('--g_output_dim', type=int, default=128)
    parser.add_argument('--d_input_dim', type=int, default=128)
    parser.add_argument('--lamda_gp', type=float, default=10.0)
    parser.add_argument('--ortho_init',type=str2bool, default=False)
    
    parser.add_argument('--sn_on_g', type=str2bool, default=False,\
                        help='Apply spectral normalisation on Generator')

    # Training setting
    parser.add_argument('--epochs_num', type=int, default=250,\
                        help='how many times to go through dataset')
    parser.add_argument('--total_step', type=int, default=50000,\
                        help='how many times to update the generator')
    
    parser.add_argument('--n_dis', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=1,\
                        help="Accumulation factor for batch_size")
    
    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0002)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    # Channel data description
    parser.add_argument('--var_names', type=str2list, default=['u','v','t2m'])#, 'orog'])
    
    #Training setting -schedulers
    parser.add_argument('--lrD_sched', type=str, default='None', \
                        choices=['None','exp', 'linear'])
    parser.add_argument('--lrG_sched', type=str, default='None', \
                        choices=['None','exp', 'linear'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
    
    # Testing and plotting settings
    parser.add_argument('--test_samples',type=int, default=128)
    parser.add_argument('--plot_samples', type=int, default=16)
    
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=0,\
                        help='step at which pretrained model have been saved')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--use_amp', type=str2bool, default=True)

    parser.add_argument('--num_cpu_workers', type=int, default=2)
    parser.add_argument('--num_gpu_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    # Path
    parser.add_argument('--data_dir', type=str, \
                        default='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/')
    parser.add_argument('--output_dir', type=str, \
                        default='/scratch/mrmn/brochetc/GAN_2D/')

    # Step size
    parser.add_argument('--log_step', type=int, default=1000) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=1000) # set to 0 if not needed
    parser.add_argument('--plot_step', type=int, default=1000) #set to 0 if not needed
    parser.add_argument('--save_step', type=int, default=1000) # set to 0 if not needed
    parser.add_argument('--test_step', type=int, default=1000) #set to 0 if not needed
    

    config=parser.parse_args()
    assert config.g_channels==len(config.var_names) and config.d_channels==len(config.var_names)
    assert config.log_step%config.test_step==0
    
    return parser

def make_dicts(ensemble, option='cartesian'):
    """
    make cartesian product of parameters used in ensemble
    
    input :
        ensemble : dict of shape { parameter : list(parameters to test) }
        
    output :
        cross product of ensemble as list of dictionaries, each entry of shape
        {parameter : value}
    """
    allowed_args=set(vars(get_expe_parameters())['_option_string_actions'])
    keys=set(ensemble)

    assert keys<=allowed_args
    prod_list=[]
    if option=='cartesian':
        for item in product(*(list(ensemble.values()))):
            prod_list.append(dict((key[2:],i) for key, i in zip(ensemble.keys(),item)))
        
    print(len(prod_list))

    return prod_list

def nameSpace2SlurmArg(config):
    """
    transform an argparse.namespace to slurm-readable chain of args
    
    input :
        config  -> an argParse.Namespace as created by argparse.parse_args()
    
    output :
        a chain of characters summing up the namespace
    """
    
    dic=vars(config)
    li=[]
    for key in dic.keys():
        if key=='var_names':
            print(dic[key], str(dic[key]))
        value=dic[key]
        li.append('--'+key+'='+str(value))
    print(li, "|".join(li))
    return "|".join(li)
    
def prepare_expe(config):
    """
    create output folders in the current directory 
    for a single experiment instance
    
    input :
        
        a config namespace as generated by get_expe_parameters
    
    ouput : 
        NAME : the name of the main experiment directory
    """
    NAME=config.version+'_'+config.train_type+'_'+str(config.latent_dim)+'_'\
    +str(config.batch_size)+'_'+str(config.n_dis)+'_'+str(config.lr_D)+'_'+str(config.lr_G)
    print(NAME)
    base_dir=os.getcwd()
    if not os.path.exists(NAME):
        os.mkdir(NAME)
    os.chdir(NAME)
    
    previous_inst=len(glob('*Instance*'))
    INSTANCE_NUM=previous_inst+1
    os.mkdir('Instance_'+str(INSTANCE_NUM))
    os.chdir('Instance_'+str(INSTANCE_NUM))
    expe_dir=os.getcwd()
    
    with open('ReadMe_'+str(INSTANCE_NUM)+'.txt', 'a') as f:
        f.write('-----------------------------------------\n')
        for arg in config.__dict__.keys():
            f.write(arg+'\t:\t'+str(config.__dict__[arg])+'\n')
        f.close
    if not os.path.exists('log'):
        os.mkdir('log')
        
    if not os.path.exists('models'):
        os.mkdir('models')
        
    if not os.path.exists('samples'):
        os.mkdir('samples')
    
    os.chdir(base_dir)
    return NAME, expe_dir

def prepare_expe_set(where,expe_list):
    """
    prepare a set of experiments through use of cross product namespaces
    
    input :
        list of dicts containing experiment parameters
    
    output :
        list of namespaces containing
    """
    base_dir=where
    config_list=[]
    with open('Memo_readme.csv', 'a') as file:
        strKeys=list(expe_list[0].keys())+['directory']
        file.write(','.join(strKeys)+'\n')

    for params in expe_list:
        
        ns=argparse.Namespace(**params)
        config=get_expe_parameters().parse_args(namespace=ns)
        print("config_var_names",config.var_names)
        NAME, expe_dir=prepare_expe(config)
        config.output_dir=expe_dir
        config_list.append(config)
        params["output_dir"]=expe_dir
        #writing in memo file
        strVals={key : str(value) for key, value in params.items()}
        strVals['directory']=NAME
        
        with open('Memo_readme.csv', 'a') as file:
            file.write(','.join(strVals.values())+'\n')

        
        
    os.chdir(base_dir)
    
    return config_list


if __name__=="__main__":
    
    dirs=get_dirs()
    previous_sets=len(glob(dirs.output_dir+'*Set*'))
    SET_NUM=previous_sets+1
    where=dirs.output_dir+'Set_'+str(SET_NUM)
    if not os.path.exists(where):
        os.mkdir(where)
    os.chdir(where)
    
    ensemble['--data_dir']=[dirs.data_dir]
    ensemble['--output_dir']=[dirs.output_dir]
    expe_list=make_dicts(ensemble)
    
    config_list=prepare_expe_set(where,expe_list)
    
    for params in expe_list:
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print('Running experiment')
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        args=nameSpace2SlurmArg(argparse.Namespace(**params))
        subprocess.run(['sbatch',script_dir+'run_GAN.slurm',args])
