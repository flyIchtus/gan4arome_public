#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:33:21 2022

@author: brochetc
"""

import argparse
import os
import trainer
import residual_nets as RN
import torch
from glob import glob



def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model architecture hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', \
                        choices=['resnet', 'coco-gan'])
    parser.add_argument('--train_type', type=str, default='wgan-gp',\
                        choices=['vanilla','wgan-gp', 'wgan-hinge'])
    parser.add_argument('--patch_size', type=int, default=128)
    
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--g_channels', type=int, default=5)
    parser.add_argument('--d_channels', type=int, default=5)
    parser.add_argument('--g_output_dim', type=int, default=256)
    parser.add_argument('--d_input_dim', type=int, default=256)
    parser.add_argument('--lamda_gp', type=float, default=10.0)
    parser.add_argument('--version', type=str, default='resnet_1')

    # Training setting
    parser.add_argument('--epochs_num', type=int, default=100,\
                        help='how many times to go through dataset')
    parser.add_argument('--total_step', type=int, default=0,\
                        help='how many times to update the generator')
    parser.add_argument('--n_dis', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0004)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    
    #Training setting -schedulers
    parser.add_argument('--lrD_sched', type=str, default=None, \
                        choices=[None,'exp', 'linear'])
    parser.add_argument('--lrG_sched', type=str, default=None, \
                        choices=[None,'exp', 'linear'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
    
    # Testing and plotting settings
    parser.add_argument('--test_samples',type=int, default=128)
    parser.add_argument('--plot_samples', type=int, default=16)
    
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None,\
                        help='step at which pretrained model have been saved')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--num_cpu_workers', type=int, default=2)
    parser.add_argument('--num_gpu_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    # Path
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=-1) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=100) # set to 0 if not needed
    parser.add_argument('--plot_step', type=int, default=100) #set to 0 if not needed
    parser.add_argument('--model_save_step', type=int, default=1) # set to 0 if not needed
    parser.add_argument('--test_step', type=int, default=1) #set to 0 if not needed
    # Channel data description
    parser.add_argument('--var_names', type=list, default=['rr','u', 'v', 't2m', 'orog'])

    config=parser.parse_args()
    assert config.g_channels==len(config.var_names) and config.d_channels==len(config.var_names)
    
    return parser.parse_args()

###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

config=get_parameters()


NAME=config.version+'_'+config.train_type+'_'+str(config.latent_dim)+'_'+str(config.batch_size)


if not os.path.exists(NAME):
    os.mkdir(NAME)
os.chdir(NAME)

previous_inst=len(glob('*Instance*'))
INSTANCE_NUM=previous_inst+1
os.mkdir('Instance_'+str(INSTANCE_NUM))
os.chdir('Instance_'+str(INSTANCE_NUM))


with open('ReadMe_'+str(INSTANCE_NUM)+'.txt', 'a') as f:
    f.write('-----------------------------------------\n')
    for arg in config.__dict__.keys():
        f.write(arg+'\t:\t'+str(config.__dict__[arg])+'\n')
    f.close
if not os.path.exists(config.log_path):
    os.mkdir(config.log_path)
    
if not os.path.exists(config.model_save_path):
    os.mkdir(config.model_save_path)
    
if not os.path.exists(config.sample_path):
    os.mkdir(config.sample_path)

###############################################################################
############################ BUILDING MODELS ##################################
###############################################################################


load_optim=False 
if config.pretrained_model is not None:
    
    modelG=torch.load(config.model_save_path+'/bestgen_{}'.format(config.pretrained_model))
    modelD=torch.load(config.model_save_path+'/bestdisc_{}'.format(config.pretrained_model))
    load_optim=True
    
else:
    
    modelG=RN.ResNet_G(config.latent_dim, config.g_output_dim, config.g_channels)
    if config.train_type=='wgan-hinge' : modelG.apply(RN.Add_Spectral_Norm)
    
    
    modelD=RN.ResNet_D(config.d_input_dim, config.d_channels)
    if config.train_type=='wgan-hinge': modelD.apply(RN.Add_Spectral_Norm)
    

if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
    
###############################################################################    
######################### LOADING models and Data #############################
###############################################################################


TRAINER=trainer.Trainer(config,device,criterion="W1_on_image_samples",\
                        test_metrics={0: 'Orography_RMSE'})
TRAINER.instantiate(modelG, modelD,load_optim)


###############################################################################
################################## TRAINING ###################################
##########################   (and online testing)  ############################
###############################################################################


if __name__=="main":
    TRAINER.fit_(modelG, modelD)

