#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:33:21 2022

@author: brochetc
"""

import torch

import horovod.torch as hvd
import trainer_horovod_v2 as trainer
import resnets_v2 as RN
import plotting_functions_v2 as plf
from exp_init import get_parameters

hvd.init()

torch.cuda.set_device(hvd.local_rank())

###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

config=get_parameters()

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
    if config.train_type=='wgan-hinge' and config.sn_on_g: 
        modelG.apply(RN.Add_Spectral_Norm)
    if config.ortho_init : modelG.apply(RN.Orthogonal_Init)
    
    
    modelD=RN.ResNet_D(config.d_input_dim, config.d_channels)
    if config.train_type=='wgan-hinge': modelD.apply(RN.Add_Spectral_Norm)
    if config.ortho_init : modelD.apply(RN.Orthogonal_Init)
 
   
###############################################################################    
######################### LOADING models and Data #############################
###############################################################################

TRAINER=trainer.Trainer(config,criterion="W1_Center",\
                        test_metrics=['IntraMapVariance', 'InterMapVariance'])
#'Orography_RMSE'])

TRAINER.instantiate(modelG, modelD,load_optim)

###############################################################################
################################## TRAINING ###################################
##########################   (and online testing)  ############################
###############################################################################

TRAINER.fit_(modelG, modelD)

###############################################################################
############################## POST-PROCESSING ################################
############################ (of training data) ###############################

plf.plot_metrics_from_csv(config.output_dir+'/log/', 'metrics.csv')
