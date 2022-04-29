#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:36:50 2021

@author: brochetc
"""

import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.datasets as dsets
from torchvision import transforms
import argparse

def denorm(x):
    return nn.ReLU((x+1)/2)

def Generator(x):
    return 0

def Discriminator(x):
    return 0

def load_pretrained_model(self):
    self.G.load_state_dict(torch.load(os.path.join(
        self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
    self.D.load_state_dict(torch.load(os.path.join(
        self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
    print('loaded trained models (step: {})..!'.format(self.pretrained_model))


def save_sample(self, data_iter):
    real_images, _ = next(data_iter)
    save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
    

def build_model(self):

    self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
    self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
    if self.parallel:
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

    # Loss and optimizer
    # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

    self.c_loss = torch.nn.CrossEntropyLoss()
    # print networks
    print(self.G)
    print(self.D)


###### extract 'real' data to compute scores on (not to train on !) #########
# AFAP this is NOT to be loaded on GPU RAM
base_path=os.getcwd()
os.chdir(config.data_path)

df=pd.read_csv('IS_method_labels.csv')
sample_names=df.sample(n=config.test_samples)



def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='coco-gan', choices=['resnet', 'coco-gan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['vanilla','wgan-gp', 'wgan-hinge'])
    parser.add_argument('--patch_size', type=int, default=128)
    
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--g_channels', type=int, default=5)
    parser.add_argument('--d_channels', type=int, default=5)
    parser.add_argument('--g_output_dim', type=int, default=128)
    parser.add_argument('--d_input_dim', type=int, default=128)
    parser.add_argument('--lamda_gp', type=float, default=10.0)
    parser.add_argument('--version', type=str, default='coco-gan_1')

    # Training setting
    parser.add_argument('--epochs_num', type=int, default=100, help='how many times to go through dataset')
    parser.add_argument('--total_step', type=int, default=0,  help='how many times to update the generator')
    parser.add_argument('--n_dis', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0004)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    
    #Training setting -schedulers
    parser.add_argument('--lrD_sched', type=str, default=None, choices=[None,'exp', 'linear'])
    parser.add_argument('--lrG_sched', type=str, default=None, choices=[None,'exp', 'linear'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
            
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=-1) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=100) # set to 0 if not needed
    parser.add_argument('--model_save_step', type=int, default=1) # set to 0 if not needed


    return parser.parse_args()





def ZeroGrad(model):   
    """
    this should be more efficient the optimizer.zero_grad()
    to be used ?
    """
    
    for param in model.parameters():
        param.grad=None

def get_memory_footprint(model):
    """
    copied from @ptrblck, NVIDIA
    output is memory footprint in bytes/octet
    """
    mem_params=sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_buff=sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params, mem_buff, mem_params+mem_buff
