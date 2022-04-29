#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:32:49 2021

@author: brochetc

following code reproduce Self-Attention module explained in Zhang et al 2018
 (Self-Attention generative Adversarial networks),
 following work of Wang et al., 2018 (Non-Local Neural networks)
 
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.functional import softmax
from time import perf_counter


############ Scale layer  deprecated as the wrap is not excalty useful

class ScaleLayer(nn.Module):
    """
    provide scale factor as learning parameter. Initialized to 1e-3.
    """
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale=nn.Parameter(torch.FloatTensor([init_value]))


    def forward(self, input):
        return input*self.scale

class SelfAttention(nn.Module):
    def __init__(self,input_channels=None,hidden_channels=None, scale_init=0.0):
        super().__init__()
        
        self.ChI=input_channels
        self.Hidd=hidden_channels
        
        self.Keys=nn.Conv2d(self.ChI, self.Hidd, kernel_size=1, bias=False)
        self.Queries=nn.Conv2d(self.ChI, self.Hidd, kernel_size=1, bias=False)
        self.Values=nn.Conv2d(self.Hidd, self.ChI, kernel_size=1, bias=False)
        self.Hidden=nn.Conv2d(self.ChI, self.Hidd, kernel_size=1, bias=False)
        
        self.Scale=nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        """
        Contracted dot product using Einstein's summation
        b symbol for batch dimension
        c symbol for channel dimension
        ij symbol for column space dimension
        kl symbol for row space dimension
        """
        
        features_loc=input.shape[2]*input.shape[3]
        
        Beta=nn.Softmax(dim=1)(torch.einsum('bci, bcj ->bij',
                       self.Keys(input).contiguous().view(-1,\
                                   self.Hidd, features_loc),
                       self.Queries(input).contiguous().view(-1,\
                                   self.Hidd, features_loc))
                       )
      
        H=self.Hidden(input).contiguous().view(-1, self.Hidd,features_loc)
        

       
        #E=torch.exp(torch.einsum('bcij, bckl -> bijkl',K,Q)) #contracted product
        #Beta=E/E.sum(dim=(1,2)).view([E.shape[0],E.shape[3], E.shape[4],1,1])
        
        output=self.Values(
                torch.einsum('bij, bcj -> bci', Beta, H).contiguous().view(-1,\
                            self.Hidd,input.shape[2], input.shape[3]))
       
        output=self.Scale*output+input
       
        return output

class ConvBlock2d(nn.Module):
    def __init__(self, ChI, ChO, kernel_size,stride, padding, activation, BN=True):
        super(ConvBlock2d, self).__init__()
        self.ChI=ChI
        self.ChO=ChO
        if activation is not None :
            self.modList=nn.ModuleList(
                [nn.Conv2d(ChI, ChO, kernel_size=kernel_size, padding=padding,\
                           stride=stride,bias=False),
                
                activation]
                )
        else :
            self.modList=nn.ModuleList(
                [nn.Conv2d(ChI, ChO, kernel_size=kernel_size, padding=padding,\
                           stride=stride,bias=False)]
                )
        if BN : self.modList.insert(index=1, module=nn.BatchNorm2d(ChO))
    def forward(self, x):
        for module in self.modList :
            x=module(x)
        return x

class ConvTransposeBlock2d(nn.Module):
    def __init__(self, ChI, ChO, kernel_size, stride, padding, activation, BN=True):
        super(ConvTransposeBlock2d, self).__init__()
        self.ChI=ChI
        self.ChO=ChO
        self.modList=nn.ModuleList(
                [nn.ConvTranspose2d(ChI, ChO, kernel_size=kernel_size,\
                                    padding=padding, stride=stride,bias=False),
                
                activation]
                )
        if BN : self.modList.insert(index=1, module=nn.BatchNorm2d(ChO))
    
    def forward(self,x):
        for module in self.modList :
            x=module(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=64,nc=1,SA_depth=0):
        super(Discriminator,self).__init__()
        self.SA_depth=SA_depth
        self.ndf=ndf  #base number of features for discriminator
        self.Mod_List=nn.ModuleList(
                [# input is nc x28x28
                ConvBlock2d(nc, ndf,4,2,3,nn.LeakyReLU(0.2, inplace=True), BN=False),
                #state size (ndf) 16 x16
                ConvBlock2d(ndf, ndf*2,4,2,1,nn.LeakyReLU(0.2, inplace=True)),
                
                #state size (ndf*2) x 8 x8
                ConvBlock2d(ndf*2, ndf*4,4,2,1, nn.LeakyReLU(0.2, inplace=True)),
              
                #state size ndf*4 x4x4
                ConvBlock2d(ndf*4, 1,4,1,0,None, BN=False)
                ])
        if SA_depth<len(self.Mod_List):
            SA_channels=self.Mod_List[SA_depth].ChI
        else : SA_channels=self.Mod_List[-1].ChO
        hidd=SA_channels
        self.Mod_List.insert(SA_depth, SelfAttention(input_channels=SA_channels,\
                                                     hidden_channels=hidd))
     
    
    def forward(self,x):
        x=x.view(-1,1,28,28)
        for i,module in enumerate(self.Mod_List) :
            x=module(x)
        return x
    
    
    
class Generator(nn.Module):
    def __init__(self, latent_dim=32, gen_feat=28, gen_chan=1, SA_depth=0):
        super(Generator, self).__init__()
        self.nz=latent_dim
        self.ngf=gen_feat #number of features used as base for generator
        self.nc=gen_chan #number of channels
        self.SA_depth=SA_depth
        self.Mod_List=nn.ModuleList([
                #z goes to transpose convolution
                ConvTransposeBlock2d(self.nz, self.ngf*8,4,1,0, nn.ReLU(True)),
        
                #state size is now (ngf*8) x 4 x4
                ConvTransposeBlock2d(self.ngf*8, self.ngf*4,4,2,1, nn.ReLU(True)),

                #state size is now (ngf*4)x8x8
                ConvTransposeBlock2d(self.ngf*4,self.ngf*2,4,2,1, nn.ReLU(True)),

        
                #state size is now (ngf*2) x 16x16
                ConvTransposeBlock2d(self.ngf*2, self.nc,4,2,3,nn.Tanh(), BN=False)
        
                #state size is now nc x 28 x 28
                ])
        if SA_depth<len(self.Mod_List):
            SA_channels=self.Mod_List[SA_depth].ChI
        else : SA_channels=self.Mod_List[-1].ChO
        hidd=SA_channels
        self.Mod_List.insert(SA_depth, SelfAttention(input_channels=SA_channels,\
                                                     hidden_channels=hidd))
    
    def forward(self,z):
        z=z.view(-1,self.nz,1,1)
        for i,module in enumerate(self.Mod_List) :
            z=module(z)
        return z

#custom_weight initialisation called on model_G and model_D
        
def weights_init(m):
    classname=m.__class__.__name__
    
    if classname.find('Conv2d')!=-1 or classname.find('ConvTranspose2d')!=-1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') !=-1 :
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
        
# adding spectral normalisation on a model
        
def Add_spectral_Norm(m):
    """
    add spectral_norm hook to each conv/transpose conv/linear layer.
    will raise error if called two times on same Module
    """
    classname=m.__class__.__name__
    if classname.find('Conv2d')!=-1 or classname.find('ConvTranspose2d')!=-1\
                                            or classname.find('Linear')!=-1:

        spectral_norm(m)
        
def weights_init_orthogonal(m):
    classname=m.__class__.__name__
    
    if classname.find('Conv2d')!=-1 or classname.find('ConTranspose2d')!=-1:
        torch.init.orthogonal_(m.weigth)
    elif classname.find('BatchNorm') !=-1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0.0)