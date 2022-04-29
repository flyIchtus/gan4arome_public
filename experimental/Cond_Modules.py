# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:05:26 2022

@author: Utilisateur

conditional modules for GANs
"""
import torch.nn as nn

class MLP3(nn.Module):
    """
    depth-3 MLP to embed input tensor
     modify to reaky relu ?
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.Model=nn.Sequential(
                nn.Linear(input_dim, input_dim*2),
                nn.Linear(input_dim*2, input_dim*4),
                nn.Linear(input_dim*4, output_dim)
                )
    def forward(self,x):
        o=self.Model(x)
        return o

class CondBatchNorm2d(nn.Module):
    """ conditional BN class in 2d"""
    def __init__(self,condition_dim, channels):
        super().__init__()
        self.channels=channels
        self.condition_dim=condition_dim
        self.Embedding=MLP3(condition_dim,2)
        
        self.BN=nn.BatchNorm2d(channels)
        for param in self.BN.parameters():
            param.requires_grad=False

    def forward(self, x, y):
        params=self.Embedding(y)
        inter=self.BN(x)
        o=params[:,0]*inter+params[:,1]
        return o

class CondBatchNorm1d(nn.Module):
    """ 
    
    conditional BN class in 1d
    
    """
    def __init__(self,condition_dim, channels):
        super().__init__()
        self.channels=channels
        self.condition_dim=condition_dim
        self.Embedding=MLP3(condition_dim,2)
        
        self.BN=nn.BatchNorm1d(channels)
        for param in self.BN.parameters():
            param.requires_grad=False
    def forward(self, x, y):
        gamma=self.EmbeddingGamma(y)
        beta=self.EmbeddingBeta(y)
        inter=self.BN(x)
        o=gamma*inter+beta
        return o
        
class Projector(nn.Module):
    def __init__(self, input_dim, inner_dim):
        super().__init__()
        self.input_dim=input_dim
        self.inner_dim=inner_dim
        self.Embedding=nn.Linear(input_dim,inner_dim)
    def forward(self,x,y):
        inter=self.Embedding(x)
        o=inter*y
        o=o.sum()
        return o

class AC_Head(nn.Module):
    """
    auxiliary classifier to guess coordinates
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim=input_dim
        self.Module=nn.Sequential(
        nn.BatchNorm1d(self.input_dim),
        nn.Linear(self.input_dim, self.input_dim//2),
        nn.BatchNorm1d(self.input_dim//2),
        nn.LeakyReLU(0.1),
        nn.Linear(self.input_dim//2,1),
        nn.Tanh())
        
    def forward(self,x):
        x=self.Module(x)
        return x
        

