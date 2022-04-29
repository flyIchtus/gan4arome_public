#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:59:36 2022

@author: brochetc

GAN logical steps for different algorithms
"""
import torch
import torch.autograd as autograd
from torch.nn.functional import relu



###############################################################################
##################### Adversarial components training steps ###################
###############################################################################


def Discrim_Step(real,modelD, modelG, optim_D, ndis, device):
    for i_dis in range(ndis):
            out_real=modelD(real)

            z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
            fake=modelG(z)
            out_fake=modelD(fake)
    
            loss=-(0.00000001+out_real).log().mean()-(1.0000001-(out_fake)).log().mean()
            optim_D.zero_grad()
            loss.backward()
            optim_D.step()
    return loss

def Generator_Step(real, modelD, modelG, optim_G, device):
    z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
    fake=modelG(z)
    out_fake=modelD(fake)
    #loss=(out_fake.mean())
    loss=(1.0000001-(out_fake).log()).mean()
    optim_G.zero_grad()
    loss.backward()
    optim_G.step()
    return loss


#############  Wasserstein step wrapper to abstract lamdaGP ###################

class Discrim_Wasserstein():
    def __init__(self, lamda):
        self.lamda=lamda

    def Discrim_Step(self,real, modelD, modelG, optim_D, n_dis, device):
        """
        perform wasserstein discriminator step
            if lipschitz constraint is enforced elsewhere -> lamda=0.0
                    time is saved not computing gradient penalty
            if lamda>0.0 : gradient penalty term is added in forward pass 
                    (and differentiated through)
        """
        for i_dis in range(n_dis):
                    
            out_real=modelD(real)
    
            z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
            
            fake=modelG(z)
                    
            out_fake=modelD(fake)
        
            ######### interpolate and compute grad for penalisation ########
            
            if self.lamda >0.0:
                ts=torch.empty(real.size(0),1).uniform_().to(device)
                interp=(ts*fake+(1.-ts)*real).requires_grad_()
    
                out_interp=modelD(interp)
                grad_out=torch.empty(real.size(0),1).fill_(1.0).to(device)
                grad_=autograd.grad(out_interp, interp, grad_out,\
                                    create_graph=True, retain_graph=True)
                grad_pen=(grad_[0].norm()-1.)*(grad_[0].norm()-1.)
                
                ############# loss computation ########################
    
                loss=-(out_real).mean()+(out_fake).mean()+self.lamda*grad_pen
                
            else :
                ############# loss computation ########################
                
                loss=-(out_real).mean()+(out_fake).mean()
                    
            optim_D.zero_grad()
            loss.backward()
            optim_D.step()
        return loss

###############################################################################

def Generator_Step_Wasserstein(real, modelD, modelG, optim_G, device):
    """
    perform wasserstein generator step
    """
    z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
    fake=modelG(z)
    out_fake=modelD(fake)
    loss=-(out_fake).mean()
    optim_G.zero_grad()
    loss.backward()
    optim_G.step()
    
    return loss

def Discrim_Step_Hinge(real, modelD, modelG, optim_D, n_dis, device):
    """
    perform hinge loss (Wasserstein) discriminator step
    """
    for i_dis in range(n_dis):
        out_real=modelD(real)
        z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
        fake=modelG(z)
        out_fake=modelD(fake)
        loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean()        
        optim_D.zero_grad()
        loss.backward()
        optim_D.step()
    return loss



