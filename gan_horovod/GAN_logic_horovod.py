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
import horovod.torch as hvd


###############################################################################
##################### Adversarial components training steps ###################
###############################################################################


# major tip if working multi-gpu -> DO NOT use model.zero_grad() 
# but set param.grad=None

# optim_X.synchronize() after optim_Y.step() is mandatory if using multigpu
# automatic mixed precision (AMP) should be used if large models are trained

""" 
TODO : adapt without optimizer step for new versions
"""

def Discrim_Step(real,modelD, modelG, optim_D, optim_G,ndis, \
                 use_amp=False, scaler=None):
    for i_dis in range(ndis):
        z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
        for param in modelD.parameters():
                param.grad=None
                
        if use_amp:
            
            assert scaler is not None
            with torch.cuda.autocast():
                out_real=modelD(real)
                fake=modelG(z)
                out_fake=modelD(fake)
    
                loss=-(out_real).log().mean()-(1.0-(out_fake)).log().mean()
            
            scaler.scale(loss).backward()
            optim_D.synchronize()
            scaler.unscale_(optim_D)
            with optim_D.skip_synchronize():
                scaler(optim_D).step()
            scaler.update()
            optim_G.synchronize()
            
        else:
            out_real=modelD(real)

            
            fake=modelG(z)
            out_fake=modelD(fake)
    
            loss=-(out_real).log().mean()-(1.0-(out_fake)).log().mean()
            
            loss.backward()
            optim_D.step()
            optim_G.synchronize()
    return loss

def Generator_Step(real, modelD, modelG, optim_D, optim_G,\
                   use_amp=False, scaler=None):
    z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
    for param in modelG.parameters():
        param.grad=None
    if use_amp:
        assert scaler is not None
        with torch.cuda.autocast():
            fake=modelG(z)
            out_fake=modelD(fake)
            loss=(1.0-(out_fake).log()).mean()
    
        scaler.scale(loss).backward()
        optim_G.synchronize()
        scaler.unscale_(optim_G)
        with optim_G.skip_synchronize():
            scaler.step(optim_G)
        scaler.update()
        optim_D.synchronize()
        
    else:
        fake=modelG(z)
        out_fake=modelD(fake)
        #loss=(out_fake.mean())
        loss=(1.0-(out_fake).log()).mean()
    
        loss.backward()
        optim_G.step()
        optim_D.synchronize()
    return loss


#############  Wasserstein step wrapper to abstract lamdaGP ###################
""" 
TODO : adapt without optimizer step for new versions
"""
class Discrim_Wasserstein():
    def __init__(self, lamda):
        self.lamda=lamda

    def Discrim_Step(self,real, modelD, modelG, optim_D, optim_G,n_dis,\
                     use_amp=False, scaler=None):
        """
        perform wasserstein discriminator step
            if lipschitz constraint is enforced elsewhere -> lamda=0.0
                    time is saved not computing gradient penalty
            if lamda>0.0 : gradient penalty term is added in forward pass 
                    (and differentiated through)
        """
        for i_dis in range(n_dis):
            z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
            for param in modelD.parameters():
                param.grad=None
            if use_amp:
                """
                TODO : check for gradient penalty scaling
                """
                assert scaler is not None
                with torch.cuda.autocast():
                    out_real=modelD(real)
                    fake=modelG(z)
                    out_fake=modelD(fake)
                
                    ######### interpolate and compute grad for penalisation ########
                    
                    if self.lamda >0.0:
                        ts=torch.empty(real.size(0),1).uniform_().cuda()
                        interp=(ts*fake+(1.-ts)*real).requires_grad_()
            
                        out_interp=modelD(interp)
                        grad_out=torch.empty(real.size(0),1).fill_(1.0).cuda()
                        grad_=autograd.grad(out_interp, interp, grad_out,\
                                            create_graph=True, retain_graph=True)
                        
                        
                        grad_pen=(grad_[0].norm()-1.)*(grad_[0].norm()-1.)
                        
                        ############# loss computation ########################
            
                        loss=-(out_real).mean()+(out_fake).mean()+self.lamda*grad_pen
                        
                    else :
                        ############# loss computation ########################
                        
                        loss=-(out_real).mean()+(out_fake).mean()
                            
                scaler.scale(loss).backward()
                optim_D.synchronize()
                scaler.unscale_(optim_D)
                
                with optim_D.skip_synchronize():
                    scaler.step(optim_D)
                scaler.update()
                optim_G.synchronize()
                
                
            else:
                out_real=modelD(real)
                fake=modelG(z)
                out_fake=modelD(fake)
            
                ######### interpolate and compute grad for penalisation ########
                
                if self.lamda >0.0:
                    ts=torch.empty(real.size(0),1).uniform_().cuda()
                    interp=(ts*fake+(1.-ts)*real).requires_grad_()
        
                    out_interp=modelD(interp)
                    grad_out=torch.empty(real.size(0),1).fill_(1.0).cuda()
                    grad_=autograd.grad(out_interp, interp, grad_out,\
                                        create_graph=True, retain_graph=True)
                    grad_pen=(grad_[0].norm()-1.)*(grad_[0].norm()-1.)
                    
                    ############# loss computation ########################
        
                    loss=-(out_real).mean()+(out_fake).mean()+self.lamda*grad_pen
                    
                else :
                    ############# loss computation ########################
                    
                    loss=-(out_real).mean()+(out_fake).mean()
                        
                
                loss.backward()
                optim_D.step()
                optim_G.synchronize()
        return loss

###############################################################################

def Generator_Step_Wasserstein(real, modelD, modelG, optim_D,optim_G,
                               use_amp=False): # scaler=None):
    """
    perform wasserstein generator step
    """
    z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
    for param in modelG.parameters():
        param.grad=None

    if use_amp :
        with torch.cuda.amp.autocast():
            fake=modelG(z)
            out_fake=modelD(fake)
            loss=-(out_fake).mean()
        """for param in modelG.parameters():
           p= param.grad[0].norm() if param.grad is not None else -1
           print('Gen grad gen step ',p)
        for param in modelD.parameters():
           p= param.grad[0].norm() if param.grad is not None else -1
           print('Disc grad gen step ',p)"""
    
    else:
        fake=modelG(z)
        out_fake=modelD(fake)
        loss=-(out_fake).mean()

    loss.backward()
            
    return loss

def Discrim_Step_Hinge(real, modelD, modelG, optim_D,optim_G,\
                       use_amp=False): #scaler=None):
    """
    perform hinge loss (Wasserstein) discriminator step
    """

    z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
    for param in modelD.parameters():
        param.grad=None
    for param in modelG.parameters():
        param.grad=None
    if use_amp:

        with torch.cuda.amp.autocast():
            out_real=modelD(real)
            with torch.no_grad():
                fake=modelG(z)
            out_fake=modelD(fake)
            loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean() 
    else:
        out_real=modelD(real)
        fake=modelG(z)
        out_fake=modelD(fake)
        loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean()

    loss.backward()
                
    return loss

