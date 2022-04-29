#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:09:13 2021

@author: brochetc

Following code implement version of look ahead optimizer as described in 
Chavdarova et al, 2021 : Taming GANs with look ahead minimax; 
please cite this work when using this code.
"""

from collections import defaultdict
import torch
import copy
import torch.optim as optim

class LookAheadOpt(optim.Optimizer):
    """
    class wrapping the optimizer into a look-ahead optimizer
    that applies look_ahead steps to the optimizer
    """
    def __init__(self, optimizer, alpha=0.3, k=100):
        self.optimizer=optimizer
        self.alpha=alpha
        self.k=k
        self.param_groups=self.optimizer.param_groups
        self.state=defaultdict(dict)
        
    def look_ahead_step(self):
        for group in self.param_groups:
            for fast in group["params"]:
                param_state=self.state[fast]
                if "slow_params" not in param_state:  #first step, slow W <- fast W
                    param_state["slow_params"]=torch.zeros_like(fast.data)
                    param_state["slow_params"].copy_(fast.data)
                slow=param_state["slow_params"]
                #updating slow params with look ahead step
                slow+=(fast.data-slow)*self.alpha
                fast.data.copy_(slow) #slow weights as new starting point for fast weights
    
    def step(self, closure=None): #wrapping inherited .step() method
        loss=self.optimizer.step(closure)
        return loss

def update_ema_gen(G, G_ema, beta_ema=0.9999):
    """
    perform exponential moving average (EMA) step with parameter beta_ema
    Note that this is good when the k_value for slow weights is not too large
    Prefer less conservative values such as 0.8 when k ~ 10³
    """
    l_param=list(G.parameters())
    l_ema_param=list(G_ema.parameters())
    for i in range(len(l_param)):
        with torch.no_grad():
            l_ema_param[i].data.copy_(l_ema_param[i].data.mul(beta_ema)) \
            .add(l_param[i].data.mul(1.-beta_ema))
