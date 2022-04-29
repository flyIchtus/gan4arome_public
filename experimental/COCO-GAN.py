# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:34:10 2022

@author: Utilisateur

implementing COCO-GAN following Lin et al., 2019
"""
import torch.nn as nn
import residual_nets as RN
import torch.distributions.normal.Normal as Normal0


###########################  From Normal to  uniform



# strategy is to instantiate "regular" blocks and then to change to conditional
# BN layers where needed

