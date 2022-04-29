#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:10:26 2022

@author: brochetc

network analysis tools

First simple code snippets and if need be, will be organized into separate modules

"""




activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name]=output.detach()
    return hook