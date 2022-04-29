#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:10:26 2022

@author: brochetc

network analysis tools

First simple code snippets and if need be, will be organized into separate modules

Owing F. Odom https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

"""
import torch
from typing import Dict,Iterable,Callable, Tuple
from torch import nn, Tensor

class FeatureExtractor(nn.Module):
    
    """
    
    class to extract features 
    of selected layers
    
    wraps the model into and forward-hookable space
    to be used at inference time mostly
    
    """
    
    def __init__(self, model: nn.Module, layers : Iterable[str]):
        super().__init__()
        self.model=model
        self.layers=layers
        
        self._features={layer :torch.empty(0) for layer in layers}
        
        def save_outputs_hook(layer_id :str) -> Callable :
            def fn(_, __, output):
                self._features[layer_id]=output
            return fn
        
        
        for layer_id in layers:
            layer=dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(save_outputs_hook(layer_id))
        
        
        
    def forward(self, x: Tensor) -> tuple([Dict[str, Tensor], Tensor]) :
        output=self.model(x)
        return self._features, output

class GradientExtractor(nn.Module):
    """
    
    class to extract gradients wrt module weights (so, output dimension)
    of selected layers
    
    wraps the model into and backward-hookable space
    to be used at inference time mostly
    
    """
    
    
    def __init__(self, model : nn.Module, loss: Callable, layers : Iterable[str], option: str):
        super().__init__()
        self.model=model
        self.layers=layers
        self._gradients={}
        self.loss=loss
        
        def save_gradients_hook(layer_id: str) -> Callable :
            def fn(model, grad_input, grad_output):
                self._gradients[layer_id]=model.weight.grad
            return fn
        
        for layer_id in layers:
            layer=dict([*self.model.named_modules()])[layer_id]
            layer.register_backward_hook(save_gradients_hook(layer_id))
            
              
                    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out=self.model(x)
        l=self.loss(out)
        l.backward()
        return self._gradients