#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:51:56 2022

@author: brochetc


Metric plots

"""


import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_DistMetrics_Dynamics(step_list,N_samples,names,directories, output_dir):
    """
    plot multi-metrics dynamics (with dynamics indexes in step list) of data present in directory
    
    """
    
    dir0=directories[0]
    
    results=pickle.load(open(dir0+str(step_list[0])+'_'+str(N_samples)+'.p','rb'))
    metrics_list=results["header"]
   
    for metric in metrics_list:
        Shape=results[metric].shape
        fig,axs=plt.subplots(1,Shape[1],figsize=(6*Shape[1],15))
        for i in range(Shape[1]):
            axs[i].plot(np.array(step_list)/1000, results[metric][:,i])
            
            axs[i].set_xlabel("Iteration step")
            axs[i].set_xticks(np.array(step_list/1000+1))
            axs[i].grid()
            axs[i].title.set_text(names[i])
        plt.savefig(output_dir+"{}_{}_dynamics_plot.png".format(metric,N_samples))
        
def plot_StandAloneMetrics_Dynamics(step_list, N_samples):
    return 0
    
    
    
