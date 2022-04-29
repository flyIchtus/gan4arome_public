#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:29:08 2022

@author: brochetc

Extract points from distribution

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import pickle

data_dir_real='/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_fake='/home/mrmn/brochetc/scratch_link/GAN_2D/Set_13/resnet_128_wgan-hinge_64_64_1_0.001_0.001/Instance_1/samples/Best_model_dataset/'
output_dir=data_dir_fake
Positions=[(56,64), (127,127),(44,65)]

Means=np.load(data_dir_real+'mean_with_orog.npy')[1:4].reshape(3)
Maxs=np.load(data_dir_real+'max_with_orog.npy')[1:4].reshape(3)

def normalize(BigMat, Mean, Max):
    res= (0.95)*(BigMat-Mean)/(Max)
    #print(res.max(), res.min())
    return  res

def extract_distrib(dico,pos_list):
    list_files_real=glob(data_dir_real+'_sample*.npy')
    list_files_fake=glob(data_dir_fake+'_Fsample*.npy')
    
    assert len(list_files_real)==len(list_files_fake)
    for pos in pos_list:
        dico[pos]=[[],[]]
    for i in range(len(list_files_real)):
        
        if i%1000==0: print(i)
        
        sample_r=np.load(list_files_real[i])
        
        sample_f=np.load(list_files_fake[i])
        for pos in pos_list:
            sr=sample_r[1:4,78+pos[0],55+pos[1]]
            sr=normalize(sr,Means,Maxs)
            sf=sample_f[:,pos[0],pos[1]]
            dico[pos][0].append(sr)
            dico[pos][1].append(sf)
    


def plot_distrib_simple(data,legend_list, var_names,title, option):
    """
    plot the distribution of data -- one distribution for each value of the last axis
    """
    fig=plt.figure(figsize=(6,8))
    st=fig.suptitle(title+" "+option, fontsize="x-large")
    data_r=data[0]
    print(data_r.shape)
    data_f=data[1]
    print(data_f.shape)
    N_var=len(var_names)
    columns=1
    for i in range(N_var):
        ax=plt.subplot(N_var, columns, i+1)
        
        distrib_r=data_r[:,i]
        distrib_f=data_f[:,i]
        ax.hist(distrib_r, bins=200, density=True, label=legend_list[0])
        ax.hist(distrib_f, bins=200, density=True, label=legend_list[1])
        
        ax.set_ylabel(var_names[i])
        ax.legend()
        
    fig.tight_layout()
    st.set_y(0.95)
    
    fig.subplots_adjust(top=0.9)
    plt.savefig(title+"_"+option+".png")
    plt.show()
    plt.close()

    return 0

if __name__=="__main__":
    dico={}
    legend_list=['PEARO', 'GAN']
    extract_distrib(dico, Positions)
    pickle.dump(dico,open('pointwise_extractions.p', 'wb'))
    for pos in Positions:
        data=(np.array(dico[pos][0]), np.array(dico[pos][1]))
        plot_distrib_simple(data,legend_list, ['u', 'v', 't2m'], 'pixel_'+str(pos)+'_W1','clim')
    
    
