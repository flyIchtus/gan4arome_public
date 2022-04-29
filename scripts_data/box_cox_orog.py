#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:57:50 2022

@author: brochetc
"""

import numpy as np
import scipy.stats as st
import os
from glob import glob
import matplotlib.pyplot as plt

data_dir="/home/mrmn/brochetc/scratch_link/GAN_2D_10/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done"

output_dir="/home/mrmn/brochetc/scratch_link/GAN_2D_10/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_boxcox"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.chdir(output_dir)
    os.mkdir('error_files')
os.chdir(data_dir)
li=glob('_sample*.npy')

for i, filename in enumerate(li):
    print(i/len(li))
    M=np.load(filename)
    S=M[-1:,:,:].shape
    if i==0:
        bc_ds,lamda=st.boxcox((M[-1:,:,:]+1.0).flatten())
    else :
        try:
            bc_ds=st.boxcox((M[-1:,:,:]+1.0).flatten(), lamda)
        except ValueError:
            print('Error !',filename)
            print(M[-1:,:,:].min())
            plt.imshow(M[-1:,:,:])
            plt.colorbar()
            plt.savefig(filename[:-3]+'.png')
            plt.close()
    N=np.concatenate((M[:-1,:,:],bc_ds.reshape(S)), axis=0)
    print(N.shape)
    np.save(os.path.join(output_dir,filename), N)
    
with open(os.path.join(output_dir,'Box_Cox_lamda_orog.txt'), 'w') as f:
    f.write(str(lamda))
    f.close()
