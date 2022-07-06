#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:58:38 2022

@author: brochetc

executble file for structure functions 

"""


import structure_functions as sfunc
import structure_plot as splot
import numpy as np
import random
from glob import glob

DATA_DIR='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/echantillons/'
DATA_DIR_F='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/echantillons/'
#output_dir='/scratch/mrmn/brochetc/GAN_2D/Set_13/resnet_128_wgan-hinge_64_64_1_0.001_0.001/Instance_1/log/'
CI=[78,206,55,183]



def load_batch(path,number,CI,Shape=(3,128,128), option='fake'):
    
    if option=='fake':
        
        list_files=glob(path+'_Fsample_*.npy')

        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[:,:Shape[1],:Shape[2]].astype(np.float32)
            
    elif option=='real':
        
        list_files=glob(path+'_sample*')
        Shape=np.load(list_files[0])[1:4,CI[0]:CI[1], CI[2]:CI[3]].shape
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1], CI[2]:CI[3]].astype(np.float32)
            
            
        Means=np.load(path+'mean_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Maxs=np.load(path+'max_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Mat=(0.95)*(Mat-Means)/Maxs
        

    return Mat


N_tests=1

for n in range(N_tests):
    
    M_real=load_batch(DATA_DIR,64,CI, option='real')
   
    M_fake=load_batch(DATA_DIR_F,64, CI, option='fake')
  
    #### increments
    inc_real,_=sfunc.increments(M_real,16)
    inc_fake,_=sfunc.increments(M_fake,16)
    
    splot.plot_increments(inc_real.mean(axis=0),inc_fake.mean(axis=0),1.3, ['u','v','t2m'],add_name='test')
    
    #### structure functions
    p_list=[1,1.5,2,2.2,2.5]
    
    struct_real=sfunc.structure_function(inc_real,p_list)
    struct_fake=sfunc.structure_function(inc_fake,p_list)
    
    splot.plot_structure(struct_real, p_list,1.3, ['u','v','t2m'],add_name='test_PEARO')
    splot.plot_structure(struct_fake, p_list,1.3, ['u','v','t2m'],add_name='test_GAN')
    
    splot.plot_structure_compare(struct_real[:,:,:3], struct_fake[:,:,:3],p_list[:3], 1.3,\
                                 ['u','v','t2m'], add_name='test')
    
