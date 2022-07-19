#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:15:34 2022

@author: brochetc

executable file for scattering coefficients

"""

import scattering_funcs as waSc
import scattering_plot as scPl
#import torch
from time import perf_counter


DATA_DIR='/home/brochetc/Bureau/Thèse/présentations_thèse/Première_année/images_des_entrainements/echantillons/'
DATA_DIR_F='/home/brochetc/Bureau/Thèse/présentations_thèse/Première_année/images_des_entrainements/'
output_dir='/home/brochetc/Bureau/Thèse/présentations_thèse/Première_année/ressources_présentations/explaining_scattering/'
CI=[78,205,55,182]

J,L=4,8

scattering_real=waSc.scatteringHandler(J,(127,127),backend='numpy',\
                                           frontend='numpy',L=L)
scattering_fake=waSc.scatteringHandler(J,(127,127),backend='numpy',\
                                           frontend='numpy',L=L)

N_tests=1
dic_fake={'S1_l' :[],'s21' :[],'s22': []}
dic_real={'S1_l' :[],'s21' :[],'s22': []}

for n in range(N_tests):
    print(n)
    M_real=waSc.load_batch(DATA_DIR,158,CI, option='real')
   
    #M_real=torch.tensor(M_real,dtype=torch.float32)

    M_fake=waSc.load_batch(DATA_DIR_F,158, CI, option='fake')
  
    #M_fake=torch.tensor(M_fake,dtype=torch.float32)

    scattering_real.reset()


    scattering_fake.reset()

    
    res_fake=scattering_fake(M_fake)

    s22_j1j2_fake=scattering_fake.shapeEstimator()
    dic_fake['s22'].append(s22_j1j2_fake)
    
    s21_j1j2_fake=scattering_fake.sparsityEstimator()
    dic_fake['s21'].append(s21_j1j2_fake)
    
    S1_l_fake=scattering_fake.S1_j1
    dic_fake['S1_l'].append(S1_l_fake)
   

   
    res_real=scattering_real(M_real)

    
    t2=perf_counter()
    s22_j1j2_real=scattering_real.shapeEstimator()
    dic_real['s22'].append(s22_j1j2_real)
    
    s21_j1j2_real=scattering_real.sparsityEstimator()
    dic_real['s21'].append(s21_j1j2_real)
    
    S1_l_real=scattering_real.S1_j1
    dic_real['S1_l'].append(S1_l_real)
    
#pickle.dump(dic_fake, open(output_dir+'fake_samples_scattering.p', 'wb'))
#pickle.dump(dic_fake, open(output_dir+'real_samples_scattering.p', 'wb'))
    
print("plotting")
scPl.plot_2o_Estimators(s22_j1j2_fake, s22_j1j2_real,"Shape", J,L,\
                        output_dir)
scPl.plot_2o_Estimators(s21_j1j2_fake, s21_j1j2_real,"Sparsity", J,L,\
                        output_dir)
scPl.plot_1o_Estimators(S1_l_fake,S1_l_real,"S1,l", J,L,\
                        output_dir)

