#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:07:31 2022

@author: brochetc

criblage des paramètres pour IS

"""

import numpy as np
import importance_sampling as IS
import IS_eval as ev
import os

N_samples=[1]

Q_mins=[0.1]
P_s=[0.5,0.2,0.8]
M_s=[0.1,0.5]

Srr=[1.0,3.0,5.0]
Sw=[5.0,10.0]


base_path='/home/mrmn/brochetc/scratch_link/'

SBpath=base_path+'Sud_Est/'

############################    Performing Baseline Tests     #################
print('######################### BASELINES ##################################')

os.chdir(SBpath)
if not os.path.exists('Baselines'):
    os.mkdir('Baselines')

prefix=SBpath+'Baselines/'

for ns in N_samples :
    print(ns)
    Sm=IS.Sampling_method(ns, 256, [1.0,0,0,0,0,0], IS.SE_indexes, prefix)
    if not os.path.exists(Sm.save_dir+'_done'):    
        IS.process_all(base_path, Sm)
        os.chdir(Sm.save_dir)
        mS=ev.Mean_Sample(256)
        np.save('MEAN_SAMPLE_AFTER_IS.npy', mS)
        vS=ev.Variance_Sample(256, mS)
        np.save('VAR_SAMPLE_AFTER_IS.npy', vS)
        os.chdir(prefix)
        os.rename(Sm.save_dir, Sm.save_dir[:-1]+'_done')
"""
###############################################################################   
########################### Performing parameter screening ####################
os.chdir(base_path)
if not os.path.exists('Screening'):
    os.mkdir('Screening')
    
prefix=base_path+'Screening/'

for ins,ns in enumerate(N_samples):
    for iqm, qmin in enumerate(Q_mins):
        for ip,p in enumerate(P_s):
            for im,m in enumerate(M_s):
                for isr,s_rr in enumerate(Srr):
                    for isw, sw in enumerate(Sw):
                        params=[qmin, m, p,1-p, s_rr, sw]
                        print(params)
                        Sm=IS.Sampling_method(ns, 128, params, IS.SE_indexes, prefix)
                        if not os.path.exists(Sm.save_dir+'_done'):    
                            IS.process_all(base_path, Sm)
                            os.chdir(Sm.save_dir)
                            mS=ev.Mean_Sample(128)
                            np.save('MEAN_SAMPLE_AFTER_IS.npy', mS)
                            vS=ev.Variance_Sample(128, mS)
                            np.save('VAR_SAMPLE_AFTER_IS.npy', vS)
                            os.chdir(prefix)
                            os.rename(Sm.save_dir, Sm.save_dir+'_done')"""
                        
