#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:23:45 2022

@author: brochetc
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np

N_repeat=[256,512,1024,2048]#,4096,8192]
metric="SWD_metric"
listmeans=[]
liststd=[]

for ns in N_repeat:
    res=np.array(pickle.load(open(metric+'_1024_'+str(ns)+'.p', 'rb'))[metric])
    listmeans.append(res[-2,0,4])
    liststd.append(res[-1,0,4])
    
plt.plot(N_repeat, listmeans, 'bo')
plt.xlabel('N of repeats')
plt.ylabel('SWD metric (1024 samples)')
plt.grid()
plt.savefig(metric+'_mean_plot.png')
plt.close()

plt.plot(N_repeat, liststd, 'bo')
plt.xlabel('N of repeats')
plt.ylabel('SWD metric (1024 samples)')
plt.grid()
plt.savefig(metric+'_std_plot.png')
plt.close()
    
"""
metric="spectral"

var_names=['u', 'v', 't2m']
NSamples=[1024,2048,4096,8192]
#fig=plt.figure(figsize=(20,20))
#st=fig.suptitle("Power Spectral Density", fontsize="x-large")
for i,ns in enumerate(NSamples) :
    for j, var in enumerate(var_names):
        fig=plt.figure(figsize=(20,20))
        res=pickle.load(open(metric+'_'+str(ns)+'.p', 'rb'))[metric]
    
        UniRad=res[0][0]
        BS=np.array([res[0][1] for m in range(len(res))])
        print(BS.shape)
        plt.plot(UniRad, BS.mean(axis=0)[j,:])
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Wavenumber (normalized)')
        plt.ylabel('Power Spectral Density')
        fig.tight_layout()
        #st.set_y(0.98)
        fig.subplots_adjust(top=0.95)
        plt.savefig(metric+'_psd'+str(ns)+'_plot_'+var+'.png')
        plt.close()"""
#NSamples=[16384,32768,65536]     
"""fig2=plt.figure(figsize=(20,20))
st=fig2.suptitle("Power Spectral Density", fontsize="x-large")
for i,ns in enumerate(NSamples) :
    ax=fig2.add_subplot(2,2,i+1)
    for j, var in enumerate(var_names):
        
        res=pickle.load(open(metric+'_fake_'+str(ns)+'_1'+'.p', 'rb'))[metric]
    
        UniRad=res[0][0]
        BS=res[0][1]
        print(BS.shape)
        ax.plot(UniRad, BS[j,:], label=var+' GAN')
        
        res=pickle.load(open(metric+'_'+str(ns)+'.p', 'rb'))[metric]
        UniRad=res[0][0]
        BS=res[0][1]
        print(BS.shape)
        ax.plot(UniRad, BS[j,:], label=var+' PEARO')

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Wavenumber (normalized) '+str(ns))
    plt.ylabel('Power Spectral Density')
fig2.tight_layout()
st.set_y(0.98)
fig2.subplots_adjust(top=0.95)
plt.savefig(metric+'_psd_large_global_plot_fake.png')
plt.close()

fig3=plt.figure(figsize=(30,10))
st=fig3.suptitle("Power Spectral Density", fontsize="x-large")
NSamples=[66048]#[16384,32768,65536]
for j, var in enumerate(var_names):
    ax=fig3.add_subplot(1,3,j+1)
    for i,ns in enumerate(NSamples) :
        res=pickle.load(open(metric+'_fake_'+str(ns)+'_1'+'.p', 'rb'))[metric]

        UniRad=res[0][0]
        BS=res[0][1]
        print(BS.shape)
        ax.plot(UniRad, BS[j,:], label='GAN '+str(ns))
        res=pickle.load(open(metric+'_'+str(ns)+'.p', 'rb'))[metric]
        UniRad=res[0][0]
        BS=res[0][1]
        print(BS.shape)
        ax.plot(UniRad, BS[j,:], label='PEARO '+str(ns))

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Wavenumber (normalized) 0.5=maximum')
    plt.ylabel(str(var), fontsize='large')
fig3.tight_layout()
st.set_y(0.98)
fig3.subplots_adjust(top=0.95)
plt.savefig(metric+'_psd_large_per_var_both.png')
plt.close()


fig4=plt.figure(figsize=(30,10))
st=fig4.suptitle("Relative Spectral Error (AE)", fontsize="x-large")
NSamples=[66048]#[16384,32768,65536]
for j, var in enumerate(var_names):
    ax=fig4.add_subplot(1,3,j+1)
    for i,ns in enumerate(NSamples) :
        res=pickle.load(open(metric+'_fake_'+str(ns)+'_1'+'.p', 'rb'))[metric]

        UniRad=res[0][0]
        BS_0=res[0][1]
        #print(BS.shape)
        #ax.plot(UniRad, BS[j,:], label='GAN '+str(ns))
        res=pickle.load(open(metric+'_'+str(ns)+'.p', 'rb'))[metric]
        UniRad=res[0][0]
        BS_1=res[0][1]
        #"print(BS.shape)"
        diff=np.abs((BS_0[j,:]-BS_1[j,:]))/BS_0[j,:]
        ax.plot(UniRad, diff) #, label='Spectral_RMSE '+str(ns))

    #plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Wavenumber (normalized) 0.5=maximum')
    plt.ylabel(str(var), fontsize='large')
fig4.tight_layout()
st.set_y(0.98)
fig4.subplots_adjust(top=0.95)
plt.savefig(metric+'_psd_large_per_var_diff.png')
plt.close()"""
