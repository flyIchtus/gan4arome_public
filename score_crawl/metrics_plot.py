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
import argparse
from metrics4arome import multivariate as mlt
from mpl_toolkits.axes_grid1 import ImageGrid

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        print(li)
        li2=li[1:-1].split(',')
        print(li2)
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


def getAndmakeDirs():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-expe_set', type=int,help='Set of experiments to dig in.')
    parser.add_argument('-batch_sizes',type=str2list, help='Set of batch sizes experimented')
    parser.add_argument('-instance_num', type=str2list, help='Instances of experiment to dig in')
    
    config=parser.parse_args()
    
    
    names=[]
    short_names=[]
    list_steps=[]
    true_batches=[int(batch) for batch in config.batch_sizes]
    true_instances=[int(insta) for insta in config.instance_num]
    for batch in config.batch_sizes :
        print(batch)
        for instance in config.instance_num:
            names.append('/scratch/mrmn/brochetc/GAN_2D/Saved_Sets_21062022/Set_'+str(config.expe_set)\
                                +'/resnet_128_wgan-hinge_64_'+str(batch)+\
                                '_1_0.001_0.001/Instance_'+str(instance))
            short_names.append('Instance_{}_Batch_{}'.format(instance, batch))
            if int(batch)<=64:
                list_steps.append([1500*k for k in range(40)]+[59999])
            else:
                list_steps.append([1500*k for k in range(22)])
    data_dir_names, log_dir_names=[f+'/samples/' for f in names],[f+'/log/' for f in names]
    
        
    return data_dir_names, log_dir_names, short_names, list_steps, true_batches, true_instances

log_dir='/scratch/mrmn/brochetc/GAN_2D/Saved_Sets_21062022/Set_38/Metrics_log/pw_W1/'
real_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'


def plot_DistMetrics_Dynamics(list_steps,N_samples,shortNames, names,directories,prefix,output_dir, coolNames):
    """
    plot multi-metrics dynamics (with dynamics indexes in step list) of data present in directory
    
    """
    
    for j, directory in enumerate(directories):
        try :  
            results=pickle.load(open(directory+prefix+'_'+str(N_samples)+'.p','rb'))
            metrics_list=results["header"]
            step_list=list_steps[j]
            for metric in metrics_list:
                print(metric)
                Shape=results[metric].squeeze().shape
                print(Shape)
                
                
                fig=plt.figure(figsize=(6*Shape[1],15))
                
                for i in range(Shape[1]):
                    plt.plot(np.array(step_list)/1000, np.log10(results[metric].squeeze()[:,i]), linewidth=2, label=names[i])
                
                plt.xlabel("Iteration k-step", fontsize='x-large')
                #plt.xticks((np.array(step_list)+1)/1000)
                plt.ylabel("Log10 metric results", fontsize='x-large')
                plt.grid()
                plt.legend()
                plt.title(coolNames[metric], fontsize='x-large')
                plt.savefig(output_dir+"{}_{}_{}_dynamics_plot.png".format(shortNames[j],metric,names[i]))
                plt.close()
        except FileNotFoundError :
            print(directory, prefix, N_samples)
            print('File not found !')
        
def plot_StandAloneMetrics_Dynamics(step_list, N_samples, names, directories, output_dir):
    """
    plot multi-metrics dynamics (with dynamics indexes in step list) of data present in directory
    for standalone metrics
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
    
    return 0
    
    
def plot_Spectrum_dynamics(indexes, names,real_dir, directories, var_names, output_dir):
    
    resreal=pickle.load(open(real_dir+'real3stand_alone_metrics_66048.p', 'rb'))
    spectrum_real=resreal['spectral_compute'][0]
    
    for k,direct in enumerate(directories) :
        
        try :
        
            res=pickle.load(open(direct+'vquantiles_stand_alone_metrics_66048.p', 'rb'))
            
            nvars=len(var_names)
            
            spectrum=np.take(res['spectral_compute'], indexes, axis=0)
            n_steps=len(indexes)
            
            fig,axs=plt.subplots(1,3,figsize=(18,15))
            
            for j in range(nvars):
                if j==0:
                    axs[j].set_ylabel(' Radially-avged PSD (log10 scale)')
                for i in range(n_steps):
                    
                    if indexes[i]>=0:
                        lab='Step {}/60000'.format(indexes[i]*1500+1)
                    else :
                        lab='Step {}/60000'.format(60000)
                    axs[j].plot(np.log10(np.arange(1,46)), np.log10(spectrum[i,j,:,0]), label=lab)
                   
                    axs[j].grid()
                    axs[j].title.set_text(var_names[j])
                    axs[j].set_xlabel('Spatial Wavenumber')
            
                axs[j].plot(np.log10(np.arange(1,46)), np.log10(spectrum_real[j,:,0]), 'b-', linewidth=2.0, label='Mean PEARO Spectrum')
                axs[j].plot(np.log10(np.arange(1,46)), np.log10(spectrum_real[j,:,1]), 'b--', linewidth=2.0,label='Q10-90 PEARO')
                axs[j].plot(np.log10(np.arange(1,46)), np.log10(spectrum_real[j,:,2]), 'b--', linewidth=2.0, label='Q10-90 PEARO')
            
            st=fig.suptitle('Spectral Power Density dynamics', fontsize='23')
            st.set_y(0.98)
            plt.legend()
            fig.tight_layout()
            plt.savefig(output_dir+'Spectral_PSD_{}_dynamics.png'.format(names[k]))
            plt.close()     
        except (FileNotFoundError, IndexError):
            print(direct+' not found !')

def plot_multivariate_dynamics(list_steps, N_samples, names, directories, output_dir):
    """
    plot bivariate histograms on specific data directories. Create one plot per step in a step series.
    
    """
    
    for path, steps,name in zip(directories, list_steps, names) :
        
        try :
            res=pickle.load(open(path+'multivar0distance_metrics_16384.p', 'rb'))
            RES=res['multivar'].squeeze()
            for step in steps :
                
                data_r,data_f=RES[step//1500,0], RES[step//1500,1]
                print(data_r.shape, data_f.shape)
            
            
                levels=mlt.define_levels(data_r,5)
                ncouples2=data_f.shape[0]*(data_f.shape[0]-1)
                bins=np.linspace(tuple([-1 for i in range(ncouples2)]), tuple([1 for i in range(ncouples2)]),101, axis=1)
            
                var_r=(np.log(data_r), bins)
                var_f=(np.log(data_f), bins)
    
                add_name=name+'_'+str(step)
                mlt.plot2D_histo(var_f, var_r, levels, output_dir, add_name)
            
        except FileNotFoundError :
            print(path + "not found !")
            
def plot_metrics_map(list_steps, N_samples, names, var_names, directories, output_dir):
    """
    plot maps of metrics on specific data directories. Create one plot per step in a step series.
    
    
    metrics must be organized as S x 1 x C x H x W
    """
    channels=len(var_names)
    
    
    for path, steps,name in zip(directories, list_steps, names) :
        
        try :
            print(path+'pw_W1_distance_metrics_16384.p')
            res=pickle.load(open(path+'pw_W1_distance_metrics_16384.p', 'rb'))
            RES=res['pw_W1'].squeeze()
            for step in steps :
                
                data=RES[step//1500]
                print(data.shape)
                assert data.shape[0]==channels
                fig=plt.figure(1,(9.,6.))
                grid=ImageGrid(fig, 111,
                  nrows_ncols=(1,channels),
                  axes_pad=0.4,
                  cbar_pad=0.25,
                  cbar_location="right",
                  cbar_mode="each",
                  cbar_size="7%",
                  label_mode='L')
                for i in range(channels):
                    ax=grid[i].imshow(1000*data[i], origin='lower',extent=(-8,8,-8,8),
                           cmap='coolwarm', vmin=5.0, vmax=70.0)
                    grid.cbar_axes[i].colorbar(ax)
                fig.tight_layout()
                add_name=name+'_'+str(step)+'_'
                plt.savefig(output_dir+add_name+'pw_W1.png')
                plt.close()
            
        except FileNotFoundError :
            print(path + "not found !")
    
if __name__=="__main__":
    
    _,directories,short_names, list_steps, true_batches, true_instances=getAndmakeDirs()
    
    
    #indexes=(0,6,12,-1)
    #N_samples=16384
    #prefix='distance_metrics'
    #names=['u', 'v', 't2m']
    #metric_coolNames={'sparse_metric' : 'Scattering Sparsity Wasserstein Distance', 'shape_metric' : 'Scattering Shape Wasserstein Distance'}
    #plot_DistMetrics_Dynamics(list_steps,N_samples,short_names,names, directories,prefix,log_dir, metric_coolNames)
    #plot_Spectrum_dynamics(indexes, names, real_dir, directories, ['u', 'v', 't2m'], log_dir)
    
    N_samples=16384
    output_dir=log_dir
    var_names=['u', 'v','t2m']
    plot_metrics_map(list_steps, N_samples, short_names, var_names,directories, output_dir)
    
    
