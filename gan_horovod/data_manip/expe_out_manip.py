#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 08:13:33 2022

@author: brochetc

experiment data manipulation

"""
import os
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
output_dir="./data_manip/"
data_dir="/home/mrmn/brochetc/scratch_link/GAN_2D/Set_10"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

BaseDir=os.getcwd()

ybounds={'criterion': [3.5e1, 5e2], 'intra_u': [1e-6,1e0],'intra_v': [1e-6,1e0],\
        'intra_t2m': [1e-6,1e0],'inter_u': [1e-6,1e0],'inter_v': [1e-6,1e0],\
        'inter_t2m': [1e-6,1e0], 'loss_D':[1e-3,3e0], 'loss_G':[1e-4, 5e3]}


xbounds={'criterion': [0, 250], 'intra_u': [0,250],'intra_v': [0,250],\
        'intra_t2m': [0,250],'inter_u': [0,250],'inter_v': [0,250],\
        'inter_t2m': [0,250], 'loss_D':[0,250], 'loss_G':[0,250]}


def ReadMe_parse(filename):
    args_dict={}
    with open(filename, 'r') as f :
        a=f.readlines()[1:]
        for st in a:
            li=st.split("\t:\t")
            args_dict[li[0]]=li[1]
    return args_dict

def make_DataFrame(data_dir):
    DirNames=glob(data_dir+'resnet*')
    
    MainDF=pd.DataFrame(columns=['version','algo', 'BatchSize', 'latent_dim', 'lrD', 'lrG','directory', 'found_nans'])
    
    dico={'found_nans':False}
    for Dirname in DirNames :
        
        dirname=Dirname[len(data_dir):]
        ic=dirname.find('128_')+4
        dico['version']=dirname[:ic-1]
        ia=dirname.find('wgan-hinge')+10
        if ia==-1:
            ia=dirname.find('wgan-gp')+7
            dico['algo']=dirname[ia-7:ia]
        else:
            dico['algo']=dirname[ia-10:ia]
        
        dico['latent_dim']=int(dirname[ia+1:ia+3])
        for i in range(1,len(dirname)):
            try:
                dico['BatchSize']=int(dirname[ia+4:ia+4+i])
            except ValueError:
                break
        print(dico['BatchSize'])
        os.chdir(Dirname)
        Instnames=os.listdir()
        for instance in Instnames:
            inst_num=int(instance[instance.find('_')+1:])
            num=len(os.listdir(instance+'/samples/'))
            dico['directory']=Dirname+'/'+instance+'/log/'
            if num >1 :
                with open(instance+'/ReadMe_'+str(inst_num)+'.txt', 'r') as f:
                    a=f.readlines()
                    dico['lrG']=float(a[18][a[18].find(':\t')+2:])
                    dico['lrD']=float(a[19][a[19].find(':\t')+2:])
                    dico['n_dis']=float(a[15][a[15].find(':\t')+2:])
            MainDF=MainDF.append(dico, ignore_index=True)
            find_nans(MainDF, dico['directory'])
    return MainDF


def find_nans(MainDF,directory):
    df=pd.read_csv(directory+'metrics.csv')
    if df.isnull().values.any():
        MainDF.loc[MainDF.directory==directory, 'found_nans']=True

def transform_ax(value,total_length,Series):
    res=Series*(4*value/total_length)
    return res

def plot_(MainDF, param_screen, metric,\
          log_scale=True, \
          output_dir=BaseDir+output_dir[1:]):
    """
    make plots of metric against param_screen for every values of other parameters
    """
    DF_plot=MainDF.loc[MainDF.found_nans==False]
    otherparams=list(DF_plot.columns.difference(DF_plot.columns[DF_plot.columns.isin([param_screen,'directory'])]))
    Groups=DF_plot.groupby(otherparams).groups
    for key in Groups.keys():
        index=Groups[key]
        floatParams=[str(f) for f in key if isinstance(f, float)]
        print(floatParams)
        
        StepList=[]
        metrList=[]
        labels=[]
        values=[]
        for z,i in enumerate(index) :
            
            value=DF_plot[param_screen][i]
            values.append(value)
            directory=DF_plot['directory'][i]
            
            data_df=pd.read_csv(directory+'metrics.csv')
            #print('df head\n',data_df.head()
            scale=value*(DF_plot['n_dis'][i])
            stepSeries=transform_ax(scale,66048,data_df['Step'])
            StepList.append(stepSeries.values)
            metrList.append(data_df[metric].values)
            labels.append(param_screen+' '+str(value))

        for v,s, m,l in sorted(zip(values,StepList, metrList, labels), key =lambda pair :pair[0]):
            plt.plot(s,m, label=l)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epoch Num')
        plt.ylabel(metric)
        plt.ylim(ybounds[metric])
        plt.xlim(xbounds[metric])
        plt.legend()
        plt.title(metric+' against '+param_screen+'_'+'_'.join(floatParams))
        plt.savefig(output_dir+metric+'_'+param_screen+'_'+'_'.join(floatParams)+'.png')
        plt.cla()
        plt.close()
    
MainDF=make_DataFrame(data_dir)
print(MainDF.columns)
plot_(MainDF, 'BatchSize', 'criterion')
plot_(MainDF, 'BatchSize', 'intra_u')
plot_(MainDF, 'BatchSize', 'intra_v')
plot_(MainDF, 'BatchSize', 'intra_t2m')
plot_(MainDF, 'BatchSize', 'inter_u')
plot_(MainDF, 'BatchSize', 'inter_v')
plot_(MainDF, 'BatchSize', 'inter_t2m')
plot_(MainDF, 'BatchSize', 'loss_D')
plot_(MainDF, 'BatchSize', 'loss_G')