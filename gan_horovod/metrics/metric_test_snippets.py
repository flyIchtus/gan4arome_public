#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import numpy as np
import __init__ as metrics
import pickle
from glob import glob
import random
import torch
from multiprocessing import Pool


CI=(78,206,55,183)

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_f='/home/mrmn/brochetc/scratch_link/GAN_2D/Set_13/resnet_128_wgan-hinge_64_64_1_0.001_0.001/Instance_1/samples/Best_model_dataset/'

original_data_dir='/scratch/mrmn/brochetc/'
output_dir='scratch/mrmn/brochetc/GAN_2D/Set_13/resnet_128_wgan-hinge_64_64_1_0.001_0.001/Instance_1/log'

ID_file='IS_method_labels.csv'
#labels=pd.read_csv(data_dir+ID_file)



def split_dataset(file_list,N_parts):
    
    inds=[i*len(file_list)//N_parts for i in range(N_parts)]+[len(file_list)]
    #print(inds)
    to_split=file_list.copy()
    random.shuffle(to_split)
    
    return [to_split[inds[i]:inds[i+1]] for i in range(N_parts)]


def gather(file_list, LatSize,var_names,output_dir=None, output_id=None, save=False):
    """
    gather all the samples present in file_List into a single big matrix
    this matrix is returned
    
    Inputs :
        file_list : list of files to be sampled from
        output_dir : str of the directory to save datasets
        output_id : name of the dataset to be saved
        var_names : list of 'channel' variables present in the dataset
    
    BigMat is the N_samples x N_variables x LatSizexLatSize matrix
    """
    
    Nsamples=len(file_list)
    Nvar=len(var_names)
    BigMat=np.zeros((Nsamples, Nvar,LatSize, LatSize))
    for i,sample_path in enumerate(file_list) :
        sample=np.float32(np.load(sample_path))
        if sample.shape[1]>LatSize:
            sample=sample[1:4,CI[0]:CI[1],CI[2]:CI[3]]
    
        BigMat[i,:,:,:]=sample
    if save :
        np.save(output_dir+output_id+'.npy', BigMat)
    return BigMat

def normalize(BigMat, Mean, Max):
    res= (0.95)*(BigMat-Mean)/(Max-Mean)
    #print(res.max(), res.min())
    return  res


def build_datasets(data_dir, program, fake=False):
    """
    
    Inputs :
        data_dir : str, the directory to get the data from
        program : dict,the datasets to be constructed
                dict { dataset_id : (parts_number, n_samples)}
    """
    if fake:
        globList=glob(data_dir+'_Fsample*')
    else:
        globList=glob(data_dir+'_sample*')
    res={}
    print(len(globList))
    for key, value in program.items():
        if value[0]==2:
            print(value[1])
            fileList=random.sample(globList,2*value[1])
            if key==0%128 : print(len(fileList))
            res[key]=split_dataset(fileList,2)
            if key==0%128 : print(len(res[key]))
        elif value[0]==1:
            fileList=random.sample(globList,value[1])
            res[key]=fileList
    #print(len(res))
    return res

def eval_distance_metrics(data):
    """
    this function should test distance metrics for datasets=[(filelist1, filelist2), ...]
    in order to avoid memory overhead, datasets are created and destroyed dynamically
    
    Inputs : 
        metrics_list : a list of metrics names to be tested
        datasets : a dict of filenames
    """
    metric, dataset, index, cuda=data
    
    if index%32==0: 
        print(index)
    
    results=[]
        
    Metric=getattr(metrics, metric)

    
    #print('gathering part 1')
    part1=gather(dataset[0],128,['u', 'v','t2m'])
    #print('part1 gathered')
    
    #print('scaling')
    Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
    Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
    
    part1=normalize(part1,Means, Maxs)
    
    #print('gathering part2')
    part2=gather(dataset[1], 128,['u','v','t2m'])
    #print('part 2 gathered')
    part2=normalize(part2,Means, Maxs)
    
    if cuda :
        part1=torch.tensor(part1, dtype=torch.float32).cuda()
        part2=torch.tensor(part2, dtype=torch.float32).cuda()
    
    results.append(Metric(part1, part2))
    return np.array(results)

def global_dataset_eval(data):
    """
    variable-wise evaluation of metric on the DataSet (treated as a single numpy matrix)
    
    """
    metric, DataSet,ind=data
    var_names=['u','v','t2m']
    Metric=getattr(metrics,metric)
    #print('gathering')
    dataset=gather(DataSet,128,var_names)
    
    #print('scaling')
    #Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
    #Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
    
    #dataset=normalize(dataset,Means, Maxs)
    
    if ind%32==0:
        print(ind)
    results=[]
    
    for i,var in enumerate(var_names):
        results.append(Metric(dataset[:,i,:,:]))
        
    if type(results[-1])==tuple:
        return results[0][0], np.array([results[i][1] for i in range(len(var_names))])
    
    return np.array(results)

def compute_and_save(metric, N_samples,ddir=data_dir,repeats=1, distance=True, full_dataset=False, cuda=False):
    """
    
    compute and save statistical metrics results on  N_samples samples
    repeats the calculation 'repeats' times
    options :
        distance is when the metric in itself is a distance
        full_dataset is used when the full available dataset is used to compute
        distance againt the full generated dataset 
        in the 'distance' context
        
        if both options are set to false, a metric is computed on the selected 
        data directory ddir (eg spectrum)
    
    """
    assert hasattr(metrics, metric)

    
    for Ns in N_samples :
        print("treating Ns {}".format(Ns))
        print('Version to scale')
        
        
        if distance and full_dataset:
            assert Ns==66048
            program={i :(1,Ns) for i in range(repeats)}
            
            dataset_r=build_datasets(data_dir, program)[0]
            dataset_f=build_datasets(data_dir_f, program, fake=True)[0]
        
            results={} 
        
            print(metric)
            datalist=[(metric, [dataset_r, dataset_f], 0)]#, (metric, dataset_f, var_names,1)]
            results[metric]=eval_distance_metrics(datalist[0])
            
            #with Pool(num_proc) as p:
            #    results[metric]=p.map(global_dataset_eval,datalist) 

            #results[metric].append(np.mean(np.array(results[metric]), axis=0))
            #results[metric].append(np.std(np.array(results[metric][:-1]), axis=0))
            
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'.p', 'wb'))
            
        elif distance:
            
            program={i :(2,Ns) for i in range(repeats)}
            
            datasets=build_datasets(data_dir, program)
          
            results={} 

            print(metric)
            datalist=[(metric, dataset[1], dataset[0], cuda) for dataset in datasets.items()]
            
            if cuda :
                num_proc=1
                with Pool(num_proc) as p:
                    results[metric]=p.map(eval_distance_metrics, datalist)
            else:
                num_proc=8
                with Pool(num_proc) as p:
                    results[metric]=p.map(eval_distance_metrics,datalist) 

            results[metric].append(np.mean(np.array(results[metric]), axis=0))
            results[metric].append(np.std(np.array(results[metric][:-1]), axis=0))
            
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'.p', 'wb'))
        
        else :
            program={i :(1,Ns) for i in range(repeats)}
            
            datasets=build_datasets(ddir, program,fake=(ddir==data_dir_f))

            results={} 

            print(metric)
            datalist=[(metric, dataset[1], 1) for dataset in datasets.items()]

            num_proc=8
            with Pool(num_proc) as p:
                results[metric]=p.map(global_dataset_eval,datalist) 

            #results[metric].append(np.mean(np.array(results[metric]), axis=0))
            #results[metric].append(np.std(np.array(results[metric][:-1]), axis=0))
            if ddir==data_dir_f:
                name='fake'
            else:
                name='real'
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+name+'_'+str(Ns)+'_'+str(repeats)+'.p', 'wb'))
            
    return 0

if __name__=="__main__":
    
    N_samples=[1024]#,16384]
    repeats=[256,512,1024,2048]
    metric='fid'
    for rep in repeats:
        print('treating repeat {}'.format(rep))
        compute_and_save(metric, N_samples,ddir=data_dir,repeats=rep, distance=True, cuda=True)
        
    """metric="spectral"
    N_samples=[66048]
    compute_and_save(metric, N_samples,ddir=data_dir_f, repeats=1, distance=False,full_dataset=True)
    
    #num_proc=12
    metrics_list=["pw_W1"] #, "SWD_metric"]
    assert hasattr(metrics, metrics_list[0])
    var_names=["u", "v", "t2m"]
    for Ns in N_samples :
        print("treating Ns {}".format(Ns))
        print('Version to scale')
        program={i :(1,Ns) for i in range(1)}
        
        dataset_r=build_datasets(data_dir, program)[0]
        dataset_f=build_datasets(data_dir_f, program, fake=True)[0]
        
        results={} 
        
        for metric in metrics_list:
            print(metric)
            datalist=[(metric, [dataset_r, dataset_f], 0)]#, (metric, dataset_f, var_names,1)]
            results[metric]=eval_distance_metrics(datalist[0])
            
            #with Pool(num_proc) as p:
            #    results[metric]=p.map(global_dataset_eval,datalist) 

            #results[metric].append(np.mean(np.array(results[metric]), axis=0))
            #results[metric].append(np.std(np.array(results[metric][:-1]), axis=0))
            
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+str(Ns)+'.p', 'wb'))"""