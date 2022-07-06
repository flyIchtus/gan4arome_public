#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import sys

if '/home/mrmn/brochetc/gan4arome' in sys.path :
    sys.path.remove('/home/mrmn/brochetc/gan4arome')
sys.path.append('/home/mrmn/brochetc/gan4arome_reorg_var/')

print('-------------BACKEND IMPORT -----------------')
print(sys.path)

import numpy as np
import metrics4arome as metrics
from glob import glob
import random


########### standard parameters #####

num_proc = 8
var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4} # do not touch unless
                                                          # you know what u are doing

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

#####################################

def split_dataset(file_list,N_parts):
    """
    randomly separate a list of files in N_parts distinct parts
    
    Inputs :
        file_list : a list of filenames
        
        N_parts : int, the number of parts to split on
    
    Returns :
         list of N_parts lists of files
    
    """
    
    inds=[i*len(file_list)//N_parts for i in range(N_parts)]+[len(file_list)]

    to_split=file_list.copy()
    random.shuffle(to_split)
    
    return [to_split[inds[i]:inds[i+1]] for i in range(N_parts)]


def normalize(BigMat, scale, Mean, Max):
    
    """
    
    Normalize samples with specific Mean and max + rescaling
    
    Inputs :
        
        BigMat : ndarray, samples to rescale
        
        scale : float, scale to set maximum amplitude of samples
        
        Mean, Max : ndarrays, must be broadcastable to BigMat
        
    Returns :
        
        res : ndarray, same dimensions as BigMat
    
    """
    
    res= scale*(BigMat-Mean)/(Max)

    return  res


def build_datasets(data_dir, program,step=None, option='real'):
    """
    
    Build file lists to get samples, as specified in the program dictionary
    
    Inputs :
        
        data_dir : str, the directory to get the data from
        
        program : dict,the datasets to be constructed
                dict { dataset_id : (N_parts, n_samples)}
                
        step : None or int -> if None, normal search among generated samples
                              if int, search among generated samples at the given step
                              (used in learning dynamics mapping)
    
    Returns :
        
        res, dictionary of the shape {dataset_id : file_list}
        
        !!! WARNING !!! : the shape of file_list depends on the number of parts
        specified in the "program" items. Can be nested.
    
    """
    if step is not None:
        name='_FsampleChunk_'+str(step)+'_'
    else:
        name='_Fsample'
        

    if option=='fake':
        globList=glob(data_dir+name+'*')
        
    else:
        globList=glob(data_dir+'_sample*')
        
    res={}
    
    
    for key, value in program.items():
        if value[0]==2:

            fileList=random.sample(globList,2*value[1])
            
            res[key]=split_dataset(fileList,2)
                        
        if value[0]==1:
            
            fileList=random.sample(globList,value[1])
            
            res[key]=fileList

    return res


def load_batch(file_list,number,\
               var_indexes=None,crop_indexes=None,
               option='real',\
               output_dir=None, output_id=None, save=False):
     
    """
    gather a fixed number of random samples present in file_list into a single big matrix

    
    Inputs :
        
        file_list : list of files to be sampled from
        
        number : int, the number of samples to draw
        
        var_indexes : iterable of ints, coordinates of variables i a given sample
        
        crop_indexes : iterable of ints, coordinates of data to be taken (only in 'real' mode)
                
        Shape : tuple, the target shape of every sample
        
        option : str, different treatment if the data is GAN generated or PEARO
        
        output_dir : str of the directory to save datasets ---> NotImplemented
        
        output_id : name of the dataset to be saved ---> NotImplemented
        
        save : bool, whether or not to save the loaded dataset ---> NotImplemented
    
    Returns :
        
        Mat : numpy array, shape  number x C x Shape[1] x Shape[2] matrix
        
    """
    
    if option=='fake':
        # in this case samples can either be in isolated files or grouped in batches
        
        assert number <= 16384  # maximum number of generated samples
        
        print(number)
        
        print('loading shape')
        Shape = np.load(file_list[0]).shape
        
        ## case : isolated files (no batching)
        if len(Shape)==3:
            
            Mat = np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
            
            list_inds=random.sample(file_list, number)
            
            for i in range(number) :
                
                print('fake file index',i)
                
                Mat[i]=np.load(list_inds[i]).astype(np.float32)
        
        ## case : batching -> select the right number of files to get enough samples
        elif len(Shape)==4:
            
            batch = Shape[0]
                        
            if batch > number : # one file is enough
                
                indices = random.sample(range(batch), number)
                k = random.randint(0,len(file_list)-1)
                
                Mat=np.load(file_list[k])[indices]
            
            else : #select multiple files and fill the number
            
                Mat=np.zeros((number, Shape[1], Shape[2], Shape[3]), \
                                                         dtype=np.float32)
                
                list_inds=random.sample(file_list, number//batch)
                
                for i in range(number//batch) :
                    Mat[i*batch: (i+1)*batch]=\
                    np.load(list_inds[i]).astype(np.float32)
                    
                if number%batch !=0 :
                    remain_inds = random.sample(range(batch),number%batch)
                
                    Mat[i*batch :] = np.load(list_inds[i+1])[remain_inds].astype(np.float32)
    
                

    elif option=='real':
        
        # in this case samples are stored once per file
        
        Shape=(len(var_indexes),
               crop_indexes[1]-crop_indexes[0],
               crop_indexes[3]-crop_indexes[2])
        
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(file_list, number) # randomly drawing samples
        
        for i in range(number):
            
            Mat[i]=np.load(list_inds[i])[var_indexes,
                                           crop_indexes[0]:crop_indexes[1],
                                           crop_indexes[2]:crop_indexes[3]].astype(np.float32)
            
    return Mat



def eval_distance_metrics(data, option='from_names'):
    
    """
    
    this function should test distance metrics for datasets=[(filelist1, filelist2), ...]
    in order to avoid memory overhead, datasets are created and destroyed dynamically
    
    Inputs :
        
       data : tuple of 
       
           metric : str, the name of a metric in the metrics4arome namespace
           
           dataset : 
               dict of file lists (option='from_names') /str (option 'from_matrix')
               
               Identifies the file names to extract data from
               
               Keys of the dataset are either : 'real', 'fake' when comparing 
                               sets of real and generated data
                               
                                                'real0', 'real1' when comparing
                               different sets of real data
           
           n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                      to compute the metric.
                                      
                                      Note : most metrics require equal numbers
                                      
                                      by default, 0 : real, 1 : fake
            
           VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                   (CI) crop indices in maps
               
           index : int, identifier to the data passed 
                   (useful only if used in multiprocessing)

       
       option : str, to choose if generated data is loaded from several files (from_names)
               or from one big Matrix (from_matrix)
                   
    Returns :
        
        results : np.array containing the calculation of the metric
        
    """
    metrics_list, dataset, n_samples_0, n_samples_1, VI, CI, index=data
    
    ## loading and normalizing data
    
    Means=np.load(data_dir+'mean_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    Maxs=np.load(data_dir+'max_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    
    if list(dataset.keys())==['real','fake']:
    
        print('index',index)
    
        if option=='from_names':
            
            assert(type(dataset['fake'])==list)
            
            fake_data = load_batch(dataset['fake'], n_samples_1,option='fake')
            
            print('fake data loaded')
            
        if option=='from_matrix':
           
            assert(type(dataset['fake']==str))
            
            fake_data = np.load(dataset['fake'], dtype=np.float32)
            
        real_data = load_batch(dataset['real'], n_samples_0, var_indexes=VI, crop_indexes=CI)
        
        print('real data loaded')
        
        real_data = normalize(real_data, 0.95, Means, Maxs)
    
    elif list(dataset.keys())==['real0', 'real1']:
    
        print(index)
    
        real_data0 = load_batch(dataset['real0'],n_samples_0, var_indexes=VI, crop_indexes=CI)
        real_data1 = load_batch(dataset['real1'], n_samples_1, var_indexes=VI, crop_indexes=CI)
        
        real_data = normalize(real_data0, 0.95, Means, Maxs)
        fake_data = normalize(real_data1, 0.95, Means, Maxs)  # not stricly "fake" but same
        
    else :
        raise ValueError("Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {}"
                         .format(list(dataset.keys())))
        
    ## the interesting part : computing each metric of metrics_list
    
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        
        Metric = getattr(metrics, metric)
    
        results[metric] = Metric(real_data, fake_data)
    
    return results, index

def global_dataset_eval(data, option='from_names'):
    
    """
    
    evaluation of metric on the DataSet (treated as a single numpy matrix)
    
    Inputs :
    
        data : iterable (tuple)of str, dict, int
            
            metric : str, the name of a metric in the metrics4arome namespace
            
            dataset :
                file list /str containing the ids of the files to get samples
                
            
            n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                          to compute the metric.
                                          
                                          Note : most metrics require equal numbers
                                          
                                          by default, 0 : real, 1 : fake
                
               VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                       (CI) crop indices in maps
                   
               index : int, identifier to the data passed 
                       (useful only if used in multiprocessing)
    
           
         option : str, to choose if generated data is loaded from several files (from_names)
                   or from one big Matrix (from_matrix)
    
    Returns :
        
        results : dictionary contining the metrics list evaluation
        
        index : the index input (to keep track on parallel execution)
        
    """
    
    metrics_list, dataset, n_samples, VI, CI, index, data_option = data
    
    print(index)
    
    if option=="from_names":
        
        assert(type(dataset)==list)
        
        rdata = load_batch(dataset, n_samples,var_indexes=VI, crop_indexes=CI,option=data_option)


    if option=='from_matrix':
       
        assert(type(dataset)==str)
        
        rdata = np.load(dataset, dtype=np.float32)
    
    if data_option=='real':
        
        Means=np.load(data_dir+'mean_with_orog.npy')[VI].reshape(1,len(VI),1,1)
        Maxs=np.load(data_dir+'max_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    
        rdata=normalize(rdata, 0.95,Means, Maxs)
    
   
    
    
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        
        Metric = getattr(metrics, metric)
                
        results[metric] = Metric(rdata)
    
    return results, index

