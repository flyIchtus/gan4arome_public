#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import numpy as np
import metrics4arome as metrics
import pickle
from glob import glob
import random



CI=(78,206,55,183)

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'

#labels=pd.read_csv(data_dir+ID_file)



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


def load_batch(file_list,number,CI,option='real',\
               output_dir=None, output_id=None, save=False):
     
    """
    gather a fixed number of random samples present in file_list into a single big matrix

    
    Inputs :
        file_list : list of files to be sampled from
        number : int, the number of samples to draw
        CI : iterable of ints, coordinates of data to be taken (only in 'real' mode)
        Shape : tuple, the target shape of every sample
        option : str, different treatment if the data is GAN generated or PEARO
        
        output_dir : str of the directory to save datasets
        output_id : name of the dataset to be saved
    
    Returns :
        Mat : numpy array, shape  number x C x Shape[1] x Shape[2] matrix
    """
    
    if option=='fake':
        # in this case samples can either be in isolated files or grouped in batches
        print('loading shape')
        Shape=np.load(file_list[0]).shape
        
        if len(Shape)==3:
            
            Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
            
        elif len(Shape)==4:
            
            Mat=np.zeros((number*Shape[0], Shape[1], Shape[2], Shape[3]), \
                                                         dtype=np.float32)
        
        list_inds=random.sample(file_list, number)
        print('list_inds',list_inds)
        for i in range(number):
            print('fake file index',i)
            if len(Shape)==3:
                
                Mat[i]=np.load(list_inds[i]).astype(np.float32)
                
            elif len(Shape)==4:
                
                Mat[i*Shape[0]: (i+1)*Shape[0]]=\
                np.load(list_inds[i]).astype(np.float32)
            
    elif option=='real':
        # in this case samples are stored once per file
        
        Shape=(3,CI[1]-CI[0], CI[3]-CI[2])
        
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(file_list, number)
        
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1], CI[2]:CI[3]].astype(np.float32)
            
    return Mat


def normalize(BigMat, Mean, Max):
    res= (0.95)*(BigMat-Mean)/(Max)
    #print(res.max(), res.min())
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
               
           index : int, identifier to the data passed 
                   (useful only if used in multiprocessing)

       
       option : str, to choose if generated data is loaded from several files (from_names)
               or from one big Matrix (from_matrix)
                   
    Returns :
        results : np.array containing the calculation of the metric
    """
    metric, dataset, index=data
    
    if list(dataset.keys())==['real','fake']:
    
        number_real=len(dataset['real'])

        number_fake=len(dataset['fake'])
    
        print('index',index)
    
        results=[]
            
        Metric=getattr(metrics, metric)
    
        if option=='from_names':
            
            assert(type(dataset['fake'])==list)
            fake_data=load_batch(dataset['fake'], number_fake,CI,option='fake')
            print('fake data loaded')
        if option=='from_matrix':
           
            assert(type(dataset['fake']==str))
            
            fake_data=np.load(dataset['fake'], dtype=np.float32)
            
        real_data=load_batch(dataset['real'],number_real, CI)
        print('real data loaded')
        Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
        Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
        real_data=normalize(real_data,Means, Maxs)
    
    elif list(dataset.keys())==['real0', 'real1']:
        
        number0=len(dataset['real0'])
        number1=len(dataset['real1'])
    
        print(index)
    
        results=[]
            
        Metric=getattr(metrics, metric)
    
       
            
        real_data0=load_batch(dataset['real0'],number0, CI)
        real_data1=load_batch(dataset['real1'], number1, CI)
        
        
        Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
        Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
        
        real_data=normalize(real_data0,Means, Maxs)
        fake_data=normalize(real_data1, Means, Maxs)
    else :
        raise ValueError("Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {}"
                         .format(list(dataset.keys())))
    
    results.append(Metric(real_data, fake_data))
    return np.array(results)

def global_dataset_eval(data, option='from_names'):
    """
    variable-wise evaluation of metric on the DataSet (treated as a single numpy matrix)
    
    data : tuple of str, dict, int
        metric : str, the name of a metric in the metrics4arome namespace
        dataset :
            file list /str
        
        ind : identifier of the call to function
    """
    metric, DataSet,ind,cuda, data_option=data
    var_names=['u','v','t2m']
    Metric=getattr(metrics,metric)

    
    if option=="from_names":
        assert(type(DataSet)==list)
        number=len(DataSet)
        data=load_batch(DataSet, number,CI,option=data_option)
       

    if option=='from_matrix':
       
        assert(type(DataSet)==str)
        
        data=np.load(DataSet, dtype=np.float32)
    
    if data_option=='real':
        Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
        Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
    
        data=normalize(data,Means, Maxs)
    
    print(ind, data.shape)
    results=[]
    
    for i,var in enumerate(var_names):
        results.append(Metric(data[:,i:(i+1),:,:]))
        
    if type(results[-1])==tuple:
        return results[0][0], np.array([results[i][1] for i in range(len(var_names))])
    
    return np.array(results).squeeze()

def save_metric_results(metric, name,arrays):
    """
    Save metric results stored in arrays list into a pickle dictionary
    
    Inputs :
        metric : str, the name of a metric in the metrics4arome namespace
    
        name : str, the additional name to add to the saved file
        
        arrays : list of np.ndarrays to be saved (each  being a results of  a metric calculation)
    """
    
    dic={}
    for i, arr in enumerate(arrays):
        dic[i]=arr
    pickle.dump(dic, open(metric+'_'+name+'.p', 'wb'))
    
    
def average_dynamics(step,N_samples,directories):
    """
    compute an average of different metrics dynamics stored in separate directories
    """
    dir0=directories[0]
    
    results=pickle.load(open(dir0+str(step)+'_'+str(N_samples)+'.p','rb'))
    metrics_list=results["header"]
    MEAN_RES={k : v for (k,v) in results.items()}
    
    for direct in directories:

        res=pickle.load(open(dir0+str(step)+'_'+str(N_samples)+'.p','rb'))
        for metric in metrics_list:
            MEAN_RES[metric]+=res[metric]
            
    for metric in metrics_list:
        MEAN_RES[metric]=MEAN_RES[metric]/len(directories)
        
    return MEAN_RES
"""
def compute_and_save(metric, N_samples,ddir=data_dir,\
                     repeats=1, distance=True,
                     full_dataset=False, auto=False, \
                     step=None, cuda=False):
    
    
    compute and save statistical metrics results on  N_samples samples
    repeats the calculation 'repeats' times
    options :
        distance is when the metric in itself is a distance
        full_dataset is used when the full available dataset is used to compute
                  distance againt the full generated dataset 
                  in the 'distance' context
        auto is when the distance is evaluated between samples from the same dataset
        otherwise distance is used to compare samples from generated and from
            real dataset
        
        if all options are set to false, a metric is computed on the selected 
        data directory ddir (eg spectrum)
    
    
    assert hasattr(metrics, metric)

    
    for Ns in N_samples :
        print("treating Ns {}".format(Ns))
        
        
        if distance and full_dataset and not auto:
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
            
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'_fakeVSreal.p', 'wb'))
            
        elif distance and auto:
            
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
            
            pickle.dump({metric :results[metric]}, open(output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'_auto.p', 'wb'))
            
        elif distance :
            program={i :(1,Ns) for i in range(repeats)}
            
            dataset_r=build_datasets(data_dir, program)
            dataset_f=build_datasets(data_dir_f, program, fake=True, step=step)
        
            results={} 
        
            print(metric)

            
            if cuda :
                for rep0 in range(repeats):
                    data=(metric, [dataset_r[rep0], dataset_f[rep0]], 0, cuda)
                    if rep0==0:
                        results[metric]=[eval_distance_metrics(data)]
                    else:
                        results[metric].append(eval_distance_metrics(data))
                results[metric].append(np.mean(np.array(results[metric])))
                results[metric].append(np.std(np.array(results[metric])))
            else:
                datalist=[(metric, [dataset_r, dataset_f], 0, cuda)]
                results[metric]=eval_distance_metrics(datalist[0])
            
            if step is not None:
                name=output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'_'+str(step)+'_fakeVSreal.p'
            else:
                name=output_dir+metric+'_'+str(Ns)+'_'+str(repeats)+'_fakeVSreal.p'
                
            pickle.dump({metric :results[metric]}, open(name, 'wb'))
        
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
            
    return 0"""

