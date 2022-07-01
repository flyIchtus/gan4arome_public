#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:04:19 2022

@author: brochetc


Metrics computation automation

"""

import sys
sys.path.append('/home/mrmn/brochetc/gan4arome/metrics4arome/')

import evaluation_backend as backend
import metrics4arome as metrics
import pickle
from glob import glob
import numpy as np
from multiprocessing import Pool
import configurate


########### standard parameters #####

num_proc = 4
var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4} # do not touch unless
                                                          # you know what u are doing
data_dir=backend.data_dir
#####################################


class MetricsCalculator() :
    
    def __init__(self, expe_config, program, add_name) :
        
        
        self.data_dir_f = expe_config.data_dir_f
        self.log_dir = expe_config.log_dir
        self.expe_dir = self.log_dir[:-4]
            
        self.steps = expe_config.steps
        self.add_name = add_name
        
        self.instance_num=expe_config.instance_num
        
        indices = configurate.retrieve_domain_parameters(self.expe_dir, 
                                                             self.instance_num)
        
        self.CI, self.var_names , self.var_inds = indices
        
    
        
    ###########################################################################
    ######################### Main class method ###############################
    ###########################################################################
    
    
    def estimation(self, metrics_list, program, parallel=False, standalone=False, real=False) :
        
        """
        
        estimate all metrics contained in metrics_list on training runs
        using specific strategies
                       -> parallel or sequential estimation
                       -> distance metrics or standalone metrics
                       -> on real samples only (if distance metrics)
                       
        Inputs :
            
            metrics_list : list, the list of metrics to be computed
            
            program : dict of shape {int : (int, int)}
                      contains all the informations about sample numbers and number of repeats
                      
                      keys index the repeats
                      
                      values[0] index the type of dataset manipulation
                      (either dividing the same dataset into parts, or selecting only one portion)
                      
                      values[1] indicate the number of samples to use in the computation
                      
                      Note : -> for tests on training dynamics, only 1 repeat is assumed
                                  (at the moment)
                             -> for tests on self-distances on real datasets,
                                many repeats are possible (to account for test variance
                                or test different sampling sizess)
            
            parallel, standalone, real : bool, the flags defining the estimation
                                         strategy
        
        Returns :
            
            None
            
            dumps the results in a pickle file
        
        """
        
        ########### sanity checks
        
        if standalone and not parallel :
            raise(ValueError, 'Estimation fo standalone metric must be done in parallel')
        
        if standalone :
            
            assert set(metrics_list) <= metrics.standalone_metrics
        
        else :
            
            assert set(metrics_list) <= metrics.distance_metrics
            
        ########################
        
        self.program = program
        
        results = {} 
        results["header"] = metrics_list
        
        for metric in metrics_list :
            print(metric)
            assert hasattr(metrics, metric)
            
            if parallel :
                if standalone :
                    if real :
                        func = lambda metric : self.parallelEstimation_standAlone(metric, option='real')
                    else :
                        func = self.parallelEstimation_standAlone
                    
                else :
                    if real :
                        func = self.parallelEstimation_realVreal
                    else :
                        func = self.parallelEstimation_realVSfake
            else :
                
                if real :
                    func = self.sequentialEstimation_realVSreal
                    
                else :
                    func = self.sequentialEstimation_realVSfake
                    
            results[metric]=func(metric)
            
        N_samples_set=set([self.program[i][1] for i in range(len(program))])
            
        N_samples_name = '_'.join([str(n) for n in N_samples_set])
        
        dumpfile=self.log_dir+self.add_name+'distance_metrics_'+str(N_samples_name)+'.p'
        
        pickle.dump(results, open(dumpfile, 'wb'))
        
        
    
    ###########################################################################
    ############################   Estimation strategies ######################
    ###########################################################################
    
    
    
    def parallelEstimation_realVSfake(self, metric):
        
        """
        
        makes a list of datasets with each item of self.steps
        and use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        data_list=[]
        
        for step in self.steps:
            
            #getting first (and only) item of the random real dataset program
            dataset_r=backend.build_datasets(data_dir, self.program)[0]
            
            N_samples=self.program[0][1]
            
            #getting files to analyze from fake dataset
            files=glob(self.data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
            
          
            data_list.append((metric, {'real':dataset_r,'fake': files},\
                              N_samples, N_samples,\
                              self.VI, self.CI, step))
        
        with Pool(num_proc) as p :
            res=p.map(backend.eval_distance_metrics, data_list)
        
        
        return np.array(res)

        
    
    def sequentialEstimation_realVSfake(self, metric):
        
        """
        
        Iterates the evaluation of the metric on each item of self.steps
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        res = []
       
        for step in self.steps:
            
            #getting first (and only) item of the random real dataset program
            dataset_r = backend.build_datasets(data_dir, self.program)[0]
            
            N_samples=self.program[0][1]
            
            #getting files to analyze from fake dataset
            files=glob(self.data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
            
          
            data=(metric, {'real':dataset_r,'fake': files},\
                  N_samples, N_samples,
                  self.VI, self.CI, step)
        
            if step==0: res = [backend.eval_distance_metrics(data)]
            else :
                
                res.append(backend.eval_distance_metrics(data))
           
            res = np.array(res)
            
        return res
        
        
    def parallelEstimation_realVSreal(self, metric):
        
        """
        
        makes a list of datasets with each pair of real datasets contained
        in self.program.
        
        Use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / real
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
                
        datasets = backend.build_datasets(data_dir, self.program)
        data_list = []         
    
        #getting the two random datasets programs
            
        for i in range(len(datasets)):
            
            N_samples = self.program[i][1]
          
            data_list.append((metric, 
                              {'real0':datasets[i][0],'real1': datasets[i][1]},
                              N_samples, N_samples,
                              self.VI, self.CI,i))
        
        with Pool(num_proc) as p :
            res = p.map(backend.eval_distance_metrics, data_list)
            
        res = np.array(res)
        
        
        return res
        
    def sequentialEstimation_realVSreal(self, metric):
        
        """
        
        Iterates the evaluation of the metric on each item of pair of real datasets 
        defined in self.program.
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
            
        #getting first (and only) item of the random real dataset program
        datasets = backend.build_datasets(data_dir, self.program)
            
        for i in range(len(datasets)):
            
            N_samples = self.program[i][1]
          
            data=(metric, {'real0':datasets[i][0],'real1': datasets[i][1]},
                           N_samples, N_samples,
                           self.VI, self.CI, i)
        
            if i==0: res = [backend.eval_distance_metrics(data)]
            else :  
                res.append(backend.eval_distance_metrics(data))
       
        res = np.array(res)
        
        return res
        
        
    def parallelEstimation_standAlone(self, metric, option='fake'):
        
        """
        
        makes a list of datasets with each dataset contained
        in self.program (case option =real) or directly from data files 
        (case option =fake)
        
        Use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / real
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        if option=='real':
            
            assert self.program is not None
            dataset_r = backend.build_datasets(data_dir, self.program)
            
            N_samples = len(dataset_r)
      
            
        for i,step in enumerate(self.steps):
            
            data_list = []
            
            #getting files to analyze from fake dataset
            if option=='fake' :
                
                files = glob(self.data_dir_f+"_FsampleChunk_"+str(step)+'_*.npy')
                
                data_list.append((metric, files, N_samples, 
                                  self.VI, self.CI, step, option))
                
            elif option=="real":
                
                data_list.append((metric, dataset_r[i], step, False, option))
                
        with Pool(num_proc) as p :
            
            res = p.map(backend.global_dataset_eval, data_list)
            
        res = np.array(res)
                            
        return res
    

