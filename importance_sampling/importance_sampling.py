#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:52:20 2022

@author: brochetc

Importance sampling algorithm for preprocessing
This algorithm is based on the ideas of Ravuri et al., 2021



"""

import numpy as np
import os
import glob
import random as rd
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from time import perf_counter

var_names=["_rrdecum","_u", "_v","_t2m"] #_rrdecum
day_name='2021-08-03T21:00:00Z'
indexes=(20,680,150,972)
SE_indexes=(120,376,540,796)



############################# Sampling method class ##########################
class Sampling_method():
    def __init__(self, N_samples, crop_size,params, indexes, prefix,channels=4):
        self.params=params
        chain=''
        for p in params:
            chain+=str(p)+'_'
        self.N_samples=N_samples
        self.crop_size=crop_size
        self.indexes=indexes
        self.save_dir=prefix+'IS_'+str(N_samples)+'_'+chain+str(crop_size)+'/'
        if not os.path.exists(self.save_dir) and not os.path.exists(self.save_dir+'_done'):
            print('creating save dir and csv fie')
            os.mkdir(self.save_dir)
            with open(self.save_dir+'/IS_method_labels.csv', 'a') as file :
                file.write('Name,Importance,PosX,PosY\n')
                file.close()
            
    def importance(self, Mat):
        """
        compute the importance score of a cropped sample according to wind and
        rr
        
        rr is assumed to be 1st channel, and u_10, v_10 the 2nd and 3rd
        """
        q_min, m, p, q, s_rr, s_w=self.params
        if q_min>=1.0 :
            return 1.0
        N=Mat.shape[0]*Mat.shape[1]
        S_r=(1-np.exp(-Mat[:,:,0]/s_rr)).sum()
        w=np.sqrt(Mat[:,:,1]**2+Mat[:,:,2]**2)
        S_w=(1-np.exp(-w/s_w)).sum()
        return min(1.0,q_min+(m/N)*(p*S_r+q*S_w))
    
    def Importance_Sampling(self, data):
        """
        perform importance sampling of the data=Map, file name
        perform saving simultaneously
        can be used in parallel form
        
        """
        M, N1, N0=data
        sample_num=N1+self.N_samples*N0
        FF_shape=M.shape[0], M.shape[1]
        for i in range(self.N_samples):
            #if i==self.N_samples-1:
            #    print(i)
            posX, posY=select_position(FF_shape,self.crop_size)
            #print(posX, posY)
            Map=crop_from_position(M, (posX, posY), self.crop_size)
            px=self.importance(Map)

            p=rd.uniform(0,1)
            if p<=px:
                sample_name='_sample'+str(sample_num)
                assert Map.shape==(self.crop_size, self.crop_size,4)
                np.save(self.save_dir+sample_name+'.npy', Map,\
                        allow_pickle=True)
                with open(self.save_dir+'/IS_method_labels.csv', 'a') as file:
                    #print('writing')
                    file.write(sample_name+','+str(px)+','+str(posX)+','+str(posY)+'\n')
                    file.close()
                sample_num+=1
        
            
    def sample_from_file(self,data):
        
        """
        importance rejection sampling from already cropped file  (faster !)
        
        """
        sample_name, posX,posY, path=data
        Map=np.load(path+'/'+sample_name+'.npy')
        px=self.importance(Map)

        p=rd.uniform(0,1)
        if p<=px:
            assert Map.shape==(self.crop_size, self.crop_size,4)
            np.save(self.save_dir+sample_name+'.npy', Map,\
                    allow_pickle=True)
            with open('IS_method_labels.csv', 'a') as filename:
                filename.write(sample_name+','+str(px)+','+str(posX)+','+str(posY))
                filename.close()
    
    def gather_samples(self, n_samples):
        """
        
        selects random (already processed) samples and gather them 
        into a single npy file to be readily loaded at network test time
        
        """
        df=pd.read_csv(self.save_dir+'/IS_method_labels.csv')
        rows=df.sample(n=n_samples)['Names']
        rows.to_csv('Gathered_samples.csv')
        BIG_MAT=np.zeros((n_samples,self.channels, self.crop_size,self.crop_size))
        for i,name in enumerate(rows):
            M=np.load(name+'.npy', allow_pickle=True)
            for j in range(self.channels):
                BIG_MAT[i,j,:,:]=M[:,:,j]
        np.save('Test_samples.npy', BIG_MAT, allow_pickle=True)

############################# Base functions #################################

def crop_arrays_in_file(filename,indexes):
    """
    M is of shape Lon x Lat x ech x Member
    """
    lb_index, rb_index, lu_index, ru_index=indexes
    print('opening', filename)
    M=np.load(filename, allow_pickle=True)
    return M[lb_index : rb_index, lu_index : ru_index,:,:]

def crop_from_position(Map, position, crop_size):
    x,y=position
    half=crop_size//2
    return Map[x-half:x+half,y-half: y+half]


def split_by_time(M):
    Nlt, Nm=M.shape[2], M.shape[3]
    li=[M[:,:,i,j] for i in range(Nlt) for j in range(Nm)]
    return li

def merge_variables(list_mat):
    for i in range(len(list_mat)) :
        M=list_mat[i]
        list_mat[i]=np.expand_dims(M,-1)
    print(list_mat[0].shape)
    return np.concatenate(list_mat, axis=4)

def select_position(FF_shape, crop_size):
    half=crop_size//2
    x=rd.randint(half, FF_shape[0]-half)
    y=rd.randint(half, FF_shape[1]-half)
    
    return (x,y)


    
######################## Dataset pass ########################################

def process_day(input_path,day_name,day_number,Sm):
    print(day_name)
    list_mat=[]
    
    print('Merging')
    for var in var_names:
        filename=input_path+day_name+var+'.npy'
        list_mat.append(crop_arrays_in_file(filename, Sm.indexes))
  
    M=merge_variables(list_mat)
 
    del list_mat
    
    print('Splitting')
    list_mat=split_by_time(M)
    N_ech=len(list_mat)
    data_list=[(M, (day_number-1)*N_ech*Sm.N_samples, n) \
               for (M, n) in zip(list_mat,range(N_ech))]
    del list_mat

    print('Sampling')
    with Pool(8) as p :
        p.map(Sm.Importance_Sampling,data_list)
    del data_list
    return 0

def process_all(input_path,Sm):
    Files=glob.glob(input_path+'*.npy')
    N=len(Files)//5
    day_name=''
    done_days=[]
    count=0
    for filename in Files :
       
        day_nameNew=filename[len(input_path):len(input_path)+20]
        if day_nameNew!=day_name:
            day_name=day_nameNew
            print(100*count/N,N) 
            if day_name not in done_days:
                count+=1
                _=process_day(input_path,day_name,count, Sm)
                done_days.append(day_name)
        print('day processed')
    return 0

def process_from_dir(input_path,Sm):
    df=pd.read_csv(input_path+'/IS_method_labels.csv')[:2641920]
    datalist=[(name, posX, posY, input_path) \
              for name, posX, posY in zip(df['Name'],df['PosX'],df['PosY'])]
    
    with Pool(8) as p:
        p.map(Sm.sample_from_file, datalist)
    

"""
var_names=["_rr","_u", "_v","_t2m"]
day_name='2021-08-03T21:00:00Z'
indexes=(20,680,150,972)
time=[]
for i in range(10):
    print(i)
    t0=perf_counter()
    _=process_day(day_name, Sm)
    time.append(perf_counter()-t0)
print(np.mean(time), np.std(time))"""
