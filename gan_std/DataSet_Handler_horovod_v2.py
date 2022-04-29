#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose
from filelock import FileLock

class ISDataset(Dataset):
    def __init__(self, data_dir, ID_file,crop_indexes,transform=None):
        
        self.data_dir=data_dir
        self.transform=transform
        self.labels=pd.read_csv(data_dir+ID_file)
        self.CI=crop_indexes
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        sample_path=os.path.join(self.data_dir, self.labels.iloc[idx,0])
        sample=np.float32(np.load(sample_path+'.npy'))[1:4,self.CI[0]:self.CI[1],self.CI[2]:self.CI[3]].transpose((1,2,0)) 
        ## transpose to get off with transform.Normalize builtin transposition
        importance=self.labels.iloc[idx,1]
        position=self.labels.iloc[idx,2]
        if self.transform:
            sample = self.transform(sample)
        return sample, importance, position


class ISData_Loader():
    def __init__(self, path, batch_size,\
                 shuf=False):
        self.path = path
        self.batch = batch_size
        self.shuf = shuf #shuffle performed once per epoch
        Means=np.load(path+'mean_with_orog.npy')[1:4]
        Maxs=np.load(path+'max_with_orog.npy')[1:4]
        self.means=list(tuple(Means))
        self.stds=list(tuple((1.0/0.95)*(Maxs-Means)))
        
    def transform(self, totensor, normalize):
        options = []
        if totensor:
            options.append(ToTensor())

        if normalize:
            options.append(Normalize(self.means, self.stds))
        transform = Compose(options)
        return transform
    
    def loader(self, hvd_size=None, hvd_rank=None, kwargs=None):
        
        if kwargs is not None :
            with FileLock(os.path.expanduser("~/.horovod_lock")):    
                dataset=ISDataset(self.path, 'IS_method_labels.csv',\
                                  (78,206,55,183),self.transform(True,True))

        self.sampler=DistributedSampler(
                    dataset, num_replicas=hvd_size,rank=hvd_rank
                    )
        if kwargs is not None:
            loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=self.shuf,
                            sampler=self.sampler,
                            **kwargs
                            )
        else:
            loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=self.shuf,
                            sampler=self.sampler,
                            )
        return loader
