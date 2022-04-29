#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet class from Importance_Sampled images

"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose


class ISDataset(Dataset):
    def __init__(self, data_dir, ID_file,device,transform=None):
        
        self.data_dir=data_dir
        self.transform=transform
        self.labels=pd.read_csv(data_dir+'/'+ID_file)

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample_path=os.path.join(self.data_dir, self.labels.iloc[idx,0])
        sample=np.load(sample_path)
        importance=self.labels.iloc[idx,1]
        position=self.labels.iloc[idx,2]
        if self.transform:
            sample = self.transform(sample)
        return sample, importance, position


class ISData_Loader():
    def __init__(self, path, batch_size,device,\
                 shuf=True, horovod=False):
        self.path = path
        self.batch = batch_size
        self.shuf = shuf #shuffle performed once per epoch
        self.device=device
        Means=np.load(path+'/'+'mean_with_orog.npy')
        Maxs=np.load(path+'/'+'max_with_orog.npy')
        self.means=tuple(Means)
        self.stds=tuple(0.90*(Maxs-Means))
        self.horovod=horovod

    def transform(self, totensor, normalize):
        options = []
        if totensor:
            options.append(ToTensor())
        if normalize:
            options.append(Normalize(self.means, self.stds))
        transform = Compose(options)
        return transform
    
    def loader(self, hvd_size=None, hvd_rank=None, kwargs=None):
        
        dataset=ISDataset(self.path, 'IS_method_labels.csv',self.device)
        if self.horovod:
            sampler=DistributedSampler(
                    dataset, num_replicas=hvd_size,rank=hvd_rank
                    )
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=self.shuf,
                            transform=self.transform(False,True),
                            sampler=sampler,
                            **kwargs
                            )
        return loader