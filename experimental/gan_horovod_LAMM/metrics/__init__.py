#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:53:46 2022

@author: brochetc

metrics version 2

File include :
    
    metric2D and criterion2D APIs to be used by Trainer class from trainer_horovod
    directly usable namespace from already implemented metrics

"""

import metrics.general_metrics as GM
import metrics.wasserstein_distances as WD
#import sliced_wasserstein as SWD

########################### High level APIs ##################################

class metric2D():
    def __init__(self,long_name,func,names):
        self.long_name=long_name
        self.func=func #func should return tensor if possible
        self.names=names # list of names for each of the func's output items

    def __call__(self, *args, **kwargs):
        return self.func(*args,**kwargs)
    
class criterion2D():
    def __init__(self,  name, func):
        self.name=name
        self.func=func
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)



###################### Usable namespace #######################################
        
Orography_RMSE=metric2D('RMS Error on orography synthesis  ',\
                        GM.orography_RMSE,['orog_rmse'])
W1_Center=criterion2D('Mean Wasserstein distance on center crop  ',\
                      WD.W1_center)
IntraMapVariance=metric2D('Mean intra-map variance of channels   ',\
                          GM.intra_map_var,['intra_u', 'intra_v', 'intra_t2m'])
InterMapVariance=metric2D('Mean Batch variance of channels   ', \
                          GM.inter_map_var,['inter_u', 'inter_v', 'inter_t2m'])