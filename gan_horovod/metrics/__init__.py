#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:53:46 2022

@author: brochetc

metrics version 2

File include :
    
    metric2D and criterion2D APIs to be used by Trainer class from trainer_horovod
    provide a directly usable namespace from already implemented metrics

"""

import metrics.general_metrics as GM
import metrics.wasserstein_distances as WD
import metrics.sliced_wasserstein as SWD
import metrics.spectrum_analysis as Spectral
import metrics.inception_metrics as inception

########################### High level APIs ##################################

class metric2D():
    def __init__(self,long_name,func,names):
        self.long_name=long_name
        self.func=func #func should return  numpy array OR tensor to benefit from parallel estimation
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
W1_Center_NUMPY=criterion2D('Mean Wasserstein distance on center crop  ',\
                      WD.W1_center_numpy)

pw_W1=metric2D('Point Wise Wasserstein distance', WD.pointwise_W1, ['pW_u', 'pW_v', 'pW_t2m'])

IntraMapVariance=metric2D('Mean intra-map variance of channels   ',\
                          GM.intra_map_var,['intra_u', 'intra_v', 'intra_t2m'])
InterMapVariance=metric2D('Mean Batch variance of channels   ', \
                          GM.inter_map_var,['inter_u', 'inter_v', 'inter_t2m'])

# Sliced Wasserstein Distance estimations

sliced_w1=SWD.SWD_API(image_shape=(128,128), numpy=True)
SWD_metric=metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1.End2End, sliced_w1.get_metric_names())

sliced_w1_torch=SWD.SWD_API(image_shape=(128,128), numpy=False)
SWD_metric_torch=metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1_torch.End2End, sliced_w1_torch.get_metric_names())

# spectral analysis

spectral=metric2D('Spectral analysis of Dataset  ',Spectral.PSD_wrap, ['PSD u', 'PSD v', 'PSD t2m'])


# FID score

#Inception_v3=inception.loadInception_v3('/home/mrmn/brochetc/gan4arome_aliasing/gan_horovod/metrics/inception_v3_weights', cuda=True)
#fid=metric2D('Fréchet Inception Distance  ', inception.FIDclass(Inception_v3).FID,['FID'])