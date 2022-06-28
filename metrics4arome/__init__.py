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

import sys

sys.path.append('/home/mrmn/brochetc/gan4arome/metrics4arome/')

import metrics4arome.general_metrics as GM
import metrics4arome.wasserstein_distances as WD
import metrics4arome.sliced_wasserstein as SWD
import metrics4arome.spectrum_analysis as Spectral
import metrics4arome.inception_metrics as inception
import metrics4arome.scattering_metric as scat
import metrics4arome.structure_functions as sfunc

########################### High level APIs ##################################

class metric2D():
    def __init__(self,long_name,func,names):
        
        self.long_name=long_name
        self.func=func #should return np.array OR tensor to benefit from parallel estimation
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

pw_W1=metric2D('Point Wise Wasserstein distance', WD.pointwise_W1,\
               ['pW_u', 'pW_v', 'pW_t2m'])

IntraMapVariance=metric2D('Mean intra-map variance of channels   ',\
                          GM.intra_map_var,['intra_u', 'intra_v', 'intra_t2m'])
InterMapVariance=metric2D('Mean Batch variance of channels   ', \
                          GM.inter_map_var,['inter_u', 'inter_v', 'inter_t2m'])

# Sliced Wasserstein Distance estimations

sliced_w1=SWD.SWD_API(image_shape=(128,128), numpy=True)
SWD_metric=metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1.End2End,\
                    sliced_w1.get_metric_names())

sliced_w1_torch=SWD.SWD_API(image_shape=(128,128), numpy=False)
SWD_metric_torch=metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1_torch.End2End,\
                    sliced_w1_torch.get_metric_names())

# spectral analysis
print('spectral_dist')
spectral_dist=metric2D('Power Spectral Density RMSE  ',\
                  Spectral.PSD_compare, ['PSD u', 'PSD v', 'PSD t2m'])
print('spectral_comp')
spectral_compute=metric2D('Power Spectral Density  ',\
                  Spectral.PowerSpectralDensity, ['PSD u', 'PSD v', 'PSD t2m'])


# FID score
print('fid')
fid=metric2D('Fréchet Inception Distance  ',\
             inception.FIDclass(inception.inceptionPath).FID,\
             ['FID'])

# scattering metrics with sparsity and shape estimators

print('scattering')
scat_sparse=scat.scattering_metric(
        J=4,L=8,shape=(127,127), estimators=['s21', 's22'],
        frontend='torch', backend='torch', cuda=True
                                   )
#two versions of the same metrics
scat_SWD_metric=metric2D('Scattering Estimators ', scat_sparse.scattering_sliced,\
                       ['s21_u', 's21_v','s21_t2m'])

scat_SWD_metric_renorm=metric2D('Scattering Estimator', scat_sparse.scattering_renorm,
                              ['s21_u', 's21_v', 's21_t2m'])

# structure functions 

struct_metric=metric2D('First order structure function', sfunc.increments,\
                       ['Sf_u', 'Sf_v', 'Sf_t2m'])