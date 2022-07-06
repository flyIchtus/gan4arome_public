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

sys.path.append('/home/mrmn/brochetc/gan4arome_reorg_var/metrics4arome/')

import metrics4arome.general_metrics as GM
import metrics4arome.wasserstein_distances as WD
import metrics4arome.sliced_wasserstein as SWD
import metrics4arome.spectrum_analysis as Spectral
import metrics4arome.inception_metrics as inception
import metrics4arome.scattering_metric as scat
import metrics4arome.structure_functions as sfunc
import metrics4arome.multivariate as multiv


###################### standard parameters

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' : 3 , 'orog' : 4}

var_dict_fake = {'u' : 0 , 'v' : 1, 't2m' : 2, 'orog' : 3}

vars_wo_orog=['u', 'v', 't2m']

######################

########################### High level APIs ##################################

class metric2D():
    def __init__(self,long_name, func, variables, names = 'metric'):
        
        
        
        self.long_name = long_name
        
        self.names = names # names for each of the func's output items
        
        self.func = func #should return np.array OR tensor to benefit from parallel estimation
        
        self.variables = variables # variables on which the metric is applied
        
    def selectVars(self, *args) :
        
        """
        select in the input data the variables to compute metric on
        """
        
        if len(args)==2 :
            
            real_data, fake_data = args
            
            VI = [var_dict[v] for v in self.variables]
            VI_f = [var_dict_fake[v] for v in self.variables]
            
            real_data = real_data[:, VI,:,:]
            
            fake_data = fake_data[:, VI_f,:,:]
        
            return real_data, fake_data
        
        else :
            
            return args
    

    def __call__(self, *args, **kwargs):
        
        data = self.selectVars(args)[0]
        
        if len(data) == 2:
            
            return self.func(data[0], data[1] ,**kwargs)
        else :
            
            return self.func(data[0], **kwargs)
   
#################
#################
    
class criterion2D():
    def __init__(self,  long_name, func, variables):
        
        self.long_name = long_name
        
        self.func = func
        
        self.variables = variables
    
    def selectVars(self, *args) :
        
        """
        select in the input data the variables to compute metric on
        """
        
        if len(args)==2 :
            
            real_data, fake_data = args
            
            VI = [var_dict[v] for v in self.variables]
            VI_f = [var_dict_fake[v] for v in self.variables]
            
            real_data = real_data[:, VI,:,:]
            
            fake_data = fake_data[:, VI_f,:,:]
        
            return real_data, fake_data
        
        else :
            
            return args
        
    def __call__(self, *args, **kwargs):
        
        data = self.selectVars(args)[0]
        
        if len(data) == 2:
            
            return self.func(data[0], data[1] ,**kwargs)
        
        else :
            
            return self.func(data[0], **kwargs)

##############################################################################
        ################## Metrics catalogue #####################
        
standalone_metrics = {'spectral_compute', 'struct_metric', 'IntraMapVariance',
                    'InterMapVariance'}

distance_metrics = {'Orography_RMSE', 'W1_Center', "W1_Center_NUMPY", "W1_random",
                    "W1_random_NUMPY",
                    "pw_W1",
                    "SWD_metric", "SWD_metric_torch","fid","scat_SWD_metric",
                    "scat_SWD_metric_renorm", "multivar"}


###################### Usable namespace #######################################
        
Orography_RMSE = metric2D('RMS Error on orography synthesis  ',\
                        GM.orography_RMSE,'orog')

IntraMapVariance = metric2D('Mean intra-map variance of channels   ',\
                          GM.intra_map_var,
                          vars_wo_orog,names = ['intra_u', 'intra_v', 'intra_t2m'])
InterMapVariance = metric2D('Mean Batch variance of channels   ', \
                          GM.inter_map_var,
                          vars_wo_orog, names = ['inter_u', 'inter_v', 'inter_t2m'])

## crude Wasserstein distances

W1_Center = criterion2D('Mean Wasserstein distance on center crop  ',\
                      WD.W1_center, vars_wo_orog)


W1_Center_NUMPY = criterion2D('Mean Wasserstein distance on center crop  ',\
                      WD.W1_center_numpy, vars_wo_orog)

W1_random = criterion2D('Mean Wasserstein distance on random selection  ',\
                        WD.W1_random, vars_wo_orog)

W1_random_NUMPY = criterion2D('Mean Wasserstein distance on random selection  ',\
                        WD.W1_random_NUMPY, vars_wo_orog)

pw_W1 = metric2D('Point Wise Wasserstein distance', WD.pointwise_W1,\
               vars_wo_orog)



# Sliced Wasserstein Distance estimations

sliced_w1 = SWD.SWD_API(image_shape=(128,128), numpy=True)
SWD_metric = metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1.End2End,\
                    vars_wo_orog, names=sliced_w1.get_metric_names())

sliced_w1_torch = SWD.SWD_API(image_shape=(128,128), numpy=False)
SWD_metric_torch = metric2D('Sliced Wasserstein Distance  ',\
                    sliced_w1_torch.End2End,\
                    vars_wo_orog, names=sliced_w1_torch.get_metric_names())

# spectral analysis

spectral_dist = metric2D('Power Spectral Density RMSE  ',\
                  Spectral.PSD_compare, vars_wo_orog)

spectral_compute = metric2D('Power Spectral Density  ',\
                  Spectral.PowerSpectralDensity, vars_wo_orog)


# FID score

fid = metric2D('Fréchet Inception Distance  ',\
             inception.FIDclass(inception.inceptionPath).FID,\
             ['FID'])

# scattering metrics with sparsity and shape estimators


scat_sparse = scat.scattering_metric(
        J=4,L=8,shape=(127,127), estimators=['s21', 's22'],
        frontend='torch', backend='torch', cuda=True
                                   )
#two versions of the same metrics
scat_SWD_metric = metric2D('Scattering Estimators ', scat_sparse.scattering_sliced,\
                       vars_wo_orog)

scat_SWD_metric_renorm = metric2D('Scattering Estimator', scat_sparse.scattering_renorm,
                              vars_wo_orog)

# structure functions 

struct_metric = metric2D('First order structure function', 
                         lambda data : sfunc.increments(data, max_length = 16),\
                       vars_wo_orog)

#multivariate_comparisons
multivar=metric2D('Multivariate data', multiv.multi_variate_correlations,\
                  vars_wo_orog,names=['Corr_f','Corr_r'])