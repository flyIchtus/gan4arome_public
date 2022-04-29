#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""
import metric_test_snippets as snip
import metrics4arome as metrics
import pickle



CI=(78,206,55,183)

data_dir='/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_fsteps='/scratch/mrmn/brochetc/GAN_2D/Set_36/resnet_aa_128_wgan-hinge_64_64_1_0.001_0.001/Instance_3/models/'
data_dir_f='/scratch/mrmn/brochetc/GAN_2D/Set_36/resnet_aa_128_wgan-hinge_64_64_1_0.001_0.001/Instance_3/samples/Best_model_dataset/'

original_data_dir='/scratch/mrmn/brochetc/'
output_dir='/scratch/mrmn/brochetc/GAN_2D/Set_36/resnet_aa_128_wgan-hinge_64_64_1_0.001_0.001/Instance_3/log/'

ID_file='IS_method_labels.csv'

if __name__=="__main__":
    """
    N_samples=[1024]#,16384]
    repeats=[32]

    list_steps=[int(ch[109:]) for ch in glob(data_dir_fsteps+'bestdisc_*')]
    print(list_steps)
    metric='fid'
    for rep in repeats:
        print('treating repeat {}'.format(rep))
        for step in list_steps:
            print(step)
            compute_and_save(metric, N_samples,ddir=data_dir,
                         repeats=rep, distance=True,\
                         step=step,cuda=True)
            
    
    metric="spectral"
    N_samples=[66048]
    compute_and_save(metric, N_samples,ddir=data_dir_f, repeats=1, distance=False,full_dataset=True)
    """
    N_samples=66048#,16384]
    #num_proc=12
    metrics_list=["pw_W1"] #, "SWD_metric"]
        
    var_names=["u", "v", "t2m"]
    
    program={i :(1,N_samples) for i in range(1)}
    
    dataset_r=snip.build_datasets(data_dir, program)[0]
    list_steps=['_Fsample_chunk'+str(1000*k)+'.npy' for k in range(50)]
    
    results={} 
    
    for metric in metrics_list:
        print(metric)
        
        assert hasattr(metrics,metric)
        
        cuda=True if metric=='FID' else False
        
        for step in list_steps:
            data=(metric, {'real':dataset_r,'fake': step}, 0, cuda)
            results[metric].append(snip.eval_distance_metrics(data))
        

        pickle.dump(results[metric], open(output_dir+metric+'_'+str(N_samples)+'.p', 'wb'))