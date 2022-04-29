import numpy as np
import os
import glob

input_path='/home/mrmn/brochetc/scratch_link/Sud_Est/Baselines/IS_1_1.0_0_0_0_0_0_256_done/'
orog_path='/home/mrmn/brochetc/scratch_link/PEARO_EURW1S40_Orography.npy'
output_path='/home/mrmn/brochetc/scratch_link/Sandbox/merge_test/'

orog=np.load(orog_path,allow_pickle=True)[120:376,540:796, np.newaxis]

listing=glob.glob(input_path+'*sample*')

for filename in listing:
    name=filename[len(input_path):]
    print(name)
    sample=np.load(filename,allow_pickle=True)
    new_sample=np.concatenate((sample,orog),axis=-1)
    np.save(output_path+name,new_sample)



