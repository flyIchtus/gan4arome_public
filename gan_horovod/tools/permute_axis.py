import numpy as np
import os
import glob

input_path='/home/mrmn/brochetc/scratch_link/Sandbox/merge_test/'

output_path='/home/mrmn/brochetc/scratch_link/Sud_Est/Baselines/IS_1_1.0_0_0_0_0_0_256_done/'

listing=glob.glob(input_path+'*sample*')
basepath=os.getcwd()
os.chdir(input_path)
if not os.path.exists('Permuted'):
    os.mkdir('Permuted')
os.chdir(basepath)
for filename in listing:
    name=filename[len(input_path):]
    M=np.load(filename,allow_pickle=True)
    new_sample=np.moveaxis(M,(0,1,2),(1,2,0))
    np.save(output_path+name,new_sample)

