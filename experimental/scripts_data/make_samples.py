import os
import numpy as np
from glob import glob
from random import randint
import csv

data_dir='/home/mrmn/brochetc/scratch_link/GAN_2D_10/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
N_samples=2048

os.chdir(data_dir)
listing=glob('*_sample*.npy')
N_reservoir=len(listing)
print(os.getcwd())
print(N_reservoir, 'possible samples')
print('Taking only (random)', N_samples)
Test_samples=np.zeros((N_samples,5,256,256))
selected=[]
i=0
while i < N_samples:
    if i%100==0: print(i)
    m=randint(0, N_reservoir-1)
    if '_sample'+str(m)+'.npy' not in selected:
        M=np.load('_sample'+str(m)+'.npy')
        Test_samples[i,:,:,:]=M
        selected.append('_sample'+str(m)+'.npy')
        i+=1
np.save('Test_samples.npy', Test_samples)
with open('_test_indexes.csv', 'w', newline='') as f:
    writer=csv.writer(f)
    writer.writerows(selected)
    f.close()

    

