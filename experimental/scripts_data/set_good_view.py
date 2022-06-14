import numpy as np
import os
import matplotlib.pyplot as plt

output_path='/home/mrmn/brochetc/scratch_link/Sud_Est/Baselines/IS_1_1.0_0_0_0_0_0_256_done/'
M=np.load(output_path+'_sample0.npy')
print(M.shape)

M1=M[-1,205:205-128:-1,55:55+128]

print(M1.shape)

M2=M1.transpose()

plt.imshow(M1)
plt.show()
plt.imshow(M2)
plt.show()
