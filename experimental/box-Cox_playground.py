#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:19:20 2022

@author: brochetc

Box-Cox transformation playground

"""

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

data_fn="/home/brochetc/lustre_link/PEARO_EURW1S40_Orography.npy"

Matrix_ds=np.load(data_fn)[120:376,540:924]

m=Matrix_ds.min()
print(m)
Matrix_ds=Matrix_ds-m+1.0
S=Matrix_ds.shape

bc_ds, lamda=st.boxcox(Matrix_ds.flatten())
print("lambda", lamda)
bc_ds=bc_ds.reshape(S)

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(1,2,1)
im=ax.imshow(Matrix_ds)
plt.colorbar(im)
ax=fig.add_subplot(1,2,2)
im2=ax.imshow(bc_ds)
plt.colorbar(im2)
plt.show()