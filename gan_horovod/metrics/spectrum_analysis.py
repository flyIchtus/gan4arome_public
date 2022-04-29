#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:17:55 2022

@author: brochetc

# DCT transform and spectral energy calculation routines


Include :
    -> 2D dct and idct transforms
    -> 'radial' energy spectrum calculation
    -> spectrum plotting configuration


"""

from scipy.fftpack import dct, idct,fft, ifft
import numpy as np
import matplotlib.pyplot as plt

################################## DCT ########################################

def dct2D(x):
    """
    2D dct transform for 2D (square) numpy array
    or for each channel of CxNxN numpy array
    
    """
    assert x.ndim in [2,3]
    if x.ndim==3:
        res=dct(dct(x.transpose((0,2,1)), norm='ortho').transpose((0,2,1)), norm='ortho')
    else :
        res=dct(dct(x.T, norm='ortho').T, norm='ortho')
    return res

def idct2D(f):
    """
    2D iverse dct transform for 2D (square) numpy array
    
    or for each channel of CxNxN numpy array
    """
    
    assert f.ndim in [2,3]
    if f.ndim==3:
        res=idct(idct(f.transpose(0,2,1), norm='ortho').transpose(0,2,1), norm='ortho')
    else :
        res=dct(dct(f.T,norm='ortho').T, norm='ortho')
    return res

def dct_var(x):
    """
    compute the bidirectional variance spectrum of the (square) numpy array x
    """
    N=x.shape[-1]
    
    fx=dct2D(x)
    Sigma=(1/N**2)*fx**2
        
    return Sigma


#################### FFT ######################################################

def fft2D(x):
    """
    2D FFT transform for 2D (square) numpy array
    or for each channel of CxNxN numpy array
    
    """
    assert x.ndim in [2,3]
    if x.ndim==3:
        res=fft(fft(x.transpose((0,2,1)), norm='ortho').transpose((0,2,1)), norm='ortho')
    else :
        res=fft(fft(x.T, norm='ortho').T, norm='ortho')
    return res


def ifft2D(x):
    
    """
    2D iverse fft transform for 2D (square) numpy array
    
    or for each channel of CxNxN numpy array
    """
    return 0
    
    


def radial_bin_dct(dct_sig, UniRad=None, Rad=None,Inds=None):
    """
    compute radial binning sum of dct_sig array
    Input :
        dct_sig : the signal to be binned (eg variance) : np array of square size
        UniRad : unique values of radii to be binned on (array)
        Rad : values of radii according to x,y int location
        Inds : indexes following the sie of dct_sig
            if the 3 latter are not provided, they are computed and returned
        
    Output :
        
        Binned_Sigma : binned dct signal along UniRad
        UniRad, Rad, Inds
        
    """
    N=dct_sig.shape[0]

    if Inds==None and UniRad==None:
        Inds=np.array([[[i,j] for i in range(N)]for j in range(N)])
        
        Rad=np.linalg.norm(Inds, axis=-1)/(N)
        UniRad=np.unique(Rad)[1:]
    
    
    Binned_Sigma=np.zeros(UniRad[UniRad<0.5].size)
    for i,r in enumerate(UniRad[UniRad<0.5]):
        #sum the contributions of dct_sig positions for which radius==r
        Binned_Sigma[i]=0.5*dct_sig[Rad==2*r-1].sum()+dct_sig[Rad==2*r].sum()+0.5*dct_sig[Rad==2*r+1].sum()
        
    return UniRad, Binned_Sigma, Inds, Rad

def radial_bin_fft(dct_sig, UniRad=None, Rad=None, Inds=None):
    return 0

def PowerSpectralDensity(x, UniRad=None, Rad=None, Inds=None):
    """
    collating previous functions
    """
    
    UniRad, Binned_Sigma, Inds, Rad=radial_bin_dct(dct_var(x).mean(axis=0),UniRad,Rad,Inds)
    return UniRad, Binned_Sigma, Inds, Rad

def PSD_wrap(x):
     UniRad, Binned_Sigma, Inds, Rad=PowerSpectralDensity(x)
     return UniRad[UniRad<0.5], Binned_Sigma

def plot_spectrum(rad, binned_spectrum, name, delta, unit):
        plt.plot((1/delta)*rad, binned_spectrum)
        plt.yscale('log')
        plt.xlabel('Wavenumber (km^{-1})')
        plt.ylabel(name+' ({})'.format(unit))
        plt.title('Power Spectral Density, '+name)
        plt.savefig('./PSD_'+name+'.png')
        plt.show()

#TODO : add parallel computing capabilities to dct /fft


    

if __name__=="__main__":
    ranges=[0.5,1.0,2.0,3.0]
    for r in ranges:
        print(r/0.25)
        t,p=np.ogrid[-r:r:0.01, -r:r:0.01]
        x=np.cos((2*np.pi/0.1)*np.sqrt(t**2+p**2))+np.cos((2*np.pi/0.0333333)*np.sqrt(t**2+p**2))
        x=x.reshape(1,x.shape[0], x.shape[1])
        UniRad, BS, Inds, Rad=PowerSpectralDensity(x)
        print(UniRad[BS.argmax()], 2*0.01/UniRad[BS.argmax()])
        plot_spectrum(UniRad[UniRad<0.5], BS, 'test', 1.0, "x")
        
    
        