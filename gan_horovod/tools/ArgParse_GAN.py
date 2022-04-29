#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:14:59 2021

@author: brochetc

#### en chantier ! #####

"""

import argparse

parser=argparse.ArgumentParser(description="Choices of architectures\
                               types of training, \
                               regularizations, optimizers, update rules\
                               initializations \
                               for GANs on MNIST")

parser.add_argument("-mlp","--Use_Of_MLP", action="store_true", help="use of mlp\
                     : default architecture is DCGAN")

parser.add_argument("train_type",choices=["vanilla", "hinge","wGAN", "wGAN-GP"])

parser.add_argument("-v", "--verbose", choices=[0,1,2], default=0)

groupSN=parser.add_argument_group()
groupSN.add_argument("-sn_d", "--specNormD", action="store_true", default=False)
groupSN.add_argument("-sn_g", "--specNormG", action="store_true", default=False)

groupSA=parser.add_argument_group()
groupSA.add_argument("-SA_d","--self_Attention_D", type=int, default=-1)
groupSA.add_argument("-SA_g","--self_Attention_G", type=int, default=-1)

args=parser.parse_args()

if args.self_Attention_D + args.self_Attention_G>-2:
    import SAGAN as GAN
elif args.mlp:
    import GAN_MNIST_vanilla as GAN
else :
    import GAN_MNIST_DCGAN as GAN

model_D=GAN.Discriminator()
model_G=GAN.Generator()
    
  
    
if x=="dc_sn" :
    y=input("SN on Disc ? (y/n) \t")
    if y=="y":
        model_D=GAN.Discriminator_SN()
    elif y=="n":
        model_D=GAN.Discriminator()
    else:
        raise ValueError('Please choose SN or not')
    
    y=input("SN on Gen ? (y/n) \t")
    if y=="y":
        model_G=GAN.Generator_SN(latent_dim=latent_dim)
    elif y=="n":
        model_G=GAN.Generator(latent_dim=latent_dim)
    else:
        raise ValueError('Please choose SN or not')
elif x=="dc_sa":
    y=input("SN on Disc ? (y/n) \t")
    SA_depth=int(input("Depth of Self Attention on Disc ?  (int) \t"))
    if y=="y":
        model_D=GAN.SA_Discriminator(SA_depth=SA_depth)
        model_D.apply(GAN.Add_spectral_Norm)
    elif y=="n":
        model_D=GAN.SA_Discriminator(SA_depth=SA_depth)
    else:
        raise ValueError('Please choose SN or not')
    
    y=input("SN on Gen ? (y/n) \t")
    SA_depth=int(input("Depth of Self Attention on Gen ?  (int) \t"))

    if y=="y":
        model_G=GAN.SA_Generator(latent_dim=latent_dim, SA_depth=SA_depth)
        model_G.apply(GAN.Add_spectral_Norm)
    elif y=="n":
        model_G=GAN.SA_Generator(latent_dim=latent_dim, SA_depth=SA_depth)
    else:
        raise ValueError('Please choose SN or not')