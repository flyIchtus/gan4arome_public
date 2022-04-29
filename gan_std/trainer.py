#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:58:19 2022

@author: brochetc

Trainer class and useful functions

"""

import torch
import torch.optim as optim
import GAN_logic as GAN
import metrics as METR
import plotting_functions as plotFunc
import DataSet_Handler as DSH
from numpy import load as LoadNumpy


###################### Scheduler choice function ##############################

def AllocScheduler(sched, optimizer, gamma):
    if sched=="exp":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched=="linear" :
        lambda0= lambda epoch: 1.0/(1.0+gamma*epoch)
        return optim.lr_scheduler.LambdaLR(optimizer, lambda0)
    else:
        print("Scheduler set to None")
        return None


###############################################################################
######################### Trainer class #######################################
###############################################################################


class Trainer():
    """
    main training class
    
    inputs :
        
        config -> model, training, optimizer, dirs, file names parameters 
                (see constructors and main.py file)
        device -> the device used, no default
        optimizer -> the type of optimizer (if working with LA-Minimax)
        test_metrics -> list of names of metrics, located in the METR module
        criterion -> specific metric name (in METR) used as save/stop criterion
        metric_verbose -> check if metric long name should be printed @ test time
                default to True
    
    outputs :
        
        models saved along with optimizers at regular steps
        plots and scores logged regularly
        return of metrics and losses lists
    """
    def __init__(self, config, device, criterion,test_metrics={},\
                 LA_optimizer=False,metric_verbose=True ):
        
        self.config=config
        self.device=device
        self.instance_flag=False
        self.LA_optimizer=LA_optimizer
        self.test_metrics=test_metrics
        self.criterion=criterion
        self.metric_verbose=metric_verbose
        
    def instantiate(self, modelG, modelD, load_optim=False):
        

        modelG.to(self.device)
        modelD.to(self.device)
        
        if load_optim:
            self.optim_G=torch.load(self.config.model_save_path+'/optimGen_{}'.format(self.config.pretrained_model))
            self.optim_D=torch.load(self.config.model_save_path+'/optimDisc_{}'.format(self.config.pretrained_model))
        else:        
            self.optim_G=optim.Adam(modelG.parameters(), lr=self.config.lr_G, 
                           betas=(self.config.beta1_G, self.config.beta2_G))
            self.optim_D=optim.Adam(modelD.parameters(), lr=self.config.lr_D,
                           betas=(self.config.beta1_D, self.config.beta2_D))
        
            self.scheduler_G=AllocScheduler(self.config.lrD_sched,\
                                            self.optim_G, self.config.lrD_gamma)
            self.scheduler_D=AllocScheduler(self.config.lrG_sched,\
                                            self.optim_D, self.config.lrG_gamma)



        ##### creating algorithm parameters and metrics to be logged
        # metrics list is absolutely agnostic, so a rough indexing { ind : name}
        # as basis for test_metrics seems good practice here
        
        self.lossD_list=[]
        self.lossG_list=[]
        self.Metrics_Lists=[[] for metr in self.test_metrics.keys()]
        self.crit_List=[]
        
        
        self.batch_size=self.config.batch_size
        self.n_dis=self.config.n_dis
        self.test_step=self.config.test_step
        
        
        with open(self.config.log_path+'/metrics.csv','a') as file:
            file.write('Step,loss_D,loss_G,criterion')
            for mname in self.test_metrics.values():
                file.write(','+mname)
            file.write('\n')
            file.close()
            
        self.plot_step=self.config.plot_step
        
        ########## Choice of algorithm #######################################
        
        if self.config.train_type=="wgan-gp":
            self.D_step=GAN.Discrim_Wasserstein(self.config.lamda_gp).Discrim_Step
            self.G_step=GAN.Generator_Step_Wasserstein
        elif self.config.train_type=="vanilla":
            self.D_step=GAN.Discrim_Step
            self.G_step=GAN.Generator_Step
        elif self.config.train_type=="wgan-hinge":
            self.D_step=GAN.Discrim_Step_Hinge
            self.G_step=GAN.Generator_Step_Wasserstein
        
        
        ########## Prepare data pipeline #####################################

        print("Loading data")
        Dl_instance=DSH.ISData_Loader(self.config.data_path,self.batch_size, 
                                      self.device)
        self.dataloader=Dl_instance.loader(None,None)
        
        #######################################################################
        
        print("Data loaded, Trainer instantiated")
        self.instance_flag=True
        
        
    def fit_(self, modelG, modelD):
        
        assert self.instance_flag #security flag avoiding meddling with uncomplete init
        
        
        modelG.train()
        modelD.train()
        Step=0
        for e in range(self.config.epochs_num):
        
            print("----------------------------------------------------------")
            print("Epoch n°", e+1, "/",self.config.epochs_num)
            
            for step, (samples, imp,pos) in enumerate(self.dataloader):
                Step=e*len(self.dataloader)+step
                
                samples,imp,pos=samples.to(self.device), imp.to(self.device),\
                pos.to(self.device)
                
                ############################### Discriminator Updates #########
                
                
                loss=self.D_step(samples, modelD, modelG, self.optim_D,\
                                              self.n_dis, self.device)
                
                ############################# Store and print value ###########
                
                
                if step==len(self.dataloader)-1:
                    self.lossD_list+=[loss.item()]
                if step%400==0: 
                    print("Percentage done :", str(100*step/len(self.dataloader))[:4])
                    print("Discrim loss",loss.item())
                        
                ############################ Generator Update #################
                
                loss=self.G_step(samples, modelD, modelG,\
                            self.optim_G, self.device)
            

                ########################### Store and print value #############
                
                if step==self.N_split-1:
                    self.lossG_list+=[loss.item()]
                if step%400==0:
                        print("Gen loss", loss.item())
        
                ################ advancing schedulers for learning rate #######
        
                if self.scheduler_G!=None :
                    self.scheduler_G.step()
                if self.scheduler_D!=None :
                    self.scheduler_D.step()

                ################# testing sample at the test step #############
                
                
                if self.test_step>0 and Step%self.test_step==0 :
                    modelG.train=False
                    modelD.train=False
                    
                    z=torch.empty((self.config.test_samples,modelG.nz)).normal_().to(self.device)
                    batch=modelG(z).detach().cpu().numpy()
                    test_samples=LoadNumpy(self.config.data_path+'/Test_samples.npy')
                    for metr in self.test_metrics.keys():
                        name=self.test_metrics[metr]
                        metric=getattr(METR,name)
                        
                        res=metric(batch,test_samples)
                        self.Metrics_Lists[metr].append(res)
                        
                        
                        if self.metrics_verbose :
                            print(metric.long_name, res)
                
                    crit=getattr(METR,self.criterion)
                    res=crit(test_samples,batch)
                    self.crit_List.append(res)
                
                    if res==min(self.crit_List): #saving models and optimizers
                        print("Best score until now, Saving")
                        torch.save(modelD.state_dict(), \
                                   self.config.model_save_path+\
                                   "bestdisc_{}".format(Step))
                        torch.save(modelG.state_dict(), \
                                   self.config.model_save_path+\
                                   "/bestgen_{}".format(Step))
                        torch.save(self.optim_D.state_dict(), \
                                   self.config.model_save_path+\
                                   "/optimDisc_{}".format(Step))
                        torch.save(self.optim_G.state_dict(), \
                                   self.config.model_save_path+\
                                   "/optimGen_{}".format(Step))
                    modelG.train()
                    modelD.train()
            
                ############### plotting distribution at plot step ############
                if self.plot_step>0 and Step%self.plot_step==0:
                     print('Plotting')
                     
                     modelG.train=False
                     z=torch.empty((self.config.plot_samples,modelG.nz)).normal_().to(self.device)
                     batch=modelG(z)
                     modelG.train=True
                     plotFunc.online_sample_plot(batch,self.config.n_samples,\
                                            e+1,self.config.var_names, \
                                            self.config.sample_path)
                    
                ############ END OF STEP ######################################
                
            ################ END OF EPOCH #####################################
            
            ############# logging experiment data at end of epoch #############
        
            data=[Step,self.lossD_list[-1], self.lossG_list[-1],\
                  self.crit_List[-1]]+[self.Metric_List[i][-1]\
                                for i in range(len(self.Metric_List))]
            data_to_write='%d,%.6f,%.6f,%.6f'
            for metr in self.test_metrics.keys():
                data_to_write=data_to_write+'%.6f'
            data_to_write=data_to_write %tuple(data)
        
            with open(self.config.log_path+'metrics.csv','a') as file:
                file.write(data_to_write)
                file.write('\n')
                file.close()
            
        ##########################
        return self.lossD_list, self.lossG_list, self.crit_List, self.Metric_List
###############################################################################