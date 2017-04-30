'''
Created on Oct 26, 2015

@author: mjiang
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy.fftpack as scifft
import pyfits as fits
import pylab

from mathTools import *
from pyWavelet.wav1d import *
from pyWavelet.wav2d import *
import pyWavelet.waveTools as pm
from pyUtils import *
from pyProx import *

import sys
import os
import glob
import re


def DecGMCA(V,M,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,postProc=0,postProcImax=50,Kend=3.0,Ksig=3.0,positivityS=False,positivityA=False):
    '''
        Deconvolution GMCA algorithm to solve simultaneously deconvolution and Blind Source Separation (BSS)
        
        @param V: Visibilities, size: bands*pixels
        
        @param M: Matrix of mask, size: sources*pixels
        
        @param n: Number of sources
        
        @param Nx: number of rows of the source
        
        @param Ny: number of columns of the source
        
        @param Imax: Number of iterations
        
        @param epsilon: Parameter of regularization
        
        @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image
        
        @param wavelet: Wavelet option
        
        @param scale: Scales of wavelet decomposition
        
        @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
        
        @param deconv: Whether in the case of blurring, default False
        
        @param wname: Wavelet type.
        
        @param thresStrtg: Strategy to find the thresholding, valid just for parameter wavelet=True
        strategy=1 means to find the thresholding level based on the percentage of all coefficients larger than 3*sigma
        strategy=2 means in terms of scales, to find the thresholding level based on the percentage of coefficients larger than 3*sigma for each scale
        
        @param FTPlane: Whether the input data are in Fourier space
        
        @param fc: Cut-off frequency, fc is normalized to numerical frequency
        
        @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
        '''
    
    (Bd,P) = np.shape(V)
    #################### Matrix completion for initilize A ###########################
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
    
    elif Ndim == 2 and FTPlane:
        V_Hi = filter_Hi(np.reshape(V,(Bd,Nx,Ny)),Ndim,fc)
        V_Hi = np.reshape(V_Hi,(Bd,P))
    else:
        V_Hi = V
    
    
    if mask:
        MM = np.zeros_like(M)
        MM[np.abs(M)>=1e-4] = 1
        MM = np.real(MM)
        Xe = SVTCompletion(V,MM,n,0.5,500)
    elif deconv and not mask and FTPlane:
        Xe = V_Hi
    else:
        Xe = V
    
    (u,d,v) = LA.svd(Xe,full_matrices=False)
    A = np.abs(u[:,:n])
    #     R = np.dot(Xe,Xe.T)
    #     d,v = LA.eig(R)
    #     A = np.abs(v[:,0:n])
    
    #     A = np.copy(Ar)
    normalize(A)
    
    ############################ Save information of wavelet (Starlet and DCT only), ################################
    if wavelet:
        pm.trHead = ''
        if wname == 'starlet':
            gen2 = False
            pm.trHead = 'star'+str(int(Ndim))+'d_gen2' if gen2 else 'star'+str(int(Ndim))+'d_gen1'
        else:
            pm.trHead = wname+'_'+str(int(Ndim))+'d'
    
    if thresStrtg == 1:
        thIter = np.zeros((Imax,n))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,n,scale-1))
    
    #     Shat = np.random.randn(n,P) + 1j*np.random.randn(n,P)
    #     tau = 10.
    P_min = 0.05
    
    if mask or deconv:
        epsilon_log = np.log10(epsilon)
        epsilonF_log = np.log10(epsilonF)
    #     dtau = float(tau - tauF)/Imax
    
    print "Main cycle"
    
    for i in np.arange(Imax):
        if mask or deconv:
            epsilon_iter = 10**(epsilon_log - (epsilon_log - epsilonF_log)*float(i)/(Imax-1))       # Exponential linearly decreased
        #                 epsilon_iter = epsilon - (epsilon - epsilonF)/(Imax-1)*float(i)                # Linear decreased
        else:
            epsilon_iter = epsilon
        
        Shat = update_S(V,A,M,mask=mask,deconv=deconv,epsilon=epsilon_iter)
        Shat[np.isnan(Shat)]=0+0j
        #             Shat = update_S(Xe,A,M,mask=mask,epsilon=epsilon_iter)
        #         Shat = np.dot(LA.inv(np.dot(A.T,A)),np.dot(A.T,V))
        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))                    # In direct plane
            if positivityS:
                S[S<0] = 0
        else:
            S = np.real(np.reshape(Shat,(n,Nx,Ny)).squeeze())
            if positivityS:
                S[S<0] = 0
        
        ############################## For wavelet representation ##########################
        if wavelet:
            
            if wname == 'starlet':
                if Ndim == 1:
                    coarseScale = np.zeros((n,1,P))
                    wt = np.zeros((n,scale-1,P))                    # For starlet transform
                elif Ndim == 2:
                    coarseScale = np.zeros((n,1,Nx,Ny))
                    wt = np.zeros((n,scale-1,Nx,Ny))                    # For starlet transform
                for sr in np.arange(n):
                    if Ndim == 1:
                        wtTmp = star1d(S[sr], scale=scale, gen2=gen2,normalization=True)         # 1d Starlet transform
                    elif Ndim == 2:
                        wtTmp = star2d(S[sr], scale=scale, gen2=gen2,normalization=True)         # 2d Starlet transform
                    # Remove coarse scale
                    coarseScale[sr] = np.copy(wtTmp[-1])
                    wt[sr] = np.copy(wtTmp[:-1])
            
            elif wname == 'dct':
                wt = np.zeros((n,Nx,Ny)).squeeze()
                coarseScale = np.zeros((n,Nx,Ny)).squeeze()
                for sr in np.arange(n):
                    if Ndim == 1:
                        wt[sr] = scifft.dct(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Ny/16] = wt[sr,:Ny/16]
                        wt[sr,:Ny/16] = 0
                    elif Ndim == 2:
                        wt[sr] = dct2(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Nx/16,:Ny/16] = wt[sr,:Nx/16,:Ny/16]
                        wt[sr,:Nx/16,:Ny/16] = 0
            ################## Noise estimate####################
            if thresStrtg == 1:
                if wname == 'dct':
                    ##################### Use the finest scale of starlet to estimate noise level ##############################
                    fineScale = np.zeros((n,P))
                    for sr in np.arange(n):
                        if Ndim == 1:
                            wtTmp = star1d(S[sr], scale=4, gen2=False,normalization=True)         # 1d Starlet transform
                        elif Ndim == 2:
                            wtTmp = star2d(S[sr], scale=4, gen2=False,normalization=True)         # 2d Starlet transform
                        fineScale[sr] = wtTmp[0].flatten()
                    sig = mad(fineScale)
                else:
                    wt = np.real(np.reshape(wt,(n,np.size(wt)/n)).squeeze())
                    sig = mad(wt[:,:P])
            #                     sig = np.array([0.01,0.01])
            elif thresStrtg == 2:
                sig = np.zeros((n,scale-1))
                for sr in np.arange(n):
                    #                 sig = mad(wt[:,:P])                                             # For starlet transform
                    sig[sr] = mad(wt[sr])
            
            thTab = find_th(wt,P_min,sig,i,Imax,strategy=thresStrtg,Ksig=Kend)
            #             thTab = tau*sig
            thIter[i] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
                hardTh(wt,thTab,weights=None,reweighted=False)
            #                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(n):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
            #                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)
            
            #             wt = np.reshape(wt,(n,np.size(wt)/(n*P),P))             # For undecimated wavelet transform
            
            if wname == 'starlet':
                if thresStrtg == 1:
                    wt = np.reshape(wt,(n,scale-1,Nx,Ny)).squeeze()
            elif wname == 'dct':
                wt = wt.reshape((n,Nx,Ny)).squeeze()
            
            if wname == 'starlet':
                for sr in np.arange(n):
                    wtTmp = np.concatenate((wt[sr],coarseScale[sr]),axis=0)
                    if Ndim == 1:
                        S[sr] = istar1d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform
                    elif Ndim == 2:
                        S[sr] = istar2d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform

            elif wname == 'dct':
                for sr in np.arange(n):
                    if Ndim == 1:
                        wt[sr,:Ny/16] = coarseScale[sr,:Ny/16]
                        S[sr] = scifft.dct(wt[sr],type=3,norm='ortho')                  # Inverse 1d dct transform
                    elif Ndim == 2:
                        wt[sr,:Nx/16,:Ny/16] = coarseScale[sr,:Nx/16,:Ny/16]
                        S[sr] = dct2(wt[sr],type=3,norm='ortho')
        #################### For non-wavelet representation ###########################
        else:
            sig = mad(S)
            #             thTab = tau*sig
            thTab = find_th(S,P_min,sig,i,Imax,strategy=thresStrtg,Ksig=Kend)
            thIter[i] = thTab
            #         print thTab
            hardTh(S,thTab,weights=None,reweighted=False)
        
        index = check_zero_sources(A,Shat.reshape((n,P)))
        if len(index) > 0:
            reinitialize_sources(V,Shat,A,index)
        else:
            if FTPlane:
                Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            else:
                S = np.reshape(S,(n,P))
        
        if FTPlane:
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
                ####################################### Update A ######################################
                A = update_A(V_Hi,Shat_Hi,M,mask=mask,deconv=deconv)
            else:
                Shat = np.reshape(Shat,(n,P))
                A = update_A(V,Shat,M,mask=mask,deconv=deconv)
        else:
            A = update_A(V,S,M,mask=mask,deconv=deconv)
        A[np.isnan(A)]=0
        A = np.real(A)
        if positivityA:
            A[A<0] = 0
        normalize(A)
        
        
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S:"
        print LA.cond(np.reshape(S,(n,P))), 'S'
    
    if Ndim == 2:
        S = np.reshape(S,(n,Nx,Ny))
    ####################### Ameliorate the estimation of the sources ##########################
    if postProc == 1:
        print 'Post processing to ameliorate the estimate S:'
        S,thIter=update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax=postProcImax,tau=0.0,eta=0.5,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivityS)
    elif postProc == 2:
        print 'Post processing to ameliorate the estimate S and estimate A:'
        inImax1 = 50
        inImax2 = 1
        tau = 0.0
        mu2 = 0.0
        eta = 0.5
        for i in np.arange(postProcImax):
            S,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax=inImax1,tau=tau,eta=eta,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivityS)
            
            if FTPlane:
                Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
                if wavelet:
                    ################# Don't take the low frequency band into account ######################
                    Shat_Hi = filter_Hi(Shat,Ndim,fc)
                    if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                        Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                        Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                    Shat_Hi = np.reshape(Shat_Hi,(n,P))
                else:
                    Shat = np.reshape(Shat,(n,P))
            else:
                S = np.reshape(S,(n,P))
            
            
            if FTPlane:
                if wavelet:
                    A = update_A_prox(V_Hi,A,Shat_Hi,M,inImax2,mu2,mask=mask,positivity=positivityA)
                else:
                    A = update_A_prox(V,A,Shat,M,inImax2,mu2,mask=mask,positivity=positivityA)
            else:
                A = update_A(V,A,S,M,inImax2,mu2,mask=mask,positivity=positivityA)
        
        S = np.real(np.reshape(S,(n,Nx,Ny)).squeeze())
    
    return (S,A)



##########################################################################
################ GMCA using proximal operators ###########################
##########################################################################

def GMCA_PALM(V,M,n,Nx,Ny,Imax,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,Ksig=0.6,positivity=False,FISTA=False,S=None,A=None):
    (Bd,P) = np.shape(V)
    
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
    
        
    if (A is None) and (S is None):
        if mask:
            MM = np.zeros_like(M)
            MM[np.abs(M)>=1e-4] = 1
            MM = np.real(MM)
            Xe = SVTCompletion(V,MM,n,1.5,200)
        elif deconv and not mask and FTPlane:
            Xe = V_Hi
        else:
            Xe = V
            
        R = np.dot(Xe,Xe.T)
        d,v = LA.eig(R)
        A = np.abs(v[:,0:n])
        normalize(A)
        
        if FTPlane:
            Shat = np.reshape(np.dot(A.T,Xe),(n,Nx,Ny)).squeeze()
            S = np.real(ifftNd1d(Shat,Ndim))
        else:
            S = np.dot(A.T,Xe)

    Imax1 = 1
    ImaxInt = 5
    Imax2 = 1
    mu1 = 0.0
    muInt = 0.0
    mu2 = 0.0
    eta = 0.0
    
    if FISTA:
        t = 1.0         # FISTA parameter

    i = 0
    tol = 1e-8
    err = 1.
    while i < Imax and err > tol:
#         S_n = update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=True,currentIter=i,globalImax=Imax)
        
#         S_n,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,eta,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,positivity=False,PALM=True,currentIter=i,globalImax=Imax)
        if i == 0:
            S_n,u,sig = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,currentIter=i,globalImax=Imax)
        else:
            S_n,u,sig = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,currentIter=i,globalImax=Imax,u=u,sig=sig)
        errS = (((S_n - S)**2).sum()) / ((S**2).sum())
        if FISTA:
            tn = (1.+np.sqrt(1+4*t*t))/2
            S = S_n + (t-1)/tn * (S_n-S)
            t = tn
        else:
            S = S_n
         
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
         
 
        if FTPlane:
            if wavelet:
                A_n = update_A_prox(V_Hi,A,Shat_Hi,M,Imax2,mu2,mask=mask,positivity=False)
            else:
                A_n = update_A_prox(V,A,Shat,M,Imax2,mu2,mask=mask,positivity=False)
        else:
            A_n = update_A(V,A,S,M,Imax2,mu2,mask=mask,positivity=False)
         
        errA = (((A_n - A)**2).sum())/((A**2).sum())
         
        A = A_n
#         err = errS
        err = max(errS,errA)
        i = i+1
        
        print "Iteration: "+str(i)
        print "Condition number of A:"
        print LA.cond(A)
        print "Condition number of S:"
        print LA.cond(S)
        print "Current error"
        print err
        
    ########### Refinement with the final thresholding ############## 
#     i = 0
#     err = 1.
#     Imax = 200
#     Imax1 = 1
#     Imax2 = 1
#     if FISTA:
#         t = 1.0
#     print "Refinement with the final thresholding"
#     while i < Imax and err > tol:
# #         S_n = update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False)
#              
# #         S_n,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,eta,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,positivity=False,PALM=False)
#         if i == 0:
#             S_n,u = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False)
#         else:
#             S_n,u = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,u=u)
#         errS = (((S_n - S)**2).sum()) / ((S**2).sum())
#         if FISTA:
#             tn = (1.+np.sqrt(1+4*t*t))/2
#             S = S_n + (t-1)/tn * (S_n-S)
#             t = tn
#         else:
#             S = S_n
#              
#         if FTPlane:
#             Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
#             if wavelet:
#                 ################# Don't take the low frequency band into account ######################
#                 Shat_Hi = filter_Hi(Shat,Ndim,fc)
#                 if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
#                     Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
#                     Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
#                 Shat_Hi = np.reshape(Shat_Hi,(n,P))
#             else:
#                 Shat = np.reshape(Shat,(n,P))
#         else:
#             S = np.reshape(S,(n,P))
#              
#      
#         if FTPlane:
#             if wavelet:
#                 A_n = update_A_prox(V_Hi,A,Shat_Hi,M,Imax2,mu2,mask=mask,positivity=False)
#             else:
#                 A_n = update_A_prox(V,A,Shat,M,Imax2,mu2,mask=mask,positivity=False)
#         else:
#             A_n = update_A(V,A,S,M,Imax2,mu2,mask=mask,positivity=False)
#              
#         errA = (((A_n - A)**2).sum())/((A**2).sum())
#              
#         A = A_n
#         err = max(errS,errA)
#         i = i+1
#              
#         print "Iteration: "+str(i)
#         print "Condition number of A:"
#         print LA.cond(A)
#         print "Condition number of S:"
#         print LA.cond(S)
#         print "Current error"
#         print err
    
    
    return (S,A) 

def GMCA_prox(V,M,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,postProc=True,Ksig=0.6,positivity=False):
    
    '''
    GMCA using proximal methods
    
    @param V: Input data, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @param n: Number of sources
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Maximum iterations
    
    @param epsilon: Tikhonov parameter 
    
    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    (Bd,P) = np.shape(V)
    
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
            
    elif Ndim == 2 and FTPlane:
        V1 = np.reshape(V,(Bd,Nx,Ny))
        V_Hi = filter_Hi(V1,Ndim,fc)
        V_Hi = np.reshape(V_Hi,(Bd,P))
    else:
        V_Hi = V
    
        
    if deconv and FTPlane:
        Xe = V_Hi
    elif mask:
        Xe = SVTCompletion(V,M,n,0.5,200)
    else:
        Xe = V
    
    (u,d,v) = LA.svd(V,full_matrices=False)
    A = u[:n,:].T
#     np.random.seed(5)
#     A = np.random.randn(Bd,n) + 0*1j*np.random.randn(Bd,n)
    normalize(A)
#     if wavelet:
#         gen2 = False
#         nsNorm(P, scale, gen2=gen2)
#     Shat = np.random.randn(n,P) + 1j*np.random.randn(n,P)
#     tau = 10.
    mu1 = 0.0
    eta = 0.5
    mu2 = 0.0
    inImax1 = 50
    inImax2 = 1
    S = np.zeros((n,P))
#     Shat = np.zeros((n,P)) + np.zeros((n,P))*1j
#     dtau = float(tau - tauF)/Imax
    for i in np.arange(Imax):
#         S,thIter = update_S_prox(V,A,S,M,Nx,Ny,Imax,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False)
#         S,thIter = update_S_prox(V,A,S,M,Nx,Ny,Imax,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False)
        
        S,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,inImax1,mu1,eta,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivity)
        
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
        

        if FTPlane:
            if wavelet:
                A = update_A_prox(V_Hi,A,Shat_Hi,M,inImax2,mu2,mask=mask,positivity=False)
#                 A = update_A(V_Hi,Shat_Hi,M,mask=mask,deconv=deconv)
            else:
                A = update_A_prox(V,A,Shat,M,inImax2,mu2,mask=mask,positivity=False)
        else:
            A = update_A(V,A,S,M,inImax2,mu2,mask=mask,positivity=False)
            
#         A = np.real(A)
#         normalize(A)    
        
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S hat:"
        print LA.cond(Shat), 'Shat'
    
#     S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))
    S = np.real(np.reshape(S,(n,Nx,Ny)).squeeze())
    return (S,A)
