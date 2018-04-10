'''
Created on Oct 26, 2015

@author: mjiang
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy.fftpack as scifft
import astropy.io.fits as fits
import pylab

from mathTools import *
from pyWavelet.wav1d import *
from pyWavelet.wav2d import *
import pyWavelet.starlet_utils as starlet
import pyWavelet.waveTools as pm
from boost_utils import *
from boost_Prox import *

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
    
    @param epsilon: Parameter of the initial regularization 
    
    @param epsilonF: Parameter of the final regularization 
    
    @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param mask: Whether in the case of masking
    
    @param deconv: Whether in the case of blurring
    
    @param wname: Wavelet type, default is starlet
    
    @param thresStrtg: Strategy to find the thresholding, valid just for parameter wavelet=True, default is 2
    strategy=1 means to find the thresholding level based on the percentage of all coefficients larger than Ksig
    strategy=2 means in terms of scales, to find the thresholding level based on the percentage of coefficients larger than Ksig for each scale
    
    @param FTPlane: Whether the input data are in Fourier space, default is true
    
    @param fc: During update of A stage, the cut-off frequency of the high-pass filter, fc is normalized to numerical frequency, default is 1./16
    
    @param logistic: Whether using logistic function for the high-pass filter, default is false 
    
    @param postProc: Whether using refinement step to ameliorate the estimate, default is 0
    
    @param postProcImax: Number of iterations for the refinement step, default is 50
    
    @param kend: The ending threshold, default is 3
    
    @param ksig: K-sigma, level of threshold
    
    @param positivityS: Positivity constraint of S, default is false
    
    @param positivityA: Positivity constraint of A, default is false
    
    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    
    (Bd,P) = np.shape(V)
    #################### Eliminate low-frequency of the data for the use of update A ###########################    
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
        #################### Matrix completion for initilize A ###########################
        MM = np.zeros_like(M)
        MM[np.abs(M)>=1e-4] = 1         # In case of not perfect mask
        MM = np.real(MM)
        Xe = SVTCompletion(V,MM,n,0.5,100)
    elif deconv and not mask and FTPlane:
        Xe = V_Hi                       # For convolution case, better use high-frequency information to initialize A
    else:
        Xe = V
    
#     (u,d,v) = LA.svd(Xe,full_matrices=False)
#     A = np.abs(u[:,:n])   
    R = np.dot(Xe,Xe.T)
    d,v = LA.eig(R)
    if positivityA:
        A = np.abs(v[:,0:n])
    else:
        A = v[:,0:n]
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
        
#     tau = 10.
    P_min = 0.05                # Initial percentage for adaptive thresholding function
    
    if mask or deconv:
        epsilon_log = np.log10(epsilon)
        epsilonF_log = np.log10(epsilonF)
        
    print "Main cycle" 
    
    for i in np.arange(Imax):
        if mask or deconv:
            epsilon_iter = 10**(epsilon_log - (epsilon_log - epsilonF_log)*float(i)/(Imax-1))       # Exponential linearly decreased
#                 epsilon_iter = epsilon - (epsilon - epsilonF)/(Imax-1)*float(i)                # Linear decreased
        else:
            epsilon_iter = epsilon
 
        Shat = UpdateS(V.reshape((Bd,Nx,Ny)),M.reshape((Bd,Nx,Ny)).astype('double'),A,epsilon=epsilon_iter)
        Shat[np.isnan(Shat)]=0+0j           # To avoid bad value
        
        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))                    # Back to image plane
        else:
            S = np.real(np.reshape(Shat,(n,Nx,Ny)).squeeze())   
            
        ############################## For wavelet representation ########################## 
        if wavelet:           
            
            if wname == 'starlet':                
                if Ndim == 1:
                    coarseScale = np.zeros((n,1,P))
                    wt = np.zeros((n,scale-1,P))                    # For starlet transform
                    # N sources starlet transform
                    wtTmp = starlet.Nstar1d(S,scale=scale,gen2=gen2,normalization=True)
                elif Ndim == 2:
                    coarseScale = np.zeros((n,1,Nx,Ny))
                    wt = np.zeros((n,scale-1,Nx,Ny))                    # For starlet transform
                    # N sources starlet transform
                    wtTmp = starlet.Nstar2d(S,scale=scale,gen2=gen2,normalization=True)                   
                
                coarseScale = np.copy(wtTmp[:,-1])      # Store coarse scale                    
                wt = np.copy(wtTmp[:,:-1])
                     
            elif wname == 'dct':
                wt = np.zeros((n,Nx,Ny)).squeeze()
                coarseScale = np.zeros((n,Nx,Ny)).squeeze()
                for sr in np.arange(n):
                    if Ndim == 1:
                        wt[sr] = scifft.dct(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Ny/16] = wt[sr,:Ny/16]              # Store coarse scale, can be do better in future
                        wt[sr,:Ny/16] = 0
                    elif Ndim == 2:
                        wt[sr] = dct2(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Nx/16,:Ny/16] = wt[sr,:Nx/16,:Ny/16]        # Store coarse scale, can be do better in future
                        wt[sr,:Nx/16,:Ny/16] = 0
            ################## Noise estimate using MAD estimator ####################
            if thresStrtg == 1:
                if wname == 'dct':
                    ##################### Use the finest scale of starlet to estimate noise level ##############################
                    fineScale = np.zeros((n,P))
                    if Ndim == 1:
                        wtTmp = starlet.Nstar1d(S,scale=scale,gen2=gen2,normalization=True)
                    elif Ndim == 2:
                        wtTmp = starlet.Nstar2d(S,scale=scale,gen2=gen2,normalization=True)
                    fineScale = wtTmp[:,0].reshape((n,P))
                    sig = mad(fineScale)
                else:
                    wt = np.real(np.reshape(wt,(n,np.size(wt)/n)).squeeze())
                    sig = mad(wt[:,:P])
            elif thresStrtg == 2:
                sig = np.zeros((n,scale-1))
                for sr in np.arange(n):
                    sig[sr] = mad(wt[sr]) 
            ################## Compute threshold for the current iteration ####################
            ######## Thresholding in terms of percentage of significant coefficients ##########
            thTab = find_th(wt,P_min,sig,i,Imax,strategy=thresStrtg,Ksig=Kend)
#             thTab = tau*sig
            thIter[i] = thTab
                       
            if thresStrtg == 1:
                hardTh(wt,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(n):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)
            
            ################## Reconstruction of S from wavelet coefficients ####################
            if wname == 'starlet':
                if thresStrtg == 1:
                    wt = np.reshape(wt,(n,scale-1,Nx,Ny)).squeeze()
            elif wname == 'dct':
                wt = wt.reshape((n,Nx,Ny)).squeeze()
            
            if wname == 'starlet':
                wtTmp = np.concatenate((wt,coarseScale[:,np.newaxis]),axis=1)
                if Ndim == 1:
                    S = starlet.Nistar1d(wtTmp, gen2=gen2, normalization=True)        # Inverse N sources starlet transform
                elif Ndim == 2:
                    S = starlet.Nistar2d(wtTmp, gen2=gen2, normalization=True)        # Inverse N sources starlet transform  
                        
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
            sig = mad(S)            # To estimate noise
#             thTab = tau*sig 
            ################## Compute threshold for the current iteration ####################
            ######## Thresholding in terms of percentage of significant coefficients ##########            
            thTab = find_th(S,P_min,sig,i,Imax,strategy=thresStrtg,Ksig=Kend)
            thIter[i] = thTab
#         print thTab
            hardTh(S,thTab,weights=None,reweighted=False)
        
        if positivityS:
            S[S<0] = 0
        
        index = check_zero_sources(A,Shat.reshape((n,P)))
        if len(index) > 0:
            reinitialize_sources(V,Shat,A,index)
        else:
            if FTPlane:
                Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            else:
                S = np.reshape(S,(n,P))
            
        ########################################## Update A ###########################################
        if FTPlane:
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
                A = UpdateA(V_Hi.reshape((Bd,Nx,Ny)),M.reshape((Bd,Nx,Ny)).astype('double'),Shat_Hi)
            else:
                Shat = np.reshape(Shat,(n,P))
                A = update_A(V,Shat,M,mask=mask,deconv=deconv)                
        else:
            A = update_A(V,S,M,mask=mask,deconv=deconv)
        A[np.isnan(A)]=0            # To avoid bad value
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
    ####################### Refinement step to ameliorate the estimates ##########################
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

def update_S_prox(V,A,S,M,Nx,Ny,Imax,mu,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,FISTA=False):
    '''
    Proximal method (Forward-Backward) to estimate sources with sparsity constraint in synthesis framework
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param S: Sources to update, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Number of iterations
    
    @param mu: Parameter of gradient step
    
    @param Ksig: k-sigma for the thresholding
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far 
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param FISTA: Whether to use FISTA algorithm
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
            coarseScale = np.zeros((N,1,P))
            wtTmp = np.zeros((N,scale,P))
            wt = np.zeros((N,scale-1,P))
            for sr in np.arange(N):
                wtTmp[sr] = star1d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)
#                 coarseScale[sr] = np.copy(wtTmp[sr,-1])                    
#                 wt[sr] = np.copy(wtTmp[sr,:-1])
    
    if thresStrtg == 1:
        thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,N,scale-1))
     
    Ndim = 1
    if FISTA:
        t = 1.0         # FISTA parameter
    it = 0
    var_rsd = 0
    errTab = np.zeros(Imax)
    
    while it<Imax:
        if wavelet:
            if wname == 'starlet':
                for sr in np.arange(N):
                    Stmp[sr] = adstar1d(wtTmp[sr], fast=True, gen2=gen2, normalization=True)
                if FTPlane:
                    Stmphat = fftNd1d(Stmp,Ndim)                    # In direct plane
                else:
                    Stmphat = Stmp
                
                rsd = V - M*np.dot(A,Stmphat)
                var_rsdN = np.var(rsd)
                rsd1 = np.real(ifftNd1d(np.dot(A.conj().transpose(),rsd),Ndim))
                rsdTr = np.zeros((N,scale,P))
                for sr in np.arange(N):
                    rsdTr[sr] = star1d(rsd1[sr],scale=scale, fast=True, gen2=gen2,normalization=True)
                wtTr = np.copy(rsdTr[:,:-1,:])
                
                wt_n = wt + mu*rsdTr
                coarseScale = wt_n[:,-1,:]
                wt_n = np.delete(wt_n,-1,axis=1)
            ################## Noise estimate####################
                if thresStrtg == 1:
                    wtTr = np.real(np.reshape(wtTr,(N,np.size(wt)/N)).squeeze())
                    sig = mad(wtTr[:,:P])
#                     sig = np.array([0.01,0.01])
                elif thresStrtg == 2:
                    sig = np.zeros((N,scale-1))
                    for sr in np.arange(N):
    #                 sig = mad(wt[:,:P])                                             # For starlet transform
                        sig[sr] = mad(wtTr[sr]) 
            
                thTab = Ksig*sig
                thIter[it] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
                hardTh(wt_n,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(N):
                    hardTh(wt_n[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)            
            wt_n = np.reshape(wt_n,(N,np.size(wt)/(N*P),P))             # For undecimated wavelet transform
            
            if FISTA:
                tn = (1.+np.sqrt(1+4*t*t))/2
                wt1 = wt_n + (t-1)/tn * (wt_n-wt)
                t = tn
            else:
                wt1 = np.copy(wt_n)
                
            for sr in np.arange(N):
                wtTmp[sr] = np.concatenate((wt1[sr],coarseScale[sr]),axis=0)
            wt = np.copy(wt_n)
            err = abs(var_rsdN-var_rsd)  
            var_rsd = var_rsdN
            errTab[it] = err
        
        it += 1
        print "Iteration:"+str(it)
        print "Current error:"+str(err)
    
    if wavelet:
        if wname == 'starlet':
            for sr in np.arange(N):
                S[sr] = adstar1d(wtTmp[sr], fast=True, gen2=gen2, normalization=True)
        
    return (S,thIter)

def update_S_prox_anal(V,A,S,M,Nx,Ny,Imax,Imax1,mu,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False):
    '''
    Proximal method (Forward-Backward) to estimate sources with sparsity constraint in analysis framework
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param S: Sources to update, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax1: Number of global iterations
    
    @param Imax2: Number of sub-iterations
    
    @param mu: Parameter of gradient step
    
    @param mu1: Parameter of sparsity constraint step
    
    @param Ksig: k-sigma for the thresholding
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far 
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @param FISTA: Whether to use FISTA algorithm
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))
            wtTmp = np.zeros((N,scale,P))
                
    if thresStrtg == 1:
        thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,N,scale-1))
     
    Ndim = 1
    it = 0
    var_rsd = 0
    errTab = np.zeros(Imax)
    
    while it<Imax:
        Shat = fftNd1d(S,Ndim)
        rsd = V - M*np.dot(A,Shat)
        var_rsdN = np.var(rsd)
        rsd1 = np.real(ifftNd1d(np.dot(A.conj().transpose(),M*rsd),Ndim))
        S_n = S + mu*rsd1
        it_int = 0
        ############### Subiteration #################
        while it_int < Imax1:
            if wavelet:
                if wname == 'starlet':
                    for sr in np.arange(N):
                        Stmp[sr] = adstar1d(wtTmp[sr], fast=True, gen2=gen2, normalization=True)
                    
                    rsdInt = S_n - Stmp
                    rsdIntTr = np.zeros((N,scale,P))
                    for sr in np.arange(N):
                        rsdIntTr[sr] = star1d(rsdInt[sr],scale=scale, fast=True, gen2=gen2,normalization=True)    
                    wtTmp_n = wtTmp + mu1*rsdIntTr
                    coarseScale = np.copy(wtTmp_n[:,-1])
                    coarseScale = coarseScale[:,np.newaxis,:]
                    wtTmp_n = np.delete(wtTmp_n,-1,axis=1)
                    rsdIntTr = np.delete(rsdIntTr,-1,axis=1)
                    wtTmp_n1 = np.copy(wtTmp_n)
                ################## Noise estimate####################
                    if thresStrtg == 1:
                        rsdIntTr = np.real(np.reshape(rsdIntTr,(N,np.size(rsdIntTr)/N)).squeeze())
                        sig = mad(rsdIntTr[:,:P])
    #                     sig = np.array([0.01,0.01])
                    elif thresStrtg == 2:
                        sig = np.zeros((N,scale-1))
                        for sr in np.arange(N):
        #                 sig = mad(wt[:,:P])                                             # For starlet transform
                            sig[sr] = mad(rsdIntTr[sr]) 
                
                    thTab = Ksig*mu1*sig
                    thIter[it] = thTab                
                    #### Thresholding in terms of percentage of significant coefficients ######
                    if thresStrtg == 1:
#                         hardTh(wtTmp_n,thTab,weights=None,reweighted=False)
                        softTh(wtTmp_n,thTab,weights=None,reweighted=False)
                    elif thresStrtg == 2:
                        for sr in np.arange(N):
#                             hardTh(wtTmp_n[sr],thTab[sr],weights=None,reweighted=False)
                            softTh(wtTmp_n[sr],thTab[sr],weights=None,reweighted=False)            
                    wtTmp_n = wtTmp_n1 - wtTmp_n
                    wtTmp_n = np.reshape(wtTmp_n,(N,np.size(wtTmp_n)/(N*P),P))             # For undecimated wavelet transform
#                     wtTmp_n = np.concatenate((wtTmp_n,coarseScale),axis=1)
                    wtTmp_n = np.concatenate((wtTmp_n,np.zeros((N,1,P))),axis=1)
                    wtTmp = np.copy(wtTmp_n)
            it_int += 1
            
        if wavelet:
            if wname == 'starlet':
                for sr in np.arange(N):
                    Stmp[sr] = adstar1d(wtTmp[sr], fast=True, gen2=gen2, normalization=True)
        
        S_n = S_n - Stmp            
        S = np.copy(S_n)
        err = abs(var_rsdN-var_rsd)  
        var_rsd = var_rsdN
        errTab[it] = err
        
        it += 1
        print "Iteration:"+str(it)
        print "Current error:"+str(err)
        
    return (S,thIter)


def update_A_prox(V,A,Shat,M,Imax,mu,mask=True,positivity=False):
    '''
    Proximal method (Forward-Backward) to estimate mixing matrix
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture to update, size: bands*sources
    
    @param Shat: Sources, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Imax: Number of iterations
    
    @param mu: Parameter of gradient step
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @return: Updated A, size: bands*sources
    '''
    (Bd,P) = np.shape(V)
    (N,P) = np.shape(Shat)
    A_n = np.zeros_like(A)  
    
    if mu == 0:
        mu = 0.5/LA.norm(np.dot(Shat,Shat.conj().transpose()))
    errTab = np.zeros(Imax)
    it = 0
    var_rsd = 0
    
    while it < Imax: 
        rsd = V - M*np.dot(A,Shat)
        var_rsdN = np.var(rsd)
        rsd1 = np.dot(M*rsd,Shat.conj().transpose())
        A_n = A + mu*rsd1
        A_n = np.real(A_n)
#         normalize(A_n)
        tab = LA.norm(A_n,axis=0)
        tab[tab<1] = 1
        if positivity:
            A_n[A_n<0] = 0
        A_n = A_n/tab                       # Division on columns, valid in python2.7
            
        A = np.copy(A_n)
        err = abs(var_rsdN-var_rsd)  
        var_rsd = var_rsdN
        errTab[it] = err
        it += 1
        print "Iteration:"+str(it)
        print "Current error:"+str(err)

    return A
        
        
