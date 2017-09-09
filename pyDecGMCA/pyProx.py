'''
Created on Oct 26, 2015

@author: mjiang
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy.fftpack as scifft
import pylab
from mathTools import *
from pyWavelet.wav1d import *
from pyWavelet.wav2d import *
import os
import glob
import re
import astropy.io.fits as fits
import pyWavelet.waveTools as pm
import sys

##########################################################################
######################### Proximal operators #############################
##########################################################################

def update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax,mu,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False,currentIter=0,globalImax=1):
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
    
    @param Ar: Mixing matrix of reference
    
    @param FISTA: Whether to use FISTA algorithm
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if mu == 0:
        Lip= LA.norm(np.dot(A,A.transpose()))
        mu = 1.0/Lip
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))
#             wtTmp = np.zeros((N,scale,P))
#             wt = np.zeros((N,scale-1,P))
#             for sr in np.arange(N):
#                 wtTmp[sr] = star1d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)
# #                 coarseScale[sr] = np.copy(wtTmp[sr,-1])                    
# #                 wt[sr] = np.copy(wtTmp[sr,:-1])
    
#     if thresStrtg == 1:
#         thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
#     elif thresStrtg == 2:
#         thIter = np.zeros((Imax,N,scale-1))
     
    if FISTA:
        t = 1.0         # FISTA parameter
    if PALM:
        P_min = 0.05
    it = 0
    var_rsd = 0
    errTab = np.zeros(Imax)
    
    while it<Imax:
        if FTPlane:
            Shat = fftNd1d(S,Ndim)
        else:
            Shat = S
        if Ndim == 2:
            Shat = np.reshape(Shat,(N,P))
        rsd = V - M*np.dot(A,Shat)
        var_rsdN = np.var(rsd)
        rsd1 = np.dot(A.conj().transpose(),M*rsd)
            
        if FTPlane: 
            rsd1 = ifftNd1d(np.reshape(rsd1,(N,Nx,Ny)).squeeze(),Ndim) 
            rsd1 = np.real(rsd1)  
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        else:
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        
        if wavelet:
            if wname == 'starlet':
                rsdTr = np.zeros((N,scale,P))
                wt = np.zeros((N,scale,P))
                for sr in np.arange(N):
                    rsdTr[sr] = star1d(rsd1[sr],scale=scale, fast=True, gen2=gen2,normalization=True)
                    wt[sr] = star1d(S_n[sr],scale=scale, fast=True, gen2=gen2,normalization=True)
                coarseScale = wt[:,-1,:]
                wt = np.delete(wt,-1,axis=1)
                rsdTr = np.delete(rsdTr,-1,axis=1)
            ################## Noise estimate####################
                if thresStrtg == 1:
                    rsdTr = np.reshape(rsdTr,(N,np.size(rsdTr)/N)).squeeze()
                    sig = mad(rsdTr[:,:P])
#                     sig = np.array([0.01,0.01])
                elif thresStrtg == 2:
                    sig = np.zeros((N,scale-1))
                    for sr in np.arange(N):
    #                 sig = mad(wt[:,:P])                                             # For starlet transform
                        sig[sr] = mad(rsdTr[sr]) 
            
                if PALM:
                    thTab = find_th(wt,P_min,sig,currentIter,globalImax,strategy=thresStrtg)
                else:
                    thTab = Ksig*sig
#                 thIter[it] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
                hardTh(wt,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(N):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)            
            wt = np.concatenate((wt,coarseScale[:,np.newaxis,:]),axis=1)
#             wt = np.reshape(wt,(N,np.size(wt)/(N*P),P))             # For undecimated wavelet transform
            if wname == 'starlet':
                for sr in np.arange(N):
                    S_n[sr] = istar1d(wt[sr],fast=True, gen2=gen2,normalization=True)
            
            if FISTA:
                tn = (1.+np.sqrt(1+4*t*t))/2
                S = S_n + (t-1)/tn * (S_n-S)
                t = tn
            else:
                S = np.copy(S_n)
                
            err = abs(var_rsdN-var_rsd)  
            var_rsd = var_rsdN
            errTab[it] = err
        
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)
            
    return S

def update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax,ImaxInt,mu,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False,currentIter=0,globalImax=1,u=None,sig=None):
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
    
    if mu == 0:
        Lip= LA.norm(np.dot(A,A.transpose()))
        mu = 1.0/Lip
    if muInt == 0:
        muInt = 1.0
    if PALM:
        P_min = 0.05
    if FISTA:
        t = 1.0
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))                
            if u is None:
                if Ndim == 1:
                    wt = np.zeros((N,scale-1,P))
                    for sr in np.arange(N):
                        wtTmp = star1d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                        wt[sr] = np.copy(wtTmp[:-1])
                elif Ndim == 2:
                    wt = np.zeros((N,scale-1,Nx,Ny))
                    for sr in np.arange(N):
                        wtTmp = star2d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                        wt[sr] = np.copy(wtTmp[:-1])
                wt = np.reshape(wt,(N,np.size(wt)/N))
                sigTmp = mad(wt[:,:P])
                u = np.zeros((N,(scale-1)*P))
                for sr in np.arange(N):
                    ind = (np.abs(wt[sr]) - Ksig*muInt*sigTmp[sr] > 0)
                    u[sr][ind] = Ksig*muInt*sigTmp[sr]*np.sign(wt[sr][ind])   
                if Ndim == 1:
                    u = np.reshape(u,(N,scale-1,P))
                    u = np.concatenate((u,np.zeros((N,1,P))),axis=1)
                elif Ndim == 2:
                    u = np.reshape(u,(N,scale-1,Nx,Ny))  
                    u = np.concatenate((u,np.zeros((N,1,Nx,Ny))),axis=1)  
                
    if thresStrtg == 1:
        thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,N,scale-1))
    
    it = 0
    errTab = np.zeros(Imax)
    err = 1.
    tol = 1e-5
    
    while it<Imax and err > tol:
        if FTPlane:
            Shat = fftNd1d(S,Ndim)
        else:
            Shat = S
        if Ndim == 2:
            Shat = np.reshape(Shat,(N,P))
        rsd = V - M*np.dot(A,Shat)
        rsd1 = np.dot(A.conj().transpose(),M*rsd)
            
        if FTPlane: 
            rsd1 = ifftNd1d(np.reshape(rsd1,(N,Nx,Ny)).squeeze(),Ndim) 
            rsd1 = np.real(rsd1)  
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        else:
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
            
        rsdTr = np.zeros((N,scale,P))
        for sr in np.arange(N):
            rsdTr[sr] = star1d(rsd1[sr],scale=scale, fast=True, gen2=gen2,normalization=True)    
        rsdTr = np.delete(rsdTr,-1,axis=1)
        ######## Test for rigorous PALM ##############
        if thresStrtg == 1 and sig is None:
            rsdTr = np.real(np.reshape(rsdTr,(N,np.size(rsdTr)/N)).squeeze())
            sig = mad(mu*rsdTr[:,:P])
        elif thresStrtg == 2 and sig is None:
            sig = np.zeros((N,scale-1))
            for sr in np.arange(N):
                sig[sr] = mad(mu*rsdTr[sr])
        
        it_int = 0
        ############### Subiteration #################
        while it_int < ImaxInt:
            if wavelet:
                if wname == 'starlet':
                    for sr in np.arange(N):
                        Stmp[sr] = adstar1d(u[sr], fast=True, gen2=gen2, normalization=True)
                    
                    rsdInt = S_n - Stmp
                    rsdIntTr = np.zeros((N,scale,P))
                    for sr in np.arange(N):
                        rsdIntTr[sr] = star1d(rsdInt[sr],scale=scale, fast=True, gen2=gen2,normalization=True)  
                    u_n = u + muInt*rsdIntTr
                    u_n = np.delete(u_n,-1,axis=1)
                    rsdIntTr = np.delete(rsdIntTr,-1,axis=1)
                    u_n1 = np.copy(u_n)
                
                    if PALM:
                        thTab = find_th(u_n,P_min,sig,currentIter,globalImax,strategy=thresStrtg,Ksig=Ksig)
                    else:
                        thTab = Ksig*sig
                    thIter[it] = thTab                
                    #### Thresholding in terms of percentage of significant coefficients ######
                    if thresStrtg == 1:
#                         hardTh(wtTmp_n,thTab,weights=None,reweighted=False)
                        softTh(u_n,thTab,weights=None,reweighted=False)
                    elif thresStrtg == 2:
                        for sr in np.arange(N):
#                             hardTh(wtTmp_n[sr],thTab[sr],weights=None,reweighted=False)
                            softTh(u_n[sr],thTab[sr],weights=None,reweighted=False)            
                    u_n = u_n1 - u_n
                    u_n = np.reshape(u_n,(N,scale-1,P))             # For undecimated wavelet transform
#                     wtTmp_n = np.concatenate((wtTmp_n,coarseScale),axis=1)
                    u_n = np.concatenate((u_n,np.zeros((N,1,P))),axis=1)
                    u = u_n
            it_int += 1
            
        if wavelet:
            if wname == 'starlet':
                for sr in np.arange(N):
                    Stmp[sr] = adstar1d(u[sr], fast=True, gen2=gen2, normalization=True)
        
        S_n = S_n - Stmp                    
        err = (((S_n-S)**2).sum())/((S**2).sum())  
        errTab[it] = err
        if FISTA:
            tn = (1.+np.sqrt(1+4*t*t))/2
            S = S_n + (t-1)/tn * (S_n-S)
            t = tn
        else:
            S = S_n
               
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)
        
    return (S,u,sig)

def update_S_prox_Condat_Vu(V,A,S,H,Nx,Ny,Ndim,Imax,tau,eta,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,positivity=False,PALM=False,currentIter=0,globalImax=1):
    '''
    Proximal method (Condat-Vu) to estimate sources with sparsity constraint in analysis framework
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param S: Sources to update, size: sources*pixels
    
    @param H: Linear operator, size: bands*pixels
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Number of iterations
    
    @param tau: Parameter of gradient step
    
    @param eta: Parameter of sparsity constraint step
    
    @param Ksig: k-sigma for the thresholding
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far 
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if tau == 0:
        Lip= LA.norm(np.dot(A,A.transpose()))
        tau = 1.0/Lip
        eta = 0.5*Lip
#     print "tau:"+str(tau)
#     print "eta:"+str(eta)
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))                
            if Ndim == 1:
                wt = np.zeros((N,scale-1,P))
                for sr in np.arange(N):
                    wtTmp = star1d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                    wt[sr] = np.copy(wtTmp[:-1])
            elif Ndim == 2:
                wt = np.zeros((N,scale-1,Nx,Ny))
                for sr in np.arange(N):
                    wtTmp = star2d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                    wt[sr] = np.copy(wtTmp[:-1])
            wt = np.reshape(wt,(N,np.size(wt)/N))
            sig = mad(wt[:,:P])
            u = np.zeros((N,(scale-1)*P))
            for sr in np.arange(N):
                ind = (np.abs(wt[sr]) - tau*eta*Ksig*sig[sr] > 0)
                u[sr][ind] = tau*eta*Ksig*sig[sr]*np.sign(wt[sr][ind])   
            if Ndim == 1:
                u = np.reshape(u,(N,scale-1,P))
            elif Ndim == 2:
                u = np.reshape(u,(N,scale-1,Nx,Ny))                   
    else:
        u = np.copy(S)
    
    if thresStrtg == 1:
        thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,N,scale-1))
     
    it = 0
#     var_rsd = 0
    err = 1.0
    tol = 1e-5
    errTab = np.zeros(Imax)
    
    if PALM:
        P_min = 0.05
    
    while it < Imax and err > tol:
        if FTPlane:
            Shat = fftNd1d(S,Ndim)
        else:
            Shat = S
        if Ndim == 2:
            Shat = np.reshape(Shat,(N,P))
        rsd = V - H*np.dot(A,Shat)
        rsd1 = np.dot(A.conj().transpose(),H*rsd)
        Stmp = np.zeros_like(S)
        if wavelet:
            for sr in np.arange(N):
                if Ndim == 1:
                    wtTmp = np.concatenate((u[sr],np.zeros((1,P))),axis=0)
                    Stmp[sr] = adstar1d(wtTmp, fast=True, gen2=gen2, normalization=True)
                elif Ndim == 2:
                    wtTmp = np.concatenate((u[sr],np.zeros((1,Nx,Ny))),axis=0)
                    Stmp[sr] = adstar2d(wtTmp,fast=True, gen2=gen2, normalization=True)
        else:
            Stmp = u
            
        if FTPlane:    
            S_n = np.real(ifftNd1d(np.reshape(Shat + tau*rsd1,(N,Nx,Ny)).squeeze(),Ndim)) - tau*Stmp
        else:
            S_n = np.reshape(Shat + tau*rsd1,(N,Nx,Ny)).squeeze() - tau*Stmp
            
        if positivity:
            S_n[S_n<0] = 0
        
        termQ1 = 2*S_n - S
        if wavelet:
            if Ndim == 1:
                wtTmp = np.zeros((N,scale,P))
            elif Ndim == 2:
                wtTmp = np.zeros((N,scale,Nx,Ny))
            
            for sr in np.arange(N):
                if Ndim == 1:
                    wtTmp[sr] = star1d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
                elif Ndim == 2:
                    wtTmp[sr] = star2d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
                    
            wtTmp = np.delete(wtTmp,-1,axis=1)
            termQ1 = u + eta*wtTmp
            termQ2 = np.copy(termQ1)
            ################## Noise estimate####################
            if thresStrtg == 1:
                wt = np.reshape(termQ2,(N,np.size(termQ2)/N))
                sig = mad(termQ2[:,:P])
    #                     sig = np.array([0.01,0.01])
            elif thresStrtg == 2:
                sig = np.zeros((N,scale-1))
                wt = np.reshape(termQ2,(N,scale-1,np.size(termQ2)/(N*(scale-1))))
#                 wtTmp = np.reshape(wtTmp,(N,scale-1,np.size(termQ2)/(N*(scale-1))))
                for sr in np.arange(N):
#                     sig = mad(wt[:,:P])                                             # For starlet transform
                    sig[sr] = mad(wt[sr])
            
            if PALM:
                thTab = find_th(wt,P_min,sig,currentIter,globalImax,strategy=thresStrtg,Ksig=tau*Ksig*eta)
            else:
                thTab = tau*Ksig*eta*sig
            thIter[it] = thTab                
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
    #                         hardTh(wtTmp_n,thTab,weights=None,reweighted=False)
                softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(N):
    #                             hardTh(wtTmp_n[sr],thTab[sr],weights=None,reweighted=False)
                    softTh(wt[sr],thTab[sr],weights=None,reweighted=False)
            if Ndim == 1:
                termQ2 = np.reshape(wt,(N,scale-1,P))
            elif Ndim == 2:        
                termQ2 = np.reshape(wt,(N,scale-1,Nx,Ny))
        else:
            termQ1 = u + eta*termQ1
            termQ2 = np.copy(termQ1)
            ################## Noise estimate####################
            wt = np.reshape(termQ2,(N,np.size(termQ2)/N))
            sig = mad(wt)
            thTab = Ksig*eta*sig
            thIter[it] = thTab                
            #### Thresholding in terms of percentage of significant coefficients ######
            softTh(wt,thTab,weights=None,reweighted=False)
            if Ndim == 1:
                termQ2 = np.reshape(wt,(N,P))
            elif Ndim == 2:        
                termQ2 = np.reshape(wt,(N,Nx,Ny))
        
                
        u_n = termQ1 - termQ2
        
        errS = (((S_n - S)**2).sum())/((S**2).sum())
        erru = (((u_n - u)**2).sum())/((u**2).sum())
        err = max(errS,erru)
#         if err < tol:
#             print "Err smaller than tol"
        S = np.copy(S_n)
        u = np.copy(u_n)
        
#         err = abs(var_rsdN-var_rsd)  
#         var_rsd = var_rsdN  
        errTab[it] = err
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:" + str(err)
#         if it%10 == 0:
#             fits.writeto('test5/estS_postProc'+str(it)+'.fits',S,clobber=True)
#             fits.writeto('test5/estS_sig'+str(it)+'.fits',sig,clobber=True)
#     fits.writeto('u.fits',u,clobber=True)
#     resWt = np.zeros((N,scale-1,Nx,Ny))
#     rsd1 = np.real(ifftNd1d(np.reshape(rsd1,(N,Nx,Ny)).squeeze(),Ndim))
#     for sr in np.arange(N):
#         wtTmp = star2d(rsd1[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
#         resWt[sr] = np.copy(wtTmp[:-1])
#     fits.writeto('res.fits',resWt,clobber=True)
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
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)

    return A

############################################################################
############################ Matrix completion #############################
############################################################################
def nearestCompletion(V,M):
    '''
    Interpolation of data using nearest neighbour method
    
    @param V: Input data, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @return: Interpolated data
    '''
    
    (Bd,P) = np.shape(V)
    V_comp = np.copy(V)
    for nu in np.arange(Bd):
        ind = tuple(np.where(M[nu,:]==0)[0])
        for ele in ind:
            dist = 1
            unComp = True 
            while unComp:
                nu_lf = nu
                if nu_lf < 0:
                    nu_lf = 0
                nu_rt = nu+1
                if nu_rt > Bd:
                    nu_rt = Bd
                ele_lf = ele-dist
                if ele_lf < 0:
                    ele_lf = 0
                ele_rt = ele+dist+1
                if ele_rt > P:
                    ele_rt = P
                vect = (V[nu_lf:nu_rt,ele_lf:ele_rt]*M[nu_lf:nu_rt,ele_lf:ele_rt]).flatten()
                vect = vect[vect!=0]
                if len(vect) > 0:
                    unComp = False
                    V_comp[nu,ele] = vect[np.random.randint(len(vect))]
                else:
                    dist += 1
    return V_comp
                
def SVTCompletion(X,M,n,delta,Nmax):
    '''
    Interpolation of data with low-rank constraint using SVT algorithm
    
    @param X: Input data, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @param n: Expected rank of the data
    
    @param delta: Parameter of gradient step 
    
    @param Nmax: Number of iterations
    
    @return: Interpolated data
    '''
    
#     (Bd,P) = np.shape(X)
    Y = np.zeros_like(X)
#     thTab = np.zeros(Nmax)
    
    for k in np.arange(Nmax):
        Y = Y + delta*M*(X-M*Y)
        U,s,V = LA.svd(Y,full_matrices=False)
        if np.size(s)>n:
            tau = s[n]
        else:
            tau = s[-1]
#         thTab[k] = tau
#         s = s - tau
        s[s<=tau] = 0                 # Thresholding to ensure Rank = n
#         s[n:] = 0
#         U[n:] = 0
#         V[n:] = 0
        S = np.diag(s)
        Y = np.dot(U,np.dot(S,V))
            
    return Y
        
        
        