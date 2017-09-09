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
##################### Boost proximal operators ###########################
##########################################################################

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
    print "tau:"+str(tau)
    print "eta:"+str(eta)
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
            #             coarseScale = np.zeros((N,1,P))
            if Ndim == 1:
                wt = starlet.Nstar1d(S, scale=scale, gen2=gen2, normalization=True)
            elif Ndim == 2:
                wt = starlet.Nstar2d(S, scale=scale, gen2=gen2, normalization=True)
            wt = np.delete(wt,-1,axis=1)
            
            #             if Ndim == 1:
            #                 wt = np.zeros((N,scale-1,P))
            #                 for sr in np.arange(N):
            #                     wtTmp = star1d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
            #                     wt[sr] = np.copy(wtTmp[:-1])
            #             elif Ndim == 2:
            #                 wt = np.zeros((N,scale-1,Nx,Ny))
            #                 for sr in np.arange(N):
            #                     wtTmp = star2d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
            #                     wt[sr] = np.copy(wtTmp[:-1])
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
            if Ndim == 1:
                wtTmp = np.concatenate((u,np.zeros((N,1,P))),axis=1)
                wt = starlet.Nadstar1d(wtTmp, gen2=gen2, normalization=True)
            elif Ndim == 2:
                wtTmp = np.concatenate((u,np.zeros((N,1,Nx,Ny))),axis=1)
                wt = starlet.Nadstar2d(wtTmp, gen2=gen2, normalization=True)
        
        #             for sr in np.arange(N):
        #                 if Ndim == 1:
        #                     wtTmp = np.concatenate((u[sr],np.zeros((1,P))),axis=0)
        #                     Stmp[sr] = adstar1d(wtTmp, fast=True, gen2=gen2, normalization=True)
        #                 elif Ndim == 2:
        #                     wtTmp = np.concatenate((u[sr],np.zeros((1,Nx,Ny))),axis=0)
        #                     Stmp[sr] = adstar2d(wtTmp,fast=True, gen2=gen2, normalization=True)
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
            #             if Ndim == 1:
            #                 wtTmp = np.zeros((N,scale,P))
            #             elif Ndim == 2:
            #                 wtTmp = np.zeros((N,scale,Nx,Ny))
            
            #             for sr in np.arange(N):
            #                 if Ndim == 1:
            #                     wtTmp[sr] = star1d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
            #                 elif Ndim == 2:
            #                     wtTmp[sr] = star2d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
            
            if Ndim == 1:
                wtTmp = starlet.Nstar1d(termQ1, scale=scale, gen2=gen2, normalization=True)
            elif Ndim == 2:
                wtTmp = starlet.Nstar2d(termQ1, scale=scale, gen2=gen2, normalization=True)
            
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
        print "Iteration:"+str(it)
        print "Current error:" + str(err)
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
