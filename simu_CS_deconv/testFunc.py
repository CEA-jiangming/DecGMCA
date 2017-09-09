'''
Created on Jan 5, 2017

@author: mjiang
'''
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pylab
import param
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyDecGMCA.algoDecG import *
from pyDecGMCA.mathTools import *
from simulationsTools.MakeExperiment import *

##################################################
########## Data simulation parameters ############
##################################################

############### Global parameters ################
Nx = param.Nx     # rows of the image
Ny = param.Ny     # colums of the image
Ndim = param.Ndim               # Dimension of the signal
############## Simulation parameters #############
pcS = param.pcS      # ratio of active peaks to creat K-sparse signal
sigS = param.sigS      # standard deviation of amplitude of sources
nArr = param.nArr           # number of sources
bd = param.bdArr         # number of bands
# dbArr = param.dbArr         # SNR in dB
numTests = param.numTests           # number of Monte-Carlo tests
########### CS simulation parameters ############
# pcArr = param.pcArr        # percentage of active data
sigmaGauss = param.sigmaGauss       # parameter of the Laplacian kernel for the creation of sparse signals
###### Deconvolutoin simulation parameters #######
# ratioArr = param.ratioArr       # ratio of the best resolved PSF and the worst PSF
##################################################
########## Algorithm parameters ##################
##################################################
FTPlane = True
mask = True
deconv = False
wavelet = True
thresStrtg = 2
Imax = 500
fc = 1./32
n = 5
db = 60
pc = 0.5
epsilonF = 1e-4
epsilon = param.epsilon
logistic=False
postProc=False
Ksig=0.6
# positivity = True

drTest = 'testFunc/'
drSources = drTest+'sources/'
drNoises = drTest+'noises/'
drMixtures = drTest+'mixtures/'
drMasks = drTest+'masks/'
drResults = drTest+'results/'
drNum = 'n'+str(n)+'/'

for r in np.arange(numTests): 
    drReal = 'r'+str(r)+'_db'+str(int(db))+'/'                    
        
    for b in bd:
        if not os.path.exists(drResults+drNum+drReal):
            os.makedirs(drResults+drNum+drReal)
        if wavelet:
            S = MakeSources(t_samp=Ny,w=sigmaGauss,n_s=n,K=50,export=False)
        else:
            S = createGrGauss(Nx,Ny,n,alpha=0.2,export=False)
#                             S = createBernGauss(Nx,Ny,pcS,sigS,n,psf=False,export=False)                            
        if not os.path.exists(drSources+drNum+drReal):
            os.makedirs(drSources+drNum+drReal) 
        fits.writeto(drSources+drNum+drReal+'S_bd'+str(b)+'.fits',S,clobber=True)
        S = S.astype('float64')
         
        A = np.random.randn(b,n)        # Mixing matrix
        while LA.cond(A) > 10:
            A = np.random.randn(b,n)
#         A /= LA.norm(A,axis=0)
        A /= np.sqrt(np.sum(A**2,axis=0))           # For numpy version 1.6
        if not os.path.exists(drMixtures+drNum+drReal):
            os.makedirs(drMixtures+drNum+drReal)
        fits.writeto(drMixtures+drNum+drReal+'A_bd'+str(b)+'.fits',A,clobber=True)
        
        N = sigS*10**(-db/20) * np.random.randn(n,Nx*Ny)
        if not os.path.exists(drNoises+drNum+drReal):
            os.makedirs(drNoises+drNum+drReal)
        fits.writeto(drNoises+drNum+drReal+'noise_bd'+str(b)+'.fits',N,clobber=True)
        
        if mask:
            M = createMask(Nx,Ny,pc,b,export=False)
            M = M.astype('float64')                                  
            if not os.path.exists(drMasks+drNum+drReal):
                os.makedirs(drMasks+drNum+drReal)
            fits.writeto(drMasks+drNum+drReal+'mask_bd'+str(b)+'.fits',M,clobber=True)
            MMat = np.reshape(M,(b,Nx*Ny))                                
        else:
            MMat = np.ones((b,Nx*Ny))

        S_N = S+N
        if FTPlane:
            S_Nhat = fftNd1d(S_N,Ndim)                   # Ft of sources
        else:
            S_Nhat = S_N
        S_NhatMat = np.reshape(S_Nhat,(n,Nx*Ny))        # Transform to 2D matrix form
        V_N = np.dot(A,S_NhatMat)*MMat
    
#         (S_est,A_est) = GMCA(V_N,n,Nx,Ny,Imax,Ndim,wavelet=wavelet,scale=4,wname='dct',thresStrtg=thresStrtg,FTPlane=FTPlane,fc=fc,logistic=logistic)
        (S_est,A_est) = DecGMCA(V_N,MMat,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet=wavelet,scale=4,mask=mask,deconv=deconv,wname='starlet',thresStrtg=2,FTPlane=FTPlane,fc=fc,logistic=logistic,postProc=postProc,Ksig=Ksig,positivityS=False)
        fits.writeto(drResults+drNum+drReal+'estS_bd'+str(b)+'.fits',S_est,clobber=True)
        fits.writeto(drResults+drNum+drReal+'estA_bd'+str(b)+'.fits',A_est,clobber=True)
