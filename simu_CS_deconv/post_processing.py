'''
Created on Feb 25, 2016

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
dbArr = param.dbArr         # SNR in dB
numTests = param.numTests           # number of Monte-Carlo tests
########### CS simulation parameters ############
pcArr = param.pcArr        # percentage of active data
sigmaGauss = param.sigmaGauss       # parameter of the Laplacian kernel for the creation of sparse signals
###### Deconvolutoin simulation parameters #######
ratioArr = param.ratioArr       # ratio of the best resolved PSF and the worst PSF
##################################################
########## Algorithm parameters ##################
##################################################
FTPlane = True
mask = True
deconv = False
Imax = 50
Ksig = 3
epsilon = param.epsilon

drTest = 'test_CS/'
drSources = drTest+'sources/'
drNoises = drTest+'noises/'
drMixtures = drTest+'mixtures/'
drMasks = drTest+'masks/'
drKernels = drTest+'kernels/'
drResults = drTest+'results/'
drPostResults = drTest+'post_results/'

for db in dbArr:
    for n in nArr:
        drNum = 'n'+str(n)+'/'
        for j in np.arange(len(epsilon))[:1]:
            for Ksig in np.arange(0.6,0.7,0.2):
            
#                 subdr = drNum+'epsilon_'+'%.e'% epsilon[j]+'_db'+str(db)+'/'
                subdr = drNum+'epsilon_'+'%.e'% epsilon[j]+'/'
                for r in np.arange(numTests):
#                     S = fits.getdata(drSources+'n'+str(n)+'/'+'S'+str(r)+'.fits')
                     
#                     S = S.astype('float64')
                    if mask:
                        drReal = 'r'+str(r)+'_mask'+str(int(pcArr[0]*100))+'_db'+str(int(db))+'/'
                    if deconv:
                        drReal = 'r'+str(r)+'_kernel'+str(int(pcArr[0]))+'_db'+str(int(db))+'/'
    #                 drReal = 'r'+str(r)+'/'
    #                 S = fits.getdata(drSources+'n'+str(n)+'/'+'S'+str(r)+'.fits')
    #                 S = S.astype('float64')
                    
    #                 N = fits.getdata(drNoise+'n'+str(n)+'/'+'noise_db'+str(int(db))+'_sigS'+str(int(sigS))+'_r'+str(r)+'.fits')
    #                 N=N.astype('float64')
                    
                    for b in bd[-1:]:
                        S = fits.getdata(drSources+drNum+drReal+'S_bd'+str(b)+'.fits')
                        S = S.astype('float64')
#                         A = fits.getdata(drMixture+'n'+str(n)+'/'+'A_bd'+str(b)+'.fits')
                        A = fits.getdata(drMixtures+drNum+drReal+'A_bd'+str(b)+'.fits')
#                         A /= LA.norm(A,axis=0)
                        A /= np.sqrt(np.sum(A**2,axis=0))           # For numpy version 1.6
                        
#                         M = fits.getdata(drMasks+str(int(pc*100))+'/mask_bd'+str(b)+'.fits')
                        if mask:
                            M = fits.getdata(drMasks+drNum+drReal+'mask_bd'+str(b)+'.fits')
                            M = M.astype('float64')
                            MMat = np.reshape(M,(b,Nx*Ny))
                        if deconv:
                            kern = fits.getdata(drKernels+drNum+drReal+'kern_bd'+str(b)+'.fits')
                            kern = kern.astype('float64')
                            kern = ifftshiftNd1d(kern,Ndim)                 # Shift ft to make sure low frequency in the corner
                            kernMat = np.reshape(kern,(b,Nx*Ny)) 
                        
#                         N = fits.getdata(drResult+subdr+drReal+'noise_db'+str(int(db))+'_sigS'+str(int(np.round(sigS)))+'_bd'+str(b)+'_r'+str(r)+'.fits')
                        N = fits.getdata(drNoises+drNum+drReal+'noise_bd'+str(b)+'.fits')
                        
                        S_N = S+N
                        if FTPlane:
                            S_Nhat = fftNd1d(S_N,Ndim)                   # Ft of sources
                        else:
                            S_Nhat = S_N
                        S_NhatMat = np.reshape(S_Nhat,(n,Nx*Ny))        # Transform to 2D matrix form
                        
                        if mask:
                            V_N = np.dot(A,S_NhatMat)*MMat
                        if deconv:
                            V_N = np.dot(A,S_NhatMat)*kernMat
                        
                        Se = fits.getdata(drResults+subdr+drReal+'estS_bd'+str(b)+'.fits')
                        Se = Se.astype('float64')
                        
                        Ae = fits.getdata(drResults+subdr+drReal+'estA_bd'+str(b)+'.fits')
    #                     Se_post,thIter=update_S_prox(V_N,Ae,Se,M,Nx,Ny,Imax,mu=1.,mask=True,wavelet=True,scale=4,wname='starlet',coarseScaleStrtg=1,thresStrtg=2,FTPlane=True,Ar=None,FISTA=False)
#                         Se_post,thIter=update_S_prox_anal(V_N,Ae,Se,kern,Nx,Ny,Imax,Imax1=30,mu=0.5,mu1=0.5,Ksig=Ksig,mask=True,wavelet=True,scale=4,wname='starlet',coarseScaleStrtg=1,thresStrtg=2,FTPlane=True,Ar=None)
                        if mask:
                            Se_post,thIter=update_S_prox_Condat_Vu(V_N,Ae,Se,MMat,Nx,Ny,Ndim,Imax,tau=0.5,eta=0.5,Ksig=Ksig,wavelet=True,scale=4,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None)
                        if deconv:
                            Se_post,thIter=update_S_prox_Condat_Vu(V_N,Ae,Se,kernMat,Nx,Ny,Ndim,Imax,tau=0.5,eta=0.5,Ksig=Ksig,wavelet=True,scale=4,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None)
#                         fits.writeto(drResult+subdr+drReal+'estS_post_bd'+str(b)+'_r'+str(r)+'.fits',Se_post,clobber=True)
                        if not os.path.exists(drPostResults+subdr+drReal):
                            os.makedirs(drPostResults+subdr+drReal)
                        fits.writeto(drPostResults+subdr+drReal+'estS_post_condat_vu_soft_bd'+str(b)+'_Ksig'+str(Ksig)+'_r'+str(r)+'.fits',Se_post,clobber=True)
#                         fits.writeto(drResult+subdr+drReal+'thIter_post_anal_soft_bd'+str(b)+'_Ksig'+str(Ksig)+'_r'+str(r)+'.fits',thIter,clobber=True)
#                         Se_post = fits.getdata(drResult+subdr+drReal+'estS_post_anal_soft_bd'+str(b)+'_Ksig'+str(Ksig)+'_r'+str(r)+'.fits')
#                         Se_post = Se_post.astype('float64')
#                         Se_post_hat = fftNd1d(Se_post,Ndim)
#                         Se_post_hat_Hi = filter_Hi(Se_post_hat,Ndim,1./32)
#                         Shat_Hi = np.reshape(Se_post_hat_Hi,(n,Nx*Ny))
#                         V_Hi = filter_Hi(V_N,Ndim,1./32)
#                         Ae_post = update_A_prox(V_Hi,Ae,Shat_Hi,M,Imax=500,mu=5e-5)
#                         fits.writeto(drResult+subdr+drReal+'estA_post_anal_soft_bd'+str(b)+'_Ksig'+str(Ksig)+'_r'+str(r)+'.fits',Ae_post,clobber=True)
