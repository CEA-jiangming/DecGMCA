'''
Created on Oct 1, 2015

@author: mjiang
'''
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pylab
import param
import os
import sys
import time

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
wavelet = True
thresStrtg = 2
Imax = 500
fc = 1./32
epsilonF = 1e-4
epsilon = param.epsilon
logistic=False
postProc=0
postProcImax=50
Ksig=3
positivityS = False
positivityA = False

DecGMCA_flag = True
MC_GMCA_flag = False
GMCA_flag = False
ForWaRD_GMCA_flag = False

runtime = 0
for db in dbArr:
    for n in nArr:
        drTest = 'test_CS/'
        drSources = drTest+'sources/'
        drNoises = drTest+'noises/'
        drMixtures = drTest+'mixtures/'
        drMasks = drTest+'masks/'
        drResults = drTest+'results_Prox/'
        drNum = 'n'+str(n)+'/'
        
        for j in np.arange(len(epsilon))[:1]:
            subdr = drNum+'epsilon_'+'%.e'% epsilon[j]+'/'
        
            for r in np.arange(numTests):
                
                for pc in pcArr:
                    drReal = 'r'+str(r)+'_mask'+str(int(pc*100))+'_db'+str(int(db))+'/'                        
                    
                    for b in bd:
                        if not os.path.exists(drResults+subdr+drReal):
                            os.makedirs(drResults+subdr+drReal)
                             
                        if DecGMCA_flag:
                            if wavelet:
                                S = MakeSources(t_samp=Ny,w=sigmaGauss,n_s=n,K=50,export=False)
                                if positivityS:
                                    S[S<0] = 0
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
#                             A /= LA.norm(A,axis=0)
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
                            
                        else:
                            S = fits.getdata(drSources+drNum+drReal+'S_bd'+str(b)+'.fits')
                            S = S.astype('float64')
                            
                            A = fits.getdata(drMixtures+drNum+drReal+'A_bd'+str(b)+'.fits')
                               
                            N = fits.getdata(drNoises+drNum+drReal+'noise_bd'+str(b)+'.fits')
                            N=N.astype('float64')
                            
                            if mask:
                                M = fits.getdata(drMasks+drNum+drReal+'mask_bd'+str(b)+'.fits')
                                M = M.astype('float64')
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
                        
                        if DecGMCA_flag:
                            start = time.time()
                            (S_est,A_est) = DecGMCA(V_N,MMat,n,Nx,Ny,Imax,epsilon[j],epsilonF,Ndim,wavelet=wavelet,scale=4,mask=mask,deconv=deconv,wname='starlet',thresStrtg=thresStrtg,FTPlane=FTPlane,fc=fc,logistic=logistic,postProc=postProc,postProcImax=postProcImax,Ksig=Ksig,positivityS=positivityS,positivityA=positivityA)
                            end = time.time()

                        runtime += end-start
                        
                        fits.writeto(drResults+subdr+drReal+'estS_bd'+str(b)+'.fits',S_est,clobber=True)
                        fits.writeto(drResults+subdr+drReal+'estA_bd'+str(b)+'.fits',A_est,clobber=True)

print runtime
pylab.show()
