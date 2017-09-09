'''
Created on Feb 19, 2017

@author: mjiang
'''
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
Imax = 1
fc = 1./32
epsilonF = 1e-4
epsilon = param.epsilon
logistic=False
postProc=0
postProcImax=50
Ksig=1.0
positivityS = False
positivityA = False

DecGMCA_flag = False
MC_GMCA_flag = False
GMCA_flag = False
ForWaRD_GMCA_flag = False

for db in dbArr:
    for n in nArr:
        drTest = 'test_CS/'
        drSources = drTest+'sources/'
        drNoises = drTest+'noises/'
        drMixtures = drTest+'mixtures/'
        drMasks = drTest+'masks/'
        drResults_DecGMCA = drTest+'results/'
        drResults = drTest+'results_Rigorous_PALM_FISTA/'
        drNum = 'n'+str(n)+'/'
        
#         for j in np.arange(len(epsilon))[:1]:
#             subdr = drNum+'epsilon_'+'%.e'% epsilon[j]+'/'
        
        for r in np.arange(numTests):
            
            for pc in pcArr:
                drReal = 'r'+str(r)+'_mask'+str(int(pc*100))+'_db'+str(int(db))+'/'                        
                
                for b in bd[-1:]:
                    if not os.path.exists(drResults+drReal):
                        os.makedirs(drResults+drReal)
                         
#                         if DecGMCA_flag:
#                             if wavelet:
#                                 S = MakeSources(t_samp=Ny,w=sigmaGauss,n_s=n,K=50,export=False)
#                                 if positivityS:
#                                     S[S<0] = 0
#                             else:
#                                 S = createGrGauss(Nx,Ny,n,alpha=0.2,export=False)
# #                             S = createBernGauss(Nx,Ny,pcS,sigS,n,psf=False,export=False)                            
#                             if not os.path.exists(drSources+drNum+drReal):
#                                 os.makedirs(drSources+drNum+drReal) 
#                             fits.writeto(drSources+drNum+drReal+'S_bd'+str(b)+'.fits',S,clobber=True)
#                             S = S.astype('float64')
#                              
#                             A = np.random.randn(b,n)        # Mixing matrix
#                             while LA.cond(A) > 10:
#                                 A = np.random.randn(b,n)
#                             A /= LA.norm(A,axis=0)
#                             if not os.path.exists(drMixtures+drNum+drReal):
#                                 os.makedirs(drMixtures+drNum+drReal)
#                             fits.writeto(drMixtures+drNum+drReal+'A_bd'+str(b)+'.fits',A,clobber=True)
#                             
#                             N = sigS*10**(-db/20) * np.random.randn(n,Nx*Ny)
#                             if not os.path.exists(drNoises+drNum+drReal):
#                                 os.makedirs(drNoises+drNum+drReal)
#                             fits.writeto(drNoises+drNum+drReal+'noise_bd'+str(b)+'.fits',N,clobber=True)
#                             
#                             if mask:
#                                 M = createMask(Nx,Ny,pc,b,export=False)
#                                 M = M.astype('float64')                                  
#                                 if not os.path.exists(drMasks+drNum+drReal):
#                                     os.makedirs(drMasks+drNum+drReal)
#                                 fits.writeto(drMasks+drNum+drReal+'mask_bd'+str(b)+'.fits',M,clobber=True)
#                                 MMat = np.reshape(M,(b,Nx*Ny))                                
#                             else:
#                                 MMat = np.ones((b,Nx*Ny))
                        
#                         else:
                    S = fits.getdata(drSources+drReal+'S_bd'+str(b)+'.fits')
                    S = S.astype('float64')
                    
                    A = fits.getdata(drMixtures+drReal+'A_bd'+str(b)+'.fits')
                       
                    N = fits.getdata(drNoises+drReal+'noise_bd'+str(b)+'.fits')
                    N=N.astype('float64')
                    
                    if mask:
                        M = fits.getdata(drMasks+drReal+'mask_bd'+str(b)+'.fits')
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
                    
#                         if DecGMCA_flag:
#                             (S_est,A_est) = DecGMCA(V_N,MMat,n,Nx,Ny,Imax,epsilon[j],epsilonF,Ndim,wavelet=wavelet,scale=4,mask=mask,deconv=deconv,wname='starlet',thresStrtg=thresStrtg,FTPlane=FTPlane,fc=fc,logistic=logistic,postProc=postProc,postProcImax=postProcImax,Ksig=Ksig,positivityS=positivityS,positivityA=positivityA)
# 
#                         elif MC_GMCA_flag and mask:
#                             Xe = SVTCompletion(V_N,M,n,1.5,1000)
#                             (S_est,A_est) = GMCA(Xe,n,Nx,Ny,Imax,Ndim,wavelet=True,scale=4,wname='starlet',thresStrtg=2,FTPlane=FTPlane,fc=fc,Ar=A)
#                         else:
#                             (S_est,A_est) = DecGMCA(V_N,MMat,n,Nx,Ny,Imax,epsilon[j],epsilonF,Ndim,wavelet=wavelet,scale=4,mask=mask,deconv=deconv,wname='starlet',thresStrtg=thresStrtg,FTPlane=FTPlane,fc=fc,logistic=logistic,postProc=2,postProcImax=postProcImax,Ksig=Ksig,positivityS=positivityS,positivityA=positivityA)
#                     Se = fits.getdata(drResults_DecGMCA+drReal+'estS_bd'+str(b)+'_r'+str(r)+'.fits')
#                     Ae = fits.getdata(drResults_DecGMCA+drReal+'estA_bd'+str(b)+'_r'+str(r)+'.fits')
#                     Se = Se.astype('float64')
#                     Ae = Ae.astype('float64')
                    (S_est,A_est) = GMCA_PALM(V_N,MMat,n,Nx,Ny,Imax,Ndim,wavelet,scale=4,mask=mask,deconv=deconv,wname='starlet',thresStrtg=thresStrtg,FTPlane=FTPlane,fc=fc,logistic=False,Ksig=Ksig,positivity=False,S=None,A=None)
                    
                    fits.writeto(drResults+drReal+'estS_bd'+str(b)+'.fits',S_est,clobber=True)
                    fits.writeto(drResults+drReal+'estA_bd'+str(b)+'.fits',A_est,clobber=True)

pylab.show()
