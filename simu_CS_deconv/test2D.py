'''
Created on Oct 1, 2015

@author: mjiang
'''
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pylab
import param
import sys
import os

# from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyDecGMCA.algoDecG import *
from pyDecGMCA.mathTools import *
from simulationsTools.MakeExperiment import *


##################################################
########## Data simulation parameters ############
##################################################
pcS = param.pcS      # ratio of active points in the image
sigS = param.sigS      # standard deviation of amplitude of sources
n = 3           # number of sources
bd = 20         # number of bands
ratio = 3        # ratio of present data
pc = 0.5     # percentage of mask
# db = param.db         # SNR in dB
db = 60
# sigN = sigS*10**(-db/20.)       # standard deviation of noise
Ndim = 2               # Dimension of the signal
##################################################
########## Algorithm parameters ##################
##################################################
FTPlane = True
mask = True
deconv = True
Imax = 500
kern_param = 100.
fc = np.array(1./32)
indice = [-9.0,-4.0,-0.2]
tauF = 3
epsilon = 1e-5
epsilonF = 1e-6


drTest = 'test_2D/'
drSources = drTest+'sources/'
drNoise = drTest+'noise/'
drMixture = drTest+'mixture/'
drMasks = drTest+'masks/'
drKernels = drTest+'kernels/'
drResult = drTest+'results/'
# drNum = 'n'+str(n)+'/'

# Noise realisations
# N = np.random.randn(n,Nx,Ny)
# Nhat = fft2d1d(N)                   # Ft of noises
# NhatMat = np.reshape(Nhat,(n,Nx*Ny))            # Transform to 2D matrix form

subdr = 'epsilon_'+'%.e'% epsilon+'_n'+str(n)+'/'
drReal = 'kernel'+str(ratio)+'_db'+str(db)+'/'
#                     S = fits.getdata(drSources+'n'+str(n)+'/'+'S'+str(r)+'.fits')
#                     S = fits.getdata(drSource+drReal+'S_bd'+str(b)+'.fits')
#                     S = S.astype('float64')
    
#                 N = fits.getdata(drNoise+'n'+str(n)+'/'+'noise_db'+str(int(db))+'_sigS'+str(int(sigS))+'_r'+str(r)+'.fits')
#                 N=N.astype('float64')
    
        
if not os.path.exists(drResult+subdr+drReal):
    os.makedirs(drResult+subdr+drReal)
S = fits.getdata(drSources+'S4.fits')
n,Nx,Ny = np.shape(S)
#                     S = S[1:]
#                     S = np.delete(S,1,axis=0)
#                     S[0,50:114,300:364] = 0
#                     S[0] *= 5
#                         
#                     S = MakeSources(t_samp=Ny,w=sigmaGauss,n_s=n,K=50,export=False)
#                     S = MakeSources2D(x_samp=Nx,y_samp=Ny,w=sigmaGauss,n_s=n,K=50,export=False)
#                     S_cyg = fits.getdata('cygayelloworangelo.fits')
#                     S = np.zeros((n,Nx,Ny))
#                     S_point = createEllipseGauss(n_s=1,x_samp=Nx,y_samp=Ny,sigS=sigS,K=10,sig=[5.,2.5],export=False)
#                     S_ext = createEllipseGauss(n_s=1,x_samp=Nx,y_samp=Ny,sigS=sigS,K=10,sig=[30.,20.],export=False)
#                     S[0] = S_cyg[1]/255.
#                     S[1] = S_point[0]
#                     S[2] = S_ext[0]
# #                     
#                     if not os.path.exists(drSources+drReal):
#                         os.makedirs(drSources+drReal) 
#                     fits.writeto(drSources+drReal+'S_bd'+str(b)+'.fits',S,clobber=True)
#                     
S = S.astype('float64')

#                     A = np.random.randn(b,n)        # Mixing matrix
#                     A /= LA.norm(A,axis=0)
#                     A = createMix(sigA,n,b,power=True,indice=indice,export=False) 
A = np.zeros((bd,n))
f = np.arange(1,1+2,2/float(bd))
A[:,0] = f**indice[0]
A[:,1] = f**indice[1]
A[:,2] = f**indice[2]
#                     A[:,1] = np.random.randn(b)
# A /= LA.norm(A,axis=0)
A /= np.sqrt(np.sum(A**2,axis=0))           # For numpy version 1.6


if not os.path.exists(drMixture+drReal):
    os.makedirs(drMixture+drReal)
fits.writeto(drMixture+drReal+'A_bd'+str(bd)+'.fits',A,clobber=True)
#                         A = fits.getdata(drMixture+'n'+str(n)+'/'+'A_bd'+str(b)+'.fits')
#                         A = fits.getdata(drMixture+drReal+'A_bd'+str(b)+'.fits')
if deconv:
#                             kern = fits.getdata(drKernels+drReal+'kern_bd'+str(b)+'.fits')
    kern = createKernel2D(x_samp=Nx,y_samp=Ny,sigma=[kern_param,kern_param/pc],bd=bd,export=False)
    kern = kern.astype('float64')
#                         M = fits.getdata(drMasks+'mask_bd'+str(b)+'.fits')
#                             kern = fits.getdata(drKernels+'kern_bd'+str(b)+'.fits')
    
#                         M = M.astype('float64')
    
    if not os.path.exists(drKernels+drReal):
        os.makedirs(drKernels+drReal)
    fits.writeto(drKernels+drReal+'kern_bd'+str(bd)+'.fits',kern,clobber=True)
    
    kern = np.real(ifftshiftNd1d(kern,Ndim))             # Shift ft to make sure low frequency in the corner
    kernMat = np.reshape(kern,(bd,Nx*Ny))
#                         kernMat = np.ones((b,Nx*Ny))
else:
    kernMat = np.ones((bd,Nx*Ny))

if mask:
#                         M = createMask(Nx,Ny,mpc,b,export=False)
    M = fits.getdata(drMasks+'mask_bd'+str(bd)+'.fits')
    M = M.astype('float64')
    
#                         if not os.path.exists(drMasks+drNum+drReal):
#                             os.makedirs(drMasks+drNum+drReal)
#                         fits.writeto(drMasks+'mask_bd'+str(b)+'.fits',M,clobber=True)
    
#                 M = ifftshiftNd1d(M,Ndim)             # Shift ft to make sure low frequency in the corner
    MMat = np.reshape(M,(bd,Nx*Ny)) 
else:
    MMat = np.ones((bd,Nx*Ny)) 

#                     N = sigS*10**(-db/20.) * np.random.randn(n,Nx,Ny)
#                     if not os.path.exists(drNoise+drReal):
#                         os.makedirs(drNoise+drReal)
#                     fits.writeto(drNoise+drReal+'noise_bd'+str(b)+'.fits',N,clobber=True)

N = fits.getdata(drNoise+'noise.fits')
#                     N = np.delete(N,1,axis=0)
#                     N = N[1:]

S_N = S+N
if FTPlane:
    S_Nhat = fftNd1d(S_N,Ndim)                   # Ft of sources
else:
    S_Nhat = S_N
S_NhatMat = np.reshape(S_Nhat,(n,Nx*Ny))        # Transform to 2D matrix form
H = kernMat*MMat
V_N = np.dot(A,S_NhatMat)*H
#     V_N = (V + N)*MMat

#                     (S_est,A_est) = GMCA_class(V_N,kernMat,n,Nx,Ny,Imax,tauF,epsilon[j],Ndim,wavelet=True,scale=4,mask=mask,wname='starlet',coarseScaleStrtg=1,thresStrtg=2,FTPlane=FTPlane,fc=fc,Ar=A)
#                     X = ForWaRD(V_N,kernMat,epsilon=1e-3)
# (S_est,A_est) = DecGMCA(V_N,H,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet=True,scale=5,mask=mask,deconv=deconv,wname='starlet',thresStrtg=2,FTPlane=FTPlane,fc=fc,postProc=False,Ksig=0.6,positivity=False)

Ae = fits.getdata(drResult+subdr+drReal+'estA_bd'+str(bd)+'.fits')
Se = fits.getdata(drResult+subdr+drReal+'estS_bd'+str(bd)+'.fits')
Se = Se.astype('float64')
for Ksig in np.arange(3.0,3.1):
    Se_postProc,thIter = update_S_prox_Condat_Vu(V_N,Ae,Se,MMat,Nx,Ny,Ndim,50,tau=0,eta=1.0,Ksig=Ksig,wavelet=True,scale=5,wname='starlet',thresStrtg=2,FTPlane=FTPlane,positivity=False)
#                     (S_est,A_est) = MC_GMCA(V_N,MMat,n,Nx,Ny,Imax,tauF,0,Ndim,wavelet=True,scale=4,mask=False,wname='starlet',coarseScaleStrtg=1,thresStrtg=2,FTPlane=FTPlane,fc=fc,Ar=A)
#                     (S_est,A_est) = GMCA_wavelet(V_N,MMat,n,Nx,Ny,Imax,tauF,epsilon[j],Ndim,wavelet=True,scale=4,mask=mask,wname='starlet')


#                     fits.writeto(drResult+subdr+'r'+str(r)+'/initA_bd'+str(b)+'_r'+str(r)+'.fits',A_init,clobber=True)
# fits.writeto(drResult+subdr+drReal+'estS_bd'+str(bd)+'.fits',S_est,clobber=True)
# fits.writeto(drResult+subdr+drReal+'estA_bd'+str(bd)+'.fits',A_est,clobber=True)

    fits.writeto(drResult+subdr+drReal+'estS_postProc_Ksig'+str(Ksig)+'_bd'+str(bd)+'.fits',Se_postProc,clobber=True)
        
        ####################################################################
        ####################### Plots ######################################
        ####################################################################
#             fig, axarr = plt.subplots(2, 2)
#             fig.suptitle(r'Estimated sources, $\epsilon$='+str(epsilon)+', mask=80%, bands='+str(b),fontsize=15)
#             im1 = axarr[0, 0].imshow(S[0])
#             axarr[0, 0].set_title('Reference 1')
#             divider1 = make_axes_locatable(axarr[0, 0])
#             cax1 = divider1.append_axes("right", size="10%", pad=0.05)
#             fig.colorbar(im1,cax=cax1)
#               
#             im2 = axarr[0, 1].imshow(S[1])
#             axarr[0, 1].set_title('Reference 2')
#             divider2 = make_axes_locatable(axarr[0, 1])
#             cax2 = divider2.append_axes("right", size="10%", pad=0.05)
#             fig.colorbar(im2,cax=cax2)
#              
#             im3 = axarr[1, 0].imshow(S_est[0])
#             axarr[1, 0].set_title('Estimation 1')
#             divider3 = make_axes_locatable(axarr[1, 0])
#             cax3 = divider3.append_axes("right", size="10%", pad=0.05)
#             fig.colorbar(im3,cax=cax3)
#              
#             im4 = axarr[1, 1].imshow(S_est[1])
#             axarr[1, 1].set_title('Estimation 2')
#             divider4 = make_axes_locatable(axarr[1, 1])
#             cax4 = divider4.append_axes("right", size="10%", pad=0.05)
#             fig.colorbar(im4,cax=cax4)

pylab.show()
