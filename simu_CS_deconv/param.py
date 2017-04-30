'''
Created on Nov 27, 2015

@author: mjiang
'''
import numpy as np

Nx = 1     # rows of the image
Ny = 4096     # colums of the image
Ndim = 1    # dimension of image (1 means 1D signal)

pcS = 0.01      # ratio of active points in the image
sigS = 1.      # standard deviation of amplitude of sources
psf = True        # sources convolved with a psf
sigmaGauss=10.0                      # parameter of Laplacian kernel
# sigA = 2.       # standard deviation of mixing matrix
kern_param = 1800            # standard deviation of Gaussian distribution (The PSF has a Gaussian-like form)
bdArr = np.array([10,20])         # number of bands
epsilon = np.array([1e0])              # Tikhonov parameter

# pcarr = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])        # ratio of present data
pcArr = np.array([0.5])        # ratio of present data
ratioArr = np.array([3])
nArr = np.array([5])           # number of sources
dbArr = np.array([60.0])

numTests = 1         # number of realizations
# dbArr = np.array([55,50,45,40,35,30,25,20,15,10])         # SNR in dB

