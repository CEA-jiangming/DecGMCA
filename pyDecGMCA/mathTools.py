'''
Created on Nov 4, 2015

@author: mjiang
'''
import numpy as np
import scipy.fftpack as scifft

def fftshift2d1d(cubefft):
    (nz,nx,ny) = np.shape(cubefft)
    cubefftSh = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j        # Convert to complex array
    for fm in np.arange(nz):
        cubefftSh[fm] = scifft.fftshift(cubefft[fm])
    return cubefftSh

def ifftshift2d1d(cubefftSh):
    (nz,nx,ny) = np.shape(cubefftSh)
    cubefft = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j           # Convert to complex array
    for fm in np.arange(nz):
        cubefft[fm] = scifft.ifftshift(cubefftSh[fm])
    return cubefft

def fft2d1d(cube):
    (nz,nx,ny) = np.shape(cube)
    cubefft = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j                     # Convert to complex array
    for fm in np.arange(nz):
        cubefft[fm] = scifft.fft2(cube[fm])
    return cubefft

def ifft2d1d(cubefft):
    (nz,nx,ny) = np.shape(cubefft)
    cube = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j                 # Convert to complex array
    for fm in np.arange(nz):
        cube[fm] = scifft.ifft2(cubefft[fm])
    return cube

def fftshiftNd1d(imagefft,N):
    if N == 1:
        (nz,ny) = np.shape(imagefft)
        imagefftSh = np.zeros((nz,ny)) + np.zeros((nz,ny)) * 1j        # Convert to complex array
    if N == 2:
        (nz,nx,ny) = np.shape(imagefft)
        imagefftSh = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j        # Convert to complex array
    for fm in np.arange(nz):
        imagefftSh[fm] = scifft.fftshift(imagefft[fm])    
    return imagefftSh

def ifftshiftNd1d(imagefftSh,N):
    if N == 1:
        (nz,ny) = np.shape(imagefftSh)
        imagefft = np.zeros((nz,ny)) + np.zeros((nz,ny)) * 1j           # Convert to complex array
    if N == 2:
        (nz,nx,ny) = np.shape(imagefftSh)
        imagefft = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j        # Convert to complex array
    for fm in np.arange(nz):
        imagefft[fm] = scifft.ifftshift(imagefftSh[fm])
    return imagefft

def fftNd1d(image,N):
    if N == 1:
        (nz,ny) = np.shape(image)
        imagefft = np.zeros((nz,ny)) + np.zeros((nz,ny)) * 1j                     # Convert to complex array
    if N == 2:
        (nz,nx,ny) = np.shape(image)
        imagefft = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j                     # Convert to complex array
    for fm in np.arange(nz):
        if N == 1:
            imagefft[fm] = scifft.fft(image[fm])
        if N == 2:
            imagefft[fm] = scifft.fft2(image[fm])
    return imagefft

def ifftNd1d(imagefft,N):
    if N == 1:
        (nz,ny) = np.shape(imagefft)
        image = np.zeros((nz,ny)) + np.zeros((nz,ny)) * 1j                 # Convert to complex array
    if N == 2:
        (nz,nx,ny) = np.shape(imagefft)
        image = np.zeros((nz,nx,ny)) + np.zeros((nz,nx,ny)) * 1j                     # Convert to complex array
    for fm in np.arange(nz):
        if N == 1:
            image[fm] = scifft.ifft(imagefft[fm])
        if N == 2:
            image[fm] = scifft.ifft2(imagefft[fm])
    return image

def mad(alpha):
    dim = np.size(np.shape(alpha)) 
    if dim == 1:
        alpha = alpha[np.newaxis,:]
#     elif dim == 2:
#         alpha = alpha[np.newaxis,:,:]
#         (nx,ny) = np.shape(alpha)
#         alpha = alpha.reshape(1,nx,ny)
    nz = np.size(alpha,axis=0)
    sigma = np.zeros(nz)
    for i in np.arange(nz):
        sigma[i] = np.median(np.abs(alpha[i] - np.median(alpha[i]))) / 0.6745   
    alpha = np.squeeze(alpha)  
    return sigma

def softTh(alpha,thTab,weights=None,reweighted=False):
    nz = np.size(thTab)
#     print weights.shape
#     print thTab.shape
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
            (alpha[i])[alpha[i]>0] -= thTab[i]
            (alpha[i])[alpha[i]<0] += thTab[i]
        else:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0
            (alpha[i])[alpha[i]>0] -= (thTab[i]*weights[i])[alpha[i]>0]
            (alpha[i])[alpha[i]<0] += (thTab[i]*weights[i])[alpha[i]<0]

def hardTh(alpha,thTab,weights=None,reweighted=False):
    nz = np.size(thTab)
#     print weights.shape
#     print thTab.shape
    if np.ndim(alpha) == 1:
        alpha = alpha[np.newaxis,:]
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
        else:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0 
    alpha = alpha.squeeze()   
    
def filter_Hi(sig,Ndim,fc,fc2=1./8):
    '''
    The function is a high-pass filter applied on signals with fc as the cut-off frequency
    
    @param sig: 1D or 2D signal as entry, the number of sources is n(n>=1)
    
    @param Ndim: The dimension of the signal, Ndim=1 means 1D signal, Ndim=2 means 2D image
    
    @param fc: The cut-off frequency is given by normalized numerical frequency
    
    @return: The high frequency part of the signal
    '''
    if Ndim == 1:
        (nz,ny) = np.shape(sig)
        sig_Hi = np.zeros_like(sig)
        sig_Hi[:,int(fc*ny):-int(fc*ny)] = sig[:,int(fc*ny):-int(fc*ny)]        
    elif Ndim == 2:
        (nz,nx,ny) = np.shape(sig)
        sig_Hi = np.zeros_like(sig)
        sig_Hi[:,int(fc*nx):-int(fc*nx),int(fc*ny):-int(fc*ny)] = sig[:,int(fc*nx):-int(fc*nx),int(fc*ny):-int(fc*ny)]
    return sig_Hi

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c