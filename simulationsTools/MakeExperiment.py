'''
Created on Oct 1, 2015

@author: mjiang
'''
import numpy as np
from scipy import ndimage
import astropy.io.fits as fits
import numpy.linalg as LA
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import special
import glob
import re
import os

def createHSCube(startfreq,stopfreq,stepfreq,k,alpha):
    '''
    @param startfreq: start frequency
    
    @param stopfreq: end frequency
    
    @param stepfreq: step frequency
    
    @param k: flux map?
    
    @param alpha: characteristic of source?
    
    @return: hyperspectral cube
    '''
    (nx,ny) = np.shape(k)
    num = (stopfreq-startfreq)/stepfreq
    mu = np.linspace(startfreq, stopfreq, num)
    cube = np.zeros((num,nx,ny))
    for i in np.arange(num):
        cube[i] = k*mu[i]**alpha
    
    return cube

def createBernGauss(Nx,Ny,pcS,sigS,n=1,psf=False,sigmaGauss = 1.0,export=True):
    '''
    Create k-sparse sources
    
    @param Nx: Rows of image
    
    @param Ny: Columes of image
    
    @param pcS: Percentage of activation
    
    @param sigS: Standard deviation of amplitude with respect to Gaussian distribution
    
    @param n: Number of sources
    
    @param psf: Option PSF, if active, k-sparse sources will be convolved with a PSF parameterized by sigmaGauss
    
    @param sigmaGauss: Standard deviation of Gaussian filter
    
    @return: k-sparse sources
    
    '''
    S = np.zeros((n,Nx,Ny))
    for i in np.arange(n):
        perm = np.random.permutation(Nx*Ny)
        num = np.int(Nx*Ny*pcS)
        act = perm[:num]
        amp = sigS*np.random.randn(num)
        tmp = np.zeros(Nx*Ny)
        tmp[act] = amp
        S[i] = np.reshape(tmp,(Nx,Ny))
        if psf:
            S[i] = ndimage.filters.gaussian_filter(S[i],sigma=sigmaGauss)
    S=S.squeeze()
    
    dr = 'test/sources/n'+str(n)+'/'
    # export to fits file
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
        t=sorted(glob.glob(dr+'S*.fits'))
        if len(t) == 0 :
            index=0
        
        else:
            spr = dr+'S|.fits|,'
            indarr=filter(bool,re.split(spr,','.join(t)))
            indarr=sorted(map(int,indarr))
            index=indarr[-1]+1
        
        outfit=fits.PrimaryHDU(S.astype(np.float32))
        outfit.writeto(dr+"S"+str(index)+".fits")
    return S

def createMask(Nx,Ny,pc,bd=1,export=True):
    '''
    @param Nx: rows of image
    
    @param Ny: columes of image
    
    @param pc: percentage of active data
    
    @param bd: number of bands
    
    @return: Created masks
    
    '''
    centerx,centery=Nx/2,Ny/2
    sig = 16
    Npixtot=Nx*Ny
    
    # shortspacing=7
    
    negflag=0
    
    Npoints=np.long(Npixtot*pc)
    print "Process of Creating mask ......"
    print "Random pixel selection ("+str(pc)+" - "+str(Npoints)+" pixels)"
    #uvrad=
    
    mask=np.zeros((bd,Nx,Ny),dtype=bool)
    
    # if (shortspacing):
    #     mask[centerx-3:centerx+3,centery-3:centery+3]=1
    
    # Random coordinate pick-up until unique Npoints has been found
    for i in np.arange(bd):
        coord = set()
        while len(coord) < Npoints:
            #             indx = round(sig*np.random.randn()) + centerx
            #             indy = round(abs(sig*np.random.randn())) + centery
            #             while indx >= Nx or indx < 0 or indy >= Ny or indy < 0:
            #                 indx = round(sig*np.random.randn())
            #                 indy = round(abs(sig*np.random.randn()))
            #             if (indx == 0 or indy == 0) and (Nx%2 == 0 or Ny%2 == 0):
            #                 coord.add((indx,indy))
            #             else:
            #                 coord.add((indx,indy))
            #                 coord.add((indx,indy))
            coord.add((np.random.randint(0,Nx), np.random.randint(0,Ny)))
        
        #         print len(coord)
        # Building the mask
        while len(coord) > 0:
            tpx,tpy=coord.pop()
            mask[i,tpx,tpy]=True
        
        if negflag == 1: mask[i]=~mask[i]
    mask = mask.squeeze()
    
    dr = 'test/masks/'
    # export in Fits format
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
        #         t=glob.glob(dr+'mask*.fits')
        #         if len(t) == 0 :
        #             index=0
        #
        #         else:
        #             spr=dr+'mask|.fits|,'
        #             indarr=filter(bool,re.split(spr,','.join(t)))
        #             indarr=sorted(map(int,indarr))
        #             index=indarr[-1]+1
        #
        outfit=fits.PrimaryHDU(mask.astype(np.float32))
        #         outfit.writeto(dr+"mask"+str(index)+".fits")
        outfit.writeto(dr+'mask_bd'+str(bd)+'.fits',clobber=True)
    print "Creation of mask finished."
    return mask

def createKernel(t_samp,sigma,bd=1,export=True):
    sigma_min = sigma[0]
    sigma_max = sigma[1]
    print "Process of creating PSFs ......"
    print "The standard deviation of PSFs ("+str(bd)+" channels, "+str(t_samp)+" samplings) range from "+str(sigma_min)+" to "+str(sigma_max)
    x = np.linspace(1,t_samp,t_samp)-t_samp/2
    kern = np.zeros((bd,t_samp))
    for i in np.arange(bd):
        sigma_i = (sigma_max - sigma_min)/bd*i + sigma_min
        kern[i] = np.exp(-0.5*(x/sigma_i)**2)
        kern[i] /= np.max(kern[i])
    
    dr = 'test/kernels/'
    # export in Fits format
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
        #         t=glob.glob(dr+'mask*.fits')
        #         if len(t) == 0 :
        #             index=0
        #
        #         else:
        #             spr=dr+'mask|.fits|,'
        #             indarr=filter(bool,re.split(spr,','.join(t)))
        #             indarr=sorted(map(int,indarr))
        #             index=indarr[-1]+1
        #
        outfit=fits.PrimaryHDU(kern)
        #         outfit.writeto(dr+"mask"+str(index)+".fits")
        outfit.writeto(dr+'kern_bd'+str(bd)+'.fits',clobber=True)
    print "Creating of PSFs finished."
    return kern

def createKernel2D(x_samp,y_samp,sigma,bd=1,export=True):
    sigma_min = sigma[0]
    sigma_max = sigma[1]
    x = np.linspace(1,x_samp,x_samp)-x_samp/2
    y = np.linspace(1,y_samp,y_samp)-y_samp/2
    X,Y = np.meshgrid(y,x)
    
    kern = np.zeros((bd,x_samp,y_samp))
    
    for i in np.arange(bd):
        sigma_i = (sigma_max - sigma_min)/(bd-1)*i + sigma_min
        kern[i] = np.exp(-0.5*(X**2+Y**2)/(sigma_i**2))
        kern[i] /= np.max(kern[i])
    dr = 'test/kernels/'    
    # export in Fits format
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
#         t=glob.glob(dr+'mask*.fits')
#         if len(t) == 0 :
#             index=0
#             
#         else:
#             spr=dr+'mask|.fits|,'
#             indarr=filter(bool,re.split(spr,','.join(t)))
#             indarr=sorted(map(int,indarr))
#             index=indarr[-1]+1
#       
        outfit=fits.PrimaryHDU(kern)
#         outfit.writeto(dr+"mask"+str(index)+".fits")
        outfit.writeto(dr+'kern_bd'+str(bd)+'.fits',clobber=True)
    return kern


def createMix(sigA,n,bd,export=True):
    print "Process of creating mixing matrix ......"
    print "Random creation of mixing matrix with standard deviation "+str(sigA)
    A = sigA * np.random.randn(bd,n)        # Mixing matrix
#     A /= LA.norm(A,axis=0)
    A /= np.sqrt(np.sum(A**2,axis=0))           # For numpy version 1.6
    dr = 'test/mixture/n'+str(n)+'/'
    if not os.path.exists(dr):
        os.makedirs(dr)
    # export in Fits format
    if export:
        fits.writeto(dr+'A_bd'+str(bd)+'.fits',A,clobber=True)
    print "Creation of mixing matrix finished."
    return A

def createNoise(Nx,Ny,n,sigS,db,numTests=1,export=True):
    '''
    @param Nx: rows of image
    
    @param Ny: columes of image
    
    @param n: number of sources
    
    @param sigS: standard deviation of amplitude with respect to Gaussian distribution
    
    @param db: SNR in decibel
    
    @param numTests: Number of realizations
    
    @return: Created noises
    '''
    print "Process of creating noises ......"
    print "Random creation with SNR = "+str(db)+"dB"
    sigN = sigS*10**(-db/20.)       # standard deviation of noise
    dr = 'test/noise/n'+str(n)+'/'
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
    for i in np.arange(numTests):
        N = sigN*np.random.randn(n,Nx,Ny)
        N = N.squeeze()
        if export:
            fits.writeto(dr+'noise_db'+str(int(db))+'_sigS'+str(int(np.round(sigS)))+'_r'+str(i)+'.fits',N,clobber=True)
    print "Creation of noises finished."
    return N

def createGrGauss(Nx,Ny,n,alpha,export=True):
    '''
    @param Nx: Rows of image
    
    @param Ny: Columes of image
    
    @param n: Number of sources
    
    @param alpha: Generalized Gaussian shape parameter for the coefficients distribution.
    '''
    S = np.zeros((n,Nx,Ny))
    
    for i in np.arange(n):
        S[i] = generate_2D_generalized_gaussian(Nx, Ny, alpha=alpha)
    
    S=S.squeeze()
    
    dr = 'test/sources/n'+str(n)+'/'
    # export to fits file
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
        t=sorted(glob.glob(dr+'S*.fits'))
        if len(t) == 0 :
            index=0
        
        else:
            spr = dr+'S|.fits|,'
            indarr=filter(bool,re.split(spr,','.join(t)))
            indarr=sorted(map(int,indarr))
            index=indarr[-1]+1
        
        outfit=fits.PrimaryHDU(S.astype(np.float32))
        outfit.writeto(dr+"S"+str(index)+".fits")
    return S

def generate_2D_generalized_gaussian(rows, columns, alpha=2):
    
    r"""
        
        from Matlab code BSSGUI by J. Petkov by Jakub Petkov
        
        adapted to Python by J. Rapin in order to have exact same simulated data
        
        between Matlab and Python versions
        
        
        
        Generates random variables with generalized Gaussian distribution
        
        with parameter alpha > 0 and variance 1.
        
        The generator is only approximate, the generated r.v. are bounded by 1000.
        
        
        
        Method: numerical inversion of the distribution function at points
        
        uniformly distributed in [0,1].
        
        
        
        Inputs
        
        ------
        
        - rows: int
        
        Number of rows of the data.
        
        - columns: int
        
        Number of columns of the data.
        
        - alpha (default: 2): float
        
        Generalized Gaussian shape parameter for the coefficients
        
        distribution.
        
        """
    
    m = rows
    
    n = columns
    
    r = 0.5 * np.random.random(m * n) + 0.5  # distribution is symmetric
    
    beta = np.sqrt(special.gamma(3.0 / alpha) /
                   
    special.gamma(1.0 / alpha))  # enough to consider r > 0.5
   
    y = r / beta
   
    ymin = 1e-20 * np.ones(m * n)
   
    ymax = 1000 * np.ones(m * n)
   
    # for simplicity, generated r.v. are bounded by 1000.
   
    for iter in range(0, 33):
       
        cdf = 0.5 + 0.5 * special.gammainc(1.0 / alpha, (beta * y) ** alpha)
       
        indplus = np.nonzero(cdf > r)
       
        if len(indplus) > 0:
           
            ymax[indplus] = y[indplus]
       
        indminus = np.nonzero(cdf < r)
       
        if len(indminus) > 0:
           
            ymin[indminus] = y[indminus]
       
        y = 0.5 * (ymax + ymin)
   
    ind = np.nonzero(np.random.random(m * n) > 0.5)
   
    if len(ind) > 0:
       
        y[ind] = -y[ind]
   
    x = y.reshape([n, m]).T.copy()
                   
    return x
def MixtMod(n=2,t=1024,K=5):

    import numpy as np

    S = np.zeros((n,t))
    
    for r in range(0,n):
        ind = randperm(t)
        S[r,ind[ind[0:K]]] = np.random.randn(K)
        
    return S
    
#
def MakeSources(n_s=2,t_samp=1024,K=8,w=5,export=True):

    import numpy as np
    import os
    import glob
    import pyfits as fits
    import re

    print "Process of creating sources ......"
    print str(n_s)+" approximate k-sparse sources of "+str(t_samp)+"samplings are created"
    S = MixtMod(n_s,t_samp,K)

    x = np.linspace(1,t_samp,t_samp)-t_samp/2

    kern = np.exp(-abs(x)/(w/np.log(2)))
    kern = kern/np.max(kern)

    for r in range(0,n_s):
        S[r,:] = np.convolve(S[r,:],kern,mode='same')
    
    # export to fits file 
    if export:    
        dr = 'test/sources/n'+str(n_s)+'/'
        if not os.path.exists(dr):
            os.makedirs(dr)
        
        t=sorted(glob.glob(dr+'S*.fits'))
        if len(t) == 0 :
            index=0
            
        else:
            spr = dr+'S|.fits|,'
            indarr=filter(bool,re.split(spr,','.join(t)))
            indarr=sorted(map(int,indarr))
            index=indarr[-1]+1            
            
        outfit=fits.PrimaryHDU(S.astype(np.float32))
        outfit.writeto(dr+"S"+str(index)+".fits")
    print "Creation of sources finished."    
    return S

def randperm(n=1):

    import numpy as np

    X = np.random.randn(n)
    I = X.argsort()
    
    return I
