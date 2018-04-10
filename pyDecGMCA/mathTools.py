'''
Created on Nov 4, 2015

@author: mjiang
'''
import numpy as np
import scipy.fftpack as scifft
import numpy.linalg as LA

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


##########################################################################
################ Paricular tools for GMCA algorithm ######################
##########################################################################

def normalize(A):
    '''
        Normalization of columns of mixing matrix
        
        @param A: Matrix to normailze (Normalization on columns)
        
        @return: void
        '''
    #     factor = LA.norm(A,axis=0)
    factor = np.sqrt(np.sum(A**2,axis=0))                  # For numpy version 1.6
    A /= factor

def check_zero_sources(A,S):
    '''
        @param A: Mixing matrix
        
        @param S: Sources
        
        @return: index tab of zero sources
        '''
    normA = np.sqrt(np.sum(A*A.conj(), axis=0))[np.newaxis,:]
    normS = np.sqrt(np.sum(S*S.conj(), axis=1))[:,np.newaxis]
    index = np.where(normA * normS.T == 0)[1]
    return index

def reinitialize_sources(V,S,A,index):
    '''
        Fast reinitialization of null sources in the factorization
        by picking one column in the residue.
        
        @param S: Sources
        Current factorization of the data.
        
        @param A: Mixing matrix
        
        @param V: numpy array
        Data array to be processed.
        
        @param index:
        index tab of zero sources
        
        @return: void
        
        '''
    if (len(index) > 0):
        for k in index:
            # compute residual
            R = V - A.dot(S)
            # compute square norm of residual to select maximum one
            res2 = np.sum(R * R.conj(), 0)
            j = np.where(res2 == np.max(res2))[0][0]
            if res2[j] > 0:
                A[:, k] = R[:, j] / np.sqrt(res2[j])
                # compute scalar product with the rest od the residual and
                # keep only positive coefficients
                S[k, :] = np.dot(A[:, k].conj().transpose(),R)

def find_th(S,P_min,sig,it,Imax,strategy=2,Ksig=3):
    '''
        Find thresholds using "percentage decreasing thresholds" function
        
        @param S: Source matrix (with real dimension), size: sources*Nx*Ny
        
        @param P_min: The initial percentage
        
        @param sig: Standard deviation of noise contaminated with sources, size: number of sources
        
        @param it: Current iteration
        
        @param Imax: Number of iterations demanded by the GMCA-like algorithms
        
        @param strategy: Strategy to find thresholds,
        strategy=1 means to find thresholds based on the percentage of all coefficients larger than Ksig
        strategy=2 means to find thresholds based on the percentage of coefficients larger than Ksig for each scale
        
        @param Ksig: K-sigma, level of significant coefficient
        
        @return: Table of thresholds, size: number of sources
        '''
    if np.ndim(S) == 1:
        S = S[np.newaxis,:]
    
    if strategy == 1:
        n = np.size(S,axis=0)
        th = np.zeros(n)
        
        for i in np.arange(n):
            # Order the pixels
            Stmp = np.reshape(S[i],np.size(S[i]))
            Stmp_sort = np.sort(abs(Stmp))[::-1]                # Descending order of coefficients
            Stmp_sort_sig = Stmp_sort[Stmp_sort>Ksig*sig[i]]        # Compared with K-sigma
            P_ind = P_min + (1.0-P_min)/(Imax-1)*it                 # Current percentage
            index = int(np.ceil(P_ind*len(Stmp_sort_sig))-1)
            
            if index < 0:
                index = 0
            if len(Stmp_sort_sig) == 0:
                th[i] = Stmp_sort[1]
            else:
                th[i] = Stmp_sort_sig[index]
    
    elif strategy == 2:
        n = np.size(S,axis=0)
        sc = np.size(S,axis=1)
        th = np.zeros((n,sc))
        for i in np.arange(n):
            Stmp = np.reshape(S[i],(sc,np.size(S[i])/sc))
            for l in np.arange(sc):
                Stmp_sort = np.sort(abs(Stmp[l]))[::-1]         # Descending order of coefficients
                Stmp_sort_sig = Stmp_sort[Stmp_sort>Ksig*sig[i,l]]      # Compared with K-sigma
                P_ind = P_min + (1.0-P_min)/(Imax-1)*it                 # Current percentage
                index = int(np.ceil(P_ind*len(Stmp_sort_sig))-1)
                if index < 0:
                    index = 0
                if len(Stmp_sort_sig) == 0:
                    th[i,l] = Stmp_sort[1]
                else:
                    th[i,l] = Stmp_sort_sig[index]
    S = S.squeeze()
    return th

def DeconvFwd(V,kern,epsilon):
    '''
        Deconvolution using ForWaRD method
        (only perform the Fourier-based regularization, the wavelet-based regularization is not included)
        
        @param V: Input data in Fourier space
        
        @param kern: Ill-conditioned linear operator
        
        @param epsilon: Regularization parameter
        
        @return: Data deconvolved from the linear operator
        '''
    (Bd,N) = np.shape(kern)
    X = np.zeros_like(V)
    for nu in np.arange(Bd):
        denom = kern[nu].conj()*kern[nu]
        rho = np.max(denom)
        denom = denom + epsilon*max(rho,1e-4)*np.ones(N)
        denom = 1./denom
        numr = kern[nu].conj() * V[nu]
        X[nu] = denom*numr
    
    return X

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

