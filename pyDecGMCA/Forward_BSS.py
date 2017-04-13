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
import pyfits as fits
import pyWavelet.waveTools as pm
import sys

def update_S(V,A,M=1,mask=True,deconv=False,epsilon=0.):
    '''
    Estimate sources using least square
    
    @param V: Visibilities, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param mask: Whether in the case of masking, default True
    
    @param deconv: Whether in the case of blurring, default False
    
    @param epsilon: Parameter of regularization
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A)
    S = np.zeros((N,P)) + np.zeros((N,P))*1j
    if mask and not deconv: 
        for k in np.arange(P):
            ind = tuple(np.where(M[:,k]==1)[0])
            numr = np.dot(A[ind,:].conj().transpose(),V[ind,k])
            denom = np.dot(A[ind,:].conj().transpose(),A[ind,:])
            rho = LA.norm(denom,2)
            denom = LA.inv(denom+epsilon*max(rho,1e-4)*np.eye(N))
            S[:,k] = np.dot(denom,numr)
    elif deconv:
        for k in np.arange(P):
            numr = np.dot(A.conj().transpose(),M[:,k]*V[:,k])
            denom = np.dot((M[:,k][:,np.newaxis]*A).conj().transpose(),M[:,k][:,np.newaxis]*A)
            rho = LA.norm(denom,2)
            denom = LA.inv(denom+epsilon*max(rho,1e-4)*np.eye(N))
            S[:,k] = np.dot(denom,numr)        
    else:
        denom = np.dot(A.conj().transpose(),A)
        if LA.cond(denom) < 1/sys.float_info.epsilon:
            denom = LA.inv(denom)
        else:
            rho = LA.norm(denom,2)
            denom = LA.inv(denom+max(1e-10*rho,1e-10)*np.eye(N)) 
        numr = np.dot(A.conj().transpose(),V)
        S = np.squeeze(np.dot(denom,numr))
    return S

def update_A(V,S,M=1,mask=True,deconv=False):
    '''
    Estimate mixing matrix using least square
    
    @param V: Visibilities, size: bands*pixels
    
    @param S: Matrix of sources, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param mask: Whether in the case of masking, default True
    
    @param deconv: Whether in the case of blurring, default False
    
    @return: Updated A, size: bands*sources
    '''
    (Bd,P) = np.shape(V)
    (N,P) = np.shape(S)
    A = np.zeros((Bd,N)) + np.zeros((Bd,N)) *1j
    if mask and not deconv:            
        for nu in np.arange(Bd):
            ind = tuple(np.where(M[nu]==1)[0])
            numr = V[nu,ind].dot(S[:,ind].conj().transpose())
            denom = np.dot(S[:,ind],S[:,ind].conj().transpose())
            if LA.cond(denom) < 1/sys.float_info.epsilon:
                denom = LA.inv(denom)
            else:
                rho = LA.norm(denom,2)
                denom = LA.inv(denom+max(1e-4*rho,1e-4)*np.eye(N))
            A[nu,:] = np.dot(numr,denom)
    elif deconv:
        for nu in np.arange(Bd):
            numr = np.dot(M[nu,:]*V[nu,:],S.conj().transpose())
            denom = np.dot(M[nu,:]*S,(M[nu,:]*S).conj().transpose())
            if LA.cond(denom) < 1/sys.float_info.epsilon:
                denom = LA.inv(denom)
            else:
                rho = LA.norm(denom,2)
                denom = LA.inv(denom+max(1e-4*rho,1e-4)*np.eye(N))
            A[nu,:] = np.dot(numr,denom)
             
    else:        
        numr = np.dot(V,S.conj().transpose())
        denom = np.dot(S,S.conj().transpose())
        if LA.cond(denom) < 1/sys.float_info.epsilon:
            denom = LA.inv(denom)
        else:
            rho = LA.norm(denom,2)
            denom = LA.inv(denom+max(1e-4*rho,1e-4)*np.eye(N))
        A = np.dot(numr,denom)
    return A

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
    
    @param iter: Current iteration
    
    @param Imax: Number of iterations demanded by the GMCA-like algorithms
    
    @param strategy: Strategy to find thresholds, 
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale 
    
    @return: Table of thresholds, size: number of sources
    '''
    if np.ndim(S) == 1:
        S = S[np.newaxis,:]
        
    if strategy == 1:
        n = np.size(S,axis=0)
    #     (n,Nx,Ny) = np.shape(S)
        th = np.zeros(n)
        
        for i in np.arange(n):
    #         if wavelet and i == n-1:
    #             th[i] = np.Inf
    #             continue
            # Order the pixels
            Stmp = np.reshape(S[i],np.size(S[i]))
    #         if wavelet and i == 0:
    #             Stmp_sort = np.sort(abs(Stmp))[::-1]
    #             Stmp_sort_sig = Stmp_sort[Stmp_sort>4*sig[i]]
    # #             Stmp_sort = np.sort(abs(Stmp[abs(Stmp)>4*sig[i]]))
    #         else:
            Stmp_sort = np.sort(abs(Stmp))[::-1]
            Stmp_sort_sig = Stmp_sort[Stmp_sort>Ksig*sig[i]]
    #         Stmp_sort_sig = Stmp_sort[Stmp_sort>(4-1.*it/(Imax-1))*sig[i]]
    #             Stmp_sort_sig = np.sort(abs(Stmp[abs(Stmp)>3*sig[i]]))
    #             Stmp_sort_sig = Stmp_sort_sig[::-1]
    #         Stmp_sort = Stmp_sort[::-1]
            ############### Percentage of pixels greater than 3 sigma ######################
    #         P_ind = int(len(Stmp_sort)*float(it+1)/Imax)
    #         P_ind = max(P_ind,P_min)
            # Percentage of pixels
    #         th[i] = max(Stmp_sort[P_ind-1],3*sig[i])
            P_ind = P_min + (1-P_min)/(Imax-1)*it
            index = int(np.ceil(P_ind*len(Stmp_sort_sig))-1)
            
    #         P_ind = P_min + float(len(Stmp_sort_sig) - P_min)/(Imax-1)*it
    #         index = np.ceil(P_ind)-1
            if index < 0:
                index = 0
            if len(Stmp_sort_sig) == 0:
                th[i] = Stmp_sort[1]
            else:
                th[i] = Stmp_sort_sig[index]
            # Percentage of amplitude
    #         th[i] = min(P_ind*Stmp_sort[0],3*sig[i])
    elif strategy == 2:
        n = np.size(S,axis=0)
        sc = np.size(S,axis=1)
        th = np.zeros((n,sc))
        for i in np.arange(n):
            Stmp = np.reshape(S[i],(sc,np.size(S[i])/sc))
            for l in np.arange(sc):
                Stmp_sort = np.sort(abs(Stmp[l]))[::-1]
                Stmp_sort_sig = Stmp_sort[Stmp_sort>Ksig*sig[i,l]]
                P_ind = P_min + (1.0-P_min)/(Imax-1)*it
                index = int(np.ceil(P_ind*len(Stmp_sort_sig))-1)
                if index < 0:
                    index = 0
                if len(Stmp_sort_sig) == 0:
                    th[i,l] = Stmp_sort[1]
                else:
                    th[i,l] = Stmp_sort_sig[index]
    S = S.squeeze()
    return th

def GMCA_fast(V,n,Imax):
    '''
    Basic GMCA algorithm using least square.
    
    @param V: Input data, size: bands*pixels
    
    @param n: Number of sources
    
    @param Imax: Number of iterations
    
    @return: Recovered sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    
    (Bd,P) = np.shape(V)
    (u,d,v) = LA.svd(V,full_matrices=False)
    A = u[:n,:].T
#     np.random.seed(5)
#     A = np.random.randn(Bd,n) + 0*1j*np.random.randn(Bd,n)
    normalize(A)
    thIter = np.zeros((Imax,n))
#     nsNorm(P, scale, gen2=gen2)
        
    P_min = 0.05
#     dtau = float(tau - tauF)/Imax
    for i in np.arange(Imax):
        S = update_S(V,A,mask=False,deconv=False)
        ############################## Coefficient processing ##########################
        sig = mad(S)    
        thTab = find_th(S,P_min,sig,i,Imax,strategy=1)
        thIter[i] = np.copy(thTab)
        hardTh(S,thTab,weights=None,reweighted=False)
                
        A = update_A(V,S,mask=False,deconv=False)
        A = np.real(A)
        normalize(A)
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S hat:"
        print LA.cond(S), 'Shat'
    return (S,A)

def AMCAWavelet(X,nOs,t, n, kend=3, nmax=300,aMCA=1, l1=1,wname='starlet',band=None):
    '''AMCA algorithm for on wavelet coefficients
    -Inputs: X  observations in the transformed domain of size m times (nOs+1)*t (undecimated wavelet transformed, nOs scales for image, coefficients of the smallest scale on the left)
    nOs: number of scales (nOs+1)
    t: number of pixels of the image
    n: number of components
    kend: final threshold k-mad
    nmax: total number of iterations
    aMCA: one for AMCA, zero for GMCA
    l1: one for l1-norm, zero for l0-norm
    wname: wavelet name
    band: wavelet bookkeeping vector/matrix, only used for decimated wavelet
    -Output: Sws wavelet coefficients of the sources (size n times (nOs*3+1)*t), A mixing matrix (size m by n)'''
    
    R = np.dot(X,X.T)
    d,v = LA.eig(R)
    A = np.abs(v[:,0:n])
    
    if wname=='starlet':
        tTot=t*nOs
        Xw=X[:,0:tTot]# Keep only wav.coeff, assume image (3D), nOs scale, non-decimated
    else:
        tTot = np.size(X[0]) - int(band[0,0]*band[0,1])
        Xw= X[:,int(band[0,0]*band[0,1]):]# Keep only wav.coeff
 
   
    S=np.zeros((n,tTot))   
    

    W=np.ones((tTot))        

    kIni=10
    dk=(kend-kIni)/nmax
    alpha = 1;
    perc = 1./nmax
 
    Go_On = 1
    it = 0
    thrd=0
    while Go_On==1:
        it += 1
        if it == nmax :
            Go_On = 0          
        
        #Estimation S
        sigA = np.sum(A*A,axis=0)
        indS = np.where(sigA > 0)[0]
        #Inversion A
        if np.size(indS) > 0: 
            Ra = np.dot(A[:,indS].T,A[:,indS])
            Ua,Sa,Va = np.linalg.svd(Ra)
            cd_Ra = np.min(Sa)/np.max(Sa)
            
            if cd_Ra > 1e-8: # If small condition number
                iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                piA = np.dot(iRa,A[:,indS].T)    
                S[indS,:] = np.dot(piA,Xw)
            else:                   
                La = np.max(Sa)
                for it_A in range(0,250):
                    S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,Xw - np.dot(A[:,indS],S[indS,:]))
             
             
            #Thresholding
            Stemp = S[indS,:] 
            if wname == 'starlet':
                madS=madAxis(S[:,0:t], axis=1) #mad of the smallest scale for each source
            else:
                k = np.size(band,axis=0) - 2
                first = int(3*np.sum(band[1:k,0]*band[1:k,1]))
                add = int(band[k,0]*band[k,1])
                madS=madAxis(S[:,first:first+3*add],axis=1)
                
            for r in range(np.size(indS)):
                
                St = Stemp[r,:]
                if wname=='starlet':
                    for i in range(np.int(nOs)):#Increase number of coefficients per scale, can be changed
                        Sint=St[i*t:(i+1)*t]
                        indNZ=np.where(abs(Sint) > ((kIni+it*dk)*madS[r]))[0]
                        if np.size(indNZ)==0:
                            indNZ=np.where(np.abs(Stemp[r,i*t:(i+1)*t])>np.percentile(np.abs(Stemp[r,i*t:(i+1)*t]),100*(t-n)/float(t)))[0]
                        Kval = np.min([np.floor(perc*(it)*len(indNZ)),t-1.])
                        I = (abs(Sint[indNZ])).argsort()[::-1]
                        Kval = min(max(Kval,n),len(I)-1)
                        thrd=abs(Sint[indNZ[I[int(Kval)]]])
                      
                      
                      
                        Sint[abs(Sint)<thrd]=0
                        indNZ = np.where(abs(Sint) >= thrd)[0]
                        if l1:
                            Sint[indNZ] = Sint[indNZ] - thrd*np.sign(Sint[indNZ]) #for l1
                        Stemp[r,i*t:(i+1)*t]=Sint.copy()
                else:
                    indNZ=np.where(abs(St) > ((kIni+it*dk)*madS[r]))[0]
                    if np.size(indNZ)==0:
                        indNZ=np.where(np.abs(St)>np.percentile(np.abs(St),100*(t-n)/float(t)))[0]
                    Kval = np.min([np.floor(perc*(it)*len(indNZ)),t-1.])
                    I = (abs(St[indNZ])).argsort()[::-1]
                    Kval = min(max(Kval,n),len(I)-1)
                    thrd=abs(St[indNZ[I[int(Kval)]]])
                    St[abs(St)<thrd]=0
                    indNZ = np.where(abs(St) >= thrd)[0]
                    if l1:
                        St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ]) #for l1
                    Stemp[r]=St.copy()
#                           
            S[indS,:] = Stemp
            
       
        Sref=S.copy()
        #Weight for AMCA
        if aMCA==1  and it>1:
            alpha=0.1**((it-1.)/(nmax-1.))/2.
            Ns = np.sqrt(np.sum(Sref*Sref,axis=1))
            IndS = np.where(Ns > 0)[0] 
            
            if len(IndS)>0:
                Sref[IndS,:] = np.dot(np.diag(1./Ns[IndS]),Sref[IndS,:])
                W = np.power(np.sum(np.power(abs(Sref[IndS,:]),alpha),axis=0),1./alpha)
                ind = np.where(W > 0)[0]
                jind = np.where(W == 0)[0]
                W[ind] = 1./W[ind];   
                W/=np.median(W[ind])
                if len(jind) > 0:
                    W[jind] = 1
                W[W>=1]=1    
                         
        Sref=S.copy()
        Ns = np.sqrt(np.sum(Sref*Sref,axis=1))
        indA = np.where(Ns > 0)[0]
        #EstimA
        if len(indA) > 0:
            Sr = Sref.copy()*W
            Rs = np.dot(Sref[indA,:],Sr[indA,:].T)
            Us,Ss,Vs = np.linalg.svd(Rs)
            cd_Rs = np.min(Ss)/np.max(Ss)
            if cd_Rs > 1e-10: #If small condition number
                piS = np.dot(Sr[indA,:].T,np.linalg.inv(Rs));
                A[:,indA] = np.dot(Xw,piS)
                A=np.abs(A)
                A = np.dot(A,np.diag(1./(1e-24 + np.sqrt(np.sum(A*A,axis=0)))));
            else:
                #--- iterative update
                Ls = np.max(Ss)
                indexSub=0
                while indexSub<250:
                    A[:,indA] = A[:,indA] + 1/Ls*np.dot(Xw - np.dot(A[:,indA],Sref[indA,:]),Sr[indA,:].T)
                    A=np.maximum(A, 0)
                    A[:,indA] = np.dot(A[:,indA],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indA]*A[:,indA],axis=0)))));
                    indexSub+=1

    
    
    if wname == 'starlet':
        Scs=np.linalg.pinv(A).dot(X[:,-t::])#Inverse coarse scall
        Sws=np.hstack((S,Scs)) #concatenate for all coefficients
    else:
        Scs=np.linalg.pinv(A).dot(X[:,0:int(band[0,0]*band[0,1])])
        Sws=np.hstack((Scs,S)) #concatenate for all coefficients
    return Sws,A

#####################################################################################

def madAxis(xin,axis='none'):
    if axis=='none':
        z = np.median(abs(xin - np.median(xin)))/0.6735

        return z
    else:
        z = np.median(abs(xin - np.median(xin,axis=1).reshape((np.shape(xin)[0],1))),axis=1)/0.6735
        return z.reshape((np.shape(xin)[0],1))    


def GMCA(V,n,Nx,Ny,Imax,Ndim,wavelet,scale,wname='starlet',thresStrtg=1,FTPlane=True,fc=1./16,logistic=False,Ar=None):
    '''
    Basic GMCA algorithm using least square with wavelet transform intergrated.
    
    @param V: Input data, size: bands*pixels
    
    @param n: Number of sources
    
    @param Nx: number of rows of the source
    
    @param Ny: number of columns of the source
    
    @param Imax: Number of iterations
    
    @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data are in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @return: Recovered sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    
    (Bd,P) = np.shape(V)
    # If the input data are in Fourier space, in order to garantee the quality of the separation, the raw data are
    # pre-processed by applying a high-pass filter. The form of the high-pass filter can be a logistic function for
    # a mono-dimensional signal or a step function for a two dimensional image.
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
            
    elif Ndim == 2 and FTPlane:
        V1 = np.reshape(V,(Bd,Nx,Ny))
        V_Hi = filter_Hi(V1,Ndim,fc)
        V_Hi = np.reshape(V_Hi,(Bd,P))
    else:
        V_Hi = V   
    
    # SVD decomposition to initialize A
    (u,d,v) = LA.svd(V,full_matrices=False)
    A = u[:n,:].T

    normalize(A)
    ############################ Save information of wavelet (Starlet and DCT only), ################################
    if wavelet:
        pm.trHead = ''
        if wname == 'starlet':
            gen2 = False
            pm.trHead = 'star'+str(int(Ndim))+'d_gen2' if gen2 else 'star'+str(int(Ndim))+'d_gen1'
        else:
            pm.trHead = wname+'_'+str(int(Ndim))+'d'    

    if thresStrtg == 1:
        thIter = np.zeros((Imax,n))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,n,scale-1))
        
    P_min = 0.1
        
    print "Main cycle" 
    
    for i in np.arange(Imax):

        Shat = update_S(V,A,mask=False,deconv=False,epsilon=0)

        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))                    # Transform in direct plane
        else:
            S = np.real(np.reshape(Shat,(n,Nx,Ny)).squeeze())
            
        ############################## For wavelet representation ########################## 
        if wavelet:
            if wname == 'starlet':
                if Ndim == 1:
                    coarseScale = np.zeros((n,1,P))
                    wt = np.zeros((n,scale-1,P))                    # For starlet transform
                elif Ndim == 2:
                    coarseScale = np.zeros((n,1,Nx,Ny))
                    wt = np.zeros((n,scale-1,Nx,Ny))                    # For starlet transform
                for sr in np.arange(n):
                    if Ndim == 1:
                        wtTmp = star1d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)         # 1d Starlet transform
                    else:
                        wtTmp = star2d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)         # 2d Starlet transform
                    # Remove coarse scale
                    coarseScale[sr] = np.copy(wtTmp[-1])                    
                    wt[sr] = np.copy(wtTmp[:-1]) 
                    
            elif wname == 'dct':
                wt = np.zeros((n,Nx,Ny)).squeeze()
                coarseScale = np.zeros(n)
                for sr in np.arange(n):
                    if Ndim == 1:
                        wt[sr] = scifft.dct(S[sr],type=2,norm='ortho')
                        coarseScale[sr] = wt[sr,0]
                        wt[sr,0] = 0
                    elif Ndim == 2:
                        wt[sr] = dct2(S[sr],type=2,norm='ortho')
                        coarseScale[sr] = wt[sr,0,0]
                        wt[sr,0,0] = 0
                        
            else:
                for sr in np.arange(n):
                    if Ndim == 1:
                        wtTmp,bd = wavOrth1d(S[sr],nz=scale,wname=wname)
                        if sr == 0:
                            wlen = len(wtTmp[bd[0]:])
                            wt = np.zeros((n,wlen))
                            coarseScale = np.zeros((n,bd[0]))
                        wt[sr] = wtTmp[bd[0]:]
                        coarseScale[sr] = wtTmp[:bd[0]]
                    elif Ndim == 2:
                        wtTmp,bd = wavOrth2d(S[sr],nz=scale,wname=wname)
                        if sr == 0:
                            wlen = len(wtTmp[bd[0,0]*bd[0,1]:])
                            wt = np.zeros((n,wlen))
                            coarseScale = np.zeros((n,bd[0,0]*bd[0,1]))
                        wt[sr] = wtTmp[bd[0,0]*bd[0,1]:]
                        coarseScale[sr] = wtTmp[:bd[0,0]*bd[0,1]]
            ################## Noise estimate####################
            if thresStrtg == 1:
                if wname == 'dct':
                    ##################### Use the finest scale of starlet to estimate noise level ##############################
                    fineScale = np.zeros((n,P))
                    for sr in np.arange(n):
                        if Ndim == 1:
                            wtTmp = star1d(S[sr], scale=4, fast=True, gen2=False,normalization=True)         # 1d Starlet transform
                        elif Ndim == 2:
                            wtTmp = star2d(S[sr], scale=4, fast=True, gen2=False,normalization=True)         # 2d Starlet transform
                        fineScale[sr] = wtTmp[0].flatten()
                    sig = mad(fineScale)
#                     sig = mad(wt[:,Nx/2:Nx,Ny/2:Ny])
                elif wname == 'starlet':
                    wt = np.real(np.reshape(wt,(n,np.size(wt)/n)).squeeze())
                    sig = mad(wt[:,:P])
                else:
                    sig = mad(wt)
            elif thresStrtg == 2:
                if wname == 'starlet':
                    sig = np.zeros((n,scale-1))
                    for sr in np.arange(n):
                        sig[sr] = mad(wt[sr]) 
           
            thTab = find_th(wt,P_min,sig,i,Imax,strategy=thresStrtg)
            thIter[i] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:                
                hardTh(wt,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(n):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)            
#             wt = np.reshape(wt,(n,np.size(wt)/(n*P),P))             # For undecimated wavelet transform
            if wname == 'starlet':
                if thresStrtg == 1:
                    wt = np.reshape(wt,(n,scale-1,Nx,Ny)).squeeze()
            elif wname == 'dct':
                wt = wt.reshape((n,Nx,Ny)).squeeze()
            else:
#                 wt = np.concatenate((coarseScale,wt),axis=1)
                wt = np.concatenate((np.zeros((n,bd[0,0]*bd[0,1])),wt),axis=1)
            for sr in np.arange(n):
                if wname == 'starlet':
                    wtTmp = np.concatenate((wt[sr],coarseScale[sr]),axis=0)
                    if Ndim == 1:
                        S[sr] = istar1d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform
                    elif Ndim == 2:
                        S[sr] = istar2d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform
                elif wname == 'dct':
                    if Ndim == 1:
                        wt[sr,0] = coarseScale[sr]
                        S[sr] = scifft.dct(wt[sr],type=3,norm='ortho')                  # Inverse 1d dct transform
                    elif Ndim == 2:
                        wt[sr,0,0] = coarseScale[sr]
                        S[sr] = dct2(wt[sr],type=3,norm='ortho')                        # Inverse 2d dct transform
                else:
                    if Ndim == 1:
                        S[sr] = iwavOrth1d(wt[sr],bd,wname=wname)
                    elif Ndim == 2:
                        S[sr] = iwavOrth2d(wt[sr],bd,wname=wname)
                    
        #################### For non-wavelet representation ###########################        
        else:
            sig = mad(S)             
            thTab = find_th(S,P_min,sig,i,Imax,strategy=1)
            hardTh(S,thTab,weights=None,reweighted=False)
        
        index = check_zero_sources(A,S)
        if len(index) > 0:
            S = np.reshape(S,(n,Nx*Ny))
            reinitialize_sources(V,S,A,index)
            S = np.reshape(S,(n,Nx,Ny))
        
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
        
        if FTPlane:
            if wavelet:
                A = update_A(V_Hi,Shat_Hi,mask=False,deconv=False)
            else:
                A = update_A(V,Shat,mask=False,deconv=False)
        else:
            A = update_A(V,S,mask=False,deconv=False)
        A = np.real(A)
        normalize(A)
        
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S hat:"
        print LA.cond(Shat), 'Shat'
    
    ####################### Add coarse scale to the final solution ##########################
#     if wavelet:
#         if wname == 'starlet':
#             Swt = np.zeros((n,scale,Nx,Ny)).squeeze()
#             Swt[:,:-1] = np.copy(wt)
#             Swt[:,np.newaxis,-1] = np.copy(coarseScale)
#             S = np.zeros((n,Nx,Ny)).squeeze()
#             for sr in np.arange(n):
#                 if Ndim == 1:
#                     S[sr] = istar1d(Swt[sr],fast=True,gen2=gen2,normalization=True)        # Inverse starlet transform
#                 elif Ndim == 2:
#                     S[sr] = istar2d(Swt[sr],fast=True,gen2=gen2,normalization=True)        # Inverse starlet transform
    if Ndim == 2:
        S = np.reshape(S,(n,Nx,Ny))
#         elif wname == 'dct':
#             for sr in np.arange(n):
#                 if Ndim == 1:
#                     wt[sr,0] = coarseScale[sr]
#                     S[sr] = scifft.dct(wt[sr],type=3,norm='ortho')
#                 elif Ndim == 2:
#                     wt[sr,0,0] = coarseScale[sr]
#                     S[sr] = dct2(wt[sr],type=3,norm='ortho') 
                                   
    return (S,A)

def DecGMCA(V,M,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,postProc=0,postProcImax=50,Ksig=3.0,positivityS=False,positivityA=False,A=None):
    '''
    Deconvolution GMCA algorithm to solve simultaneously deconvolution and Blind Source Separation (BSS)
    
    @param V: Visibilities, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @param n: Number of sources
    
    @param Nx: number of rows of the source
    
    @param Ny: number of columns of the source
    
    @param Imax: Number of iterations
    
    @param epsilon: Parameter of regularization 
    
    @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @param deconv: Whether in the case of blurring, default False
    
    @param wname: Wavelet type.
    
    @param thresStrtg: Strategy to find the thresholding, valid just for parameter wavelet=True
    strategy=1 means to find the thresholding level based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find the thresholding level based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    
    (Bd,P) = np.shape(V)
    #################### Matrix completion for initilize A ###########################
#     if mask:
#         Xe = SVTCompletion(V,M,n,1.5,200)
#     elif deconv:
#         flogist1 = 1./(1+np.exp(-0.03*np.linspace(-(1./8*P-fc*P)/2,(1./8*P-fc*P)/2,1./8*P-fc*P)))
#         flogist2 = 1./(1+np.exp(0.03*np.linspace(-(1./8*P-fc*P)/2,(1./8*P-fc*P)/2,1./8*P-fc*P)))
#         V_Hi = filter_Hi(V,Ndim,fc)
#         V_Hi[:,int(fc*P):int(1./8*P)] = V_Hi[:,int(fc*P):int(1./8*P)]*flogist1
#         V_Hi[:,int(-1./8*P):int(-fc*P)] = V_Hi[:,int(-1./8*P):int(-fc*P)]*flogist2
#         Xe = V_Hi
#     else:
#         Xe = V
        
#     logistic = True
    
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
            
    elif Ndim == 2 and FTPlane:
        V_Hi = filter_Hi(np.reshape(V,(Bd,Nx,Ny)),Ndim,fc)
        V_Hi = np.reshape(V_Hi,(Bd,P))
    else:
        V_Hi = V
    
        
    if mask:
        MM = np.zeros_like(M)
        MM[np.abs(M)>=1e-4] = 1
        MM = np.real(MM)
        Xe = SVTCompletion(V,MM,n,0.5,250)
    elif deconv and not mask and FTPlane:
        Xe = V_Hi
    else:
        Xe = V
     
#     (u,d,v) = LA.svd(Xe,full_matrices=False)
#     A = np.abs(u[:,:n])   
    R = np.dot(Xe,Xe.T)
    d,v = LA.eig(R)
    A = np.abs(v[:,0:n])
       
#     A = np.copy(Ar)
    normalize(A)
    
    ############################ Save information of wavelet (Starlet and DCT only), ################################
    if wavelet:
        pm.trHead = ''
        if wname == 'starlet':
            gen2 = False
            pm.trHead = 'star'+str(int(Ndim))+'d_gen2' if gen2 else 'star'+str(int(Ndim))+'d_gen1'
        else:
            pm.trHead = wname+'_'+str(int(Ndim))+'d'  
    
    if thresStrtg == 1:
        thIter = np.zeros((Imax,n))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,n,scale-1))
        
#     Shat = np.random.randn(n,P) + 1j*np.random.randn(n,P)
#     tau = 10.
    P_min = 0.05
    
    if mask or deconv:
        epsilon_log = np.log10(epsilon)
        epsilonF_log = np.log10(epsilonF)
#     dtau = float(tau - tauF)/Imax  
        
    print "Main cycle" 
    
    for i in np.arange(Imax):
        if mask or deconv:
            epsilon_iter = 10**(epsilon_log - (epsilon_log - epsilonF_log)*float(i)/(Imax-1))       # Exponential linearly decreased
#                 epsilon_iter = epsilon - (epsilon - epsilonF)/(Imax-1)*float(i)                # Linear decreased
        else:
            epsilon_iter = epsilon
 
        Shat = update_S(V,A,M,mask=mask,deconv=deconv,epsilon=epsilon_iter)
#             Shat = update_S(Xe,A,M,mask=mask,epsilon=epsilon_iter)
#         Shat = np.dot(LA.inv(np.dot(A.T,A)),np.dot(A.T,V))
        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))                    # In direct plane
        else:
            S = np.real(np.reshape(Shat,(n,Nx,Ny)).squeeze())    
            
        ############################## For wavelet representation ########################## 
        if wavelet:           
            
            if wname == 'starlet':                
                if Ndim == 1:
                    coarseScale = np.zeros((n,1,P))
                    wt = np.zeros((n,scale-1,P))                    # For starlet transform
                elif Ndim == 2:
                    coarseScale = np.zeros((n,1,Nx,Ny))
                    wt = np.zeros((n,scale-1,Nx,Ny))                    # For starlet transform
                for sr in np.arange(n):
                    if Ndim == 1:
                        wtTmp = star1d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)         # 1d Starlet transform
                    elif Ndim == 2:
                        wtTmp = star2d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)         # 2d Starlet transform
                    # Remove coarse scale
                    coarseScale[sr] = np.copy(wtTmp[-1])                    
                    wt[sr] = np.copy(wtTmp[:-1]) 
                     
            elif wname == 'dct':
                wt = np.zeros((n,Nx,Ny)).squeeze()
                coarseScale = np.zeros((n,Nx,Ny)).squeeze()
                for sr in np.arange(n):
                    if Ndim == 1:
                        wt[sr] = scifft.dct(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Ny/16] = wt[sr,:Ny/16]
                        wt[sr,:Ny/16] = 0
                    elif Ndim == 2:
                        wt[sr] = dct2(S[sr],type=2,norm='ortho')
                        coarseScale[sr,:Nx/16,:Ny/16] = wt[sr,:Nx/16,:Ny/16]
                        wt[sr,:Nx/16,:Ny/16] = 0
            ################## Noise estimate####################
            if thresStrtg == 1:
                if wname == 'dct':
                    ##################### Use the finest scale of starlet to estimate noise level ##############################
                    fineScale = np.zeros((n,P))
                    for sr in np.arange(n):
                        if Ndim == 1:
                            wtTmp = star1d(S[sr], scale=4, fast=True, gen2=False,normalization=True)         # 1d Starlet transform
                        elif Ndim == 2:
                            wtTmp = star2d(S[sr], scale=4, fast=True, gen2=False,normalization=True)         # 2d Starlet transform
                        fineScale[sr] = wtTmp[0].flatten()
                    sig = mad(fineScale)
                else:
                    wt = np.real(np.reshape(wt,(n,np.size(wt)/n)).squeeze())
                    sig = mad(wt[:,:P])
#                     sig = np.array([0.01,0.01])
            elif thresStrtg == 2:
                sig = np.zeros((n,scale-1))
                for sr in np.arange(n):
#                 sig = mad(wt[:,:P])                                             # For starlet transform
                    sig[sr] = mad(wt[sr]) 
           
            thTab = find_th(wt,P_min,sig,i,Imax,strategy=thresStrtg)
#             thTab = tau*sig
            thIter[i] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
                hardTh(wt,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(n):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)            
            
#             wt = np.reshape(wt,(n,np.size(wt)/(n*P),P))             # For undecimated wavelet transform
            
            if wname == 'starlet':
                if thresStrtg == 1:
                    wt = np.reshape(wt,(n,scale-1,Nx,Ny)).squeeze()
            elif wname == 'dct':
                wt = wt.reshape((n,Nx,Ny)).squeeze()
            
            for sr in np.arange(n):
                if wname == 'starlet':
                    wtTmp = np.concatenate((wt[sr],coarseScale[sr]),axis=0)
                    if Ndim == 1:
                        S[sr] = istar1d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform
                    elif Ndim == 2:
                        S[sr] = istar2d(wtTmp, fast=True, gen2=gen2, normalization=True)        # Inverse starlet transform
                    
                elif wname == 'dct':
                    if Ndim == 1:
                        wt[sr,:Ny/16] = coarseScale[sr,:Ny/16]
                        S[sr] = scifft.dct(wt[sr],type=3,norm='ortho')                  # Inverse 1d dct transform
                    elif Ndim == 2:
                        wt[sr,:Nx/16,:Ny/16] = coarseScale[sr,:Nx/16,:Ny/16]
                        S[sr] = dct2(wt[sr],type=3,norm='ortho')
        #################### For non-wavelet representation ###########################        
        else:
            sig = mad(S)
#             thTab = tau*sig             
            thTab = find_th(S,P_min,sig,i,Imax,strategy=thresStrtg)
            thIter[i] = thTab
#         print thTab
            hardTh(S,thTab,weights=None,reweighted=False)
            
        if positivityS:
            S[S<0] = 0
        
        index = check_zero_sources(A,Shat)
        if len(index) > 0:
            reinitialize_sources(V,Shat,A,index)
        
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
        
        if FTPlane:
            if wavelet:
                A = update_A(V_Hi,Shat_Hi,M,mask=mask,deconv=deconv)
            else:
                A = update_A(V,Shat,M,mask=mask,deconv=deconv)
        else:
            A = update_A(V,S,M,mask=mask,deconv=deconv)
        
        A = np.real(A)
        if positivityA:
            A[A<0] = 0
        normalize(A)
        
        if i%100 == 0:
            fits.writeto('test5/estA'+str(i)+'.fits',A,clobber=True)
            fits.writeto('test5/estS'+str(i)+'.fits',S,clobber=True)
        
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S:"
        print LA.cond(np.reshape(S,(n,P))), 'S'    
    
    if Ndim == 2:
        S = np.reshape(S,(n,Nx,Ny))
#     ####################### Add coarse scale to the final solution ##########################
#     if wavelet:
#         if wname == 'starlet':
#             Swt = np.zeros((n,scale,P))
#             Swt[:,:-1] = np.copy(wt)
#             Swt[:,np.newaxis,-1] = np.copy(coarseScale)   
#         S = np.zeros((n,P))
#         for sr in np.arange(n):
#             if wname == 'starlet':
#                 S[sr] = istar1d(Swt[sr],fast=True,gen2=gen2,normalization=True)                  # Starlet transform 
#     ####################### Ameliorate the estimation of the sources ##########################
    if postProc == 1:
        print 'Post processing to ameliorate the estimate S:'
        S,thIter=update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax=postProcImax,tau=0.0,eta=0.5,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivityS)
    elif postProc == 2:
        print 'Post processing to ameliorate the estimate S and estimate A:'
        inImax1 = 50
        inImax2 = 1
        tau = 0.0
        mu2 = 0.0
        eta = 0.5
        for i in np.arange(postProcImax):
            S,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax=inImax1,tau=tau,eta=eta,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivityS)
            
            if FTPlane:
                Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
                if wavelet:
                    ################# Don't take the low frequency band into account ######################
                    Shat_Hi = filter_Hi(Shat,Ndim,fc)
                    if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                        Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                        Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                    Shat_Hi = np.reshape(Shat_Hi,(n,P))
                else:
                    Shat = np.reshape(Shat,(n,P))
            else:
                S = np.reshape(S,(n,P))
            
    
            if FTPlane:
                if wavelet:
                    A = update_A_prox(V_Hi,A,Shat_Hi,M,inImax2,mu2,mask=mask,positivity=positivityA)
                else:
                    A = update_A_prox(V,A,Shat,M,inImax2,mu2,mask=mask,positivity=positivityA)
            else:
                A = update_A(V,A,S,M,inImax2,mu2,mask=mask,positivity=positivityA)
        
        S = np.real(np.reshape(S,(n,Nx,Ny)).squeeze())
    
    return (S,A)

# def DecGMCA_prox(V,M,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,postProc=True,Ksig=0.6,positivity=False):
    

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



##########################################################################
################ GMCA using proximal operators ###########################
##########################################################################

def update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax,mu,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False,currentIter=0,globalImax=1):
    '''
    Proximal method (Forward-Backward) to estimate sources with sparsity constraint in synthesis framework
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param S: Sources to update, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Number of iterations
    
    @param mu: Parameter of gradient step
    
    @param Ksig: k-sigma for the thresholding
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far 
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @param FISTA: Whether to use FISTA algorithm
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if mu == 0:
        Lip= LA.norm(np.dot(A,A.transpose()))
        mu = 1.0/Lip
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))
#             wtTmp = np.zeros((N,scale,P))
#             wt = np.zeros((N,scale-1,P))
#             for sr in np.arange(N):
#                 wtTmp[sr] = star1d(S[sr], scale=scale, fast=True, gen2=gen2,normalization=True)
# #                 coarseScale[sr] = np.copy(wtTmp[sr,-1])                    
# #                 wt[sr] = np.copy(wtTmp[sr,:-1])
    
#     if thresStrtg == 1:
#         thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
#     elif thresStrtg == 2:
#         thIter = np.zeros((Imax,N,scale-1))
     
    if FISTA:
        t = 1.0         # FISTA parameter
    if PALM:
        P_min = 0.05
    it = 0
    var_rsd = 0
    errTab = np.zeros(Imax)
    
    while it<Imax:
        if FTPlane:
            Shat = fftNd1d(S,Ndim)
        else:
            Shat = S
        if Ndim == 2:
            Shat = np.reshape(Shat,(N,P))
        rsd = V - M*np.dot(A,Shat)
        var_rsdN = np.var(rsd)
        rsd1 = np.dot(A.conj().transpose(),M*rsd)
            
        if FTPlane: 
            rsd1 = ifftNd1d(np.reshape(rsd1,(N,Nx,Ny)).squeeze(),Ndim) 
            rsd1 = np.real(rsd1)  
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        else:
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        
        if wavelet:
            if wname == 'starlet':
                rsdTr = np.zeros((N,scale,P))
                wt = np.zeros((N,scale,P))
                for sr in np.arange(N):
                    rsdTr[sr] = star1d(rsd1[sr],scale=scale, fast=True, gen2=gen2,normalization=True)
                    wt[sr] = star1d(S_n[sr],scale=scale, fast=True, gen2=gen2,normalization=True)
                coarseScale = wt[:,-1,:]
                wt = np.delete(wt,-1,axis=1)
                rsdTr = np.delete(rsdTr,-1,axis=1)
            ################## Noise estimate####################
                if thresStrtg == 1:
                    rsdTr = np.reshape(rsdTr,(N,np.size(rsdTr)/N)).squeeze()
                    sig = mad(rsdTr[:,:P])
#                     sig = np.array([0.01,0.01])
                elif thresStrtg == 2:
                    sig = np.zeros((N,scale-1))
                    for sr in np.arange(N):
    #                 sig = mad(wt[:,:P])                                             # For starlet transform
                        sig[sr] = mad(rsdTr[sr]) 
            
                if PALM:
                    thTab = find_th(wt,P_min,sig,currentIter,globalImax,strategy=thresStrtg)
                else:
                    thTab = Ksig*sig
#                 thIter[it] = thTab
            
            #### Thresholding in terms of percentage of significant coefficients ######
            if thresStrtg == 1:
                hardTh(wt,thTab,weights=None,reweighted=False)
#                 softTh(wt,thTab,weights=None,reweighted=False)
            elif thresStrtg == 2:
                for sr in np.arange(N):
                    hardTh(wt[sr],thTab[sr],weights=None,reweighted=False)
#                     softTh(wt[sr],thTab[sr],weights=None,reweighted=False)            
            wt = np.concatenate((wt,coarseScale[:,np.newaxis,:]),axis=1)
#             wt = np.reshape(wt,(N,np.size(wt)/(N*P),P))             # For undecimated wavelet transform
            if wname == 'starlet':
                for sr in np.arange(N):
                    S_n[sr] = istar1d(wt[sr],fast=True, gen2=gen2,normalization=True)
            
            if FISTA:
                tn = (1.+np.sqrt(1+4*t*t))/2
                S = S_n + (t-1)/tn * (S_n-S)
                t = tn
            else:
                S = np.copy(S_n)
                
            err = abs(var_rsdN-var_rsd)  
            var_rsd = var_rsdN
            errTab[it] = err
        
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)
            
    return S

def update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax,ImaxInt,mu,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False,currentIter=0,globalImax=1,u=None,sig=None):
    '''
    Proximal method (Forward-Backward) to estimate sources with sparsity constraint in analysis framework
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture, size: bands*sources
    
    @param S: Sources to update, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax1: Number of global iterations
    
    @param Imax2: Number of sub-iterations
    
    @param mu: Parameter of gradient step
    
    @param mu1: Parameter of sparsity constraint step
    
    @param Ksig: k-sigma for the thresholding
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @param wavelet: Wavelet option
    
    @param scale: Scales of wavelet decomposition
    
    @param wname: Wavelet type. Only one-dimensional starlet has been implemented so far 
    
    @param thresStrtg: Strategy to find thresholds, valid just for parameter wavelet=True
    strategy=1 means to find thresholds based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find thresholds based on the percentage of coefficients larger than 3*sigma for each scale
    
    @param FTPlane: Whether the input data in Fourier space
    
    @param fc: Cut-off frequency, fc is normalized to numerical frequency
    
    @param Ar: Mixing matrix of reference
    
    @param FISTA: Whether to use FISTA algorithm
    
    @return: Updated S, size: sources*pixels
    '''
    (Bd,P) = np.shape(V)
    (Bd,N) = np.shape(A) 
    Stmp = np.zeros_like(S)  
    
    if mu == 0:
        Lip= LA.norm(np.dot(A,A.transpose()))
        mu = 1.0/Lip
    if muInt == 0:
        muInt = 1.0
    if PALM:
        P_min = 0.05
    if FISTA:
        t = 1.0
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))                
            if u is None:
                if Ndim == 1:
                    wt = np.zeros((N,scale-1,P))
                    for sr in np.arange(N):
                        wtTmp = star1d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                        wt[sr] = np.copy(wtTmp[:-1])
                elif Ndim == 2:
                    wt = np.zeros((N,scale-1,Nx,Ny))
                    for sr in np.arange(N):
                        wtTmp = star2d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                        wt[sr] = np.copy(wtTmp[:-1])
                wt = np.reshape(wt,(N,np.size(wt)/N))
                sigTmp = mad(wt[:,:P])
                u = np.zeros((N,(scale-1)*P))
                for sr in np.arange(N):
                    ind = (np.abs(wt[sr]) - Ksig*muInt*sigTmp[sr] > 0)
                    u[sr][ind] = Ksig*muInt*sigTmp[sr]*np.sign(wt[sr][ind])   
                if Ndim == 1:
                    u = np.reshape(u,(N,scale-1,P))
                    u = np.concatenate((u,np.zeros((N,1,P))),axis=1)
                elif Ndim == 2:
                    u = np.reshape(u,(N,scale-1,Nx,Ny))  
                    u = np.concatenate((u,np.zeros((N,1,Nx,Ny))),axis=1)  
                
    if thresStrtg == 1:
        thIter = np.zeros((Imax,N))                                 # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax,N,scale-1))
    
    it = 0
    errTab = np.zeros(Imax)
    err = 1.
    tol = 1e-5
    
    while it<Imax and err > tol:
        if FTPlane:
            Shat = fftNd1d(S,Ndim)
        else:
            Shat = S
        if Ndim == 2:
            Shat = np.reshape(Shat,(N,P))
        rsd = V - M*np.dot(A,Shat)
        rsd1 = np.dot(A.conj().transpose(),M*rsd)
            
        if FTPlane: 
            rsd1 = ifftNd1d(np.reshape(rsd1,(N,Nx,Ny)).squeeze(),Ndim) 
            rsd1 = np.real(rsd1)  
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
        else:
            S_n = S + mu*np.reshape(rsd1,(N,Nx,Ny)).squeeze()
            
        rsdTr = np.zeros((N,scale,P))
        for sr in np.arange(N):
            rsdTr[sr] = star1d(rsd1[sr],scale=scale, fast=True, gen2=gen2,normalization=True)    
        rsdTr = np.delete(rsdTr,-1,axis=1)
        ######## Test for rigorous PALM ##############
        if thresStrtg == 1 and sig is None:
            rsdTr = np.real(np.reshape(rsdTr,(N,np.size(rsdTr)/N)).squeeze())
            sig = mad(mu*rsdTr[:,:P])
        elif thresStrtg == 2 and sig is None:
            sig = np.zeros((N,scale-1))
            for sr in np.arange(N):
                sig[sr] = mad(mu*rsdTr[sr])
        
        it_int = 0
        ############### Subiteration #################
        while it_int < ImaxInt:
            if wavelet:
                if wname == 'starlet':
                    for sr in np.arange(N):
                        Stmp[sr] = adstar1d(u[sr], fast=True, gen2=gen2, normalization=True)
                    
                    rsdInt = S_n - Stmp
                    rsdIntTr = np.zeros((N,scale,P))
                    for sr in np.arange(N):
                        rsdIntTr[sr] = star1d(rsdInt[sr],scale=scale, fast=True, gen2=gen2,normalization=True)  
                    u_n = u + muInt*rsdIntTr
                    u_n = np.delete(u_n,-1,axis=1)
                    rsdIntTr = np.delete(rsdIntTr,-1,axis=1)
                    u_n1 = np.copy(u_n)
                
                    if PALM:
                        thTab = find_th(u_n,P_min,sig,currentIter,globalImax,strategy=thresStrtg,Ksig=Ksig)
                    else:
                        thTab = Ksig*sig
                    thIter[it] = thTab                
                    #### Thresholding in terms of percentage of significant coefficients ######
                    if thresStrtg == 1:
#                         hardTh(wtTmp_n,thTab,weights=None,reweighted=False)
                        softTh(u_n,thTab,weights=None,reweighted=False)
                    elif thresStrtg == 2:
                        for sr in np.arange(N):
#                             hardTh(wtTmp_n[sr],thTab[sr],weights=None,reweighted=False)
                            softTh(u_n[sr],thTab[sr],weights=None,reweighted=False)            
                    u_n = u_n1 - u_n
                    u_n = np.reshape(u_n,(N,scale-1,P))             # For undecimated wavelet transform
#                     wtTmp_n = np.concatenate((wtTmp_n,coarseScale),axis=1)
                    u_n = np.concatenate((u_n,np.zeros((N,1,P))),axis=1)
                    u = u_n
            it_int += 1
            
        if wavelet:
            if wname == 'starlet':
                for sr in np.arange(N):
                    Stmp[sr] = adstar1d(u[sr], fast=True, gen2=gen2, normalization=True)
        
        S_n = S_n - Stmp                    
        err = (((S_n-S)**2).sum())/((S**2).sum())  
        errTab[it] = err
        if FISTA:
            tn = (1.+np.sqrt(1+4*t*t))/2
            S = S_n + (t-1)/tn * (S_n-S)
            t = tn
        else:
            S = S_n
               
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)
        
    return (S,u,sig)

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
#     print "tau:"+str(tau)
#     print "eta:"+str(eta)
    
    if wavelet:
        if wname == 'starlet':
            pm.trHead = ''
            gen2 = False
#             coarseScale = np.zeros((N,1,P))                
            if Ndim == 1:
                wt = np.zeros((N,scale-1,P))
                for sr in np.arange(N):
                    wtTmp = star1d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                    wt[sr] = np.copy(wtTmp[:-1])
            elif Ndim == 2:
                wt = np.zeros((N,scale-1,Nx,Ny))
                for sr in np.arange(N):
                    wtTmp = star2d(S[sr],scale=scale,fast=True, gen2=gen2,normalization=True)
                    wt[sr] = np.copy(wtTmp[:-1])
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
            for sr in np.arange(N):
                if Ndim == 1:
                    wtTmp = np.concatenate((u[sr],np.zeros((1,P))),axis=0)
                    Stmp[sr] = adstar1d(wtTmp, fast=True, gen2=gen2, normalization=True)
                elif Ndim == 2:
                    wtTmp = np.concatenate((u[sr],np.zeros((1,Nx,Ny))),axis=0)
                    Stmp[sr] = adstar2d(wtTmp,fast=True, gen2=gen2, normalization=True)
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
            if Ndim == 1:
                wtTmp = np.zeros((N,scale,P))
            elif Ndim == 2:
                wtTmp = np.zeros((N,scale,Nx,Ny))
            
            for sr in np.arange(N):
                if Ndim == 1:
                    wtTmp[sr] = star1d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
                elif Ndim == 2:
                    wtTmp[sr] = star2d(termQ1[sr],scale=scale,fast=True, gen2=gen2, normalization=True)
                    
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
#         print "Iteration:"+str(it)
#         print "Current error:" + str(err)
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
        
def update_A_prox(V,A,Shat,M,Imax,mu,mask=True,positivity=False):
    '''
    Proximal method (Forward-Backward) to estimate mixing matrix
    
    @param V: Input data, size: bands*pixels
    
    @param A: Matrix of mixture to update, size: bands*sources
    
    @param Shat: Sources, size: sources*pixels
    
    @param M: Matrix of mask, size: bands*pixels
    
    @param Imax: Number of iterations
    
    @param mu: Parameter of gradient step
    
    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask
    
    @return: Updated A, size: bands*sources
    '''
    (Bd,P) = np.shape(V)
    (N,P) = np.shape(Shat)
    A_n = np.zeros_like(A)  
    
    if mu == 0:
        mu = 0.5/LA.norm(np.dot(Shat,Shat.conj().transpose()))
    errTab = np.zeros(Imax)
    it = 0
    var_rsd = 0
    
    while it < Imax: 
        rsd = V - M*np.dot(A,Shat)
        var_rsdN = np.var(rsd)
        rsd1 = np.dot(M*rsd,Shat.conj().transpose())
        A_n = A + mu*rsd1
        A_n = np.real(A_n)
#         normalize(A_n)
        tab = LA.norm(A_n,axis=0)
        tab[tab<1] = 1
        if positivity:
            A_n[A_n<0] = 0
        A_n = A_n/tab                       # Division on columns, valid in python2.7
            
        A = np.copy(A_n)
        err = abs(var_rsdN-var_rsd)  
        var_rsd = var_rsdN
        errTab[it] = err
        it += 1
#         print "Iteration:"+str(it)
#         print "Current error:"+str(err)

    return A

def GMCA_PALM(V,M,n,Nx,Ny,Imax,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,Ksig=0.6,positivity=False,FISTA=False,S=None,A=None):
    (Bd,P) = np.shape(V)
    
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
    
        
    if (A is None) and (S is None):
        if mask:
            MM = np.zeros_like(M)
            MM[np.abs(M)>=1e-4] = 1
            MM = np.real(MM)
            Xe = SVTCompletion(V,MM,n,1.5,200)
        elif deconv and not mask and FTPlane:
            Xe = V_Hi
        else:
            Xe = V
            
        R = np.dot(Xe,Xe.T)
        d,v = LA.eig(R)
        A = np.abs(v[:,0:n])
        normalize(A)
        
        if FTPlane:
            Shat = np.reshape(np.dot(A.T,Xe),(n,Nx,Ny)).squeeze()
            S = np.real(ifftNd1d(Shat,Ndim))
        else:
            S = np.dot(A.T,Xe)

    Imax1 = 1
    ImaxInt = 5
    Imax2 = 1
    mu1 = 0.0
    muInt = 0.0
    mu2 = 0.0
    eta = 0.0
    
    if FISTA:
        t = 1.0         # FISTA parameter

    i = 0
    tol = 1e-8
    err = 1.
    while i < Imax and err > tol:
#         S_n = update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=True,currentIter=i,globalImax=Imax)
        
#         S_n,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,eta,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,positivity=False,PALM=True,currentIter=i,globalImax=Imax)
        if i == 0:
            S_n,u,sig = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,currentIter=i,globalImax=Imax)
        else:
            S_n,u,sig = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,currentIter=i,globalImax=Imax,u=u,sig=sig)
        errS = (((S_n - S)**2).sum()) / ((S**2).sum())
        if FISTA:
            tn = (1.+np.sqrt(1+4*t*t))/2
            S = S_n + (t-1)/tn * (S_n-S)
            t = tn
        else:
            S = S_n
         
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
         
 
        if FTPlane:
            if wavelet:
                A_n = update_A_prox(V_Hi,A,Shat_Hi,M,Imax2,mu2,mask=mask,positivity=False)
            else:
                A_n = update_A_prox(V,A,Shat,M,Imax2,mu2,mask=mask,positivity=False)
        else:
            A_n = update_A(V,A,S,M,Imax2,mu2,mask=mask,positivity=False)
         
        errA = (((A_n - A)**2).sum())/((A**2).sum())
         
        A = A_n
#         err = errS
        err = max(errS,errA)
        i = i+1
        
        print "Iteration: "+str(i)
        print "Condition number of A:"
        print LA.cond(A)
        print "Condition number of S:"
        print LA.cond(S)
        print "Current error"
        print err
        
    ########### Refinement with the final thresholding ############## 
#     i = 0
#     err = 1.
#     Imax = 200
#     Imax1 = 1
#     Imax2 = 1
#     if FISTA:
#         t = 1.0
#     print "Refinement with the final thresholding"
#     while i < Imax and err > tol:
# #         S_n = update_S_prox_peusdo_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False,PALM=False)
#              
# #         S_n,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,Imax1,mu1,eta,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,positivity=False,PALM=False)
#         if i == 0:
#             S_n,u = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False)
#         else:
#             S_n,u = update_S_prox_anal(V,A,S,M,Nx,Ny,Ndim,Imax1,ImaxInt,mu1,muInt,Ksig,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,PALM=False,u=u)
#         errS = (((S_n - S)**2).sum()) / ((S**2).sum())
#         if FISTA:
#             tn = (1.+np.sqrt(1+4*t*t))/2
#             S = S_n + (t-1)/tn * (S_n-S)
#             t = tn
#         else:
#             S = S_n
#              
#         if FTPlane:
#             Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
#             if wavelet:
#                 ################# Don't take the low frequency band into account ######################
#                 Shat_Hi = filter_Hi(Shat,Ndim,fc)
#                 if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
#                     Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
#                     Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
#                 Shat_Hi = np.reshape(Shat_Hi,(n,P))
#             else:
#                 Shat = np.reshape(Shat,(n,P))
#         else:
#             S = np.reshape(S,(n,P))
#              
#      
#         if FTPlane:
#             if wavelet:
#                 A_n = update_A_prox(V_Hi,A,Shat_Hi,M,Imax2,mu2,mask=mask,positivity=False)
#             else:
#                 A_n = update_A_prox(V,A,Shat,M,Imax2,mu2,mask=mask,positivity=False)
#         else:
#             A_n = update_A(V,A,S,M,Imax2,mu2,mask=mask,positivity=False)
#              
#         errA = (((A_n - A)**2).sum())/((A**2).sum())
#              
#         A = A_n
#         err = max(errS,errA)
#         i = i+1
#              
#         print "Iteration: "+str(i)
#         print "Condition number of A:"
#         print LA.cond(A)
#         print "Condition number of S:"
#         print LA.cond(S)
#         print "Current error"
#         print err
    
    
    return (S,A) 

def GMCA_prox(V,M,n,Nx,Ny,Imax,epsilon,epsilonF,Ndim,wavelet,scale,mask,deconv,wname='starlet',thresStrtg=2,FTPlane=True,fc=1./16,logistic=False,postProc=True,Ksig=0.6,positivity=False):
    
    '''
    GMCA using proximal methods
    
    @param V: Input data, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @param n: Number of sources
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Maximum iterations
    
    @param epsilon: Tikhonov parameter 
    
    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    '''
    (Bd,P) = np.shape(V)
    
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(V,Ndim,fc)
        if logistic:
            fc_pass = 2*fc
            steep = 0.1/32/fc
            flogist1 = 1./(1+np.exp(-steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            flogist2 = 1./(1+np.exp(steep*np.linspace(-(fc_pass*P-fc*P)/2,(fc_pass*P-fc*P)/2,fc_pass*P-fc*P)))
            V_Hi[:,int(fc*P):int(fc_pass*P)] = V_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
            V_Hi[:,int(-fc_pass*P):int(-fc*P)] = V_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
            
    elif Ndim == 2 and FTPlane:
        V1 = np.reshape(V,(Bd,Nx,Ny))
        V_Hi = filter_Hi(V1,Ndim,fc)
        V_Hi = np.reshape(V_Hi,(Bd,P))
    else:
        V_Hi = V
    
        
    if deconv and FTPlane:
        Xe = V_Hi
    elif mask:
        Xe = SVTCompletion(V,M,n,0.5,200)
    else:
        Xe = V
    
    (u,d,v) = LA.svd(V,full_matrices=False)
    A = u[:n,:].T
#     np.random.seed(5)
#     A = np.random.randn(Bd,n) + 0*1j*np.random.randn(Bd,n)
    normalize(A)
#     if wavelet:
#         gen2 = False
#         nsNorm(P, scale, gen2=gen2)
#     Shat = np.random.randn(n,P) + 1j*np.random.randn(n,P)
#     tau = 10.
    mu1 = 0.0
    eta = 0.5
    mu2 = 0.0
    inImax1 = 50
    inImax2 = 1
    S = np.zeros((n,P))
#     Shat = np.zeros((n,P)) + np.zeros((n,P))*1j
#     dtau = float(tau - tauF)/Imax
    for i in np.arange(Imax):
#         S,thIter = update_S_prox(V,A,S,M,Nx,Ny,Imax,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False)
#         S,thIter = update_S_prox(V,A,S,M,Nx,Ny,Imax,mu1,Ksig,mask,wavelet,scale,wname='starlet',thresStrtg=2,FTPlane=True,Ar=None,FISTA=False)
        
        S,thIter = update_S_prox_Condat_Vu(V,A,S,M,Nx,Ny,Ndim,inImax1,mu1,eta,Ksig=Ksig,wavelet=wavelet,scale=scale,wname=wname,thresStrtg=thresStrtg,FTPlane=FTPlane,positivity=positivity)
        
        if FTPlane:
            Shat = fftNd1d(S,Ndim)              # Transform in Fourier space
            if wavelet:
                ################# Don't take the low frequency band into account ######################
                Shat_Hi = filter_Hi(Shat,Ndim,fc)
                if Ndim == 1 and logistic:              # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:,int(fc*P):int(fc_pass*P)] = Shat_Hi[:,int(fc*P):int(fc_pass*P)]*flogist1
                    Shat_Hi[:,int(-fc_pass*P):int(-fc*P)] = Shat_Hi[:,int(-fc_pass*P):int(-fc*P)]*flogist2
                Shat_Hi = np.reshape(Shat_Hi,(n,P))
            else:
                Shat = np.reshape(Shat,(n,P))
        else:
            S = np.reshape(S,(n,P))
        

        if FTPlane:
            if wavelet:
                A = update_A_prox(V_Hi,A,Shat_Hi,M,inImax2,mu2,mask=mask,positivity=False)
#                 A = update_A(V_Hi,Shat_Hi,M,mask=mask,deconv=deconv)
            else:
                A = update_A_prox(V,A,Shat,M,inImax2,mu2,mask=mask,positivity=False)
        else:
            A = update_A(V,A,S,M,inImax2,mu2,mask=mask,positivity=False)
            
#         A = np.real(A)
#         normalize(A)    
        
        print "Iteration: "+str(i+1)
        print "Condition number of A:"
        print LA.cond(A), 'A'
        print "Condition number of S hat:"
        print LA.cond(Shat), 'Shat'
    
#     S = np.real(ifftNd1d(np.reshape(Shat,(n,Nx,Ny)).squeeze(),Ndim))
    S = np.real(np.reshape(S,(n,Nx,Ny)).squeeze())
    return (S,A)



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
        
        
        