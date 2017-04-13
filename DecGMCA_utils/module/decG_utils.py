# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:18:38 2017

@author: jbobin
"""

import decG
import numpy as np

#
# Updating S
#
#
#  INPUTS:
#   
#       X_f : Fourier coefficients of the data (complex array of size (nobs x npix x npix) )
#       Beam : Beam in the Fourier domain (real array of size (nobs x npix x npix) ) - BEWARE : IT IS ASSUMED TO BE REAL
#       MixMat : Mixing matrix (real array of size (nobs x nsources) )
#       epsilon: regularization parameter
#
#   OUTPUTS
#
#       S_f : Fourier coefficients of the sources (complex array of size (nsources x npix x npix) )
#


def UpdateS(X_f,Beam,MixMat,epsilon=0.1):
    
    nX = np.shape(X_f)
    nobs = nX[0]
    npix = nX[1]
    npiy = nX[2]
    
    nA = np.shape(MixMat)
    ns = nA[1]
    
    S_fr = np.zeros((ns,npix,npiy))
    S_fi = np.zeros((ns,npix,npiy))
    
    S_fr = decG.PLK().applyH_Pinv_S(np.real(X_f).reshape((nobs,npix*npiy)).astype('double'),Beam.reshape((nobs,npix*npiy)).astype('double'),MixMat.astype('double'),np.array([epsilon]).astype('double'),np.array([ns]).astype('double'),np.array([nobs]).astype('double'),np.array([npix*npiy]).astype('double')).reshape((ns,npix,npiy))
    S_fi = decG.PLK().applyH_Pinv_S(np.imag(X_f).reshape((nobs,npix*npiy)).astype('double'),Beam.reshape((nobs,npix*npiy)).astype('double'),MixMat.astype('double'),np.array([epsilon]).astype('double'),np.array([ns]).astype('double'),np.array([nobs]).astype('double'),np.array([npix*npiy]).astype('double')).reshape((ns,npix,npiy))
    
    return S_fr + 1j*S_fi

def UpdateA(X_f,Beam,Shat):
    
    nX = np.shape(X_f)
    nobs = nX[0]
    npix = nX[1]
    npiy = nX[2]
    
    nA = np.shape(Shat)
    ns = nA[0]
    
    A = decG.PLK().applyH_Pinv_A(np.real(X_f).reshape((nobs,npix*npiy)).astype('double'),np.imag(X_f).reshape((nobs,npix*npiy)).astype('double'),Beam.reshape((nobs,npix*npiy)).astype('double'),np.real(Shat).reshape((ns,npix*npiy)).astype('double'),np.imag(Shat).reshape((ns,npix*npiy)).astype('double'),np.array([ns]).astype('double'),np.array([nobs]).astype('double'),np.array([npix*npiy]).astype('double')).reshape((nobs,ns))
    
    return A

#######################################
#######################################
#######################################

def Test_UpdateS(nobs=5,ns=3,tsamp = 1024,epsilon=0.1):
    
    S = np.random.randn(ns,tsamp*tsamp)
    M = np.sign(np.random.rand(nobs,tsamp*tsamp)-0.1)
    M = np.sign(M*(M > 0))
    A = np.random.randn(nobs,ns)
    X = np.dot(A,S)
    X_f = np.fft.fft(X,axis=1)
    mX_f = X_f*M
    
    Se_f = UpdateS(mX_f.reshape((nobs,tsamp,tsamp)),M.reshape((nobs,tsamp,tsamp)).astype('double'),A,epsilon=epsilon)
    
    Se = np.real(np.fft.ifft(Se_f.reshape((ns,tsamp**2)),axis=1))
    
    return S,Se

def Test_UpdateA(nobs=5,ns=3,tsamp = 5):
    
    S = np.random.randn(ns,tsamp*tsamp)
    S_f = np.fft.fft(S,axis=1)
    M = np.sign(np.random.rand(nobs,tsamp*tsamp)-0.1)
    M = np.sign(M*(M > 0))
    A = np.random.randn(nobs,ns)
    X_f = np.dot(A,S_f)
    mX_f = X_f*M
    
    Ae = UpdateA(mX_f.reshape((nobs,tsamp,tsamp)),M.reshape((nobs,tsamp,tsamp)).astype('double'),S_f)
    
    return A,Ae