# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:18:38 2017

@author: jbobin
"""

import starlet
import numpy as np

def star1d(sig,scale,gen2=False,normalization=False):
    N = np.size(sig)
    obj = starlet.Starlet1D(N,scale)
    if gen2:
        wt = obj.transform(sig,normalization)
    else:
        wt = obj.transform_gen1(sig,normalization)

    return wt

def istar1d(wt,gen2=False,normalization=False):
    (scale,N) = np.shape(wt)
    obj = starlet.Starlet1D(N,scale)
    if gen2:
        sig = obj.reconstruct(wt,normalization)
    else:
        sig = obj.reconstruct_gen1(wt,normalization)
    
    return sig

def adstar1d(wt,gen2=False,normalization=False):
    (scale,N) = np.shape(wt)
    obj = starlet.Starlet1D(N,scale)
    if gen2:
        sig = obj.trans_adjoint(wt,normalization)
    else:
        sig = obj.trans_adjoint_gen1(wt,normalization)
    
    return sig

def Nstar1d(sig,scale,gen2=False,normalization=False):
    N,Nx = np.shape(sig)
    obj = starlet.Starlet1D(Nx,scale)
    if gen2:
        wt = obj.stack_transform(np.array([N]).astype('float64'),sig,normalization)
    else:
        wt = obj.stack_transform_gen1(np.array([N]).astype('float64'),sig,normalization)
    
    return wt

def Nistar1d(wt,gen2=False,normalization=False):
    (N,scale,Nx) = np.shape(wt)
    obj = starlet.Starlet1D(Nx,scale)
    if gen2:
        sig = obj.stack_reconstruct(np.array([N]).astype('float64'),wt,normalization)
    else:
        sig = obj.stack_reconstruct_gen1(np.array([N]).astype('float64'),wt,normalization)
    
    return sig

def Nadstar1d(wt,gen2=False,normalization=False):
    (N,scale,Nx) = np.shape(wt)
    obj = starlet.Starlet1D(Nx,scale)
    if gen2:
        sig = obj.stack_trans_adjoint(np.array([N]).astype('float64'),wt,normalization)
    else:
        sig = obj.stack_trans_adjoint_gen1(np.array([N]).astype('float64'),wt,normalization)
    
    return sig


def star2d(sig,scale,gen2=False,normalization=False):
    Nx,Ny = np.shape(sig)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        wt = obj.transform(sig,normalization)
    else:
        wt = obj.transform_gen1(sig,normalization)
    
    return wt

def istar2d(wt,gen2=False,normalization=False):
    (scale,Nx,Ny) = np.shape(wt)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        sig = obj.reconstruct(wt,normalization)
    else:
        sig = obj.reconstruct_gen1(wt,normalization)
    
    return sig

def adstar2d(wt,gen2=False,normalization=False):
    (scale,Nx,Ny) = np.shape(wt)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        sig = obj.trans_adjoint(wt,normalization)
    else:
        sig = obj.trans_adjoint_gen1(wt,normalization)
    
    return sig

def Nstar2d(sig,scale,gen2=False,normalization=False):
    N,Nx,Ny = np.shape(sig)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        wt = obj.stack_transform(np.array([N]).astype('float64'),sig,normalization)
    else:
        wt = obj.stack_transform_gen1(np.array([N]).astype('float64'),sig,normalization)
    
    return wt

def Nistar2d(wt,gen2=False,normalization=False):
    (N,scale,Nx,Ny) = np.shape(wt)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        sig = obj.stack_reconstruct(np.array([N]).astype('float64'),wt,normalization)
    else:
        sig = obj.stack_reconstruct_gen1(np.array([N]).astype('float64'),wt,normalization)
    
    return sig

def Nadstar2d(wt,gen2=False,normalization=False):
    (N,scale,Nx,Ny) = np.shape(wt)
    obj = starlet.Starlet2D(Nx,Ny,scale)
    if gen2:
        sig = obj.stack_trans_adjoint(np.array([N]).astype('float64'),wt,normalization)
    else:
        sig = obj.stack_trans_adjoint_gen1(np.array([N]).astype('float64'),wt,normalization)
    
    return sig


