# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:18:38 2017

@author: jbobin
"""

from starlet_utils import *
import numpy as np
import wav1d
import wav2d
import pyfits as fits
import time

def testFunc():
    sig = fits.getdata('S_bd5.fits')
    sig = sig.astype('float64')

    wt = Nstar1d(sig,4,gen2=False,normalization=False)
    wtn = Nstar1d(sig,4,gen2=False,normalization=True)
    wt_gen2 = Nstar1d(sig,4,gen2=True,normalization=False)
    wtn_gen2 = Nstar1d(sig,4,gen2=True,normalization=True)

    rec = Nistar1d(wt,gen2=False,normalization=False)
    recn = Nistar1d(wtn,gen2=False,normalization=True)
    rec_gen2 = Nistar1d(wt_gen2,gen2=True,normalization=False)
    recn_gen2 = Nistar1d(wtn_gen2,gen2=True,normalization=True)

    ad = Nadstar1d(wt,gen2=False,normalization=False)
    adn = Nadstar1d(wtn,gen2=False,normalization=True)
    ad_gen2 = Nadstar1d(wt_gen2,gen2=True,normalization=False)
    adn_gen2 = Nadstar1d(wtn_gen2,gen2=True,normalization=True)

    return (wt,wtn,wt_gen2,wtn_gen2,rec,recn,rec_gen2,recn_gen2,ad,adn,ad_gen2,adn_gen2)


def testFuncRec():
    sig = fits.getdata('S_bd5.fits')
    sig = sig.astype('float64')
    Nx,Ny = np.shape(sig)

    wt = np.zeros((Nx,4,Ny))
    wtn = np.zeros((Nx,4,Ny))
    wt_gen2 = np.zeros((Nx,4,Ny))
    wtn_gen2 = np.zeros((Nx,4,Ny))

    ad = np.zeros((Nx,Ny))
    adn = np.zeros((Nx,Ny))
    ad_gen2 = np.zeros((Nx,Ny))
    adn_gen2 = np.zeros((Nx,Ny))
    
    for i in np.arange(Nx):
        wt[i] = wav1d.star1d(sig[i],4,gen2=False,normalization=False)
        wtn[i] = wav1d.star1d(sig[i],4,gen2=False,normalization=True)
        wt_gen2[i] = wav1d.star1d(sig[i],4,gen2=True,normalization=False)
        wtn_gen2[i] = wav1d.star1d(sig[i],4,gen2=True,normalization=True)
    
    for i in np.arange(Nx):
        ad[i] = wav1d.adstar1d(wt[i],gen2=False,normalization=False)
        adn[i] = wav1d.adstar1d(wtn[i],gen2=False,normalization=True)
        ad_gen2[i] = wav1d.adstar1d(wt_gen2[i],gen2=True,normalization=False)
        adn_gen2[i] = wav1d.adstar1d(wtn_gen2[i],gen2=True,normalization=True)

    return (wt,wtn,wt_gen2,wtn_gen2,ad,adn,ad_gen2,adn_gen2)


N = 200
gen2=False
normalization=False
fast=False
sig = np.random.randn(N,256,512)
#sig = fits.getdata('im_bd5.fits')
sig = sig.astype('float64')
#    fits.writeto('ompTest.fits',sig)
start = time.time()
wt = Nstar2d(sig,6,gen2=gen2,normalization=normalization)
end = time.time()
print end - start

#wt1 = np.zeros_like(wt)
#start1 = time.time()
#for i in np.arange(N):
#    wt1[i] = wav2d.star2d(sig[i],4,fast=False,gen2=gen2,normalization=normalization)
#end1 = time.time()
#print end1 - start1
#
#diff = wt - wt1
#print diff.max(),diff.min()
#
##### inverse operator ###
#sig_rec = Nistar2d(wt,gen2=gen2,normalization=normalization)
#diff = sig_rec - sig
#print diff.max(),diff.min()
#
#### adjoint operator ###
#start = time.time()
#sig_ad = Nadstar2d(wt,gen2=gen2,normalization=normalization)
#end = time.time()
#print end - start
#
#sig_ad1 = np.zeros_like(sig_ad)
#start1 = time.time()
#for i in np.arange(N):
#    sig_ad1[i] = wav2d.adstar2d(wt1[i],fast=False,gen2=gen2,normalization=normalization)
#end1 = time.time()
#print end1 - start1
#
#diff = sig_ad - sig_ad1
#print diff.max(),diff.min()
#    N = 1
#    sig = np.random.randn(N,16384)
#    sig = sig.astype('float64')
#    start = time.time()
#    wt = Nstar1d(sig,4,gen2=False,normalization=False)
#    end = time.time()
#    print end - start
#
#    start1 = time.time()
#    for i in np.arange(N):
#        wt[i] = wav1d.star1d(sig[i],4,gen2=False,normalization=False)
#    end1 = time.time()
#    print end1 - start1


def testFunc2D():
    sig = fits.getdata('im_bd5.fits')
    sig = sig.astype('float64')
    
    wt = Nstar2d(sig,4,gen2=False,normalization=False)
    wtn = Nstar2d(sig,4,gen2=False,normalization=True)
    wt_gen2 = Nstar2d(sig,4,gen2=True,normalization=False)
    wtn_gen2 = Nstar2d(sig,4,gen2=True,normalization=True)
    
    rec = Nistar2d(wt,gen2=False,normalization=False)
    recn = Nistar2d(wtn,gen2=False,normalization=True)
    rec_gen2 = Nistar2d(wt_gen2,gen2=True,normalization=False)
    recn_gen2 = Nistar2d(wtn_gen2,gen2=True,normalization=True)
    
    ad = Nadstar2d(wt,gen2=False,normalization=False)
    adn = Nadstar2d(wtn,gen2=False,normalization=True)
    ad_gen2 = Nadstar2d(wt_gen2,gen2=True,normalization=False)
    adn_gen2 = Nadstar2d(wtn_gen2,gen2=True,normalization=True)
    
    return (wt,wtn,wt_gen2,wtn_gen2,rec,recn,rec_gen2,recn_gen2,ad,adn,ad_gen2,adn_gen2)


def testFuncRec2D():
    sig = fits.getdata('im_bd5.fits')
    sig = sig.astype('float64')
    N,Nx,Ny = np.shape(sig)
    
    wt = np.zeros((N,4,Nx,Ny))
    wtn = np.zeros((N,4,Nx,Ny))
    wt_gen2 = np.zeros((N,4,Nx,Ny))
    wtn_gen2 = np.zeros((N,4,Nx,Ny))
    
    ad = np.zeros((N,Nx,Ny))
    adn = np.zeros((N,Nx,Ny))
    ad_gen2 = np.zeros((N,Nx,Ny))
    adn_gen2 = np.zeros((N,Nx,Ny))
    
    for i in np.arange(N):
        wt[i] = wav2d.star2d(sig[i],4,fast=False,gen2=False,normalization=False)
        wtn[i] = wav2d.star2d(sig[i],4,fast=False,gen2=False,normalization=True)
        wt_gen2[i] = wav2d.star2d(sig[i],4,fast=False,gen2=True,normalization=False)
        wtn_gen2[i] = wav2d.star2d(sig[i],4,fast=False,gen2=True,normalization=True)
    
    for i in np.arange(N):
        ad[i] = wav2d.adstar2d(wt[i],fast=False,gen2=False,normalization=False)
        adn[i] = wav2d.adstar2d(wtn[i],fast=False,gen2=False,normalization=True)
        ad_gen2[i] = wav2d.adstar2d(wt_gen2[i],fast=False,gen2=True,normalization=False)
        adn_gen2[i] = wav2d.adstar2d(wtn_gen2[i],fast=False,gen2=True,normalization=True)
    
    return (wt,wtn,wt_gen2,wtn_gen2,ad,adn,ad_gen2,adn_gen2)




