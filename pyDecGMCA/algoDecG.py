'''
Created on Oct 26, 2015

@author: mjiang
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy.fftpack as scifft
import astropy.io.fits as fits
import pylab

from pyDecGMCA.mathTools import *
import pyWavelet.wav1d as wt1d
import pyWavelet.wav2d as wt2d
import pyWavelet.waveTools as pm
from pyDecGMCA.pyUtils import *
from pyDecGMCA.pyProx import *
from scipy.stats import pearsonr

import sys
import os
import glob
import re


def DecGMCA(V, M, n, Nx, Ny, Imax, epsilon, epsilonF, Ndim, wavelet, scale, mask, deconv, wname='starlet', thresStrtg=2,
            FTPlane=True, fc=1. / 16, logistic=False, postProc=0, postProcImax=50, Kend=3.0, Ksig=3.0,
            positivityS=False, positivityA=False, mixCube=None, csCube=None):
    """
    Deconvolution GMCA algorithm to solve simultaneously deconvolution and Blind Source Separation (BSS)

    @param V: Visibilities, size: bands*pixels

    @param M: Matrix of mask, size: sources*pixels

    @param n: Number of sources

    @param Nx: number of rows of the source

    @param Ny: number of columns of the source

    @param Imax: Number of iterations

    @param epsilon: Parameter of regularization

    @param epsilonF: Parameter of the final regularization

    @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image

    @param wavelet: Wavelet option

    @param scale: Scales of wavelet decomposition

    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask

    @param deconv: Whether in the case of blurring, default False

    @param wname: Wavelet type.

    @param thresStrtg: Strategy to find the thresholding, valid just for parameter wavelet=True
    strategy=1 means to find the thresholding level based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find the thresholding level based on the percentage of coefficients larger than 3*sigma for each scale

    @param FTPlane: Whether the input data are in Fourier space

    @param fc: Cut-off frequency, fc is normalized to numerical frequency

    @param logistic: Whether using logistic function for the high-pass filter, default is false

    @param postProc: Whether using refinement step to ameliorate the estimate, default is 0

    @param postProcImax: Number of iterations for the refinement step, default is 50

    @param Kend: The ending threshold, default is 3

    @param Ksig: K-sigma, level of threshold

    @param positivityS: Positivity constraint of S, default is false

    @param positivityA: Positivity constraint of A, default is false

    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    """

    (Bd, P) = np.shape(V)
    # Matrix completion to initialize A
    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(ifftshiftNd1d(V, Ndim), Ndim, fc)
        if logistic:
            fc_pass = 2 * fc
            steep = 0.1 / 32 / fc
            flogist1 = 1. / (1 + np.exp(
                -steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, int(fc_pass * P - fc * P))))
            flogist2 = 1. / (1 + np.exp(
                steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, int(fc_pass * P - fc * P))))
            V_Hi[:, int(fc * P):int(fc_pass * P)] = V_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
            V_Hi[:, int(-fc_pass * P):int(-fc * P)] = V_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
        V_Hi = fftshiftNd1d(V_Hi, Ndim)

    elif Ndim == 2 and FTPlane:
        V_Hi = filter_Hi(ifftshiftNd1d(np.reshape(V, (Bd, Nx, Ny)), Ndim), Ndim, fc)
        V_Hi = fftshiftNd1d(V_Hi, Ndim)
        V_Hi = np.reshape(V_Hi, (Bd, P))
    else:
        V_Hi = V

    if mask:
        MM = np.zeros_like(M)
        MM[np.abs(M) >= 1e-4] = 1
        MM = np.real(MM)
        Xe = SVTCompletion(V, MM, n, 0.5, 10)
    elif deconv and not mask and FTPlane:
        Xe = V_Hi
    else:
        Xe = V

    (u, d, v) = LA.svd(Xe, full_matrices=False)
    A = np.abs(u[:, :n])
    # R = np.dot(Xe, Xe.T)
    # d, v = LA.eig(R)
    # A = np.abs(v[:, 0:n])
    #
    # A = np.copy(Ar)
    normalize(A)
    # A = fits.getdata('estA_ref.fits')

    # Save information of wavelet (Starlet and DCT only)
    if wavelet:
        if wname == 'starlet':
            if Ndim == 1:
                starlet1d = wt1d.Starlet1D(nele=P, scale=scale, fast=True, gen2=False, normalization=True)
            elif Ndim == 2:
                starlet2d = wt2d.Starlet2D(nx=Nx, ny=Ny, scale=scale, fast=True, gen2=False, normalization=True)
        # pm.trHead = ''
        # if wname == 'starlet':
        #     gen2 = False
        #     pm.trHead = 'star' + str(int(Ndim)) + 'd_gen2' if gen2 else 'star' + str(int(Ndim)) + 'd_gen1'
        # else:
        #     pm.trHead = wname + '_' + str(int(Ndim)) + 'd'

    if thresStrtg == 1:
        thIter = np.zeros((Imax, n))  # Stock of thresholding in terms of iteration
    elif thresStrtg == 2:
        thIter = np.zeros((Imax, n, scale - 1))

    # Shat = np.random.randn(n, P) + 1j * np.random.randn(n, P)
    # tau = 10.
    P_min = 0.05

    if mask or deconv:
        epsilon_log = np.log10(epsilon)
        epsilonF_log = np.log10(epsilonF)
        # dtau = float(tau - tauF) / Imax

    print("Main cycle")

    for i in range(Imax):
        if mask or deconv:
            # epsilon_iter = 10 ** (epsilon_log - (epsilon_log - epsilonF_log) * float(i) / (
            #             Imax - 1))  # Exponential linearly decreased
            epsilon_iter = 0
            # epsilon_iter = epsilon - (epsilon - epsilonF) / (Imax - 1) * float(i)  # Linear decreased
        else:
            epsilon_iter = epsilon

        Shat = update_S(V, A, M, mask=mask, deconv=deconv, epsilon=epsilon_iter)
        Shat[np.isnan(Shat)] = np.zeros(1).astype('complex64')
        #             Shat = update_S(Xe,A,M,mask=mask,epsilon=epsilon_iter)
        #         Shat = np.dot(LA.inv(np.dot(A.T,A)),np.dot(A.T,V))
        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat, (n, Nx, Ny)).squeeze(), Ndim))  # In direct plane
            if positivityS:
                S[S < 0] = 0
        else:
            S = np.real(np.reshape(Shat, (n, Nx, Ny)).squeeze())
            if positivityS:
                S[S < 0] = 0

        # For wavelet representation
        if wavelet:

            if wname == 'starlet':
                if Ndim == 1:
                    coarseScale = np.zeros((n, 1, P))
                    wt = np.zeros((n, scale - 1, P))  # For starlet transform
                elif Ndim == 2:
                    coarseScale = np.zeros((n, 1, Nx, Ny))
                    wt = np.zeros((n, scale - 1, Nx, Ny))  # For starlet transform
                for sr in range(n):
                    if Ndim == 1:
                        wtTmp = starlet1d.decomposition(S[sr])          # 1d Starlet transform
                    elif Ndim == 2:
                        wtTmp = starlet2d.decomposition(S[sr])            # 2d Starlet transform
                    # Remove coarse scale
                    coarseScale[sr] = np.copy(wtTmp[-1])
                    wt[sr] = np.copy(wtTmp[:-1])

            elif wname == 'dct':
                wt = np.zeros((n, Nx, Ny)).squeeze()
                coarseScale = np.zeros((n, Nx, Ny)).squeeze()
                for sr in range(n):
                    if Ndim == 1:
                        wt[sr] = scifft.dct(S[sr], type=2, norm='ortho')
                        coarseScale[sr, :Ny / 16] = wt[sr, :Ny / 16]
                        wt[sr, :Ny / 16] = 0
                    elif Ndim == 2:
                        wt[sr] = wt2d.dct2(S[sr], type=2, norm='ortho')
                        coarseScale[sr, :Nx / 16, :Ny / 16] = wt[sr, :Nx / 16, :Ny / 16]
                        wt[sr, :Nx / 16, :Ny / 16] = 0
            # Noise estimate
            if thresStrtg == 1:
                if wname == 'dct':
                    # Use the finest scale of starlet to estimate noise level
                    fineScale = np.zeros((n, P))
                    for sr in range(n):
                        if Ndim == 1:
                            wtTmp = starlet1d.decomposition(S[sr])                # 1d Starlet transform
                        elif Ndim == 2:
                            wtTmp = starlet2d.decomposition(S[sr])                  # 2d Starlet transform
                        fineScale[sr] = wtTmp[0].flatten()
                    sig = mad(fineScale)
                else:
                    wt = np.real(np.reshape(wt, (n, np.size(wt) // n)).squeeze())
                    sig = mad(wt[:, :P])
                    # sig = np.array([0.01, 0.01])
            elif thresStrtg == 2:
                sig = np.zeros((n, scale - 1))
                for sr in range(n):
                    # sig = mad(wt[:,:P])                                             # For starlet transform
                    sig[sr] = mad(wt[sr])

            thTab = find_th(wt, P_min, sig, i, Imax, strategy=thresStrtg, Ksig=Kend)
            # thTab = tau*sig
            # thTab = [804.82486547, 2480.76016982, 2072.01364484]
            thIter[i] = thTab

            # Thresholding in terms of percentage of significant coefficients
            if thresStrtg == 1:
                wt = hardTh(wt, thTab, weights=None, reweighted=False)
                # softTh(wt, thTab, weights=None, reweighted=False)
            elif thresStrtg == 2:
                for sr in range(n):
                    wt[sr] = hardTh(wt[sr], thTab[sr], weights=None, reweighted=False)
                    # softTh(wt[sr], thTab[sr], weights=None, reweighted=False)

                    # wt = np.reshape(wt, (n, np.size(wt)/(n*P), P))             # For undecimated wavelet transform

            if wname == 'starlet':
                if thresStrtg == 1:
                    wt = np.reshape(wt, (n, scale - 1, Nx, Ny)).squeeze()
            elif wname == 'dct':
                wt = wt.reshape((n, Nx, Ny)).squeeze()

            if wname == 'starlet':
                for sr in range(n):
                    wtTmp = np.concatenate((wt[sr], coarseScale[sr]), axis=0)
                    if Ndim == 1:
                        S[sr] = starlet1d.reconstruction(wtTmp)  # Inverse starlet transform
                    elif Ndim == 2:
                        S[sr] = starlet2d.reconstruction(wtTmp)  # Inverse starlet transform

            elif wname == 'dct':
                for sr in range(n):
                    if Ndim == 1:
                        wt[sr, :Ny / 16] = coarseScale[sr, :Ny / 16]
                        S[sr] = scifft.dct(wt[sr], type=3, norm='ortho')  # Inverse 1d dct transform
                    elif Ndim == 2:
                        wt[sr, :Nx / 16, :Ny / 16] = coarseScale[sr, :Nx / 16, :Ny / 16]
                        S[sr] = wt2d.dct2(wt[sr], type=3, norm='ortho')
        # For non-wavelet representation
        else:
            sig = mad(S)
            # thTab = tau*sig
            thTab = find_th(S, P_min, sig, i, Imax, strategy=thresStrtg, Ksig=Kend)
            thIter[i] = thTab
            # print(thTab)
            S = hardTh(S, thTab, weights=None, reweighted=False)

        index = check_zero_sources(A, Shat.reshape((n, P)))
        if len(index) > 0:
            reinitialize_sources(V, Shat, A, index)
        else:
            if FTPlane:
                Shat = fftNd1d(S, Ndim)  # Transform in Fourier space
            else:
                S = np.reshape(S, (n, P))

        if FTPlane:
            if wavelet:
                # Don't take the low frequency band into account
                Shat_Hi = filter_Hi(ifftshiftNd1d(Shat, Ndim), Ndim, fc)
                if Ndim == 1 and logistic:  # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:, int(fc * P):int(fc_pass * P)] = Shat_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
                    Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] = Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
                Shat_Hi = fftshiftNd1d(Shat_Hi, Ndim)
                Shat_Hi = np.reshape(Shat_Hi, (n, P))
                # Update A
                A = update_A(V_Hi, Shat_Hi, M, mask=mask, deconv=deconv)
            else:
                Shat = np.reshape(Shat, (n, P))
                A = update_A(V, Shat, M, mask=mask, deconv=deconv)
        else:
            A = update_A(V, S, M, mask=mask, deconv=deconv)
        A[np.isnan(A)] = 0
        A = np.real(A)
        if positivityA:
            A[A < 0] = 0
        index = check_zero_sources(A, Shat.reshape((n, P)))
        if len(index) > 0:
            reinitialize_sources(V, Shat.reshape((n, P)), A, index)
            A[np.isnan(A)] = 0
            A = np.real(A)
        normalize(A)

        # test evaluation
        # piA = np.linalg.inv(A.T @ A) @ A.T
        # S_est = piA @ mixCube.reshape((Bd, -1))
        # res = mixCube.reshape(Bd, -1) - A @ S_est.reshape(n, -1)
        # # res = mixCube.reshape(Bd, -1) - A@S.reshape(n, -1)
        # coef = np.zeros(Bd)
        # for ii in range(Bd):
        #     coef[ii], _ = pearsonr(res[ii].flatten(), csCube[ii].flatten())

        print("Iteration: " + str(i + 1))
        print("Condition number of A:")
        print(LA.cond(A), 'A')
        print("Condition number of S:")
        print(LA.cond(np.reshape(S, (n, P))), 'S')

    if Ndim == 2:
        S = np.reshape(S, (n, Nx, Ny))
    # Ameliorate the estimation of the sources
    if postProc == 1:
        print('Post processing to ameliorate the estimate S:')
        # Shat = update_S(V, A, M, mask=mask, deconv=deconv, epsilon=0.)
        # Shat[np.isnan(Shat)] = np.zeros(1).astype('complex64')
        # deltaV = V - M * (A @ Shat)
        # deltaShat = update_S(deltaV, A, M, mask=mask, deconv=deconv, epsilon=0.)
        # deltaShat[np.isnan(deltaShat)] = np.zeros(1).astype('complex64')
        # Shat += deltaShat
        # deltaV_n = V - M * (A @ Shat)
        # deltaV = deltaV_n
        # # Compare residual and G-T
        # dec_eor = DeconvFwd(deltaV, M, epsilon=sys.float_info.epsilon)
        # dec_eor = np.real(ifft2d1d(dec_eor.reshape(Bd, Nx, Ny)))
        # errEoR = (np.abs(dec_eor - csCube)).sum() / (np.abs(csCube)).sum()
        # coef = np.zeros(Bd)
        # for ii in range(Bd):
        #     coef[ii], _ = pearsonr(dec_eor[ii].flatten(), csCube[ii].flatten())

        # S = np.real(ifft2d1d(Shat.reshape(n, Nx,Ny)))
        S, thIter = update_S_prox_Condat_Vu(V, A, S, M, Nx, Ny, Ndim, Imax=postProcImax, tau=0.0, eta=0.5, Ksig=Ksig,
                                            wavelet=wavelet, scale=scale, wname=wname, thresStrtg=thresStrtg,
                                            FTPlane=FTPlane, positivity=positivityS, GT=csCube)
    elif postProc == 2:
        print('Post processing to ameliorate the estimate S and estimate A:')
        inImax1 = 50
        inImax2 = 1
        tau = 0.0
        mu2 = 0.0
        eta = 0.5
        for i in range(postProcImax):
            S, thIter = update_S_prox_Condat_Vu(V, A, S, M, Nx, Ny, Ndim, Imax=inImax1, tau=tau, eta=eta, Ksig=Ksig,
                                                wavelet=wavelet, scale=scale, wname=wname, thresStrtg=thresStrtg,
                                                FTPlane=FTPlane, positivity=positivityS)

            if FTPlane:
                Shat = fftNd1d(S, Ndim)  # Transform in Fourier space
                if wavelet:
                    # Don't take the low frequency band into account
                    Shat_Hi = filter_Hi(ifftshiftNd1d(Shat, Ndim), Ndim, fc)
                    if Ndim == 1 and logistic:  # If logistic function is activated, particular processing is designed for 1d signal
                        Shat_Hi[:, int(fc * P):int(fc_pass * P)] = Shat_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
                        Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] = Shat_Hi[:,
                                                                     int(-fc_pass * P):int(-fc * P)] * flogist2
                    Shat_Hi = fftshiftNd1d(Shat_Hi, Ndim)
                    Shat_Hi = np.reshape(Shat_Hi, (n, P))
                else:
                    Shat = np.reshape(Shat, (n, P))
            else:
                S = np.reshape(S, (n, P))

            if FTPlane:
                if wavelet:
                    A = update_A_prox(V_Hi, A, Shat_Hi, M, inImax2, mu2, mask=mask, positivity=positivityA)
                else:
                    A = update_A_prox(V, A, Shat, M, inImax2, mu2, mask=mask, positivity=positivityA)
            else:
                A = update_A(V, A, S, M, inImax2, mu2, mask=mask, positivity=positivityA)

        S = np.real(np.reshape(S, (n, Nx, Ny)).squeeze())

    elif postProc == 3:
        print('Post processing to ameliorate the estimate S:')

        for i in range(3):
            Shat = update_S(V, A, M, mask=mask, deconv=deconv, epsilon=0.)
            Shat[np.isnan(Shat)] = np.zeros(1).astype('complex64')
            deltaV = V - M * (A @ Shat)
            # if FTPlane:
            #     S = np.real(ifftNd1d(np.reshape(Shat, (n, Nx, Ny)).squeeze(), Ndim))  # In direct plane
            #
            # deltaS, thIter = update_S_prox_Condat_Vu(deltaV, A, np.zeros_like(S), M, Nx, Ny, Ndim, Imax=10, tau=0.0,
            #                                          eta=0.5,
            #                                          Ksig=Kend,
            #                                          wavelet=wavelet, scale=scale, wname=wname, thresStrtg=thresStrtg,
            #                                          FTPlane=FTPlane, positivity=positivityS)
            # if FTPlane:
            #     deltaShat = fftNd1d(deltaS, Ndim)  # Transform in Fourier space
            # Shat += deltaShat.reshape(n, P)

            errRes = 1.
            while errRes > 1.e-6:
                deltaShat = update_S(deltaV, A, M, mask=mask, deconv=deconv, epsilon=0.)
                deltaShat[np.isnan(deltaShat)] = np.zeros(1).astype('complex64')
                Shat += deltaShat
                deltaV_n = V - M * (A @ Shat)
                # errRes = LA.norm(deltaV_n - deltaV) / LA.norm(deltaV)
                errRes = (deltaV_n - deltaV).std()
                deltaV = deltaV_n

                # Compare residual and G-T
                dec_eor = DeconvFwd(deltaV, M, epsilon=sys.float_info.epsilon)
                dec_eor = np.real(ifft2d1d(dec_eor.reshape(Bd, Nx, Ny)))
                errEoR = (np.abs(dec_eor - csCube)).sum() / (np.abs(csCube)).sum()
                coef = np.zeros(Bd)
                for ii in range(Bd):
                    coef[ii], _ = pearsonr(dec_eor[ii].flatten(), csCube[ii].flatten())
                print("Post-proc: Iteration {}, Error: {}, Correlation: {}".format(i, errEoR, coef[::10]))
                print('Relative error of residual observation: {}'.format(errRes))

            deltaV_Hi = filter_Hi(ifftshiftNd1d(deltaV.reshape(Bd, Nx, Ny), Ndim), Ndim, fc)
            deltaV_Hi = deltaV_Hi.reshape(Bd, P)
            Shat_Hi = filter_Hi(ifftshiftNd1d(Shat.reshape(n, Nx, Ny), Ndim), Ndim, fc)
            Shat_Hi = Shat_Hi.reshape(n, P)
            deltaA = update_A(deltaV_Hi, Shat_Hi, M, mask=mask, deconv=deconv)
            # if FTPlane:
            #     # Shat = fftNd1d(Shat.reshape(n, Nx, Ny), Ndim)  # Transform in Fourier space
            #     Shat = Shat.reshape(n, Nx, Ny)
            #     # Don't take the low frequency band into account
            #     Shat_Hi = filter_Hi(ifftshiftNd1d(Shat, Ndim), Ndim, fc)
            #     if Ndim == 1 and logistic:  # If logistic function is activated, particular processing is designed for 1d signal
            #         Shat_Hi[:, int(fc * P):int(fc_pass * P)] = Shat_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
            #         Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] = Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
            #     Shat_Hi = fftshiftNd1d(Shat_Hi, Ndim)
            #     Shat_Hi = np.reshape(Shat_Hi, (n, P))
            #     # Update A
            #     # A = update_A(V_Hi, Shat_Hi, M, mask=mask, deconv=deconv)
            # deltaA = update_A_prox(deltaV_Hi, np.zeros_like(A), Shat_Hi, M, 5, 0.0, mask=mask, positivity=positivityA)
            A += np.real(deltaA)
            A = np.real(A)
            normalize(A)

        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat, (n, Nx, Ny)).squeeze(), Ndim))  # In direct plane
            if positivityS:
                S[S < 0] = 0
        deltaV = deltaV.reshape(Bd, Nx, Ny)

    else:
        Shat = update_S(V, A, M, mask=mask, deconv=deconv, epsilon=0.)
        Shat[np.isnan(Shat)] = np.zeros(1).astype('complex64')
        deltaV = V - M * (A @ Shat)
        if FTPlane:
            S = np.real(ifftNd1d(np.reshape(Shat, (n, Nx, Ny)).squeeze(), Ndim))  # In direct plane
            if positivityS:
                S[S < 0] = 0
        deltaV = deltaV.reshape(Bd, Nx, Ny)

    return S, A, deltaV, thIter


##########################################################################
# GMCA using proximal operators
##########################################################################
def GMCA_PALM(V, M, n, Nx, Ny, Imax, Ndim, wavelet, scale, mask, deconv, wname='starlet', thresStrtg=2, FTPlane=True,
              fc=1. / 16, logistic=False, Ksig=0.6, positivity=False, FISTA=False, S=None, A=None):
    (Bd, P) = np.shape(V)

    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(ifftshiftNd1d(V, Ndim), Ndim, fc)
        if logistic:
            fc_pass = 2 * fc
            steep = 0.1 / 32 / fc
            flogist1 = 1. / (1 + np.exp(
                -steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, fc_pass * P - fc * P)))
            flogist2 = 1. / (1 + np.exp(
                steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, fc_pass * P - fc * P)))
            V_Hi[:, int(fc * P):int(fc_pass * P)] = V_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
            V_Hi[:, int(-fc_pass * P):int(-fc * P)] = V_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
        V_Hi = fftshiftNd1d(V_Hi, Ndim)

    if (A is None) and (S is None):
        if mask:
            MM = np.zeros_like(M)
            MM[np.abs(M) >= 1e-4] = 1
            MM = np.real(MM)
            Xe = SVTCompletion(V, MM, n, 1.5, 200)
        elif deconv and not mask and FTPlane:
            Xe = V_Hi
        else:
            Xe = V

        R = np.dot(Xe, Xe.T)
        d, v = LA.eig(R)
        A = np.abs(v[:, 0:n])
        normalize(A)

        if FTPlane:
            Shat = np.reshape(np.dot(A.T, Xe), (n, Nx, Ny)).squeeze()
            S = np.real(ifftNd1d(Shat, Ndim))
        else:
            S = np.dot(A.T, Xe)

    Imax1 = 1
    ImaxInt = 5
    Imax2 = 1
    mu1 = 0.0
    muInt = 0.0
    mu2 = 0.0
    eta = 0.0

    if FISTA:
        t = 1.0  # FISTA parameter

    i = 0
    tol = 1e-8
    err = 1.
    while i < Imax and err > tol:
        # S_n = update_S_prox_peusdo_anal(V, A, S, M, Nx, Ny, Ndim, Imax1, mu1, Ksig, mask, wavelet, scale,
        #                                 wname='starlet', thresStrtg=2, FTPlane=True, Ar=None, FISTA=False, PALM=True,
        #                                 currentIter=i, globalImax=Imax)

        # S_n, thIter = update_S_prox_Condat_Vu(V, A, S, M, Nx, Ny, Ndim, Imax1, mu1, eta, Ksig, wavelet, scale,
        #                                       wname='starlet', thresStrtg=2, FTPlane=True, Ar=None, positivity=False,
        #                                       PALM=True, currentIter=i, globalImax=Imax)
        if i == 0:
            S_n, u, sig = update_S_prox_anal(V, A, S, M, Nx, Ny, Ndim, Imax1, ImaxInt, mu1, muInt, Ksig, wavelet, scale,
                                             wname='starlet', thresStrtg=2, FTPlane=True, PALM=False, currentIter=i,
                                             globalImax=Imax)
        else:
            S_n, u, sig = update_S_prox_anal(V, A, S, M, Nx, Ny, Ndim, Imax1, ImaxInt, mu1, muInt, Ksig, wavelet, scale,
                                             wname='starlet', thresStrtg=2, FTPlane=True, PALM=False, currentIter=i,
                                             globalImax=Imax, u=u, sig=sig)
        errS = (((S_n - S) ** 2).sum()) / ((S ** 2).sum())
        if FISTA:
            tn = (1. + np.sqrt(1 + 4 * t * t)) / 2
            S = S_n + (t - 1) / tn * (S_n - S)
            t = tn
        else:
            S = S_n

        if FTPlane:
            Shat = fftNd1d(S, Ndim)  # Transform in Fourier space
            if wavelet:
                # Don't take the low frequency band into account
                Shat_Hi = filter_Hi(ifftshiftNd1d(Shat, Ndim), Ndim, fc)
                if Ndim == 1 and logistic:  # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:, int(fc * P):int(fc_pass * P)] = Shat_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
                    Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] = Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
                Shat_Hi = fftshiftNd1d(Shat_Hi, Ndim)
                Shat_Hi = np.reshape(Shat_Hi, (n, P))
            else:
                Shat = np.reshape(Shat, (n, P))
        else:
            S = np.reshape(S, (n, P))

        if FTPlane:
            if wavelet:
                A_n = update_A_prox(V_Hi, A, Shat_Hi, M, Imax2, mu2, mask=mask, positivity=False)
            else:
                A_n = update_A_prox(V, A, Shat, M, Imax2, mu2, mask=mask, positivity=False)
        else:
            A_n = update_A(V, A, S, M, Imax2, mu2, mask=mask, positivity=False)

        errA = (((A_n - A) ** 2).sum()) / ((A ** 2).sum())

        A = A_n
        # err = errS
        err = max(errS, errA)
        i = i + 1

        print("Iteration: " + str(i))
        print("Condition number of A:")
        print(LA.cond(A))
        print("Condition number of S:")
        print(LA.cond(S))
        print("Current error")
        print(err)

    # Refinement with the final thresholding
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

    return S, A


def GMCA_prox(V, M, n, Nx, Ny, Imax, epsilon, epsilonF, Ndim, wavelet, scale, mask, deconv, wname='starlet',
              thresStrtg=2, FTPlane=True, fc=1. / 16, logistic=False, postProc=True, Ksig=0.6, positivity=False):
    """
    GMCA using proximal methods
    
    @param V: Input data, size: bands*pixels
    
    @param M: Matrix of mask, size: sources*pixels
    
    @param n: Number of sources
    
    @param Nx: Number of rows of the source
    
    @param Ny: Number of columns of the source
    
    @param Imax: Maximum iterations
    
    @param epsilon: Tikhonov parameter

    @param epsilonF: Parameter of the final regularization

    @param Ndim: The dimension of initial source. Value 1 for signal, value 2 for image

    @param wavelet: Wavelet option

    @param scale: Scales of wavelet decomposition

    @param mask: Application of mask. Value True for the case with mask, value False for the case without mask

    @param deconv: Whether in the case of blurring, default False

    @param wname: Wavelet type.

    @param thresStrtg: Strategy to find the thresholding, valid just for parameter wavelet=True
    strategy=1 means to find the thresholding level based on the percentage of all coefficients larger than 3*sigma
    strategy=2 means in terms of scales, to find the thresholding level based on the percentage of coefficients larger than 3*sigma for each scale

    @param FTPlane: Whether the input data are in Fourier space

    @param fc: Cut-off frequency, fc is normalized to numerical frequency

    @param logistic: Whether using logistic function for the high-pass filter, default is false

    @param postProc: Whether using refinement step to ameliorate the estimate, default is 0

    @param postProcImax: Number of iterations for the refinement step, default is 50

    @param Ksig: K-sigma, level of threshold

    @param positivity: Positivity constraint, default is false
    
    @return: Sources S and mixing matrix A, size sources*pixels and size bands*sources respectively
    """
    (Bd, P) = np.shape(V)

    if Ndim == 1 and FTPlane:
        V_Hi = filter_Hi(ifftshiftNd1d(V, Ndim), Ndim, fc)
        if logistic:
            fc_pass = 2 * fc
            steep = 0.1 / 32 / fc
            flogist1 = 1. / (1 + np.exp(
                -steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, fc_pass * P - fc * P)))
            flogist2 = 1. / (1 + np.exp(
                steep * np.linspace(-(fc_pass * P - fc * P) / 2, (fc_pass * P - fc * P) / 2, fc_pass * P - fc * P)))
            V_Hi[:, int(fc * P):int(fc_pass * P)] = V_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
            V_Hi[:, int(-fc_pass * P):int(-fc * P)] = V_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
        V_Hi = fftshiftNd1d(V_Hi, Ndim)

    elif Ndim == 2 and FTPlane:
        V1 = np.reshape(V, (Bd, Nx, Ny))
        V_Hi = filter_Hi(ifftshiftNd1d(V1, Ndim), Ndim, fc)
        V_Hi = fftshiftNd1d(V_Hi, Ndim)
        V_Hi = np.reshape(V_Hi, (Bd, P))
    else:
        V_Hi = V

    if deconv and FTPlane:
        Xe = V_Hi
    elif mask:
        Xe = SVTCompletion(V, M, n, 0.5, 200)
    else:
        Xe = V

    (u, d, v) = LA.svd(V, full_matrices=False)
    A = u[:n, :].T
    # np.random.seed(5)
    # A = np.random.randn(Bd, n) + 0 * 1j * np.random.randn(Bd, n)
    normalize(A)
    # if wavelet:
    #     gen2 = False
    #     nsNorm(P, scale, gen2=gen2)
    # Shat = np.random.randn(n, P) + 1j * np.random.randn(n, P)
    # tau = 10.
    mu1 = 0.0
    eta = 0.5
    mu2 = 0.0
    inImax1 = 50
    inImax2 = 1
    S = np.zeros((n, P))
    # Shat = np.zeros((n, P)) + np.zeros((n, P)) * 1j
    # dtau = float(tau - tauF) / Imax
    for i in range(Imax):
        # S,thIter = update_S_prox(V, A, S, M, Nx, Ny, Imax, mu1, Ksig, mask, wavelet, scale, wname='starlet',
        #                          thresStrtg=2, FTPlane=True, Ar=None, FISTA=False)
        # S, thIter = update_S_prox(V, A, S, M, Nx, Ny, Imax, mu1, Ksig, mask, wavelet, scale, wname='starlet',
        #                           thresStrtg=2, FTPlane=True, Ar=None, FISTA=False)

        S, thIter = update_S_prox_Condat_Vu(V, A, S, M, Nx, Ny, Ndim, inImax1, mu1, eta, Ksig=Ksig, wavelet=wavelet,
                                            scale=scale, wname=wname, thresStrtg=thresStrtg, FTPlane=FTPlane,
                                            positivity=positivity)

        if FTPlane:
            Shat = fftNd1d(S, Ndim)  # Transform in Fourier space
            if wavelet:
                # Don't take the low frequency band into account
                Shat_Hi = filter_Hi(ifftshiftNd1d(Shat), Ndim, fc)
                if Ndim == 1 and logistic:  # If logistic function is activated, particular processing is designed for 1d signal
                    Shat_Hi[:, int(fc * P):int(fc_pass * P)] = Shat_Hi[:, int(fc * P):int(fc_pass * P)] * flogist1
                    Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] = Shat_Hi[:, int(-fc_pass * P):int(-fc * P)] * flogist2
                Shat_Hi = fftshiftNd1d(Shat_Hi, Ndim)
                Shat_Hi = np.reshape(Shat_Hi, (n, P))
            else:
                Shat = np.reshape(Shat, (n, P))
        else:
            S = np.reshape(S, (n, P))

        if FTPlane:
            if wavelet:
                A = update_A_prox(V_Hi, A, Shat_Hi, M, inImax2, mu2, mask=mask, positivity=False)
                # A = update_A(V_Hi, Shat_Hi, M, mask=mask, deconv=deconv)
            else:
                A = update_A_prox(V, A, Shat, M, inImax2, mu2, mask=mask, positivity=False)
        else:
            A = update_A(V, A, S, M, inImax2, mu2, mask=mask, positivity=False)

        # A = np.real(A)
        # normalize(A)

        print("Iteration: " + str(i + 1))
        print("Condition number of A:")
        print(LA.cond(A), 'A')
        print("Condition number of S hat:")
        print(LA.cond(Shat), 'Shat')

    # S = np.real(ifftNd1d(np.reshape(Shat, (n, Nx, Ny)).squeeze(), Ndim))
    S = np.real(np.reshape(S, (n, Nx, Ny)).squeeze())
    return S, A


def GMCA(X, n=2, maxts=7, mints=3, nmax=100, L0=0, UseP=1, verb=0, Init=0, Aposit=False, BlockSize= None, NoiseStd=[],
         IndNoise=[], Kmax=1., AInit=None, tol=1e-6, ColFixed=None):
    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    nX = np.shape(X)
    m = nX[0]
    t = nX[1]
    Xw = cp.copy(X)

    if BlockSize == None:
        BlockSize = n

    if verb:
        print("Initializing ...")
    if Init == 0:
        R = np.dot(Xw, Xw.T)
        D, V = lng.eig(R)
        A = V[:, 0:n]
    if Init == 1:
        A = np.random.randn(m, n)
    if AInit is not None:
        A = cp.deepcopy(AInit)

    for r in range(0, n):
        A[:, r] = A[:, r] / lng.norm(A[:, r])  # - We should do better than that

    if ColFixed is not None:
        p_fixed = np.shape(ColFixed)[1]
        A[:, 0:p_fixed] = ColFixed

    S = np.dot(A.T, Xw)

    # Call the core algorithm

    S, A = Core_GMCA(X=Xw, A=A, S=S, n=n, maxts=maxts, BlockSize=BlockSize, Aposit=Aposit, tol=tol, kend=mints,
                     nmax=nmax, L0=L0, UseP=UseP, verb=verb, IndNoise=IndNoise, Kmax=Kmax, NoiseStd=NoiseStd,
                     ColFixed=ColFixed)

    Results = {"sources": S, "mixmat": A}

    return Results


################# AMCA internal code (Weighting the sources only)

def Core_GMCA(X=0, n=0, A=0, S=0, maxts=7, kend=3, nmax=100, BlockSize=2, L0=1, Aposit=False, UseP=0, verb=0,
              IndNoise=[], Kmax=0.5, NoiseStd=[], tol=1e-6, ColFixed=None):
    # --- Import useful modules
    import numpy as np
    import scipy.linalg as lng
    import copy as cp
    import scipy.io as sio
    import time

    # --- Init

    n_X = np.shape(X)
    n_S = np.shape(S)

    if ColFixed is not None:
        p_fixed = np.shape(ColFixed)[1]

    k = maxts
    dk = (k - kend) / (nmax - 1)
    perc = Kmax / nmax
    Aold = cp.deepcopy(A)
    #
    Go_On = 1
    it = 1
    #
    if verb:
        print("Starting main loop ...")
        print(" ")
        print("  - Final k: ", kend)
        print("  - Maximum number of iterations: ", nmax)
        if UseP:
            print("  - Using support-based threshold estimation")
        if L0:
            print("  - Using L0 norm rather than L1")
        if Aposit:
            print("  - Positivity constraint on A")
        print(" ")
        print(" ... processing ...")
        start_time = time.time()

    if Aposit:
        A = abs(A)

    Resi = cp.deepcopy(X)

    # --- Main loop
    while Go_On:
        it += 1

        if it == nmax:
            Go_On = 0

        # --- Estimate the sources

        sigA = np.sum(A * A, axis=0)
        indS = np.where(sigA > 0)[0]

        if np.size(indS) > 0:

            # Using blocks

            # IndBatch = randperm(len(indS))  #---- mini-batch amongst available sources

            # if BlockSize+1 < len(indS):
            #     indS = indS[IndBatch[0:BlockSize]]

            # Resi = Resi + np.dot(A[:,indS],S[indS,:])   # Putting back the sources

            Ra = np.dot(A[:, indS].T, A[:, indS])
            Ua, Sa, Va = np.linalg.svd(Ra)
            Sa[Sa < np.max(Sa) * 1e-9] = np.max(Sa) * 1e-9
            iRa = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
            piA = np.dot(iRa, A[:, indS].T)
            S[indS, :] = np.dot(piA, Resi)

            # Propagation of the noise statistics

            if len(NoiseStd) > 0:
                SnS = 1. / (n_X[1] ** 2) * np.diag(np.dot(piA, np.dot(np.diag(NoiseStd ** 2), piA.T)))

            # Thresholding

            Stemp = S[indS, :]
            Sref = cp.copy(S)

            for r in range(len(indS)):

                St = Stemp[r, :]

                if len(IndNoise) > 0:
                    tSt = kend * mad(St[IndNoise])
                elif len(NoiseStd) > 0:
                    tSt = kend * np.sqrt(SnS[indS[r]])
                else:
                    tSt = kend * mad(St)

                indNZ = np.where(abs(St) - tSt > 0)[0]
                thrd = mad(St[indNZ])

                if UseP == 0:
                    thrd = k * thrd

                if UseP == 1:
                    Kval = np.min([np.floor(np.max([2. / n_S[1], perc * it]) * len(indNZ)), n_S[1] - 1.])
                    I = abs(St[indNZ]).argsort()[::-1]
                    Kval = np.int(np.min([np.max([Kval, 5.]), len(I) - 1.]))
                    IndIX = np.int(indNZ[I[Kval]])
                    thrd = abs(St[IndIX])

                if UseP == 2:
                    t_max = np.max(abs(St[indNZ]))
                    t_min = np.min(abs(St[indNZ]))
                    thrd = (0.5 * t_max - t_min) * (1 - (it - 1.) / (nmax - 1)) + t_min  # We should check that

                St[(abs(St) < thrd)] = 0
                indNZ = np.where(abs(St) > thrd)[0]

                if L0 == 0:
                    St[indNZ] = St[indNZ] - thrd * np.sign(St[indNZ])

                Stemp[r, :] = St

            S[indS, :] = Stemp

            k = k - dk

        # --- Updating the mixing matrix

        Xr = cp.deepcopy(Resi)

        if ColFixed is not None:
            sigA = np.sum(S * S, axis=1)
            sigA[0:p_fixed] = 0
            indS = np.where(sigA > 0)[0]

        Rs = np.dot(S[indS, :], S[indS, :].T)
        Us, Ss, Vs = np.linalg.svd(Rs)
        Ss[Ss < np.max(Ss) * 1e-9] = np.max(Ss) * 1e-9
        iRs = np.dot(Us, np.dot(np.diag(1. / Ss), Vs))
        piS = np.dot(S[indS, :].T, iRs)
        A[:, indS] = np.dot(Resi, piS)

        if Aposit:
            for r in range(len(indS)):
                if A[(abs(A[:, indS[r]]) == np.max(abs(A[:, indS[r]]))), indS[r]] < 0:
                    A[:, indS[r]] = -A[:, indS[r]]
                    S[indS[r], :] = -S[indS[r], :]
                A = A * (A > 0)

        A = A / np.maximum(1e-24, np.sqrt(np.sum(A * A, axis=0)))

        DeltaA = np.max(abs(1. - abs(np.sum(A * Aold, axis=0))))  # Angular variations

        if DeltaA < tol:
            if it > 500:
                Go_On = 0

        if verb:
            print("Iteration #", it, " - Delta = ", DeltaA)

        Aold = cp.deepcopy(A)

        # Resi = Resi - np.dot(A[:,indS],S[indS,:])  # Re-defining the residual

    if verb:
        elapsed_time = time.time() - start_time
        print("Stopped after ", it, " iterations, in ", elapsed_time, " seconds")
    #
    return S, A


def run_GMCA(X_wt, AInit, n_s, mints, nmax, L0, ColFixed, whitening, epsi):
    # First guess mixing matrix (could be set to None or not provided at all)
    if AInit is None:
        AInit = np.random.rand(len(X_wt), n_s)

    print('\nNow running GMCA . . .')

    if whitening:

        R = X_wt @ X_wt.T
        L, U = np.linalg.eig(R)
        ## whitening the data

        Q = np.diag(1. / (L + epsi * np.max(L))) @ U.T
        iQ = U @ np.diag((L + epsi * np.max(L)))

        if ColFixed is None:
            CL = None
        else:
            CL = Q @ ColFixed

        # start_w = time.time()
        Results = GMCA(Q @ X_wt, n=n_s, mints=mints, nmax=nmax, L0=L0, Init=0, AInit=AInit, ColFixed=CL)
        # end_w = time.time()

        Ae = iQ @ Results["mixmat"]  # estimated mixing matrix

    else:
        # start_w = time.time()
        Results = GMCA(X_wt, n=n_s, mints=mints, nmax=nmax, L0=L0, Init=0, AInit=AInit, ColFixed=ColFixed)
        # end_w = time.time()

        Ae = Results["mixmat"]

    # tw = end_w - start_w
    # print('. . completed in %.2f minutes\n' % (tw / 60))

    return Ae

