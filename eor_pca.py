import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pyWavelet.wav2d as wt2d
import pylab
import os
import sys
import time
import eorEvaluation as eorEval
import argparse
import pymaster as nmt
import numpy.linalg as LA

from os.path import join
from pyDecGMCA.algoDecG import *
from pyDecGMCA.mathTools import *
from radial_data import radial_data
from sklearn.decomposition import FastICA, PCA
from scipy.stats import pearsonr


def main():
    DECGMCA = True
    GMCA = False
    FASTICA = False
    FORWARD = False

    Nx = 256
    Ny = 256
    step = 0.1
    freq1 = 50
    freq2 = 200
    band1 = 60
    band2 = 70
    bandstep = 1
    freqs = np.arange(band1, band2, step)
    bands = range(int((band1 - freq1) / step), int((band2 - freq1) / step), bandstep)
    noisyCase = True
    kernelCase = 3  # 0 no kernel, 1 mask, 2 psf, 3 realistic psf

    cube = np.zeros((np.size(freqs), Nx, Ny))
    kern = np.zeros((np.size(freqs), Nx, Ny))
    if noisyCase:
        noise = 0.005 + 0.005 * np.random.randn(np.size(freqs), Nx, Ny)
        fits.writeto('noise.fits', noise, overwrite=True)
    components = ['21cm', 'ff', 'sync', 'ptr']

    projectPath = '../../EoR/'
    skymapPath = 'skymap/'
    typebeam = 'kernel'
    beamPath = typebeam + '_0.1MHz_r5000/'
    freqbin = 'wide/'
    # Read sky maps and beam
    for cmp in components:
        if cmp == '21cm':
            filename = join(projectPath, skymapPath, cmp + '.fits')
            cube21cm = fits.getdata(filename)
        elif cmp == 'ff':
            filename = join(projectPath, skymapPath, cmp + '.fits')
        elif cmp == 'halo':
            filename = join(projectPath, skymapPath, cmp + '.fits')
        elif cmp == 'ptr':
            filename = join(projectPath, skymapPath, cmp + '.fits')
        elif cmp == 'sync':
            filename = join(projectPath, skymapPath, cmp + '.fits')
        else:
            print("Invalid component!")
        tmp = fits.getdata(filename)
        # if cmp == 'ptr':
        #     tmp[tmp>1] = 0
        cube += tmp[bands]

    # noisy cube
    if noisyCase:
        cube += noise

    bd = len(bands)
    cube21cm = cube21cm[bands]
    X = cube.reshape(bd, -1).T

    # For comparison, compute PCA from built-in library
    # n_components = 3
    # pca = PCA(n_components=n_components)
    # H = pca.fit_transform(cube)  # Reconstruct signals based on orthogonal components

    # Compute PCA manually
    n_components = 3
    n_samples, n_features = X.shape
    # mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = X
    eig_val, eig_vec = LA.eig(norm_X.T @ norm_X)
    ind_sort = np.argsort(np.abs(eig_val))[::-1]
    feature = eig_vec[:, ind_sort]
    feature[n_components:] = 0
    fg_est = norm_X @ feature
    residual = (norm_X - np.abs(fg_est)).T.reshape(bd, Nx, Ny)
    coef = eorEval.eval_pearson(residual, cube21cm, bd)

    U, S, V = LA.svd(X, full_matrices=False)
    S[n_components:] = 0
    fg_est1 = (U * S) @ V
    residual1 = (X - fg_est1).T.reshape(bd, Nx, Ny)
    coef1 = eorEval.eval_pearson(residual1, cube21cm, bd)

    plt.figure()
    plt.plot(freqs, coef)
    plt.xlabel('freqs / MHz')
    plt.ylabel('coef')

    plt.figure()
    plt.plot(freqs, coef1)
    plt.xlabel('freqs / MHz')
    plt.ylabel('coef1')

    plt.show()


if __name__ == '__main__':
    main()
