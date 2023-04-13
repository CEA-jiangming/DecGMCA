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

from os.path import join
from pyDecGMCA.algoDecG import *
from pyDecGMCA.mathTools import *
from radial_data import radial_data
from sklearn.decomposition import FastICA, PCA
from scipy.stats import pearsonr


def createKernel2D(Nx, Ny, sigma, bd=1, export=True):
    sigma_min = sigma[0]
    sigma_max = sigma[1]
    print("Process of creating PSFs ......")
    print("The standard deviation of PSFs (" + str(bd) + " channels, " + str(Nx) + " * " + str(Ny) + " samplings) range from " + str(
        sigma_min) + " to " + str(sigma_max))
    x = np.linspace(1, Nx, Nx) - Nx / 2
    y = np.linspace(1, Ny, Ny) - Ny / 2
    kern = np.zeros((bd, Nx, Ny))
    X, Y = np.meshgrid(x, y)
    for i in np.arange(bd):
        sigma_i = (sigma_max - sigma_min) / bd * i + sigma_min
        kern[i] = np.exp(-0.5 * (X / sigma_i) ** 2 - 0.5 * (Y / sigma_i) ** 2)
        kern[i] /= np.max(kern[i])
    dr = 'EoR/kernels/'
    # export in Fits format
    if export:
        if not os.path.exists(dr):
            os.makedirs(dr)
            # t = glob.glob(dr + 'mask*.fits')
            # if len(t) == 0:
            #     index = 0
            #
            # else:
            #     spr = dr + 'mask|.fits|,'
            #     indarr = filter(bool, re.split(spr, ','.join(t)))
            #     indarr = sorted(map(int, indarr))
            #     index = indarr[-1] + 1

        outfit = fits.PrimaryHDU(kern)
        # outfit.writeto(dr + "mask" + str(index) + ".fits")
        outfit.writeto(dr + 'kern_bd' + str(bd) + '.fits', clobber=True)
    print("Creating of PSFs finished.")
    return kern


def main(args):
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

    if kernelCase == 1:
        ratio = 0.3
        kern = np.zeros((np.size(freqs), Nx, Ny))
        nnz = int(np.floor(Nx * Ny * ratio))
        for i, _ in enumerate(freqs):
            cordx = np.random.randint(0, Nx, nnz)
            cordy = np.random.randint(0, Ny, nnz)
            kern[i, cordx, cordy] = 1
            ratio = float(kern[i].sum()) / np.size(kern[i])
            print("Sampling rate: {}".format(ratio))
        kern = np.ones((np.size(freqs), Nx, Ny))
    elif kernelCase == 2:
        kern = createKernel2D(Nx, Ny, [30, 45], bd, export=False)
        # kern[kern >= 0.1] = 1
        # kern[kern < 0.1] = 0
    elif kernelCase == 3:
        for i, f in enumerate(freqs):
            beamname = join(projectPath, beamPath, typebeam + '_freq' + '{:.2f}'.format(f) + '.fits')
            beam = fits.getdata(beamname)
            kern[i] = beam / beam.max()
            # std_f = 100 + 120 * (f/150.)**(-2.55)
            # noise[i] = 1/np.sqrt(2) * (std_f * np.random.randn(Nx, Ny) + 1j * std_f * np.random.randn(Nx, Ny))
            # cube[i] = cube[i] - cube[i].mean()

    # Observation (Mixtures convolved with beams in Fourier space)
    if kernelCase > 0:
        Y = fft2d1d(cube) * kern
    else:
        Y = fft2d1d(cube)
    invY = np.real(ifft2d1d(Y))
    fits.writeto('inv_obs.fits', invY, overwrite=True)
    fits.writeto('mix_cube.fits', cube, overwrite=True)
    Y = np.reshape(Y, (bd, Nx * Ny))
    kern = np.reshape(kern, (bd, Nx * Ny))

    # DecGMCA
    if DECGMCA:
        FTPlane = True
        mask = False
        deconv = True
        wavelet = True
        n = args.n  # 4
        Ndim = 2
        scale = args.s  # 4
        thresStrtg = args.t  # 1
        Imax = 10
        fc = 1. / 32
        epsilonF = args.e  # 1 * 1.e-3
        epsilon = 10 * epsilonF
        logistic = True
        postProc = 3
        postProcImax = 10
        kend = args.k  # 0.4
        Ksig = args.k
        positivity = False
        (S_est_ori, A_est, deltaY, thIter) = DecGMCA(Y, kern, n, Nx, Ny, Imax, epsilon, epsilonF, Ndim,
                                 wavelet=wavelet, scale=scale, mask=mask, deconv=deconv,
                                 wname='starlet', thresStrtg=thresStrtg, FTPlane=FTPlane, fc=fc, Kend=kend,
                                 logistic=logistic, postProc=postProc, postProcImax=postProcImax, Ksig=Ksig, positivityA=positivity, positivityS=positivity, mixCube=cube, csCube=cube21cm)

        fits.writeto('estS_ori.fits', S_est_ori, overwrite=True)
        fits.writeto('estA.fits', A_est, overwrite=True)

        # Residuals computed from estimated A of DecGMCA solution
        X_ori = A_est @ S_est_ori.reshape(n, -1)
        res_dec_ori = cube.reshape(bd, -1) - X_ori
        fits.writeto('res_ori.fits', res_dec_ori, overwrite=True)

        # Residuals computed from estimated A and ground-truth mixtures
        piA = np.linalg.inv(A_est.T @ A_est) @ A_est.T
        S_est = piA @ cube.reshape((bd, -1))
        S_est = S_est.reshape(n, Nx, Ny)
        fits.writeto('estS.fits', S_est, overwrite=True)
        X = A_est@S_est.reshape(n, -1)
        res_dec = cube.reshape(bd, -1) - X
        res_dec = res_dec.reshape((bd, Nx, Ny))
        fits.writeto('res.fits', res_dec, overwrite=True)
        fits.writeto('thIter.fits', thIter, overwrite=True)

        # Beamed ground-truth
        cube21cm_vis = np.real(ifft2d1d(fft2d1d(cube21cm) * kern.reshape(bd, Nx, Ny)))
        if noisyCase:
            noise_vis = ifft2d1d(fft2d1d(noise) * kern.reshape(bd, Nx, Ny))
        res_vis = np.real(ifft2d1d(deltaY))
        fits.writeto('res_vis.fits', res_vis, overwrite=True)
        fits.writeto('21cm_beamed.fits', cube21cm_vis, overwrite=True)

        # Estimated decovolved EoR
        dec_eor = DeconvFwd(deltaY.reshape(bd, -1), kern, epsilon=sys.float_info.epsilon)
        dec_eor = np.real(ifft2d1d(dec_eor.reshape(bd, Nx, Ny)))
        fits.writeto('res_deconv.fits', dec_eor, overwrite=True)

        if thresStrtg == 2:
            plt.figure()
            plt.plot(thIter[:, :, 0])
            plt.xlabel('Iteration')
            plt.ylabel('Th / scale 0')
            plt.legend(['s1', 's2', 's3'])
            plt.show()

            plt.figure()
            plt.plot(thIter[:, :, 1])
            plt.xlabel('Iteration')
            plt.ylabel('Th / scale 1')
            plt.legend(['s1', 's2', 's3'])
            plt.show()

            plt.figure()
            plt.plot(thIter[:, :, 2])
            plt.xlabel('Iteration')
            plt.ylabel('Th / scale 2')
            plt.legend(['s1', 's2', 's3'])
            plt.show()
        elif thresStrtg == 1:
            plt.figure()
            plt.plot(thIter)
            plt.xlabel('Iteration')
            plt.ylabel('Th')
            plt.legend(['s1', 's2', 's3'])
            # plt.show()
            plt.savefig(
                'DecGMCA_real_threshold_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF, kend))

    # Deconvolution using ForWaRD
    if FORWARD:
        X = DeconvFwd(Y, kern, epsilon=1.e-10)
        X = np.real(ifft2d1d(X.reshape(bd, Nx, Ny)))
        plt.figure()
        plt.imshow(X[0], vmax=1e4)
        plt.colorbar()
        plt.figure()
        plt.imshow(invY[0], vmax=1e4)
        plt.colorbar()
        plt.figure()
        plt.imshow(cube[0], vmax=1e4)
        plt.colorbar()
        fits.writeto('Y_forward.fits', X.reshape(bd, Nx, Ny), overwrite=True)
    else:
        X = cube

    # #
    # GMCA
    # wavelet space
    if GMCA:
        scale = 4
        starlet2D = wt2d.Starlet2D(nx=Nx, ny=Ny, scale=scale, fast=True, gen2=False, normalization=True)
        X_wt = np.zeros((bd, (scale-1)*Nx*Ny))
        for i in range(bd):
            tmp = starlet2D.decomposition(X[i])
            X_wt[i] = tmp[:-1].reshape((1, -1))

        mints = 0.5
        nmax = 100
        L0 = 0
        AInit = None
        ColFixed = None
        whitening = False
        epsi = 1e-3
        n = args.n  # 4

        Ae = run_GMCA(X_wt, AInit, n, mints, nmax, L0, ColFixed, whitening, epsi)
        # Ae = Results["mixmat"]
        # fits.writeto('estA_forward_gmca.fits', Ae, overwrite=True)
        fits.writeto('estA_gmca.fits', Ae, overwrite=True)

        piA = np.linalg.inv(Ae.T @ Ae) @ Ae.T
        Se = piA @ cube.reshape((bd, -1))
        X_gmca = Ae @ Se
        res = cube.reshape((bd, -1)) - X_gmca
        Se = Se.reshape((n, Nx, Ny))
        res = res.reshape((bd, Nx, Ny))
        fits.writeto('estS_gmca.fits', Se, overwrite=True)
        fits.writeto('res_gmca.fits', res, overwrite=True)

    # Compute ICA
    if FASTICA:
        ica = FastICA(n_components=n)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix

        # # For comparison, compute PCA
        # pca = PCA(n_components=3)
        # H = pca.fit_transform(cube)  # Reconstruct signals based on orthogonal components
        # ps = []
        # std = []
        # for i in range(300):
        #     im = np.fft.fft2(res[i])
        #     im = abs(im) ** 2
        #     rd = radial_data(im)
        #     ps.append(rd.mean)
        #     std.append(rd.std)
        # ps = np.array(ps)
        # std = np.array(std)

    # Evaluation section
    # Pearson correlation coefficients
    if DECGMCA:
        # Compare residuals computed from estimated A of DecGMCA solution with the ground-truth
        coef_ori = eorEval.eval_pearson(res_dec_ori, cube21cm, bd)
        # Compare residuals computed from estimated A and ground-truth mixtures with the ground-truth
        coef = eorEval.eval_pearson(res_dec, cube21cm, bd)
        # Compare original residual with beamed ground-truth
        coef_vis = eorEval.eval_pearson(res_vis, cube21cm_vis, bd)
        # Compare deconvolved residual with the ground-truth
        coef_dec = eorEval.eval_pearson(dec_eor, cube21cm, bd)

        plt.figure()
        plt.plot(freqs, coef_ori)
        plt.xlabel('freqs / MHz')
        plt.ylabel('coef')
        plt.savefig('DecGMCA_post_coef_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF, kend))

        plt.figure()
        plt.plot(freqs, coef)
        plt.xlabel('freqs / MHz')
        plt.ylabel('coef')
        plt.savefig(
            'DecGMCA_coef_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF, kend))

        plt.figure()
        plt.plot(freqs, coef_vis)
        plt.xlabel('freqs / MHz')
        plt.ylabel('coef')
        plt.savefig(
            'DecGMCA_vis_coef_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF, kend))

        plt.figure()
        plt.plot(freqs, coef_dec)
        plt.xlabel('freqs / MHz')
        plt.ylabel('coef')
        plt.savefig(
            'DecGMCA_dec_coef_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF,
                                                                                       kend))

        # if noisyCase:
        #     eorEval.eval_power_flat(Nx, Ny, cube21cm, dec_eor, freqs, [0, 50], noise_vis, auto=True, cross=False, show=False, args=[n, scale, thresStrtg, epsilonF, kend])
        # else:
        eorEval.eval_power_flat(Nx, Ny, cube21cm, dec_eor, freqs, [0, 50], None, auto=True, cross=True, show=False, args=[n, scale, thresStrtg, epsilonF, kend])

    if GMCA:
        coef_gmca = eorEval.eval_pearson(res, cube21cm, bd)
        plt.figure()
        plt.plot(freqs, coef_gmca)
        plt.xlabel('freqs / MHz')
        plt.ylabel('coef')
        plt.savefig('GMCA_coef_correlation.png')
        if noisyCase:
            eorEval.eval_power_flat(Nx, Ny, cube21cm, res, freqs, [0, 50], noise, auto=True, cross=False, show=False, name='GMCA')
        else:
            eorEval.eval_power_flat(Nx, Ny, cube21cm, res, freqs, [0, 50], None, auto=True, cross=True, show=False, name='GMCA')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default='3', type=int, help='source number')
    parser.add_argument('-s', default='4', type=int, help='scale')
    parser.add_argument('-t', default='1', type=int, help='threshold strategy')
    parser.add_argument('-e', default='1.e-4', type=float, help='final epsilon')
    parser.add_argument('-k', default='0.5', type=float, help='final k')
    args = parser.parse_args()
    print('Number of sources: {0}, Scale: {1}, Threshold strategy: {2}, Final regularization parameter: {3},'
          ' Final k: {4}'.format(args.n, args.s, args.t, args.e, args.k))
    main(args)
