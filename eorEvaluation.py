import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pymaster as nmt


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def eval_pearson(estimation, groundTruth, bd):
    coef = np.zeros(bd)
    for i in range(bd):
        coef[i], _ = pearsonr(estimation[i].flatten(), groundTruth[i].flatten())
    return coef


def eval_power_flat(Nx, Ny, cube21cm, res, freqs, freqsInd, noise=None, auto=True, cross=True, show=False, name='DecGMCA', args=None):
    Lx = Nx / 1800. * 10 * np.pi / 180
    Ly = Ny / 1800. * 10 * np.pi / 180

    l0_bins = np.arange(Nx / 8.) * 8 * np.pi / Lx
    lf_bins = (np.arange(Nx / 8.) + 1) * 8 * np.pi / Lx

    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()

    for ind in freqsInd:
        f0 = nmt.NmtFieldFlat(Lx, Ly, np.ones_like(cube21cm[ind]), [cube21cm[ind]])
        f2 = nmt.NmtFieldFlat(Lx, Ly, np.ones_like(res[ind]), [res[ind]])

        # Computing power spectra:
        # As in the full-sky case, you compute the pseudo-CL estimator by
        # computing the coupled power spectra and then decoupling them by
        # inverting the mode-coupling matrix. This is done in two steps below,
        # but pymaster provides convenience routines to do this
        # through a single function call
        if auto:
            w00 = nmt.NmtWorkspaceFlat()
            w00.compute_coupling_matrix(f0, f0, b)
            cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
            cl00_uncoupled = w00.decouple_cell(cl00_coupled)
            w22 = nmt.NmtWorkspaceFlat()
            w22.compute_coupling_matrix(f2, f2, b)
            cl22_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
            cl22_uncoupled = w22.decouple_cell(cl22_coupled)
        if cross:
            w02 = nmt.NmtWorkspaceFlat()
            w02.compute_coupling_matrix(f0, f2, b)
            cl02_coupled = nmt.compute_coupled_cell_flat(f0, f2, b)
            cl02_uncoupled = w02.decouple_cell(cl02_coupled)

        if noise is not None:
            f0_noise = nmt.NmtFieldFlat(Lx, Ly, np.ones_like(noise[ind]), [noise[ind]])
            w00_noise = nmt.NmtWorkspaceFlat()
            w00_noise.compute_coupling_matrix(f0_noise, f0_noise, b)
            cl00_coupled_noise = nmt.compute_coupled_cell_flat(f0_noise, f0_noise, b)
            cl00_uncoupled_noise = w00_noise.decouple_cell(cl00_coupled_noise)
            auto_spec = cl22_uncoupled[0] - cl00_uncoupled_noise[0]

        plt.figure()
        plt.loglog(ells_uncoupled, cl00_uncoupled[0], 'r-')
        plt.loglog(ells_uncoupled, cl22_uncoupled[0], 'g-')
        if cross:
            plt.loglog(ells_uncoupled, cl02_uncoupled[0], 'c-.')
        if noise is not None:
            plt.loglog(ells_uncoupled, auto_spec, 'b.')
        plt.grid('on')
        plt.xlabel('$\ell$')
        plt.ylabel('$C_\ell$')
        if cross and (noise is None):
            plt.legend(['Auto-power spectrum of estimated 21cm', 'Auto-power spectrum of Ground-Truth',
                        'Cross-power spectrum'])
        elif (not cross) and (noise is not None):
            plt.legend(['Auto-power spectrum of estimated 21cm', 'Auto-power spectrum of Ground-Truth',
                        'Corrected auto-power spectrum'])
        # plt.savefig(
        #     'DecGMCA_power_spec_chan1_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale, thresStrtg, epsilonF,
        #                                                                                kend))
        plt.savefig('{}_power_spec_{:.2f}MHz_n{}_scale{}_thStrtg{}_epsilonF{}_kend{}.png'.format(name, freqs[ind],
                                                                                                 args[0], args[1], args[2], args[3], args[4]))

        if cross:
            norm_cross_spec = cl02_uncoupled[0] / np.sqrt(cl00_uncoupled[0] * cl22_uncoupled[0])
            plt.figure()
            plt.plot(ells_uncoupled, norm_cross_spec)
            plt.grid('on')
            plt.xlabel('$\ell$')
            plt.ylabel('Coef')
            # plt.savefig(
            #     'DecGMCA_normalized_power_spec_chan1_n{0}_scale{1}_thStrtg{2}_epsilonF{3}_kend{4}.png'.format(n, scale,
            #                                                                         thresStrtg, epsilonF, kend))
            plt.savefig('{}_normalized_spec_{:.2f}MHz_n{}_scale{}_thStrtg{}_epsilonF{}_kend{}.png'.format(name, freqs[ind],
                                                                                                          args[0], args[1], args[2], args[3], args[4]))
    if show:
        plt.show()
