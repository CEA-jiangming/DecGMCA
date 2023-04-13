import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from os.path import join


def main(cmp):
    freqs = range(50, 201, 10)
    Nx = 1800
    Ny = 1800
    cube = np.zeros((np.size(freqs), Nx, Ny))

    projectPath = '../../EoR/skymap/'
    freqbin = 'wide'

    for i, f in enumerate(freqs):
        if cmp == '21cm':
            filename = join(projectPath, cmp, 'slice', freqbin, 'deltaTb_f' +
                            '{:03d}'.format(f) + '.00_N1800_fov10.0.fits')
        elif cmp == 'ff':
            filename = join(projectPath, cmp, freqbin, 'gfree_' +
                            '{:03d}'.format(f) + '.00.fits')
        elif cmp == 'halo':
            filename = join(projectPath, cmp, freqbin, 'cluster_' +
                            '{:03d}'.format(f) + '.00.fits')
        elif cmp == 'ptr':
            filename = join(projectPath, cmp, freqbin, 'ptr_' +
                            '{:d}'.format(f) + '.00.fits')
        elif cmp == 'sync':
            filename = join(projectPath, cmp, freqbin, 'gsync_' +
                            '{:03d}'.format(f) + '.00.fits')
        else:
            print("Invalid component!")
        cube[i] = fits.getdata(filename)

    if (cube>=0).all():
        print("All values are non-negative!")
    A = np.sum(cube, axis=(1, 2))
    plt.figure()
    plt.plot(freqs, A)
    plt.title(cmp)
    plt.xlabel('freq (MHz)')
    plt.ylabel('Total brightness temperature (K)')
    plt.show()


if __name__ == '__main__':
    cmp = 'ptr'
    main(cmp)
