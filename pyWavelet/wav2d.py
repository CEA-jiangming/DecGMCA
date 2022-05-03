"""
Created on Mar 30, 2015

@author: mjiang
"""
import numpy as np
import scipy.fftpack as scifft
import scipy.signal as psg
from pyWavelet import wav1d


def test_ind(ind, N):
    res = ind
    if ind < 0:
        res = -ind
        if res >= N:
            res = 2 * N - 2 - ind
    if ind >= N:
        res = 2 * N - 2 - ind
        if res < 0:
            res = -ind
    return res


def b3splineTrans(im_in, step):
    (nx, ny) = np.shape(im_in)
    im_out = np.zeros((nx, ny))
    c1 = 1. / 16
    c2 = 1. / 4
    c3 = 3. / 8

    buff = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            jl = test_ind(j - step, ny)
            jr = test_ind(j + step, ny)
            jl2 = test_ind(j - 2 * step, ny)
            jr2 = test_ind(j + 2 * step, ny)
            buff[i, j] = c3 * im_in[i, j] + c2 * (im_in[i, jl] + im_in[i, jr]) + c1 * (im_in[i, jl2] + im_in[i, jr2])

    for j in range(ny):
        for i in range(nx):
            il = test_ind(i - step, nx)
            ir = test_ind(i + step, nx)
            il2 = test_ind(i - 2 * step, nx)
            ir2 = test_ind(i + 2 * step, nx)
            im_out[i, j] = c3 * buff[i, j] + c2 * (buff[il, j] + buff[ir, j]) + c1 * (buff[il2, j] + buff[ir2, j])

    return im_out


def b3spline_fast(step_hole):
    c1 = 1. / 16
    c2 = 1. / 4
    c3 = 3. / 8
    length = 4 * step_hole + 1
    kernel1d = np.zeros((1, length))
    kernel1d[0, 0] = c1
    kernel1d[0, -1] = c1
    kernel1d[0, step_hole] = c2
    kernel1d[0, -1 - step_hole] = c2
    kernel1d[0, 2 * step_hole] = c3
    kernel2d = np.dot(kernel1d.T, kernel1d)
    return kernel2d


class Starlet2D(object):
    """
    Class of two-dimensional starlet

    List of atributes:
    self.scale, self.nx, self.ny, self.fast, self.gen2, self.normalization, self.trTab

    List of methods:
    decomposition, reconstruction, adjoint
    """
    def __init__(self, nx, ny, scale, fast=True, gen2=False, normalization=False):
        self.scale = scale
        self.nx = nx
        self.ny = ny
        self.fast = fast
        self.gen2 = gen2
        self.normalization = normalization  # Normalization of the noise variance of each scale
        im = np.zeros((nx, ny))
        im[nx // 2, ny // 2] = 1
        wt = self.decomposition(im, computewt=False)
        tmp = wt ** 2
        self.trTab = np.sqrt(np.sum(tmp, axis=(1, 2)))

    def decomposition(self, im, computewt=True):
        wt = np.zeros((self.scale, self.nx, self.ny))
        step_hole = 1
        im_in = np.copy(im)

        for i in range(self.scale - 1):
            if self.fast:
                kernel2d = b3spline_fast(step_hole)
                im_out = psg.convolve2d(im_in, kernel2d, boundary='symm', mode='same')
            else:
                im_out = b3splineTrans(im_in, step_hole)

            if self.gen2:
                if self.fast:
                    im_aux = psg.convolve2d(im_out, kernel2d, boundary='symm', mode='same')
                else:
                    im_aux = b3splineTrans(im_out, step_hole)
                wt[i] = im_in - im_aux
            else:
                wt[i] = im_in - im_out

            if self.normalization and computewt:
                wt[i] /= self.trTab[i]
            im_in = np.copy(im_out)
            step_hole *= 2

        wt[self.scale - 1] = np.copy(im_out)
        if self.normalization and computewt:
            wt[self.scale - 1] /= self.trTab[self.scale - 1]

        return wt

    def reconstruction(self, coef):
        if self.normalization:
            wt = np.copy(coef)
            for i in range(self.scale):
                wt[i] *= self.trTab[i]
        else:
            wt = coef

        if self.gen2:
            '''
            h' = h, g' = Dirac
            '''
            step_hole = pow(2, self.scale - 2)
            im = np.copy(wt[self.scale - 1])
            for k in range(self.scale - 2, -1, -1):
                if self.fast:
                    kernel2d = b3spline_fast(step_hole)
                    im_out = psg.convolve2d(im, kernel2d, boundary='symm', mode='same')
                else:
                    im_out = b3splineTrans(im, step_hole)
                im = im_out + wt[k]
                step_hole //= 2
        else:
            '''
            h' = Dirac, g' = Dirac
            '''
            #         im = np.sum(wt,axis=0)
            '''
            h' = h, g' = Dirac + h
            '''
            im = np.copy(wt[self.scale - 1])
            step_hole = pow(2, self.scale - 2)
            for k in range(self.scale - 2, -1, -1):
                if self.fast:
                    kernel2d = b3spline_fast(step_hole)
                    im = psg.convolve2d(im, kernel2d, boundary='symm', mode='same')
                    im_out = psg.convolve2d(wt[k], kernel2d, boundary='symm', mode='same')
                else:
                    im = b3splineTrans(im, step_hole)
                    im_out = b3splineTrans(wt[k], step_hole)
                im += wt[k] + im_out
                step_hole //= 2
        return im

    def adjoint(self, coef):
        if self.normalization:
            wt = np.copy(coef)
            for i in range(self.scale):
                wt[i] *= self.trTab[i]
        else:
            wt = coef

        im = np.copy(wt[self.scale - 1])
        step_hole = pow(2, self.scale - 2)
        for k in range(self.scale - 2, -1, -1):
            if self.fast:
                kernel2d = b3spline_fast(step_hole)
                im = psg.convolve2d(im, kernel2d, boundary='symm', mode='same')
                im_out = psg.convolve2d(wt[k], kernel2d, boundary='symm', mode='same')
                if self.gen2:
                    im_out2 = psg.convolve2d(im_out, kernel2d, boundary='symm', mode='same')
                    im += wt[k] - im_out2
                else:
                    im += wt[k] - im_out
            else:
                im = b3splineTrans(im, step_hole)
                im_out = b3splineTrans(wt[k], step_hole)
                if self.gen2:
                    im_out2 = b3splineTrans(im_out, step_hole)
                    im += wt[k] - im_out2
                else:
                    im += wt[k] - im_out
            step_hole //= 2
        return im


# def star2d(im, scale, fast=True, gen2=True, normalization=False):
#     (nx, ny) = np.shape(im)
#     nz = scale
#     # Normalized transfromation
#     head = 'star2d_gen2' if gen2 else 'star2d_gen1'
#     trans = 1 if gen2 else 2
#     if wavtl.trHead != head:
#         wavtl.trHead = head
#     if normalization:
#         wavtl.trTab = nsNorm(nx, ny, nz, trans)
#     wt = np.zeros((nz, nx, ny))
#     step_hole = 1
#     im_in = np.copy(im)
#
#     for i in range(nz - 1):
#         if fast:
#             kernel2d = b3spline_fast(step_hole)
#             im_out = psg.convolve2d(im_in, kernel2d, boundary='symm', mode='same')
#         else:
#             im_out = b3splineTrans(im_in, step_hole)
#
#         if gen2:
#             if fast:
#                 im_aux = psg.convolve2d(im_out, kernel2d, boundary='symm', mode='same')
#             else:
#                 im_aux = b3splineTrans(im_out, step_hole)
#             wt[i, :, :] = im_in - im_aux
#         else:
#             wt[i, :, :] = im_in - im_out
#
#         if normalization:
#             wt[i, :, :] /= wavtl.trTab[i]
#         im_in = np.copy(im_out)
#         step_hole *= 2
#
#     wt[nz - 1, :, :] = np.copy(im_out)
#     if normalization:
#         wt[nz - 1, :, :] /= wavtl.trTab[nz - 1]
#
#     return wt
#
#
# def istar2d(wtOri, fast=True, gen2=True, normalization=False):
#     (nz, nx, ny) = np.shape(wtOri)
#     wt = np.copy(wtOri)
#     # Unnormalization step
#     head = 'star2d_gen2' if gen2 else 'star2d_gen1'
#     trans = 1 if gen2 else 2
#     if normalization:
#         if wavtl.trHead != head:
#             wavtl.trHead = head
#             wavtl.trTab = nsNorm(nx, ny, nz, trans)
#         for i in range(nz):
#             wt[i, :, :] *= wavtl.trTab[i]
#
#     if gen2:
#         '''
#         h' = h, g' = Dirac
#         '''
#         step_hole = pow(2, nz - 2)
#         imRec = np.copy(wt[nz - 1, :, :])
#         for k in range(nz - 2, -1, -1):
#             if fast:
#                 kernel2d = b3spline_fast(step_hole)
#                 im_out = psg.convolve2d(imRec, kernel2d, boundary='symm', mode='same')
#             else:
#                 im_out = b3splineTrans(imRec, step_hole)
#             imRec = im_out + wt[k, :, :]
#             step_hole /= 2
#     else:
#         '''
#         h' = Dirac, g' = Dirac
#         '''
#         #         imRec = np.sum(wt,axis=0)
#         '''
#         h' = h, g' = Dirac + h
#         '''
#         imRec = np.copy(wt[nz - 1, :, :])
#         step_hole = pow(2, nz - 2)
#         for k in range(nz - 2, -1, -1):
#             if fast:
#                 kernel2d = b3spline_fast(step_hole)
#                 imRec = psg.convolve2d(imRec, kernel2d, boundary='symm', mode='same')
#                 im_out = psg.convolve2d(wt[k, :, :], kernel2d, boundary='symm', mode='same')
#             else:
#                 imRec = b3splineTrans(imRec, step_hole)
#                 im_out = b3splineTrans(wt[k, :, :], step_hole)
#             imRec += wt[k, :, :] + im_out
#             step_hole /= 2
#     return imRec
#
#
# def adstar2d(wtOri, fast=True, gen2=True, normalization=False):
#     (nz, nx, ny) = np.shape(wtOri)
#     wt = np.copy(wtOri)
#     # Unnormalization step
#     # !Attention: wt is not the original wt after unnormalization
#     head = 'star2d_gen2' if gen2 else 'star2d_gen1'
#     trans = 1 if gen2 else 2
#     if normalization:
#         if wavtl.trHead != head:
#             wavtl.trHead = head
#             wavtl.trTab = nsNorm(nx, ny, nz, trans)
#         for i in range(nz):
#             wt[i, :, :] *= wavtl.trTab[i]
#
#     imRec = np.copy(wt[nz - 1, :, :])
#     step_hole = pow(2, nz - 2)
#     for k in range(nz - 2, -1, -1):
#         if fast:
#             kernel2d = b3spline_fast(step_hole)
#             imRec = psg.convolve2d(imRec, kernel2d, boundary='symm', mode='same')
#             im_out = psg.convolve2d(wt[k, :, :], kernel2d, boundary='symm', mode='same')
#             if gen2:
#                 im_out2 = psg.convolve2d(im_out, kernel2d, boundary='symm', mode='same')
#                 imRec += wt[k, :, :] - im_out2
#             else:
#                 imRec += wt[k, :, :] - im_out
#         else:
#             imRec = b3splineTrans(imRec, step_hole)
#             im_out = b3splineTrans(wt[k, :, :], step_hole)
#             if gen2:
#                 im_out2 = b3splineTrans(im_out, step_hole)
#                 imRec += wt[k, :, :] - im_out2
#             else:
#                 imRec += wt[k, :, :] - im_out
#         step_hole /= 2
#     return imRec
#
#
# def nsNorm(nx, ny, nz, trans=1):
#     im = np.zeros((nx, ny))
#     im[nx / 2, ny / 2] = 1
#     if trans == 1:  # starlet transform 2nd generation
#         wt = star2d(im, nz, fast=True, gen2=True, normalization=False)
#         tmp = wt ** 2
#     elif trans == 2:  # starlet transform 1st generation
#         wt = star2d(im, nz, fast=True, gen2=False, normalization=False)
#         tmp = wt ** 2
#     tabNs = np.sqrt(np.sum(np.sum(tmp, 1), 1))
#     return tabNs


def pstar2d(im, nz, Niter, fast=True, gen2=True, hard=False, den=False):
    """
    Iteratively apply positivity constraint on (2nd generation) starlet coefficients
    :param im: input image
    :param nz: number of scales
    :param Niter: number of iterations
    :param fast: fast computation of starlet, default is True
    :param gen2: second generation of starlet, default is True
    :param hard: hard-thresholding, default is False, and soft-shresholding is used
    :param den: the thresholding level is decided by noise level, otherwise decided by the peak coefficient.
    default is False
    :return: positive starlet coefficients
    """
    from pyDecGMCA import mathTools
    (nx, ny) = np.shape(im)
    rsd = np.copy(im)
    star2d = Starlet2D(nx, ny, nz, fast, gen2, normalization=True)
    wt = star2d.decomposition(rsd)
    mwt = wt.max()
    wt = np.zeros((nz, nx, ny))

    for it in range(Niter):
        tmp = star2d.decomposition(rsd)
        wt += tmp
        if den:
            noise = mathTools.mad(wt[0])
            print("sigma of noise: {}".format(noise))
            if hard:
                mathTools.hardTh(wt, 3 * noise)
            else:
                mathTools.softTh(wt, 3 * noise)
        else:
            ld = mwt * (1. - (it + 1.) / Niter)
            if ld < 0:
                ld = 0
            print('lamda=' + str(ld))
            if hard:
                mathTools.hardTh(wt, ld)
            else:
                mathTools.softTh(wt, ld)
        wt[wt < 0] = 0
        rec = star2d.reconstruction(wt)
        print("Check whether the reconstructed image is non-negative: {}".format((rec >= 0).all()))
        rsd = im - rec
        print("Residual level: {}".format((np.abs(rsd)).sum()))
    return wt


#########################
# (Bi-)orthogonal wavelet
#########################
def dct2(im, type=2, n=None, norm=None):
    coef1 = scifft.dct(im, type=type, n=n, axis=0, norm=norm)
    coef = scifft.dct(coef1, type=type, n=n, axis=1, norm=norm)
    return coef


def wavOrth2d(im, nz, wname='haar'):
    sx, sy = np.shape(im)
    scale = nz
    if scale > np.ceil(np.log2(sx)) + 1 or scale > np.ceil(np.log2(sy)) + 1:
        print("Too many decomposition scales! The decomposition scale will be set to default value: 1!")
        scale = 1
    elif scale < 1:
        print(
            "Decomposition scales should not be smaller than 1! The decomposition scale will be set to default value: 1!")
        scale = 1

    band = np.zeros((scale + 1, len(np.shape(im))))
    band[-1] = np.shape(im)

    if wname == 'haar' or wname == 'db1' or wname == 'db2' or wname == 'db3' or wname == 'db4' or wname == 'db5':
        wtype = 1
    else:
        wtype = 2

    (h0, g0) = wav1d.wavFilters(wname, wtype, 'd')
    lf = np.size(h0)
    wt = np.array([])
    start = np.array([1, 1])

    for sc in range(scale - 1):
        end = np.array([sx + lf - 1, sy + lf - 1])
        lenExt = lf - 1
        imExtCol = np.lib.pad(im, ((0, 0), (lenExt, lenExt)), 'symmetric')  # Extension of columns
        tmp = psg.convolve2d(imExtCol, h0[np.newaxis], 'valid')
        im = convdown(tmp, h0[np.newaxis], lenExt, start, end)  # Approximation
        hor = convdown(tmp, g0[np.newaxis], lenExt, start, end)  # Horizontal details
        tmp = psg.convolve2d(imExtCol, g0[np.newaxis], 'valid')
        vet = convdown(tmp, h0[np.newaxis], lenExt, start, end)  # Vertical details
        dig = convdown(tmp, g0[np.newaxis], lenExt, start, end)  # Diagonal details
        wt = np.hstack([hor.flatten(), vet.flatten(), dig.flatten(), wt])
        sx, sy = np.shape(im)
        band[-2 - sc] = np.array([sx, sy])
    wt = np.hstack([im.flatten(), wt])
    band[0] = np.shape(im)
    return wt, band


def convdown(x, F, lenExt, start, end):
    im = np.copy(x[:, start[1]:end[1]:2])  # Downsampling
    y = np.lib.pad(im, ((lenExt, lenExt), (0, 0)), 'symmetric')  # Extension of rows
    y = psg.convolve2d(y.T, F, 'valid')
    y = y.T
    y = y[start[0]:end[0]:2, :]
    return y


def iwavOrth2d(wt, band, wname='haar'):
    scale = np.size(band, axis=0) - 1

    if wname == 'haar' or wname == 'db1' or wname == 'db2' or wname == 'db3' or wname == 'db4' or wname == 'db5':
        wtype = 1
    else:
        wtype = 2

    (h1, g1) = wav1d.wavFilters(wname, wtype, 'r')
    h1 = h1[np.newaxis]
    g1 = g1[np.newaxis]

    sx = band[0, 0]
    sy = band[0, 1]

    im = np.reshape(wt[:sx * sy], (sx, sy))

    for sc in range(scale - 1, 0, -1):
        h, v, d = detcoef2('a', wt, band, sc)
        im = upsconv2(im, h1, h1, band[scale - sc + 1]) + upsconv2(h, g1, h1, band[scale - sc + 1]) + \
             upsconv2(v, h1, g1, band[scale - sc + 1]) + upsconv2(d, g1, g1, band[scale - sc + 1])
    return im


def upsconv2(x, F, G, s):
    sx, sy = np.shape(x)
    y = np.zeros((2 * sx - 1, sy))
    y[::2] = np.copy(x)
    y = psg.convolve2d(y.T, F, 'full')
    y = y.T
    ytmp = np.copy(y)
    sx = np.size(ytmp, axis=0)
    y = np.zeros((sx, 2 * sy - 1))
    y[:, ::2] = np.copy(ytmp)
    y = psg.convolve2d(y, G, 'full')
    first = (np.floor(float(np.size(y, axis=0) - s[0]) / 2.), np.floor(float(np.size(y, axis=1) - s[1]) / 2.))
    last = (np.size(y, axis=0) - np.ceil(float(np.size(y, axis=0) - s[0]) / 2.),
            np.size(y, axis=1) - np.ceil(float(np.size(y, axis=1) - s[1]) / 2.))
    y = y[first[0]:last[0], first[1]:last[1]]
    return y


def detcoef2(o, wt, band, sc):
    nmax = np.size(band, axis=0) - 1
    if sc > nmax or sc < 0:
        print("The scale is not valid and will be set to default value: 1!")
        sc = 1

    k = np.size(band, axis=0) - sc - 1
    first = band[0, 0] * band[0, 1] + 3 * np.sum(band[1:k, 0] * band[1:k, 1])
    add = band[k, 0] * band[k, 1]
    first = {
        'h': int(first),
        'v': int(first + add),
        'd': int(first + 2 * add),
        'a': int(first),
        'c': int(first),
    }

    last = {
        'h': int(first['h'] + add),
        'v': int(first['v'] + add),
        'd': int(first['d'] + add),
        'a': (int(first['a'] + add), int(first['a'] + 2 * add), int(first['a'] + 3 * add)),
        'c': int(first['c'] + 3 * add),
    }

    if o == 'h' or o == 'v' or o == 'd':
        return np.reshape(wt[first[o]:last[o]], tuple(band[k]))
    elif o == 'a':
        h = np.reshape(wt[first['a']:last['a'][0]], tuple(band[k]))
        v = np.reshape(wt[(last['a'][0]):last['a'][1]], tuple(band[k]))
        d = np.reshape(wt[(last['a'][1]):last['a'][2]], tuple(band[k]))
        return h, v, d
    elif o == 'c':
        return wt[first['c']:last['c']]
