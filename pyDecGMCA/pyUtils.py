"""
Created on Oct 26, 2015

@author: mjiang
"""

import numpy as np
import numpy.linalg as LA
import sys


def update_A(V, S, M=1, mask=True, deconv=False, epsilon=0., strat=2):
    """
    Estimate mixing matrix using least square

    @param V: Visibilities, size: bands*pixels

    @param S: Matrix of sources, size: sources*pixels

    @param M: Matrix of mask, size: bands*pixels

    @param mask: Whether in the case of masking, default True

    @param deconv: Whether in the case of blurring, default False

    @return: Updated A, size: bands*sources
    """
    (Bd, P) = np.shape(V)
    (N, P) = np.shape(S)
    A = np.zeros((Bd, N)) + np.zeros((Bd, N)) * 1j
    # To avoid element by element looping, the procedure is written differently for different cases
    if strat == 2:
        normSS = LA.norm(S@S.T, ord=-2)
    if mask and not deconv:
        for nu in range(Bd):
            ind = tuple(np.where(M[nu] == 1)[0])
            numr = V[nu, ind].dot(S[:, ind].conj().transpose())
            denom = np.dot(S[:, ind], S[:, ind].conj().transpose())
            #
            # if strat == 1:
            #     rho = LA.norm(denom, ord=2)
            #     denom = denom + epsilon * max(rho, 1e-4) * np.eye(N)
            # elif strat == 2:
            #     rho = LA.norm(denom, ord=-2)
            #     denom = denom + np.maximum(0,  rho) * np.eye(N)
            # try:
            #     Ua, Sa, Va = LA.svd(denom)
            # except LA.LinAlgError:
            #     print('SVD did not converge, abort')
            #     return 1
            # denom = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
            # A[nu, :] = np.dot(numr, denom)

            if LA.cond(denom) < 1 / sys.float_info.epsilon:
                denom = LA.inv(denom)
            else:
                rho = LA.norm(denom, 2)
                denom = LA.inv(denom + max(1e-4 * rho, 1e-4) * np.eye(N))
            A[nu, :] = np.dot(numr, denom)
    elif deconv:
        for nu in range(Bd):
            numr = np.dot(M[nu, :] * V[nu, :], S.conj().transpose())
            denom = np.dot(M[nu, :] * S, (M[nu, :] * S).conj().transpose())
            if LA.cond(denom) < 1 / sys.float_info.epsilon:
                denom = LA.inv(denom)
            else:
                rho = LA.norm(denom, 2)
                denom = LA.inv(denom + max(1e-4 * rho, 1e-4) * np.eye(N))
            A[nu, :] = np.dot(numr, denom)

    else:
        numr = np.dot(V, S.conj().transpose())
        denom = np.dot(S, S.conj().transpose())
        if LA.cond(denom) < 1 / sys.float_info.epsilon:
            denom = LA.inv(denom)
        else:
            rho = LA.norm(denom, 2)
            denom = LA.inv(denom + max(1e-4 * rho, 1e-4) * np.eye(N))
        A = np.dot(numr, denom)
    return A


def update_S(V, A, M=1, mask=True, deconv=False, epsilon=0., strat=2):
    """
    Estimate sources using least square

    @param V: Visibilities, size: bands*pixels

    @param A: Matrix of mixture, size: bands*sources

    @param M: Matrix of mask, size: bands*pixels

    @param mask: Whether in the case of masking, default True

    @param deconv: Whether in the case of blurring, default False

    @param epsilon: Parameter of regularization

    @return: Updated S, size: sources*pixels
    """
    (Bd, P) = np.shape(V)
    (Bd, N) = np.shape(A)
    S = np.zeros((N, P)) + np.zeros((N, P)) * 1j
    # To avoid element by element looping, the procedure is written differently for different cases
    if strat == 2:
        normAA = LA.norm(A.T@A, ord=-2)
    if mask and not deconv:
        for k in range(P):
            ind = tuple(np.where(M[:, k] == 1)[0])
            numr = np.dot(A[ind, :].conj().transpose(), V[ind, k])
            denom = np.dot(A[ind, :].conj().transpose(), A[ind, :])
            if strat == 1:
                rho = LA.norm(denom, ord=2)
                denom = denom + epsilon * max(rho, 1e-4) * np.eye(N)
            elif strat == 2:
                rho = LA.norm(denom, ord=-2)
                denom = denom + np.maximum(0, epsilon - rho / normAA) * np.eye(N)
            try:
                Ua, Sa, Va = LA.svd(denom)
            except LA.LinAlgError:
                print('SVD did not converge, abort')
                return 1
            denom = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
            S[:, k] = np.dot(denom, numr)
    elif deconv:
        for k in range(P):
            numr = np.dot(A.conj().transpose(), M[:, k] * V[:, k])
            denom = np.dot((M[:, k][:, np.newaxis] * A).conj().transpose(), M[:, k][:, np.newaxis] * A)
            rho = LA.norm(denom, 2)
            if strat == 1:
                denom = denom + epsilon * max(rho, 1e-4) * np.eye(N)
            elif strat == 2:
                rho = LA.norm(denom, ord=-2)
                denom = denom + np.maximum(0, epsilon - rho / normAA) * np.eye(N)
            try:
                Ua, Sa, Va = LA.svd(denom)
            except LA.LinAlgError:
                print('SVD did not converge, abort')
                return 1
            denom = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
            S[:, k] = np.dot(denom, numr)
    else:
        denom = np.dot(A.conj().transpose(), A)
        if LA.cond(denom) < 1 / sys.float_info.epsilon:
            denom = denom
        else:
            rho = LA.norm(denom, 2)
            denom = denom + max(1e-10 * rho, 1e-10) * np.eye(N)
        try:
            Ua, Sa, Va = LA.svd(denom)
        except LA.LinAlgError:
            print('SVD did not converge, abort')
            return 1
        denom = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
        numr = np.dot(A.conj().transpose(), V)
        S = np.squeeze(np.dot(denom, numr))
    return S
