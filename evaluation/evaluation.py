'''
Created on Nov 25, 2015

@author: mjiang
'''

import numpy as np
from scipy import linalg
from munkres import munkres

def evaluation(result, reference, verbose=0):
    r"""
    Evaluate BSS results using criteria from Vincent et al.
    This function reorders the sources and mixtures so as to match
    the reference factorization.
    
    Inputs
    ------
    - result: dict
        output of a BSS algorithm, with field "factorization".
    - reference: dict
        dictionary containing the reference factorization as well as the noise and
        other relevant information if need be.
    - verbose (default: 0): bool
        Display important parameters
    
    Outputs
    ------
    criteria: dict
        value of each of the criteria.
    decomposition: dict
        decomposition of the estimated sources under target, interference,
        noise and artifacts.
    """
    # get column signals
    Ar = reference[0].copy()
    Sr = reference[1].T.copy()
    noise = reference[2].T.copy()
    Ae = result[0]
    Se = result[1].T
    r = Sr.shape[1]
    # set nan values to 0
    Ar[np.isnan(Ar)] = 0
    Sr[np.isnan(Sr)] = 0
    Ae[np.isnan(Ae)] = 0
    Se[np.isnan(Se)] = 0
    # precomputation
    SDR_A = compute_sdr_matrix(Ar, Ae)
    # order computation
    costMatrix = -SDR_A
    hungarian = munkres.Munkres()
    ind_list = hungarian.compute(costMatrix.tolist())
    indices = np.zeros(r, dtype=int)
    for k in range(0, r):
        indices[k] = ind_list[k][1]
    # reorder the factorization
    Ae = Ae[:, indices]
    Se = Se[:, indices]
    # get reordered results
#     Ae = result[0].copy()
#     Ae[np.isnan(Ae)] = 0
#     Se = result[1].T.copy()
#     Se[np.isnan(Se)] = 0
#     result_ordered = (Ae, Se)
    # compute criteria
    delta=abs(abs(linalg.inv(Ae.T.dot(Ae)).dot(Ae.T).dot(Ar)) - np.eye(r)).sum() / (r*r)
    criteria = {}
    # on S
    output = decomposition_criteria(Se, Sr, noise)
    decomposition = output[1]
    criteria['SDR_S'] = output[0]['SDR']
    criteria['SIR_S'] = output[0]['SIR']
    criteria['SNR_S'] = output[0]['SNR']
    criteria['SAR_S'] = output[0]['SAR']
    # on A
#     output = decomposition_criteria(Ae, Ar, reference['noise'])
#     criteria['SDR_A'] = output[0]['SDR']
#     criteria['SIR_A'] = output[0]['SIR']
#     criteria['SNR_A'] = output[0]['SNR']
#     criteria['SAR_A'] = output[0]['SAR']
#     if verbose != 0:
#         print("Results of the reconstruction:")
#         print("Decomposition criteria on S:")
#         print("    - Mean SDR: " + str(criteria['SDR_S']) + ".")
#         print("    - Mean SIR: " + str(criteria['SIR_S']) + ".")
#         print("    - Mean SNR: " + str(criteria['SNR_S']) + ".")
#         print("    - Mean SAR: " + str(criteria['SAR_S']) + ".")
#         print("Decomposition criteria on A:")
#         print("    - Mean SDR: " + str(criteria['SDR_A']) + ".")
#         print("    - Mean SIR: " + str(criteria['SIR_A']) + ".")
#         print("    - Mean SNR: " + str(criteria['SNR_A']) + ".")
#         print("    - Mean SAR: " + str(criteria['SAR_A']) + ".")
    return (criteria, decomposition, delta,Ae,Se.T)


def compute_sdr_matrix(X, Y):
    r"""
    Computes the SDR of each couple reference/estimate sources.
    
    Inputs
    ------
    X: numpy array
        reference of column signals.
    Y: numpy array
        estimate of column signals.
    
    Output
    ------
    MSDR: numpy array
        numpy array such that MSDR(i,j) is the SNR between the i-th row of
        X with the j-th column of Y.
    """
    # normalize the reference
    X = X / dim_norm(X, 0)
    # get shape and initialize
    n_x = X.shape[1]
    n_y = Y.shape[1]
    L = X.shape[0]
    MSDR = np.zeros([n_x, n_y])
    # computation
    for n in range(0, n_x):
        targets = X[:, n].reshape([L, 1]) * (X[:, n].T.dot(Y))
        diff = Y - targets
        norm_diff_2 = np.maximum(np.sum(diff * diff, 0), np.spacing(1))
        norm_targets_2 = np.maximum(
            np.sum(targets * targets, 0), np.spacing(1))
        MSDR[n, :] = -10 * np.log10(norm_diff_2 / norm_targets_2)
    return MSDR

def dim_norm(data, dim=0, norm_type=2):
    r"""
    Computes the norm of X along a given dimension.
    
    Inputs
    ------
    - data: numpy array
        Data array to be processed.
    - dim (default: 0): int
        Dimension on which to process the data.
    - norm_type (default: 2): int
        Norm type to be used for the computation (norms 1 or 2).
    """
    if norm_type == 2:
        norms = np.sqrt(np.sum(data * data, axis=dim))
    else:
        if norm_type == 1:
            norms = np.sum(np.abs(data), axis=dim)
        else:
            raise Exception("Norm type can be either \"1\" or \"2\" (not \"" +
                            str(norm_type) + "\).")
    shape = np.array(data.shape)
    shape[dim] = 1
    return norms.reshape(shape)


def decomposition_criteria(Se, Sr, noise):
    r"""
    Computes the SDR of each couple reference/estimate sources.
    
    Inputs
    ------
    Se: numpy array
        estimate of column signals.
    Sr: numpy array
        reference of column signals.
    noise: numpy array
        noise matrix cotaminating the data.
    
    Outputs
    ------
    criteria: dict
        value of each of the criteria.
    decomposition: dict
        decomposition of the estimated sources under target, interference,
        noise and artifacts.
    """
    # compute projections
    Sr = Sr / dim_norm(Sr, 0)
    pS = Sr.dot(np.linalg.lstsq(Sr, Se)[0])
    SN = np.hstack([Sr, noise])
    #pSN = SN.dot(np.linalg.lstsq(SN, Se)[0])  # this crashes on MAC
    pSN = SN.dot(linalg.lstsq(SN, Se)[0])
    eps = np.spacing(1)
    # compute decompositions
    decomposition = {}
    # targets
    decomposition['target'] = np.sum(Se * Sr, 0) * Sr   # Sr is normalized
    # interferences
    decomposition['interferences'] = pS - decomposition['target']
    # noise
    decomposition['noise'] = pSN - pS
    # artifacts
    decomposition['artifacts'] = Se - pSN
    # compute criteria
    criteria = {}
    # SDR: source to distortion ratio
    num = decomposition['target']
    den = decomposition['interferences'] +\
        (decomposition['noise'] + decomposition['artifacts'])
    norm_num_2 = np.sum(num * num, 0)
    norm_den_2 = np.sum(den * den, 0)
    criteria['SDR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    criteria['SDR median'] = np.median(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SIR: source to interferences ratio
    num = decomposition['target']
    den = decomposition['interferences']
    norm_num_2 = sum(num * num, 0)
    norm_den_2 = sum(den * den, 0)
    criteria['SIR'] = np.mean(10 * np.log10(
        np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    # SNR: source to noise ratio
    if np.max(np.abs(noise)) > 0:  # only if there is noise
        num = decomposition['target'] + decomposition['interferences']
        den = decomposition['noise']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SNR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SNR'] = np.inf
    # SAR: sources to artifacts ratio
    if (noise.shape[1] + Sr.shape[1] < Sr.shape[0]):
        # if noise + sources form a basis, there is no "artifacts"
        num = decomposition['target'] +\
            (decomposition['interferences'] + decomposition['noise'])
        den = decomposition['artifacts']
        norm_num_2 = sum(num * num, 0)
        norm_den_2 = sum(den * den, 0)
        criteria['SAR'] = np.mean(10 * np.log10(
            np.maximum(norm_num_2, eps) / np.maximum(norm_den_2, eps)))
    else:
        criteria['SAR'] = np.inf
    return (criteria, decomposition)
