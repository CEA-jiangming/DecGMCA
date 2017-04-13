'''
Created on Nov 27, 2015

@author: mjiang
'''

import numpy as np

trHead = ''                         # Transform head
trTab = []                          # Normalization table

def mad(alpha,align=1):
    '''
    @param alpha: input vector or matrix
    
    @param align: data structure
    align=1(default): the first dimension of the data denotes the index of object(source,etc.)
    align=2: mad will treat on the entire data
    
    @return: standard deviation of the noise
    '''
    dim = np.size(np.shape(alpha))
    if align == 1: 
        if dim == 1:
            alpha = alpha[np.newaxis,:]
        nz = np.size(alpha,axis=0)
        sigma = np.zeros(nz)
        for i in np.arange(nz):
            sigma[i] = np.median(np.abs(alpha[i] - np.median(alpha[i]))) / 0.6745   
        alpha = np.squeeze(alpha)
    elif align == 2:
        if dim == 2:
            sigma = np.median(np.abs(alpha - np.median(alpha))) / 0.6745          
    return sigma

def softTh(alpha,thTab,weights=None,reweighted=False):
    '''
    @param alpha: wavelet coefficients. 
    Att:alpha will be modified after processing
    
    @param thTab: threshold level, the same size as the size of the first dimension of alpha
    
    @param weights: weights matrix (for l_1 reweighted scheme)
    
    @param reweighted: l_1 reweighted scheme
    
    '''
    nz = np.size(thTab)
#     print weights.shape
#     print thTab.shape
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
            (alpha[i])[alpha[i]>0] -= thTab[i]
            (alpha[i])[alpha[i]<0] += thTab[i]
        else:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0
            (alpha[i])[alpha[i]>0] -= (thTab[i]*weights[i])[alpha[i]>0]
            (alpha[i])[alpha[i]<0] += (thTab[i]*weights[i])[alpha[i]<0]

def hardTh(alpha,thTab,weights=None,reweighted=False):
    '''
    @param alpha: wavelet coefficients. 
    Att:alpha will be modified after processing
    
    @param thTab: threshold level, the same size as the size of the first dimension of alpha
    
    @param weights: weights matrix (for l_1 reweighted scheme), default is None
    
    @param reweighted: l_1 reweighted scheme, default is False
    
    '''
    nz = np.size(thTab)
#     print weights.shape
#     print thTab.shape
    if np.ndim(alpha) == 1:
        alpha = alpha[np.newaxis,:]
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
        else:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0 
    alpha = alpha.squeeze()   