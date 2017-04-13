'''
Created on Mar 30, 2015

@author: mjiang
'''
import numpy as np
import scipy.signal as psg
import waveTools as wavtl
import arraypad

def scaleFilter(wname,wtype):
    if wtype == 1:
        if wname =='haar' or wname == 'db1':
            F = np.array([0.5,0.5])
            
        elif wname == 'db2':
            F = np.array([0.34150635094622,0.59150635094587,0.15849364905378,-0.09150635094587])
            
        elif wname == 'db3':
            F = np.array([0.23523360389270,0.57055845791731,0.32518250026371,-0.09546720778426,\
                          -0.06041610415535,0.02490874986589])
            
        elif wname == 'db4':
            F = np.array([0.16290171402562,0.50547285754565,0.44610006912319,-0.01978751311791,\
                          -0.13225358368437,0.02180815023739,0.02325180053556,-0.00749349466513])
            
        elif wname == 'db5':
            F = np.array([0.11320949129173,0.42697177135271,0.51216347213016,0.09788348067375,\
                           -0.17132835769133,-0.02280056594205,0.05485132932108,-0.00441340005433,\
                           -0.00889593505093,0.00235871396920])
        return F
    elif wtype == 2:
        if wname == '9/7' or wname == 'bior4.4':
            Df = np.array([0.0267487574110000,-0.0168641184430000,-0.0782232665290000,0.266864118443000,\
                           0.602949018236000,0.266864118443000,-0.0782232665290000,-0.0168641184430000,\
                           0.0267487574110000])
            Rf = np.array([-0.0456358815570000,-0.0287717631140000,0.295635881557000,0.557543526229000,\
                           0.295635881557000,-0.0287717631140000,-0.0456358815570000])
        elif wname == 'bior1.1':
            Df = np.array([0.5])
            Rf = np.array([0.5])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior1.3':
            Df = np.array([-1./16,1./16,1./2])
            Rf = np.array([0.5])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior1.5':
            Df = np.array([3./256,-3./256,-11./128,11./128,1./2])
            Rf = np.array([0.5])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior2.2':
            Df = np.array([-1./8,1./4,3./4,1./4,-1./8])
            Rf = np.array([1./4,1./2,1./4])
        elif wname == 'bior2.4':
            Df = np.array([3./128,-3./64,-1./8,19./64,45./64,19./64,-1./8,-3./64,3./128])
            Rf = np.array([1./4,1./2,1./4])
        elif wname == 'bior2.6':
            Df = np.array([-5./1024,5./512,17./512,-39./512,-123./1024,81./256,175./256,81./256,-123./1024,-39./512,17./512,5./512,-5./1024])
            Rf = np.array([1./4,1./2,1./4])
        elif wname == 'bior2.8':
            Df = 1./(2**15)*np.array([35,-70,-300,670,1228,-3126,-3796,10718,22050,10718,-3796,-3126,1228,670,-300,-70,35])
            Rf = np.array([1./4,1./2,1./4])
        elif wname == 'bior3.1':
            Df = 1./4*np.array([-1,3])
            Rf = 1./8*np.array([1,3])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior3.3':
            Df = 1./64*np.array([3,-9,-7,45])
            Rf = 1./8*np.array([1,3])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior3.5':
            Df = 1./512*np.array([-5,15,19,-97,-26,350])
            Rf = 1./8*np.array([1,3])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior3.7':
            Df = 1./(2**14)*np.array([35,-105,-195,865,363,-3489,-307,11025])
            Rf = 1./8*np.array([1,3])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        elif wname == 'bior3.9':
            Df = 1./(2**17)*np.array([-63,189,469,-1911,-1308,9188,1140,-29676,190,87318])
            Rf = 1./8*np.array([1,3])
            Df = np.hstack((Df,Df[::-1]))
            Rf = np.hstack((Rf,Rf[::-1]))
        return (Rf,Df)
 
def orthWavFilter(F):       
    p = 1
#     h1 = np.copy(F)
    Lo_R = np.sqrt(2)*F/np.sum(F)
#     Lo_R = F/np.sqrt(np.sum(F**2))
    Hi_R = np.copy(Lo_R[::-1])
    first = 2-p%2
#     print first 
#     print tmp
    Hi_R[first::2] = -Hi_R[first::2]
    Hi_D=np.copy(Hi_R[::-1])
    Lo_D=np.copy(Lo_R[::-1])
    return (Lo_D,Hi_D,Lo_R,Hi_R)


def biorWavFilter(Rf,Df):
    lr = len(Rf)
    ld = len(Df)
    lmax = max(lr,ld)
    if lmax%2:
        lmax += 1
    Rf = np.hstack([np.zeros(int((lmax-lr)/2)),Rf,np.zeros(int(np.ceil((lmax-lr)/2.)))])
    Df = np.hstack([np.zeros(int((lmax-ld)/2)),Df,np.zeros(int(np.ceil((lmax-ld)/2.)))])
    
    [Lo_D1,Hi_D1,Lo_R1,Hi_R1] = orthWavFilter(Df)
    [Lo_D2,Hi_D2,Lo_R2,Hi_R2] = orthWavFilter(Rf)
    
    return (Lo_D1,Hi_D1,Lo_R1,Hi_R1,Lo_D2,Hi_D2,Lo_R2,Hi_R2)

def wavFilters(wname,wtype,mode):
    if wtype == 1:
        F = scaleFilter(wname,1)
        (Lo_D,Hi_D,Lo_R,Hi_R) = orthWavFilter(F)
    elif wtype == 2:
        (Rf,Df) = scaleFilter(wname,2)
        [Lo_D,Hi_D1,Lo_R1,Hi_R,Lo_D2,Hi_D,Lo_R,Hi_R2] = biorWavFilter(Rf,Df)
    if mode =='d':
        return (Lo_D,Hi_D)
    elif mode =='r':
        return (Lo_R,Hi_R)
    elif mode == 'l':
        return (Lo_D,Lo_R)
    elif mode == 'h':
        return (Hi_D,Hi_R)
        
def wavOrth1d(x,nz,wname='haar',wtype=1):
    N = np.size(x)
    scale = nz
    if scale > np.ceil(np.log2(N))+1:
        print "Too many decomposition scales! The decomposition scale will be set to default value: 1!"
        scale = 1
    if scale < 1:
        print "Decomposition scales should not be smaller than 1! The decomposition scale will be set to default value: 1!"
        scale = 1        
     
    band = np.zeros(scale+1)
    band[-1] = len(x)   
      
    if wname =='haar' or wname == 'db1' or wname == 'db2' or wname == 'db3' or wname == 'db4' or wname == 'db5':
        wtype = 1
    else:
        wtype = 2
               
    (h0,g0) = wavFilters(wname,wtype,'d')
    lf = np.size(h0)
    wt = np.array([])
#     end = N
    start = 1
    for sc in np.arange(scale-1):
#             start = np.ceil(float(end)/2)
        lsig = np.size(x)
        end = lsig + lf - 1
        lenExt = lf - 1
        xExt = np.lib.pad(x, (lenExt,lenExt), 'symmetric')
        app = np.convolve(xExt,h0,'valid')
        x = np.copy(app[start:end:2])
        detail = np.convolve(xExt,g0,'valid')
        wt = np.hstack([detail[start:end:2],wt])     
        band[-2-sc] = len(detail[start:end:2])
    wt = np.hstack([x,wt]) 
    band[0] = len(x) 
    return (wt,band)
        
def iwavOrth1d(wt,band,wname='haar',wtype=1):
    if wname =='haar' or wname == 'db1' or wname == 'db2' or wname == 'db3' or wname == 'db4' or wname == 'db5':
        wtype = 1
    else:
        wtype = 2
        
    (h1,g1) = wavFilters(wname,wtype,'r')
    sig = np.copy(wt[:band[0]])
    start = band[0]
#         lf = np.size(h1)
    for sc in np.arange(np.size(band)-2):
        last = start+band[sc+1]
        detail = np.copy(wt[start:last])
        lsig = 2*np.size(sig)
#             s = lsig - lf + 2
        s = band[sc+2]
        appInt = np.zeros(lsig-1)
        appInt[::2] = np.copy(sig)
        appInt = np.convolve(appInt,h1,'full')
        first = int(np.floor(float(np.size(appInt) - s)/2.))
        last = np.size(appInt) - int(np.ceil(float(np.size(appInt) - s)/2.))
        appInt = appInt[first:last]            
        detailInt = np.zeros(lsig-1)
        detailInt[::2] = np.copy(detail)
        detailInt = np.convolve(detailInt,g1,'full')
        detailInt = detailInt[first:last]           
        sig = appInt + detailInt 
        start = last          
    return sig

#######################################################
############### Starlet 1d ############################
#######################################################
def test_ind(ind,N):
    res = ind
    if ind < 0 : 
        res = -ind
        if res >= N: 
            res = 2*N - 2 - ind
    if ind >= N : 
        res = 2*N - 2 - ind
        if res < 0:
            res = -ind
    return res
    

def b3splineTrans(sig_in,step):
    n = np.size(sig_in)
    sig_out = np.zeros(n)
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    
    for i in np.arange(n):
        il = test_ind(i-step,n)
        ir = test_ind(i+step,n)
        il2 = test_ind(i-2*step,n)
        ir2 = test_ind(i+2*step,n)
        sig_out[i] = c3 * sig_in[i] + c2 * (sig_in[il] + sig_in[ir]) + c1 * (sig_in[il2] + sig_in[ir2])
    
    return sig_out

def b3spline_fast(step_hole):
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = 4*step_hole+1
    kernel1d = np.zeros(length)
    kernel1d[0] = c1
    kernel1d[-1] = c1
    kernel1d[step_hole] = c2
    kernel1d[-1-step_hole] = c2
    kernel1d[2*step_hole] = c3
    return kernel1d

def star1d(sig,scale,fast = True,gen2=True,normalization=False):
    n = np.size(sig)
    ns = scale
    # Normalized transfromation
    head = 'star1d_gen2' if gen2 else 'star1d_gen1'
    if wavtl.trHead != head:
        wavtl.trHead = head
    if normalization:
        wavtl.trTab = nsNorm(n,ns,gen2)
    wt = np.zeros((ns,n))
    step_hole = 1
    sig_in = np.copy(sig)
    
    for i in np.arange(ns-1):
        if fast:
            kernel1d = b3spline_fast(step_hole)
            sig_pad = np.lib.pad(sig_in, (2*step_hole,2*step_hole), 'reflect')
            sig_out = psg.convolve(sig_pad, kernel1d, mode='valid')
        else:
            sig_out = b3splineTrans(sig_in,step_hole)
            
        if gen2:
            if fast:
                sig_pad = np.lib.pad(sig_out, (2*step_hole,2*step_hole), 'reflect')
                sig_aux = psg.convolve(sig_pad, kernel1d, mode='valid')
            else:
                sig_aux = b3splineTrans(sig_out,step_hole)
            wt[i] = sig_in - sig_aux
        else:        
            wt[i] = sig_in - sig_out
            
        if normalization:
            wt[i] /= wavtl.trTab[i]
        sig_in = np.copy(sig_out)
        step_hole *= 2
        
    wt[ns-1] = np.copy(sig_out)
    if normalization:
        wt[ns-1] /= wavtl.trTab[ns-1]
    
    return wt

   
def istar1d(wtOri,fast=True,gen2=True,normalization=False):
    (ns,n) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    head = 'star1d_gen2' if gen2 else 'star1d_gen1' 
    if wavtl.trHead != head:
        wavtl.trHead = head  
    if normalization:
        for i in np.arange(ns):
            wt[i] *= wavtl.trTab[i]
    
    if gen2:
        '''
        h' = h, g' = Dirac
        '''
        step_hole = pow(2,ns-2)
        sigRec = np.copy(wt[ns-1])
        for k in np.arange(ns-2,-1,-1):            
            if fast:
                kernel1d = b3spline_fast(step_hole)
                sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
                sig_out = psg.convolve(sig_pad, kernel1d, mode='valid')
            else:
                sig_out = b3splineTrans(sigRec,step_hole)
            sigRec = sig_out + wt[k]
            step_hole /= 2            
    else:
        '''
        h' = Dirac, g' = Dirac
        '''
        sigRec = np.sum(wt,axis=0)
#         '''
#         h' = h, g' = Dirac + h
#         '''
#         sigRec = np.copy(wt[ns-1])
#         step_hole = pow(2,ns-2)
#         for k in np.arange(ns-2,-1,-1):
#             if fast:
#                 kernel1d = b3spline_fast(step_hole)
#                 sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
#                 sigRec = psg.convolve(sig_pad, kernel1d, mode='valid')
#                 wt_pad = np.lib.pad(wt[k], (2*step_hole,2*step_hole), 'reflect')
#                 sig_out = psg.convolve(wt_pad, kernel1d, mode='valid')
#             else:
#                 sigRec = b3splineTrans(sigRec,step_hole)
#                 sig_out = b3splineTrans(wt[k],step_hole)
#             sigRec += wt[k]+sig_out
#             step_hole /= 2
    return sigRec        

def adstar1d(wtOri,fast=True,gen2=True,normalization=False):
    (ns,n) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    # !Attention: wt is not the original wt after unnormalization
    head = 'star1d_gen2' if gen2 else 'star1d_gen1' 
    if normalization:
        if wavtl.trHead != head:
            wavtl.trHead = head
            wavtl.trTab = nsNorm(n,ns,gen2)
        for i in np.arange(ns):
            wt[i] *= wavtl.trTab[i]
     
    sigRec = np.copy(wt[ns-1])
    step_hole = pow(2,ns-2)
    for k in np.arange(ns-2,-1,-1):
        if fast:
            kernel1d = b3spline_fast(step_hole)
            sig_pad = np.lib.pad(sigRec, (2*step_hole,2*step_hole), 'reflect')
            sigRec = psg.convolve(sig_pad, kernel1d, mode='valid')
            wt_pad = np.lib.pad(wt[k], (2*step_hole,2*step_hole), 'reflect')
            sig_out = psg.convolve(wt_pad, kernel1d, mode='valid')
            if gen2:
                sig_pad = np.lib.pad(sig_out, (2*step_hole,2*step_hole), 'reflect')
                sig_out2 = psg.convolve(sig_pad, kernel1d, mode='valid')
                sigRec += wt[k] -sig_out2
            else: sigRec += wt[k] -sig_out
        else:
            sigRec = b3splineTrans(sigRec,step_hole)
            sig_out = b3splineTrans(wt[k],step_hole)
            if gen2:
                sig_out2 = b3splineTrans(sig_out,step_hole)
                sigRec += wt[k] -sig_out2
            else: sigRec += wt[k]-sig_out
        step_hole /= 2
    return sigRec

def nsNorm(nx,nz,gen2=True):
    sig = np.zeros(nx)
    sig[nx/2] = 1                      
    wt = star1d(sig,nz,fast=True,gen2=gen2,normalization=False)      
    tmp = wt**2
    tabNs = np.sqrt(np.sum(tmp,1)) 
    head = 'star1d_gen2' if gen2 else 'star1d_gen1' 
    if wavtl.trHead != head:
        wavtl.trHead = head
        wavtl.trTab = tabNs     
    return tabNs