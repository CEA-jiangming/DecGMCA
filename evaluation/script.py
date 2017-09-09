'''
Created on Nov 25, 2015

@author: mjiang
'''
import astropy.io.fits as fits
import numpy as np
from evaluation import *
import matplotlib.pyplot as plt
import os
import pylab

epsilon=1e0
# epsilon=1e-1
# epsilon=1e-2
# epsilon=1e-3
# epsilon=1e-4
# epsilon=0

dbArr = np.array([55,50,45,40,35,30,25,20,15,10]) 
nArr=np.array([5])
# dbArr=np.array([55])
rl=1
sigS=1
pcArr = np.array([0.5])
# bds=np.arange(5,26,5)
bds=np.array([20])
varArr=dbArr
# varArr=np.array([epsilon])

drTest = '../simu_CS_deconv/test_CS/'
# drTest = '../test/'
drResult = drTest+'results_PALM_FISTA_Refinement/'
# drPostResult = drTest+'post_results/'
# drResult = drTest+'results/'
drMixture = drTest+'mixtures/'
drSources = drTest+'sources/'
drNoise = drTest+'noises/'

drEvalResult = drTest+'eval_results/'

delta = np.zeros((len(varArr),len(bds)))
delta_rl = np.zeros((len(varArr),len(bds),rl))

SDR = np.zeros((len(varArr),len(bds)))
SDR_rl = np.zeros((len(varArr),len(bds),rl))

SIR = np.zeros((len(varArr),len(bds)))
SIR_rl = np.zeros((len(varArr),len(bds),rl))

SNR = np.zeros((len(varArr),len(bds)))
SNR_rl = np.zeros((len(varArr),len(bds),rl))

SAR = np.zeros((len(varArr),len(bds)))
SAR_rl = np.zeros((len(varArr),len(bds),rl))


for j in np.arange(len(varArr)):
#     subdr = 'n'+str(n)+'/epsilon_'+'%.e'% epsilon+'_db_'+str(db[0])+'_n'+str(n)+'/'
#     subdr = 'n'+str(varArr[j])+'/epsilon_'+'%.e'% epsilon +'/'
#     if not os.path.exists(subdr):
#         os.makedirs(subdr)
    for i in np.arange(len(bds)):
        for r in np.arange(rl):
            rlpath='r'+str(r)+'_mask'+str(int(pcArr[0]*100))+'_db'+str(dbArr[j])+'/'
#             rlpath='r'+str(r)+'_kernel_db'+str(db[0])+'/'
#             rlpath='r'+str(r)+'/'
#             if not os.path.exists(subdr+rlpath):
#                 os.makedirs(subdr+rlpath)
#             AeName = drResult+rlpath+'estA_bd'+str(bds[i])+'_r'+str(r)+'.fits'
            AeName = drResult+rlpath+'estA_bd'+str(bds[i])+'.fits'
#             AeName = drResult+subdr+rlpath+'estA_post_anal_soft_bd'+str(bds[i])+'_Ksig'+str(varArr[j])+'_r'+str(r)+'.fits'
#             SeName = drResult+subdr+rlpath+'estS_post_anal_soft_bd'+str(bds[i])+'_Ksig1'+'_r'+str(r)+'.fits'
#             SeName = drResult+rlpath+'estS_bd'+str(bds[i])+'_r'+str(r)+'.fits'
            SeName = drResult+rlpath+'estS_bd'+str(bds[i])+'.fits'
#             SeName = drResult+subdr+rlpath+'estS_post_bd'+str(bds[i])+'_r'+str(r)+'.fits'
#             SeName = drPostResult+subdr+rlpath+'estS_post_condat_vu_soft_bd'+str(bds[i])+'_Ksig0.6'+'_r'+str(r)+'.fits'
            noiseName = drNoise+rlpath+'noise_bd'+str(bds[i])+'.fits'
            AName = drMixture+rlpath+'A_bd'+str(bds[i])+'.fits'
            SName = drSources+rlpath+'S_bd'+str(bds[i])+'.fits'
            Ae = fits.getdata(AeName)
            A = fits.getdata(AName)
            Se = fits.getdata(SeName)
            Se = np.reshape(Se,(Se.shape[0],np.size(Se[0])))
            S = fits.getdata(SName)
            S = np.reshape(S,(S.shape[0],np.size(S[0])))
            noise = fits.getdata(noiseName)
#             noise = fits.getdata(drNoise+'/n'+str(n)+'/noise_db'+str(db[0])+'_sigS'+str(sigS)+'_r'+str(r)+'.fits')
            criteria, decomposition, delta_rl[j,i,r],Ae_ord,Se_ord = evaluation((Ae,Se), (A,S,noise), verbose=0)
            
            SDR_rl[j,i,r] = criteria['SDR_S']
            SIR_rl[j,i,r] = criteria['SIR_S']
            SNR_rl[j,i,r] = criteria['SNR_S']
            SAR_rl[j,i,r] = criteria['SAR_S']
            
#             deltaRl[j,i,r]=abs(abs(linalg.inv(Ae.T.dot(Ae)).dot(Ae.T).dot(A)) - np.eye(n)).sum() / (n*n)
#             fits.writeto(AeName,Ae_ord,clobber=True)
#             fits.writeto(SeName,Se_ord,clobber=True)
            if not os.path.exists(drEvalResult+rlpath):
                os.makedirs(drEvalResult+rlpath) 
            fits.writeto(drEvalResult+rlpath+'estA_bd'+str(bds[i])+'.fits',Ae_ord,clobber=True)
            fits.writeto(drEvalResult+rlpath+'estS_bd'+str(bds[i])+'.fits',Se_ord,clobber=True)

# delta = np.median(delta_rl,axis=2)
fits.writeto('delta_rl_FISTA_Ref.fits',delta_rl,clobber=True)

# SDR = np.median(SDR_rl,axis=2)
fits.writeto('SDR_rl_FISTA_Ref.fits',SDR_rl,clobber=True)

# SIR = np.median(SIR_rl,axis=2)
# fits.writeto('SIR_rl.fits',SIR_rl,clobber=True)

# SNR = np.median(SNR_rl,axis=2)
# fits.writeto('SNR_rl.fits',SNR_rl,clobber=True)

# SAR = np.median(SAR_rl,axis=2)
# fits.writeto('SAR_rl.fits',SAR_rl,clobber=True)

# #### Plots for criterion of A ############
#   
# for j in np.arange(len(varArr)):
#     plt.figure()
# #     plt.plot(bds,delta[j,:])
#     plt.plot(bds,np.log10(delta[j,:]))
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'$\frac{A_{est}^\dagger A_{ref}-I_n}{n^2}$')
#   
#   
# for j in np.arange(len(varArr)):
#     plt.figure()
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'$\frac{A_{est}^\dagger A_{ref}-I_n}{n^2}$')
#     for r in np.arange(rl):
#         plt.plot(bds,np.log10(delta_rl[j,:,r]))
# #         plt.plot(bds,deltaRl[j,:,r])
#     plt.legend(('r0','r1','r2','r3','r4'))
# 
#       
# #### Plots for criteria of S ############ 
# # SDR
# for j in np.arange(len(varArr)):
#     plt.figure()
#     plt.plot(bds,SDR[j,:])
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SDR of S')
#   
# for j in np.arange(len(varArr)):
#     plt.figure()
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SDR of S')
#     for r in np.arange(rl):
#         plt.plot(bds,SDR_rl[j,:,r])
#     plt.legend(('r0','r1','r2','r3','r4'))

# # SIR
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.plot(bds,SIR[j,:])
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SIR of S')
#     plt.savefig('SIR_median_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')
# 
# 
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SIR of S')
#     for r in np.arange(rl):
#         plt.plot(bds,SIR_rl[j,:,r])
#     plt.legend(('r0','r1','r2','r3','r4'))
#     plt.savefig('SIR_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')
#     
# # SNR
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.plot(bds,SNR[j,:])
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SNR of S')
#     plt.savefig('SNR_median_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')
# 
# 
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SNR of S')
#     for r in np.arange(rl):
#         plt.plot(bds,SNR_rl[j,:,r])
#     plt.legend(('r0','r1','r2','r3','r4'))
#     plt.savefig('SNR_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')
#     
# # SAR
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.plot(bds,SAR[j,:])
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SAR of S')
#     plt.savefig('SAR_median_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')
# 
# for j in np.arange(len(epsilon)):
#     plt.figure()
#     plt.title(r'Error of mixing matrix,$\epsilon$='+str(epsilon[j]))
#     plt.xlabel('Number of bands')
#     plt.ylabel(r'SAR of S')
#     for r in np.arange(rl):
#         plt.plot(bds,SAR_rl[j,:,r])
#     plt.legend(('r0','r1','r2','r3','r4'))
#     plt.savefig('SAR_n'+str(n)+'_epsilon_'+'%.e'% epsilon[j]+'.png')

# pylab.show()
# A45e=fits.getdata('/Users/mjiang/Documents/workspace/python/HSSRec/results/epsilon_e4/estA_bd45.fits')
# A45=A45=fits.getdata('/Users/mjiang/Documents/workspace/python/HSSRec/A_bd45.fits')
# 
# S45=fits.getdata('/Users/mjiang/Documents/workspace/python/HSSRec/S0.fits')
# S45=np.reshape(S45,(S45.shape[0],S45.shape[1]*S45.shape[2]))
# S45e=fits.getdata('/Users/mjiang/Documents/workspace/python/HSSRec/results/epsilon_e4/estS_bd45.fits')
# S45e=np.reshape(S45e,(S45e.shape[0],S45e.shape[1]*S45e.shape[2]))
# 
# print S45.shape
# print S45e.shape

# SDR_S = compute_sdr_matrix(S45, S45e)
# print SDR_S
# costMatrix = -SDR_S
# r=S45.shape[1]
# hungarian = munkres.Munkres()
# ind_list = hungarian.compute(costMatrix.tolist())
# indices = np.zeros(r, dtype=int)
# for k in range(0, r):
#     indices[k] = ind_list[k][1]
# # reorder the factorization
# A_ord = A45e[:, indices]
# 
# print "reference A"
# print A45
# print "Ordered A:"
# print A_ord
# print "Original A:"
# print A45e
# 
# delta1=abs(linalg.inv(A_ord.T.dot(A_ord)).dot(A_ord.T).dot(A45) - np.eye(2)).sum() / (2*2)
# print delta1
# delta2=abs(abs(linalg.inv(A_ord.T.dot(A_ord)).dot(A_ord.T).dot(A45)) - np.eye(2)).sum() / (2*2)
# print delta2

# delta=evaluation((A45e,S45e), (A45,S45), verbose=0)
# 
pylab.show()
print SDR_rl.squeeze()
print np.log10(delta_rl).squeeze()

