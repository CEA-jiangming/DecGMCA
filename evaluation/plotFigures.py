'''
Created on Feb 20, 2017

@author: mjiang
'''
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pylab

# delta_rl_Dec = fits.getdata('delta_rl_DecGMCA.fits')
# SDR_rl_Dec = fits.getdata('SDR_rl_DecGMCA.fits')
#  
# sampleA = delta_rl_Dec[:,-1,0]
# sampleS = SDR_rl_Dec[:,-1,0]
# dec_a = np.log10(sampleA).squeeze()
# dec_s = sampleS[::-1].squeeze()

delta_rl_Dec = fits.getdata('delta_rl_PostProc.fits')
SDR_rl_Dec = fits.getdata('SDR_rl_PostProc.fits')
dec_a = np.log10(delta_rl_Dec).squeeze()
dec_s = SDR_rl_Dec.squeeze()

delta_rl_PALM = fits.getdata('delta_rl_PALM.fits')
SDR_rl_PALM = fits.getdata('SDR_rl_PALM.fits')

palm_a = np.log10(delta_rl_PALM).squeeze()
palm_s = SDR_rl_PALM.squeeze()

delta_rl_Rig_PALM = fits.getdata('delta_rl_Rigorous_PALM.fits')
SDR_rl_Rig_PALM = fits.getdata('SDR_rl_Rigorous_PALM.fits')

rig_palm_a = np.log10(delta_rl_Rig_PALM).squeeze()
rig_palm_s = SDR_rl_Rig_PALM.squeeze()

delta_rl_Ref_PALM = fits.getdata('delta_rl_FISTA_Ref.fits')
SDR_rl_Ref_PALM = fits.getdata('SDR_rl_FISTA_Ref.fits')

ref_palm_a = np.log10(delta_rl_Ref_PALM).squeeze()
ref_palm_s = SDR_rl_Ref_PALM.squeeze()

colorArr = ['b','g','r','c','m','y']
dbArr = np.arange(10,54,5)

plt.figure()
plt.plot(dbArr,-dec_a[:0:-1],colorArr[0]+'^-',linewidth=2,markersize=10)
#plt.plot(dbArr,-palm_a[::-1],colorArr[1]+'s--',linewidth=2,markersize=10)
plt.plot(dbArr,-rig_palm_a[:0:-1],colorArr[2]+'o:',linewidth=2,markersize=10)
#plt.plot(dbArr,-ref_palm_a[::-1],colorArr[1]+'s--',linewidth=2,markersize=10)
lab=[]
lab.append('DecGMCA')
lab.append('PALM')
# lab.append('Rigorous PALM')
lab=tuple(lab)
plt.legend(lab,loc=0,fontsize=10)
plt.xlabel('SNR(dB)',fontsize=15)
plt.ylabel('Criterion of A',fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(5,55)
# plt.savefig('Z_SDR_sup_Pp_SNR.png')

plt.figure()
plt.plot(dbArr,dec_s[:0:-1],colorArr[0]+'^-',linewidth=2,markersize=10)
#plt.plot(dbArr,palm_s[:0:-1],colorArr[1]+'s--',linewidth=2,markersize=10)
plt.plot(dbArr,rig_palm_s[:0:-1],colorArr[2]+'o:',linewidth=2,markersize=10)
#plt.plot(dbArr,ref_palm_s[::-1],colorArr[1]+'s--',linewidth=2,markersize=10)
lab=[]
lab.append('DecGMCA')
lab.append('PALM')
# lab.append('Rigorous PALM')
lab=tuple(lab)
plt.legend(lab,loc=0,fontsize=10)
plt.xlabel('SNR(dB)',fontsize=15)
plt.ylabel('SDR(dB)',fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(5,55)
# plt.savefig('Z_SDR_sup_Pp_SNR.png')

pylab.show()