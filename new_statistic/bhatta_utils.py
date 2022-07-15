import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import add
import seaborn as sns
import pycbc.conversions as convert



def gaussian_dist(x,mu=0,sig=1):
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sig)**2)



def bhattacharya_coeff(mu1,mu2,sig1,sig2,theta=np.linspace(1,1001,10000)):
    bc=[]
    for i in range(len(mu1)):
        gd1=gaussian_dist(theta,mu1[i],sig1[i])
        gd2=gaussian_dist(theta,mu2[i],sig2[i])
        bc.append((np.sqrt(gd1*gd2)*np.diff(theta)[0]).sum())  
    return np.array(bc)
def bhattacharya_dist(mu1,mu2,sig1,sig2):
    return 1/4 * np.log(1/4 * ((sig1/sig2)**2 + (sig2/sig1)**2 + 2 ) ) + 1/4 * ((mu1-mu2)**2) / (sig1**2 + sig2**2)


    
    
def get_Mobs_injection(snr_inj, m_inj,s=0.08*8,seed=None):
    np.random.seed(seed)
    snr_obs = snr_inj - np.random.randn(len(snr_inj))
    sig_M_guess = m_inj*(s/snr_obs)
    M_mf = m_inj + np.random.randn(len(snr_inj))*sig_M_guess
    sig_mf = M_mf*(s/snr_obs)
    return M_mf, sig_mf