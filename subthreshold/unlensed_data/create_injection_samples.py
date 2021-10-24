#!/usr/bin/python
__author__ = 'haris.k'
#----------------------------------------------------------------
import argparse
import sys,csv,os
import numpy as np
from cosmology_models import LCDM
from scipy import special
from random import shuffle
#----------------------------------------------------------------
lcdm = LCDM(0.3)

parser = argparse.ArgumentParser(description='Create a dat file with BBH injection samples with a given z and component masses distribution')
parser.add_argument('-outfile', '--outfile', help='name of output file', required=True)
parser.add_argument('-z_pdf_model', '--z_pdf_model', help='z_pdf_models: Dominik, Belczynski, Pop-III and Primordial (see Liang Dai et al. PRD,2017 for details)', required=True)
parser.add_argument('-mass_pdf_model', '--mass_pdf_model', help='comp. mass_pdf_models: powerlaw1, powerlaw2, (see doi:10.3847/2041-8205/833/1/L1) ,schechter, lognormal, gaussian  (see Liang Dai et al. PRD,2017) ', required=True)
parser.add_argument('-z_max', '--z_max', help='readshift upper limit', type=float, required=True)
parser.add_argument('-n', '--n', type=int, help='number of samples', required=True)


args = parser.parse_args()
outfile = args.outfile
z_pdf_model = args.z_pdf_model
mass_pdf_model = args.mass_pdf_model
z_max = args.z_max
n = args.n


def Heaviside(x):
    ''' Heaviside step function '''
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])

def m1m2_to_mchirpeta(mass1, mass2):
    eta = mass1*mass2/(mass1+mass2)/(mass1+mass2)
    mc = (mass1+mass2) * np.power(eta, 3./5.)
    return (mc, eta)

def mchirpeta_to_m1m2(mchirp, eta):
    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    return (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)

def schechter_mass_distribution(z,m_lower=5.,gamma_z=6.):
    '''
    The function return Schechter source frame probability distribution
    of mass ms for given a redshift z. 
    Ref: Eq.(21), Liang Dai et al. PRD(2017)
    '''
    m = np.linspace(m_lower,200.,5000)
    m_prime=3*np.power((1+z)/2.5,0.5)
    pdf = Heaviside(m-m_lower)/(m_prime * special.gamma(1+gamma_z)) \
        * np.power((m-m_lower)/m_prime,gamma_z) * np.exp(-(m-m_lower)/m_prime)
    norm = np.sum(pdf)*(m[1]-m[0])
    return m, pdf/norm

z_pdf_data=np.genfromtxt('./z_PDF.dat',dtype=None,names=True)
z = z_pdf_data['z']
delta_z = z[1]-z[0]
z_pdf = z_pdf_data[z_pdf_model]
idx = idx = (np.abs(z-z_max)).argmin() + 1
z = z[:idx]
z_pdf_nomr = np.sum(z_pdf[:idx])* delta_z
z_pdf = z_pdf[:idx] / z_pdf_nomr

n_buffer = 2*n
z_samples = []
if  mass_pdf_model=='schechter':
    m1s_samples = []
    m2s_samples = []
    for ii in range(len(z)):
        n_zbin =  int(n_buffer*z_pdf[ii]*delta_z)
        z_samples=np.append(z_samples,z[ii] - 0.5*delta_z + delta_z*np.random.rand(n_zbin))
        ms,pdf_ms = schechter_mass_distribution(z[ii])
        delta_ms= ms[1]-ms[0]
        m1s_samples=np.append(m1s_samples,np.random.choice(ms,size=n_zbin,p=pdf_ms*delta_ms))
        m2s_samples=np.append(m2s_samples,np.random.choice(ms,size=n_zbin,p=pdf_ms*delta_ms))
    index=range(len(z_samples))
    np.random.shuffle(index)
    z_samples = z_samples[index[:n]]
    m1s_samples = m1s_samples[index[:n]]
    m2s_samples = m2s_samples[index[:n]]
else:
    for ii in range(len(z)):
        n_zbin =  int(n_buffer*z_pdf[ii]*delta_z)
        z_samples=np.append(z_samples,z[ii] - 0.5*delta_z + delta_z*np.random.rand(n_zbin))
    index=np.arange(len(z_samples))
    np.random.shuffle(index)
    z_samples = z_samples[index[:n]]
    if (mass_pdf_model=='lognormal'):
        temp1= np.random.normal(np.log(25.), 0.18, n)
        temp2= np.random.normal(np.log(25.), 0.18, n)
        m1s_samples,m2s_samples = np.exp(temp1),np.exp(temp2)
    elif (mass_pdf_model=='gaussian'):
        m1s_samples= np.random.normal(8., 1.5, 2*n)
        index=np.where(m1s_samples>=5)
        m1s_samples=m1s_samples[index][:n]
        m2s_samples= np.random.normal(8., 1.5, 2*n)
        index=np.where(m2s_samples>=5)
        m2s_samples=m2s_samples[index][:n]
    elif (mass_pdf_model=='powerlaw1'):
        mmin=5.
        mmax=100.
        u1=np.random.rand(4*n)
        u2=np.random.rand(4*n)
        m1s_samples=np.exp(u1*(np.log(mmax/mmin))+np.log(mmin))
        m2s_samples=np.exp(u2*(np.log(mmax/mmin))+np.log(mmin))
        index=np.where(m1s_samples+m2s_samples<=100)
        m1s_samples =m1s_samples[index][:n]
        m2s_samples =m2s_samples[index][:n]
    elif (mass_pdf_model=='powerlaw2'):
        mmin=5.
        mmax=60.
        u1=np.random.rand(4*n)
        qmin=5./60.
        q=np.random.rand(4*n)*(1-qmin)+qmin
        c= 1.35/(np.power(mmin,-1.35)-np.power(mmax,-1.35))
        m1s_samples=1/np.power(np.power(mmin,-1.35)-1.35*u1/c,1/1.35)
        m2s_samples=q*m1s_samples
        index=np.where(m1s_samples+m2s_samples<=100)
        m1s_samples=m1s_samples[index]
        m2s_samples=m2s_samples[index]
        index=np.where(m2s_samples>5)
        m1s_samples = m1s_samples[index][:n]
        m2s_samples = m2s_samples[index][:n]
            
mcs_samples,eta_samples = m1m2_to_mchirpeta(m1s_samples,m2s_samples)
mcz_samples = mcs_samples*(1+z_samples)
m1z_samples = m1s_samples*(1+z_samples)
m2z_samples = m2s_samples*(1+z_samples)

ldistance=np.zeros(len(z_samples))
for ii in range(len(z_samples)):
    ldistance[ii] = lcdm.luminosity_distance_z(z_samples[ii])
    
    
with open('%s_%s_%s_inj_samples.dat'%(outfile,z_pdf_model,mass_pdf_model),'w') as f:
    writer = csv.writer(f,delimiter='\t')
    writer.writerow(['z\tldistance\tm1z\tm2z\tmcz\tm1s\tm2s\tmcs\teta'])
    writer.writerows(zip(z_samples,ldistance,m1z_samples,m2z_samples,mcz_samples,m1s_samples,m2s_samples,mcs_samples,eta_samples))
