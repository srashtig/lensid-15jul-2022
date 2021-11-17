__author__='haris.k'

"""
This is a script to compute snr using pycbc modules
Last modified on 2020-03-07
"""
import numpy as np, argparse
from pycbc import waveform, filter, psd
from pycbc.waveform.generator import FDomainDetFrameGenerator
from pycbc.waveform.generator import FDomainCBCGenerator
import csv


parser = argparse.ArgumentParser(description='This is stand alone code for computing snr for injection with O3a PSD')
parser.add_argument('-ifile', '--ifile', help="Input dat file name")

args = parser.parse_args()
ifile = args.ifile

delta_f = 1./64
f_lower = 20.
f_high = 3000.
psd_length = int(2000./delta_f)
t0 = 1251752040 # reference time

o3a_psd = {}
o3a_psd['V1']= psd.analytical.AdvVirgo(psd_length, delta_f, 18.)
o3a_psd['H1']= psd.analytical.aLIGOZeroDetHighPower(psd_length, delta_f, 18.)
o3a_psd['L1']= psd.analytical.aLIGOZeroDetHighPower(psd_length, delta_f, 18.)

'''
psd_dir = '/home1/haris.k/runs/O3/O3a_representative_psd'
h1_asd_data = np.loadtxt('%s/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'%psd_dir) 
l1_asd_data = np.loadtxt('%s/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'%psd_dir) 
v1_asd_data = np.loadtxt('%s/O3-Virgo_sensitivity_asd.txt'%psd_dir)

o3a_psd = {}
o3a_psd['H1'] = psd.read.from_numpy_arrays(h1_asd_data[:,0],\
        np.power(h1_asd_data[:,1],2), psd_length, delta_f, 18.)
o3a_psd['L1'] = psd.read.from_numpy_arrays(l1_asd_data[:,0],\
        np.power(l1_asd_data[:,1],2), psd_length, delta_f, 18.)
o3a_psd['V1'] = psd.read.from_numpy_arrays(v1_asd_data[:,0],\
        np.power(v1_asd_data[:,1],2), psd_length, delta_f, 18.)
'''
generator = FDomainDetFrameGenerator(FDomainCBCGenerator, 0.,\
        variable_args=['mass1', 'mass2', 'spin1z', 'spin2z', \
        'tc', 'ra', 'dec', 'inclination', 'polarization', 'distance'],\
        detectors=['H1', 'L1','V1'],delta_f=delta_f,f_lower=f_lower,\
        approximant='IMRPhenomPv2')

samples = np.genfromtxt(ifile,names=True)
m1z,m2z,z,dist = samples['m1z'],samples['m2z'],samples['z'],samples['ldistance']
n = len(m1z)
ra = np.random.rand(n)*2*np.pi
dec = np.arcsin(np.random.rand(n)*2-1)

phi_tmp  = np.random.rand(n)*2*np.pi
theta_tmp = np.arccos(np.random.rand(n)*2-1)
a = np.random.rand(n)
s1x,s1y,s1z = a*np.sin(theta_tmp)*np.cos(phi_tmp),a*np.sin(theta_tmp)*np.sin(phi_tmp),a*np.cos(theta_tmp)

phi_tmp  = np.random.rand(n)*2*np.pi
theta_tmp = np.arccos(np.random.rand(n)*2-1)
a = np.random.rand(n)
s2x,s2y,s2z = a*np.sin(theta_tmp)*np.cos(phi_tmp),a*np.sin(theta_tmp)*np.sin(phi_tmp),a*np.cos(theta_tmp)

iota = np.arccos(np.random.rand(n))
pol = np.random.rand(n)*2*np.pi
tc  =t0 + np.random.rand(n)*24*60.*60.

with open('%s_withsnr.dat'%(ifile.split('.')[0]), "w") as file:
    file.write('z\tldistance\tm1z\tm2z\tra\tdec\tiota\tpol\ttc\ts1x\ts1y\ts1z\ts2x\ts2y\ts2z\tsnr\n')

snr = np.zeros(n)
for ii in range(n):
        data = generator.generate(mass1=m1z[ii],mass2=m2z[ii],spin1z=s1z[ii],spin2z=s2z[ii], tc=tc[ii], ra=ra[ii], dec=dec[ii], inclination=iota[ii], polarization=pol[ii], distance = dist[ii])
        snrsq = 0
        for det in ['H1','L1','V1']:
            snrsq = snrsq + filter.matchedfilter.sigmasq(data[det], psd=o3a_psd[det],\
                                low_frequency_cutoff=20., high_frequency_cutoff=1500)
        snr[ii] = np.sqrt(snrsq)
        if ((snr[ii]<8) and (snr[ii]>5)):
            #print('%s---%s'%(ii,snr[ii]))
            with open('%s_withsnr.dat'%(ifile.split('.')[0]), "a") as file:
                    file.write('%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n'%(z[ii],dist[ii],m1z[ii],m2z[ii],ra[ii],dec[ii],iota[ii],pol[ii],tc[ii],s1x[ii],s1y[ii],s1z[ii],s2x[ii],s2y[ii],s2z[ii],snr[ii]))
