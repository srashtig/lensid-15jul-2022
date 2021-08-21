import pylab 
import pycbc.noise
import pycbc.psd
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from numpy.core.defchararray import add
import os
import numpy as np
import pandas as pd
import sys
from lensid.utils.qt_utils import *
import argparse
def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for generating QTs for unlensed injection dataset')


    parser.add_argument('-odir','--odir', help='Output directory',default='check')
    parser.add_argument('-start','--start', type=int, help='unlensed inj start index',default=0)
    parser.add_argument('-whitened','--whitened',type=bool,help='True/False',default = False)
    parser.add_argument('-n','--n', type=int, help='no. of unlensed injs',default = 0)
    parser.add_argument('-mode','--mode', type = int, help='enter no : 1. default \t 2. test',default = 1)
    parser.add_argument('-infile','--infile', help='.npz unlensed injs file path to load tags from',required = True)
    parser.add_argument('-psd_mode','--psd_mode', type = int, help='enter no : 1. analytical \t 2.load from files',default = 1)
    parser.add_argument('-asd_dir','--asd_dir',help='optional, give directory where psd files are in format H1.txt, L1.txt, V1.txt',default=None)
    parser.add_argument('-qrange', '--qrange',type = int, help= '1. default is 3,7 for m1>60 and 4,10 otherwise. 2. wide is 3,30', default = 1)
    args = parser.parse_args()

    if args.psd_mode ==1:
        psd_mode = 'analytical'
        psd_H, psd_L, psd_V = inj_psds_HLV(psd_mode=psd_mode)

    elif args.psd_mode==2:
        psd_mode='load'
        psd_H, psd_L, psd_V = inj_psds_HLV(psd_mode=psd_mode,asd_dir=args.asd_dir)
    else: 
        print('invalid psd_mode choice')
    duration=64


    data=np.load(args.infile)

    if args.mode == 1:
        lost_ids = []
    elif args.mode ==2:
        lost_ids=np.array([55,57,68,76,182,207,225,277])
    else:
        print('mode not found')


    ntot=data['m1z'].shape[0]
    if args.n == 0:
        args.n = ntot

    whitened=args.whitened

    odir =  args.odir

    if not os.path.exists(odir):
        os.makedirs(odir)
        os.makedirs(odir+'/H1')
        os.makedirs(odir+'/L1')
        os.makedirs(odir+'/V1')

    pow_H1=np.zeros(args.n) 
    pow_L1=np.zeros(args.n)    
    pow_V1=np.zeros(args.n)    
    fnames=np.zeros(args.n).astype('U256') 

    for count,i in enumerate(range(args.start,args.start+args.n)):
        if args.qrange == 1:
            if data["m1z"][i]<60:
                q = q_msmall
            else:
                q = q_mlarge
        else:
            q=q_wide
        hp,hc = get_td_waveform(approximant = "IMRPhenomPv2",mass1=data["m1z"][i],mass2=data['m2z'][i],inclination=data["iota"][i],delta_t=1.0/2**12,f_lower=15,f_higher = 1000, distance =data["ldistance"][i],coa_phase = data['phi0'][i])
        # flower=30 works without whitening
        print(data["m1z"][i],data['m2z'][i])
        det_h1 = Detector("H1")
        det_l1 = Detector("L1")
        det_v1 = Detector("V1")
        end_time = data["tc"][i]
        declination = data['dec'][i]
        right_ascension = data["ra"][i]
        polarization = data["pol"][i]
        hp.start_time += end_time
        hc.start_time +=end_time
        signal_h1 = det_h1.project_wave(hp,hc,right_ascension,declination,polarization)
        signal_l1 = det_l1.project_wave(hp,hc,right_ascension,declination,polarization)
        signal_v1 = det_v1.project_wave(hp,hc,right_ascension,declination,polarization)

        noise_signal_h1 = inject_noise_signal(signal_h1, psd_H, duration=duration,whitened=whitened)
        noise_signal_l1 = inject_noise_signal(signal_l1, psd_L, duration=duration,whitened=whitened)
        noise_signal_v1 = inject_noise_signal(signal_v1, psd_V, duration=duration,whitened=whitened)

        fname=str(data['event_tag'][i])
        if whitened== True:
            fname=fname+'-whitened'
        pow_H1[count]=plot_qt_from_ts(noise_signal_h1, data["tc"][i], q, outfname=odir+ "/H1/"+fname)
        pow_L1[count]=plot_qt_from_ts(noise_signal_l1, data["tc"][i], q, outfname=odir+ "/L1/"+fname)
        pow_V1[count]=plot_qt_from_ts(noise_signal_v1, data["tc"][i], q, outfname=odir+ "/V1/"+fname)
        fnames[count] = fname


        print(i)
    fname = odir+'/qt_power_unlensed'
    if whitened== True:
        fname=fname+'-whitened'
    np.savez(fname+'.npz',pow_H1=pow_H1,pow_L1=pow_L1,pow_V1=pow_V1,fnames=fnames)
