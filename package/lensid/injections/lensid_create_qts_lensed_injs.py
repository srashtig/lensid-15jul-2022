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
import lensid.utils.qt_utils as qtils
import argparse

def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for generating QTs for lensed injection dataset')

    parser.add_argument('-odir','--odir', help='Output directory',default='check')

    parser.add_argument('-start','--start', type=int, help='lensed inj start index',default=0)
    parser.add_argument('-whitened','--whitened',help='True/False',type=bool,default = False)
    parser.add_argument('-n','--n', type=int, help='no. of unlensed injs',default = 0)
    parser.add_argument('-mode','--mode', type = int, help='enter no : 1. default \t 2. test',default = 1)
    parser.add_argument('-infile','--infile', help='.npz lensed injs file path to load tags from',required = True)
    parser.add_argument('-psd_mode','--psd_mode', type = int, help='enter no : 1. analytical \t 2.load from files',default = 1)
    parser.add_argument('-asd_dir','--asd_dir',help='optional,give directory where psd files are in format H1.txt, L1.txt, V1.txt',default=None)
    parser.add_argument('-qrange', '--qrange',type = int, help= '1. default is 3,7 for m1>60 and 4,10 otherwise. 2. wide is 3,30', default = 1)
    args = parser.parse_args()

    if args.psd_mode ==1:
        psd_mode = 'analytical'
        psd_H, psd_L, psd_V = qtils.inj_psds_HLV(psd_mode=psd_mode)

    elif args.psd_mode==2:
        psd_mode='load'
        psd_H, psd_L, psd_V = qtils.inj_psds_HLV(psd_mode=psd_mode,asd_dir=args.asd_dir)
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


    ntot=data['m1'].shape[0]-len(lost_ids)
    if args.n == 0:
        args.n = ntot

    whitened=args.whitened

    odir =  args.odir

    if not os.path.exists(odir):
        os.makedirs(odir)
    try:    
        os.makedirs(odir+'/H1')
        os.makedirs(odir+'/L1')
        os.makedirs(odir+'/V1')
    except:
        print(odir, ' already exists. saving Qtransforms to it.')


    ids=np.arange(data['m1'].shape[0])
    ids=np.delete(ids,lost_ids)

    pow_H1=np.zeros([args.n,2]) 
    pow_L1=np.zeros([args.n,2])    
    pow_V1=np.zeros([args.n,2])    
    fnames=np.zeros([args.n,2]).astype('U256')  

    for count,i in enumerate(ids[args.start:args.start+args.n]):
        if args.qrange == 1:
            if data["m1"][i]<60:
                q = qtils.q_msmall
            else:
                q = qtils.q_mlarge
        else:
            q=qtils.q_wide

        for j in range(0,2):
            hp,hc = get_td_waveform(approximant = "IMRPhenomPv2",mass1=data["m1"][i],mass2=data['m2'][i],inclination=data["incl"][i],delta_t=1.0/2**12,f_lower=15,f_higher = 1000, distance =data["dist"][i][j],coa_phase = data['phi0'][i])
            det_h1 = Detector("H1")
            det_l1 = Detector("L1")
            det_v1 = Detector("V1")
            end_time = data["t0"][i][j]
            declination = data['dec'][i]
            right_ascension = data["ra"][i]
            polarization = data["pol"][i]
            hp.start_time += end_time
            hc.start_time +=end_time

            signal_h1 = det_h1.project_wave(hp,hc,right_ascension,declination,polarization)
            signal_l1 = det_l1.project_wave(hp,hc,right_ascension,declination,polarization)
            signal_v1 = det_v1.project_wave(hp,hc,right_ascension,declination,polarization)


            noise_signal_h1 = qtils.inject_noise_signal(signal_h1, psd_H, duration=duration,whitened=whitened)
            noise_signal_l1 = qtils.inject_noise_signal(signal_l1, psd_L, duration=duration,whitened=whitened)
            noise_signal_v1 = qtils.inject_noise_signal(signal_v1, psd_V, duration=duration,whitened=whitened)

            fname=str(data['event_tag'][i])+'_'+str(data['img_tag'][i,j])
            if whitened== True:
                fname=fname+'-whitened'
            pow_H1[count,j]=qtils.plot_qt_from_ts(noise_signal_h1, data["t0"][i][j], q, outfname=odir+ "/H1/"+fname)
            pow_L1[count,j]=qtils.plot_qt_from_ts(noise_signal_l1, data["t0"][i][j], q, outfname=odir+ "/L1/"+fname)
            pow_V1[count,j]=qtils.plot_qt_from_ts(noise_signal_v1, data["t0"][i][j], q, outfname=odir+ "/V1/"+fname)
            fnames[count,j]=fname

        print(i)
    fname = odir+'/qt_power_lensed'
    if whitened== True:
        fname=fname+'-whitened'
    np.savez(fname+'.npz',pow_H1=pow_H1,pow_L1=pow_L1,pow_V1=pow_V1,fnames=fnames)