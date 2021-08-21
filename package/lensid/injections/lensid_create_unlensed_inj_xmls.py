import os
import numpy as np
import pandas as pd
import argparse
import sys

def main():                
    parser = argparse.ArgumentParser(description='This is stand alone code for generating injection for bayestar localisation')
    parser.add_argument('-odir','--odir', help='Output directory',default='unlensed_skymaps')
    parser.add_argument('-index','--index', type=int, help='index of injection file',required=True)
    parser.add_argument('-infile','--infile', help='.npz unlensed injs file path to load tags from',required = True)

    args = parser.parse_args()

    data=np.load(args.infile)
    index = args.index
    outdir = args.odir

    odir = outdir+'/'+str(index) + '/'
    if not os.path.exists(odir):
        os.makedirs(odir)


    m1=data["m1z"][index]
    m2=data['m2z'][index]
    iota=data["iota"][index]
    dL=data["ldistance"][index]
    dec = data['dec'][index]
    ra = data["ra"][index]
    pol = data["pol"][index]
    phi0=data["phi0"][index]
    t0=data["tc"][index]

    os.system('lalapps_inspinj -o %sinj.xml --m-distr fixMasses --fixed-mass1 %s --fixed-mass2 %s--t-distr uniform --time-step 7200 --gps-start-time %f --gps-end-time %f --d-distr volume --polarization %f --min-distance %f --max-distance %f --l-distr fixed --latitude %f --longitude %f --i-distr fixed --fixed-inc %f --f-lower 30 --disable-spin --coa-phase-distr fixed --fixed-coa-phase %f --waveform SEOBNRv4 ' %(odir,m1,m2,t0,t0+200,pol*180/np.pi,dL*1e3,dL*1e3,dec*180/np.pi,ra*180/np.pi-180,iota*180/np.pi,phi0*180/np.pi))