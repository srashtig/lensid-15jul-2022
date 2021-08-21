import os
import numpy as np
import pandas as pd
from numpy.core.defchararray import add
import sys

import argparse

def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for converting .fits skymaps to .npz for lensed unlensed injections')


    parser.add_argument('-odir','--odir', help='Output directory ',default='test')
    parser.add_argument('-indir','--indir', help='Input directory ',default='test')
    parser.add_argument('-index','--index', type=int, help='index of injection file',required=True)
    parser.add_argument('-lensed','--lensed',  help='lensed injection? y/n',default='n')
    parser.add_argument('-infile','--infile', help='.npz lensed injs file path to load tags from',required = True)
    args = parser.parse_args()
    index = args.index
    odir = args.odir
    indir =args.indir

    if args.lensed == 'n':
        unlensed_data=np.load(args.infile)       
        outfile = odir + '/' + str(unlensed_data['event_tag'][index]) + '.npz'
        infile = indir + '/unlensed/' + str(index) + '/0.fits'
        os.system('fits_to_cart.py -infile %s -outfile %s'%(infile,outfile))
    else:
        lensed_data=np.load(args.infile)

        fname_img1= str(lensed_data['event_tag'][index])+'_'+str(lensed_data['img_tag'][index,0])
        fname_img2 =str(lensed_data['event_tag'][index])+'_'+str(lensed_data['img_tag'][index,1])

        outfile1 = odir + '/' + str(fname_img1)+ '.npz'
        outfile2 = odir + '/' + str(fname_img2)+ '.npz'
        infile1 = indir + '/lensed/' + str(index) + '/0/0.fits'
        infile2 = indir + '/lensed/' + str(index) + '/1/0.fits'

        os.system('lensid_fits_to_cart -infile %s -outfile %s'%(infile1,outfile1))
        os.system('lensid_fits_to_cart -infile %s -outfile %s'%(infile2,outfile2))
