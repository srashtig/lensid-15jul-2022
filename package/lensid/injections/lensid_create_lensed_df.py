import os
import numpy as np
import pandas as pd
from numpy.core.defchararray import add
import argparse
def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for generating lensed dataframe for testing ML lensing classification')

    parser.add_argument('-odir','--odir', help='Output directory',default='check')
    parser.add_argument('-outfile','--outfile', help='dataframe odir/outfile',default = 'lensed.csv')

    parser.add_argument('-start','--start', type=int, help='lensed inj start index',default=0)
    parser.add_argument('-n','--n', type=int, help='no. of unlensed injs',default = 0)
    parser.add_argument('-mode','--mode', type = int, help='enter no : 1. default \t 2. test',default = 1)
    parser.add_argument('-infile','--infile', help='.npz lensed injs file path to load tags from',required = True)

    args = parser.parse_args()
    print('\n Arguments used:- \n')
    
    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))

        
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


    odir = args.odir

    fnames_img1=add(add(data['event_tag'].astype(str),'_'),data['img_tag'][:,0].astype(str))
    fnames_img2=add(add(data['event_tag'].astype(str),'_'),data['img_tag'][:,1].astype(str))

    fnames_img1=np.delete(fnames_img1,lost_ids)
    fnames_img2=np.delete(fnames_img2,lost_ids)
    fnames_img1,fnames_img2 = fnames_img1[args.start:args.start+args.n],fnames_img2[args.start:args.start+args.n]

    Lensed_df=pd.DataFrame()
    Lensed_df=Lensed_df.assign(img_0=fnames_img1)
    Lensed_df=Lensed_df.assign(img_1=fnames_img2)
    Lensed_df=Lensed_df.assign(Lensing=np.ones(fnames_img1.shape).astype('int'))
    print(ntot , 'no. of lensed event pairs in total, ' , args.n,  ' considered' )
    if not os.path.exists(odir):
        os.makedirs(odir)
    outfile = odir + '/' +  args.outfile
    Lensed_df.to_csv(outfile)
    print('Dataframe saved at ', outfile)
