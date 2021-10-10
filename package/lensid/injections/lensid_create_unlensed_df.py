import os
import numpy as np
import pandas as pd
from numpy.core.defchararray import add
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for generating unlensed dataframe for testing ML lensing classification')

    parser.add_argument(
        '-odir',
        '--odir',
        help='Output directory',
        default='check')
    parser.add_argument(
        '-outfile',
        '--outfile',
        help='dataframe name(.csv)',
        default='unlensed.csv')
    parser.add_argument(
        '-start',
        '--start',
        type=int,
        help='unlensed inj start index',
        default=0)
    parser.add_argument(
        '-n',
        '--n',
        type=int,
        help='no. of unlensed injs',
        default=0)
    parser.add_argument(
        '-infile',
        '--infile',
        help='input unlensed inj file path to load tags from',
        required=True)
    args = parser.parse_args()
    print('\n Arguments used:- \n')

    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))

    data = np.load(args.infile)

    ntot = data['m1z'].shape[0]
    if args.n == 0:
        args.n = ntot

    odir = args.odir
    cols = ['img_0', 'img_1', 'Lensing']
    from itertools import combinations
    ul_ids = np.arange(args.start, args.start + args.n)
    comb = np.array(list(combinations(ul_ids, 2)))
    npairs = comb.shape[0]

    img1 = data['event_tag'][comb[:, 0]]
    img2 = data['event_tag'][comb[:, 1]]

    Unlensed_df_test = pd.DataFrame(columns=cols, index=range(npairs))
    Unlensed_df_test['img_0'] = img1.astype('U256')
    Unlensed_df_test['img_1'] = img2.astype('U256')
    Unlensed_df_test['Lensing'] = np.zeros(npairs).astype('int')
    print(ntot, 'no. of events in total, ', args.n, ' considered')
    if not os.path.exists(odir):
        os.makedirs(odir)
    outfile = odir + '/' + args.outfile
    Unlensed_df_test.to_csv(outfile)
    print('Dataframe saved at ', outfile)
