import sys
import lensid.utils.ml_utils as ml
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for calculating features from the skymaps pairs(.npz)')
    parser.add_argument(
        '-infile',
        '--infile',
        help='input Dataframe path',
        default='train/lensed.csv')
    parser.add_argument(
        '-outfile',
        '--outfile',
        help='output Dataframe path ',
        default='train/lensed_sky.csv')
    parser.add_argument(
        '-data_dir',
        '--data_dir',
        help='sky .npz files folder path',
        default='train')
    parser.add_argument(
        '-data_dir_0',
        '--data_dir_0',
        help='sky .npz files folder path of images 0',
        default=None)
    parser.add_argument(
        '-data_dir_1',
        '--data_dir_1',
        help='sky .npz files folder path of images 1',
        default=None) 

    parser.add_argument(
        '-start',
        '--start',
        type=int,
        help=' input DF start index',
        default=0)

    parser.add_argument('-n', '--n', type=int, help='no. of  pairs', default=0)

    parser.add_argument(
        '-pe_skymaps',
        '--pe_skymaps',
        help='use PE skymaps 1/0',
        type=int,
        default=0)
    args = parser.parse_args()
    print('\n Arguments used:- \n')

    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))

    data_dir = args.data_dir + '/'
    n = args.n
    infile = args.infile
    start = args.start
    pe_skymaps = args.pe_skymaps
    outfile = args.outfile
    data_dir_0 = args.data_dir_0
    data_dir_1 = args.data_dir_1
    _main(data_dir,start, n,infile,outfile,pe_skymaps,data_dir_0,data_dir_1)
    
def _main(data_dir,start, n,infile,outfile,pe_skymaps, data_dir_0=None,data_dir_1=None):
    if n == 0:
        df = pd.read_csv(infile, index_col=[0])[start:]
        print(len(df['img_0']), ' event pairs ')
    else:
        df = pd.read_csv(infile, index_col=[0])[
            start:start + n]

    dl = 1000
    l = len(df.img_0.values)

    if pe_skymaps == 0:
        data_mode_xgb = 'current'
        df['bayestar_skymaps_blu'] = ''
        df['bayestar_skymaps_d2'] = ''
        df['bayestar_skymaps_d3'] = ''
        df['bayestar_skymaps_lsq'] = ''
    else:
        data_mode_xgb = 'pe'
        df['pe_skymaps_blu'] = ''
        df['pe_skymaps_d2'] = ''
        df['pe_skymaps_d3'] = ''
        df['pe_skymaps_lsq'] = ''

    for i in range(0, l, dl):
        if i + dl < l:
            print(i)
            features, xgb_sky_labels, df[i:i + dl], missing_ids = ml.generate_skymaps_fm(
                df[i:i + dl]).XGBoost_input_matrix(data_mode_xgb=data_mode_xgb, data_dir=data_dir,data_dir_0=data_dir_0,data_dir_1=data_dir_1)

        else:
            features, xgb_sky_labels, df[i:l], missing_ids = ml.generate_skymaps_fm(
                df[i:l]).XGBoost_input_matrix(data_mode_xgb=data_mode_xgb, data_dir=data_dir,data_dir_0=data_dir_0,data_dir_1=data_dir_1)

    print(df.tail())
    df.to_csv(outfile)


if __name__ == '__main__':
    main()
