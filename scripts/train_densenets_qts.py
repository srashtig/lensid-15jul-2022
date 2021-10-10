import lensid.utils.ml_utils as ml
import pandas as pd
import warnings
import os
import argparse
import numpy as np
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for training Densenets for H or L or V detector using the lensed and unlensed simulated events qtransforms')
    parser.add_argument(
        '-lensed_df',
        '--lensed_df',
        help='input lensed Dataframe path',
        default='train/lensed.csv')
    parser.add_argument(
        '-unlensed_df',
        '--unlensed_df',
        help='input unlensed Dataframe path',
        default='train/unlensed_half.csv')
    parser.add_argument(
        '-data_dir',
        '--data_dir',
        help='QTs images folder path',
        required=True)

    parser.add_argument('-whitened', '--whitened', help='1/0', default=0)
    parser.add_argument(
        '-odir',
        '--odir',
        help='output trained densenet models H1.h5, L1.h5, V1.h5 directory path ',
        required=1)
    parser.add_argument(
        '-det',
        '--det',
        help='which detector(H1 or L1 or V1)',
        default='H1')

    parser.add_argument(
        '-size_lensed',
        '--size_lensed',
        help='no. of lensed events to train on',
        type=int,
        default=1400)
    parser.add_argument(
        '-size_unlensed',
        '--size_unlensed',
        help='no. of unlensed events to train on',
        type=int,
        default=1400)
    parser.add_argument(
        '-epochs',
        '--epochs',
        help='no. of epochs to train for',
        type=int,
        default=20)
    parser.add_argument(
        '-lr',
        '--lr',
        help='initial learing rate for training',
        type=float,
        default=0.01)

    args = parser.parse_args()

    data_dir = args.data_dir

    odir = args.odir

    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Arguments used:- \n')

    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))

    df_lensed = pd.read_csv(args.lensed_df)
    df_lensed = df_lensed.drop(columns=['Unnamed: 0'])
    df_lensed['img_0'] = df_lensed['img_0'].values
    df_lensed['img_1'] = df_lensed['img_1'].values
    df_lensed = df_lensed[:args.size_lensed]
    df_unlensed = pd.read_csv(args.unlensed_df)
    df_unlensed = df_unlensed.drop(columns=['Unnamed: 0'])
    df_unlensed = df_unlensed.sample(
        frac=1, random_state=42).reset_index(
        drop=True)[
            :args.size_unlensed]
    df_train = pd.concat([df_lensed, df_unlensed], ignore_index=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    det = args.det
    X, y, missing_ids, df_train = ml.generate_resize_densenet_fm(df_train).DenseNet_input_matrix(
        det=det, data_mode_dense="current", data_dir=data_dir, phenom=1)

    dense_model_trained = ml.train_densenet(
        X, y, det, args.epochs, args.lr)  # 20,0.01, .005
    dense_model_trained.save(odir + det + '.h5')


if __name__ == "__main__":
    main()
