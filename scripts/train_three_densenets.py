import yaml
import matplotlib
import numpy as np
import matplotlib.pylab as plt
import configparser
import os
import pandas as pd
import argparse
import lensid.utils.ml_utils as ml
import joblib
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for training the three densenets for H1, L1, V1 detectors, given the simulated lensed and unlensed event pairs, their QTransform images for each detector.')
    parser.add_argument(
        '-config',
        '--config',
        help='input CONFIG.yaml file',
        default='config_train_test_workflow.yaml')
    args = parser.parse_args()

    def set_var(var_name, value):
        globals()[var_name] = value

    stream = open(args.config, 'r')
    dictionary = yaml.load_all(stream)

    for doc in dictionary:
        for key, value in doc.items():
            print(key + " : " + str(value))
            set_var(key, value)

    if train_densenets == 1:
        print('\n ##   Training Densenets ...  ## \n')

        if train_h1 == 1:
            os.system(
                'python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %d -size_lensed %d -size_unlensed %d -lr %f -det H1' %
                (df_dir_train +
                 lensed_df,
                 df_dir_train +
                 unlensed_df,
                 base_out_dir +
                 dense_model_dir_out,
                 epochs,
                 data_dir_qts_train,
                 whitened,
                 size_lensed,
                 size_unlensed,
                 h1_lr))
        if train_l1 == 1:
            os.system(
                'python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %d -size_lensed %d -size_unlensed %d -lr %f -det L1' %
                (df_dir_train +
                 lensed_df,
                 df_dir_train +
                 unlensed_df,
                 base_out_dir +
                 dense_model_dir_out,
                 epochs,
                 data_dir_qts_train,
                 whitened,
                 size_lensed,
                 size_unlensed,
                 l1_lr))
        if train_v1 == 1:
            os.system(
                'python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %d -size_lensed %d -size_unlensed %d -lr %f -det V1' %
                (df_dir_train +
                 lensed_df,
                 df_dir_train +
                 unlensed_df,
                 base_out_dir +
                 dense_model_dir_out,
                 epochs,
                 data_dir_qts_train,
                 whitened,
                 size_lensed,
                 size_unlensed,
                 v1_lr))


if __name__ == "__main__":
    main()
