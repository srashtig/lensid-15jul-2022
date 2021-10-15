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
from lensid.train_test import train_densenets_qts 
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
            #_main(odir, data_dir, lensed_df, unlensed_df, size_lensed, size_unlensed, det, epochs, lr, whitened)
            train_densenets_qts._main(base_out_dir + dense_model_dir_out,data_dir_qts_train, df_dir_train + lensed_df, df_dir_train + unlensed_df, size_lensed, size_unlensed, 'H1', epochs, h1_lr, whitened)
            
        if train_l1 == 1:
            train_densenets_qts._main(base_out_dir + dense_model_dir_out,data_dir_qts_train, df_dir_train + lensed_df, df_dir_train + unlensed_df, size_lensed, size_unlensed, 'L1', epochs, l1_lr, whitened)
            
        if train_v1 == 1:
            train_densenets_qts._main(base_out_dir + dense_model_dir_out,data_dir_qts_train, df_dir_train + lensed_df, df_dir_train + unlensed_df, size_lensed, size_unlensed, 'V1', epochs, v1_lr, whitened)


if __name__ == "__main__":
    main()
