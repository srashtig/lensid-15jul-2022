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
import train_crossvalidate_test_XGB_qts , train_crossvalidate_test_XGB_sky, test_combined_ML_results

def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for training, testing with cross validation and optionally compare to blu the machine learning models for the lensing identification, given the features extracted out of QTs and skymaps of simulated lensed and unlensed event pairs.')
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
    if tag_sky_in == '':
        tag_sky = 'None'
    else:
        tag_sky = tag_sky_in

    if tag_qts_in == '':
        tag_qts = 'None'
    else:
        tag_qts = tag_qts_in

    if (train_test_qts == 1):
        print('\n ##   ML QTs ...  ## \n')
        #_main(odir,df_dir_train,df_dir_test,tag, train_size_lensed, cv_size_lensed, scale_pos_weight, max_depth, n_estimators, cv_splits, compare_to_blu, path_to_blu )
        train_crossvalidate_test_XGB_qts._main(base_out_dir,df_dir_train_features_in, df_dir_test_features_in, tag_qts,train_size_lensed_xgbqts, cv_size_lensed_xgbqts, scale_pos_weight_xgbqts, max_depth_xgbqts, n_estimators_xgbqts, cv_splits, compare_to_blu, path_to_blu)
    if (train_test_sky == 1):
        print('\n ##  ML Skymaps ... ## \n')
        #_main(odir,df_dir_train,df_dir_test,tag, train_size_lensed, cv_size_lensed, scale_pos_weight, max_depth, n_estimators, cv_splits, compare_to_blu, path_to_blu)
        train_crossvalidate_test_XGB_sky._main(base_out_dir,df_dir_train_features_in, df_dir_test_features_in, tag_sky,train_size_lensed_xgbsky, cv_size_lensed_xgbsky, scale_pos_weight_xgbsky, max_depth_xgbsky, n_estimators_xgbsky, cv_splits, compare_to_blu, path_to_blu)
                
    if (test_combined_ml == 1):
        if (train_test_qts == 1) and (train_test_sky == 1):
            indir_combined = base_out_dir + '/dataframes'
        else:
            indir_combined = indir_df
        print('\n ## Testing Combined ML ## \n')
        #_main(odir,indir, tag_sky, tag_qts, cv_splits, compare_to_blu)
        test_combined_ML_results._main(base_out_dir, indir_combined, tag_sky, tag_qts, cv_splits, compare_to_blu)


if __name__ == "__main__":
    main()
