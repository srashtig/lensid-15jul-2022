import os
import argparse
import configparser

config = configparser.ConfigParser()
config.read('train_test_wrokflow.ini')

# train densenets
# feature extraction QT
# feature extraction Skymaps

# train_test_cv QTs
# train_test_cv Skymaps
# test_combined_ml

# investigations?


import os
import pandas as pd
import argparse
import lensid.utils.ml_utils as ml
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pylab as plt
import numpy as np
import matplotlib
import yaml


def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for training, testing with cross validation and optionally compare to blu the machine learning models for the lensing identification, given the features extracted out of QTs and skymaps of simulated lensed and unlensed event pairs.')
    parser.add_argument('-config','--config', help='input CONFIG.yaml file',default='config_O3_events.yaml')
    args = parser.parse_args()
  
    def set_var(var_name, value):
        globals()[var_name] = value
        
    stream = open(args.config, 'r')
    dictionary = yaml.load_all(stream)

    for doc in dictionary:
        for key, value in doc.items():
            print(key + " : " + str(value))
            set_var(key,value)
    if tag_sky_in == '':
        tag_sky = 'None'
    else:
        tag_sky = tag_sky_in
        
    if tag_qts_in == '':
        tag_qts = 'None'
    else:
        tag_qts = tag_qts_in

        
    if (train_test_qts == True): 
        print('\n ##   ML QTs ...  ## \n')
        os.system('python train_crossvalidate_test_XGB_qts.py -df_dir_train %s -df_dir_test %s -odir %s -tag %s -compare_to_blu %s -path_to_blu %s -train_size_lensed %d -cv_size_lensed %d -cv_splits %d -scale_pos_weight %f -max_depth %d -n_estimators %d'%(df_dir_train_features_in, df_dir_test_features_in, base_out_dir, tag_qts, compare_to_blu, path_to_blu, train_size_lensed_xgbqts, cv_size_lensed_xgbqts, cv_splits_xgbqts, scale_pos_weight_xgbqts, max_depth_xgbqts, n_estimators_xgbqts))
    if (train_test_sky == True): 
        print('\n ##  ML Skymaps ... ## \n')
        os.system('python train_crossvalidate_test_XGB_sky.py -df_dir_train %s -df_dir_test %s -odir %s -tag %s -compare_to_blu %s -path_to_blu %s -train_size_lensed %d -cv_size_lensed %d -cv_splits %d -scale_pos_weight %f -max_depth %d -n_estimators %d'%(df_dir_train_features_in, df_dir_test_features_in, base_out_dir, tag_sky, compare_to_blu, path_to_blu, train_size_lensed_xgbsky, cv_size_lensed_xgbsky, cv_splits_xgbsky, scale_pos_weight_xgbsky, max_depth_xgbsky, n_estimators_xgbsky))

    if (test_combined_ml == True):
        if (train_test_qts == True) and (train_test_sky == True ) : 
            indir_combined = base_out_dir + '/dataframes'
        else:
            indir_combined = indir_df
        print('\n ## Testing Combined ML ## \n')
        os.system('python test_combined_ML_results.py -indir %s -tag_sky %s -tag_qts %s -odir %s -compare_to_blu %s'%(indir_combined,tag_sky,tag_qts,base_out_dir,compare_to_blu))
        
    
if __name__ == "__main__":
    main()