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
    if tag_sky == '':
        tag_sky_in = 'None'
    else:
        tag_sky_in = tag_sky
        
    if tag_qts == '':
        tag_qts_in = 'None'
    else:
        tag_qts_in = tag_qts
    
    if train_densenets == True: 
        print('\n ##   Training Densenets ...  ## \n')

        if train_h1 == True:
            os.system('python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %s -size_lensed %d -size_unlensed %d -lr %f -det H1'%(df_dir_train+lensed_df,df_dir_train+unlensed_df, dense_model_dir, epochs,data_dir_qts_train, whitened, size_lensed, size_unlensed, h1_lr ))
        if train_l1 == True:
            os.system('python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %s -size_lensed %d -size_unlensed %d -lr %f -det L1'%(df_dir_train+lensed_df,df_dir_train+unlensed_df, dense_model_dir, epochs,data_dir_qts_train, whitened, size_lensed, size_unlensed, l1_lr ))
        if train_v1 == True:
            os.system('python train_densenets_qts.py -lensed_df %s -unlensed_df %s -odir %s -epochs %d  -data_dir %s -whitened %s -size_lensed %d -size_unlensed %d -lr %f -det V1'%(df_dir_train+lensed_df,df_dir_train+unlensed_df, dense_model_dir, epochs,data_dir_qts_train, whitened, size_lensed, size_unlensed, v1_lr ))
        
    if (train_test_qts == True): 
        print('\n ##   ML QTs ...  ## \n')
        os.system('python train_crossvalidate_test_XGB_qts.py -df_dir_train %s -df_dir_test %s -odir %s -tag %s -compare_to_blu %s -path_to_blu %s -train_size_lensed %d -cv_size_lensed %d -cv_splits %d -scale_pos_weight %f -max_depth %d -n_estimators %d'%(df_dir_train_features, df_dir_test_features, work_dir, tag_qts_in, compare_to_blu, path_to_blu, train_size_lensed_xgbqts, cv_size_lensed_xgbqts, cv_splits_xgbqts, scale_pos_weight_xgbqts, max_depth_xgbqts, n_estimators_xgbqts))
    if (train_test_sky == True): 
        print('\n ##  ML Skymaps ... ## \n')
        os.system('python train_crossvalidate_test_XGB_sky.py -df_dir_train %s -df_dir_test %s -odir %s -tag %s -compare_to_blu %s -path_to_blu %s -train_size_lensed %d -cv_size_lensed %d -cv_splits %d -scale_pos_weight %f -max_depth %d -n_estimators %d'%(df_dir_train_features, df_dir_test_features, work_dir, tag_sky_in, compare_to_blu, path_to_blu, train_size_lensed_xgbsky, cv_size_lensed_xgbsky, cv_splits_xgbsky, scale_pos_weight_xgbsky, max_depth_xgbsky, n_estimators_xgbsky))

    if (test_combined_ml == True):
        if (train_test_qts == True) and (train_test_sky == True ) : 
            indir_combined = work_dir + '/dataframes'
        else:
            indir_combined = indir_df
        print('\n ## Testing Combined ML ## \n')
        os.system('python test_combined_ML_results.py -indir %s -tag_sky %s -tag_qts %s -odir %s -compare_to_blu %s'%(indir_combined,tag_sky_in,tag_qts_in,work_dir,compare_to_blu))
        
    
if __name__ == "__main__":
    main()