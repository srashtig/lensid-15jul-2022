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
    if not os.path.exists(base_out_dir+df_dir_test_features_out):
            os.makedirs(base_out_dir+df_dir_test_features_out)
    if not os.path.exists(base_out_dir+df_dir_train_features_out):
            os.makedirs(base_out_dir+df_dir_train_features_out)
    if calc_features_sky ==True:
        print('Calculating Sky features...') 
        for df in train_dfs_dict.keys() :
            print('train ', df)
            os.system('nohup lensid_get_features_sky_ml -infile %s -outfile %s -data_dir %s  -n %d >%s.out &'%(df_dir_train+df+'.csv',(base_out_dir+df_dir_train_features_out+df+'_sky'+tag_sky_out+'.csv'),data_dir_sky_train, train_dfs_dict[df],base_out_dir+'/train_'+df+'_sky'+tag_sky_out))
        for df in test_dfs_dict.keys() :
            print('test ', df)
            os.system('nohup lensid_get_features_sky_ml -infile %s -outfile %s -data_dir %s -n %d >%s.out &'%(df_dir_test+df+'.csv',(base_out_dir+df_dir_test_features_out+df+'_sky'+tag_sky_out+'.csv'),data_dir_sky_test ,test_dfs_dict[df],base_out_dir+'/test_'+df+'_sky'+tag_sky_out))

    if cal_features_qts == True:
        print('Calculating Qtransform features...') 
        for df in train_dfs_dict.keys() :
            print('train ', df)
            os.system('nohup lensid_get_features_qts_ml -infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %s -n %d >%s.out &'%(df_dir_train+df + '.csv',(base_out_dir+df_dir_train_features_out+df+'_qts'+tag_qts_out+'.csv'),data_dir_qts_train, dense_model_dir_in, whitened,train_dfs_dict[df], base_out_dir+'/train_'+df+'_qts'+tag_qts_out ))
        for df in test_dfs_dict.keys() :
            print('test ', df)
            os.system('nohup lensid_get_features_qts_ml -infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %s -n %d >%s.out &'%(df_dir_test+df + '.csv',(base_out_dir+df_dir_test_features_out+df+'_qts'+tag_qts_out+'.csv'),data_dir_qts_test, dense_model_dir_in, whitened, test_dfs_dict[df], base_out_dir+'/test_'+df+'_qts'+tag_qts_out))
    
if __name__ == "__main__":
    main()