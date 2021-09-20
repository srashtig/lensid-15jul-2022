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
    train_dfs = ['lensed','unlensed_half','unlensed_second_half']
    test_dfs = ['lensed','unlensed']
    if calc_features_sky ==True:
        print('Calculating Sky features...') 
        for df in train_dfs :
            os.system('nohup lensid_get_features_sky_ml -infile %s -outfile %s -data_dir %s >%s.out &'%(df_dir_train+df+'.csv',(df_dir_train_features+df+'_sky'+tag_sky+'.csv'),data_dir_sky_train,'train_'+df+'_sky'+tag_sky))
        for df in test_dfs :
            os.system('nohup lensid_get_features_sky_ml -infile %s -outfile %s -data_dir %s >%s.out &'%(df_dir_test+df+'.csv',(df_dir_test_features+df+'_sky'+tag_sky+'.csv'),data_dir_sky_test,'test_'+df+'_sky'+tag_sky))

    if cal_features_qts == True:
        print('Calculating Qtransform features...') 
        for df in train_dfs :
            os.system('nohup lensid_get_features_qts_ml -infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %s >%s.out &'%(df_dir_train+df + '.csv',(df_dir_train_features+df+'_qts'+tag_qts+'.csv'),data_dir_qts_train, dense_model_dir, whitened, work_dir+'/train_'+df+'_qts'+tag_qts))
        for df in test_dfs :
            os.system('nohup lensid_get_features_qts_ml -infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %s >%s.out &'%(df_dir_test+df + '.csv',(df_dir_test_features+df+'_qts'+tag_qts+'.csv'),data_dir_qts_test, dense_model_dir, whitened, work_dir+'/test_'+df+'_qts'+tag_qts))
    
if __name__ == "__main__":
    main()