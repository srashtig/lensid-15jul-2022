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


os.system('python train_densenets_QTs.py -lensed_df ~/strong-lensing-ml/data/dataframes/train/lensed.csv -unlensed_df ~/strong-lensing-ml/data/dataframes/train/unlensed_half.csv -odir dense_out/cit/ -epochs 10 -data_dir ~/alice_data_lensid/qts/train/')

os.system('lensid_get_features_qts_ml -infile check/lensed.csv -outfile check/lensed_QTs.csv -dense_models_dir ~/lensid/saved_models/ -data_dir check')

os.system('python train_crossvalidate_test_XGB_sky.py')
print('xgbsky done')
os.system('python train_crossvalidate_test_XGB_qts.py')
print('xgbqt done')

os.system('python test_combined_ML_results.py')


print('ml combined done')
