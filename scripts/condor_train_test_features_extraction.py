import configparser
import os
import pandas as pd
import argparse
import joblib
import matplotlib.pylab as plt
import numpy as np
import matplotlib
import yaml
from pycondor import Dagman, Job


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
    if not os.path.exists(base_out_dir + df_dir_test_features_out):
        os.makedirs(base_out_dir + df_dir_test_features_out)
    if not os.path.exists(base_out_dir + df_dir_train_features_out):
        os.makedirs(base_out_dir + df_dir_train_features_out)

    error = os.path.abspath(base_out_dir + '/condor/error')
    output = os.path.abspath(base_out_dir + '/condor/output')
    log = os.path.abspath(base_out_dir + '/condor/log')
    submit = os.path.abspath(base_out_dir + '/condor/submit')

    dagman = Dagman(name='features_extraction_dagman',
                    submit=submit)
    '''

    if calc_features_sky ==1:
        print('Calculating Sky features...')
        for df in train_dfs_dict.keys() :
            print('train ', df)
            arguments = '-infile %s -outfile %s -data_dir %s  -n %d >%s.out &'%(df_dir_train+df+'.csv',(base_out_dir+df_dir_train_features_out+df+'_sky'+tag_sky_out+'.csv'),data_dir_sky_train, train_dfs_dict[df],base_out_dir+'/train_'+df+'_sky'+tag_sky_out)
            print(arguments)
            job_date = Job(name='sky_train',
               executable='lensid_get_features_sky_ml',
               submit=submit,
               error=error,
               output=output,
               log=log,
               dag=dagman)

            #os.system('nohup lensid_get_features_sky_ml )

        for df in test_dfs_dict.keys() :
            print('test ', df)
            os.system('nohup lensid_get_features_sky_ml -infile %s -outfile %s -data_dir %s -n %d >%s.out &'%(df_dir_test+df+'.csv',(base_out_dir+df_dir_test_features_out+df+'_sky'+tag_sky_out+'.csv'),data_dir_sky_test ,test_dfs_dict[df],base_out_dir+'/test_'+df+'_sky'+tag_sky_out))
'''
    exec_file = '/home/srashti.goyal/lensid/package/lensid/feature_extraction/lensid_get_features_qts_ml.py'
    exec_file_1 = '/home/srashti.goyal/.conda/envs/igwn-py37-hanabi/bin/lensid_get_features_qts_ml'
    execu = 'lensid_get_features_qts_ml'
    if cal_features_qts == 1:
        print('Calculating Qtransform features...')
        for df in train_dfs_dict.keys():
            print('train ', df)
            arguments = '--infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %d -n %d' % (df_dir_train + df + '.csv', (
                base_out_dir + df_dir_train_features_out + df + '_qts' + tag_qts_out + '.csv'), data_dir_qts_train, dense_model_dir_in, whitened, train_dfs_dict[df])
            print(arguments)
            job = Job(
                name='qts_features_train_' + df,
                executable=exec_file_1,
                submit=submit,
                error=error,
                output=output,
                log=log,
                arguments=arguments,
                universe='vanilla',
                getenv=True,
                extra_lines=['accounting_group = ligo.prod.o3.cbc.testgr.tiger'],
                request_memory='2GB')
            dagman.add_job(job)

        for df in test_dfs_dict.keys():
            print('test ', df)
        #    os.system('nohup lensid_get_features_qts_ml -infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %d -n %d >%s.out &'%(df_dir_test+df + '.csv',(base_out_dir+df_dir_test_features_out+df+'_qts'+tag_qts_out+'.csv'),data_dir_qts_test, dense_model_dir_in, whitened, test_dfs_dict[df], base_out_dir+'/test_'+df+'_qts'+tag_qts_out))
    dagman.build_submit()


if __name__ == "__main__":
    main()
