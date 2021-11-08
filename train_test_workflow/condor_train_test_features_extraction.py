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


    error = os.path.abspath(base_out_dir + '/condor/error')
    output = os.path.abspath(base_out_dir + '/condor/output')
    log = os.path.abspath(base_out_dir + '/condor/log')
    submit = os.path.abspath(base_out_dir + '/condor/submit')

    dagman = Dagman(name='features_extraction_dagman',
                    submit=submit)
    if calc_features_custom == 0:
        if not os.path.exists(base_out_dir + df_dir_test_features_out):
            os.makedirs(base_out_dir + df_dir_test_features_out)
        if not os.path.exists(base_out_dir + df_dir_train_features_out):
            os.makedirs(base_out_dir + df_dir_train_features_out)
        
        if (calc_features_sky ==1) and (train_features_extract ==1):
            print('Calculating training set sky features... with following arguments')
            for df in train_dfs_dict.keys() :
                print('train ', df)
                arguments = '-infile %s -outfile %s -data_dir %s  -n %d '%(df_dir_train + df+'.csv',(base_out_dir+ df_dir_train_features_out+ df+'_sky'+ tag_sky_out+'.csv'),data_dir_sky_train, train_dfs_dict[df])

                print(arguments)
                job = Job(
                    name='sky_features_train_' + df,
                    executable=exec_file_loc + 'lensid_get_features_sky_ml',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='4GB')
                dagman.add_job(job)

        if (calc_features_sky ==1) and (test_features_extract ==1):
            print('Calculating testing set sky features... with following arguments')
            for df in test_dfs_dict.keys() :
                print('test ', df)
                arguments = '-infile %s -outfile %s -data_dir %s -n %d '%(df_dir_test+ df+ '.csv',(base_out_dir + df_dir_test_features_out+ df+'_sky'+ tag_sky_out+ '.csv'), data_dir_sky_test, test_dfs_dict[df])
                print(arguments)
                job = Job(
                    name='sky_features_test_' + df,
                    executable=exec_file_loc + 'lensid_get_features_sky_ml',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='4GB')
                dagman.add_job(job)


        if (cal_features_qts == 1 and (train_features_extract ==1)):
            print('Calculating training set Qtransform features... with following arguments:')
            for df in train_dfs_dict.keys():
                print('train ', df)
                arguments = '--infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %d -n %d' % (df_dir_train + df + '.csv', (
                    base_out_dir + df_dir_train_features_out + df + '_qts' + tag_qts_out + '.csv'), data_dir_qts_train, dense_model_dir_in, whitened, train_dfs_dict[df])
                print(arguments)
                job = Job(
                    name='qts_features_train_' + df,
                    executable=exec_file_loc + 'lensid_get_features_qts_ml',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='4GB')
                dagman.add_job(job)

        if (cal_features_qts == 1 and (test_features_extract ==1)):
            print('Calculating testing set Qtransform features... with following arguments:')
            for df in test_dfs_dict.keys():
                print('test ', df)
                arguments = '-infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %d -n %d '%(df_dir_test+df + '.csv',(base_out_dir+ df_dir_test_features_out+ df+ '_qts'+ tag_qts_out +'.csv'),data_dir_qts_test, dense_model_dir_in, whitened, test_dfs_dict[df])
                print(arguments)
                job = Job(
                    name='qts_features_test_' + df,
                    executable=exec_file_loc + 'lensid_get_features_qts_ml',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='4GB')
                dagman.add_job(job)
    #custom
    
    if (calc_features_sky ==1) and (calc_features_custom ==1):
        print('Calculating custom given set sky features... with following arguments')

        arguments = '-infile %s -outfile %s -data_dir %s  -n %d '%(in_dir+custom_df+'.csv',(base_out_dir+ out_df_dir+custom_df+'_sky'+ tag_custom_sky_out+'.csv'),data_dir_sky_custom, num_pairs)

        print(arguments)
        job = Job(
            name='sky_features_custom_' + custom_df,
            executable=exec_file_loc + 'lensid_get_features_sky_ml',
            submit=submit,
            error=error,
            output=output,
            log=log,
            arguments=arguments,
            universe='vanilla',
            getenv=True,
            extra_lines=['accounting_group = ' + accounting_tag],
            request_memory='4GB')
        dagman.add_job(job)
            
    if (cal_features_qts == 1 and (calc_features_custom ==1)):
        print('Calculating custom given set Qtransform features... with following arguments:')
        arguments = '--infile %s -outfile %s -data_dir %s -dense_models_dir %s -whitened %d -n %d' % (in_dir+custom_df + '.csv', (
                base_out_dir + out_df_dir+ custom_df + '_qts' + tag_custom_qts_out + '.csv'), data_dir_qts_custom, dense_model_dir_in, whitened, num_pairs)
        print(arguments)
        job = Job(
            name='qts_features_custom_' + custom_df,
            executable=exec_file_loc + 'lensid_get_features_qts_ml',
            submit=submit,
            error=error,
            output=output,
            log=log,
            arguments=arguments,
            universe='vanilla',
            getenv=True,
            extra_lines=['accounting_group = ' + accounting_tag],
            request_memory='4GB')
        dagman.add_job(job)
                
                
    if submit_dag ==1:       
        'Dag created submitting the jobs...'
        dagman.build_submit()
        
    else:
        dagman.build()
        print('\n \n Dag saved at: %s'%(submit +'/' + dagman.name))
if __name__ == "__main__":
    main()
