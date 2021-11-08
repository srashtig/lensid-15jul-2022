import os
import argparse
import yaml
from pycondor import Dagman, Job

def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for generating Qtranforms, bayestar skymaps and dataframes for the lensed and the unlensed injections given a config file. The data can be used for training, testing and background computation.')
    parser.add_argument(
        '-config',
        '--config',
        help='input CONFIG.yaml file',
        required=True)
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

    dagman = Dagman(name='data_gen_dagman',
                    submit=submit)

    if data_gen_qts == 1:

        if not os.path.exists(base_out_dir + out_dir_qts):
            os.makedirs(base_out_dir + out_dir_qts)
            
        if data_gen_custom == 0:

            if (data_gen_lensed == 1) and (data_gen_train == 1):

                print('## Generating lensed events QTs for training... ## ')
                arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' %(base_out_dir + out_dir_qts + '/train', start_idx_train_lensed, num_train_lensed, train_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
                print(arguments)
                job = Job(
                    name= 'qts_train_lensed',
                    executable=exec_file_loc + 'lensid_create_qts_lensed_injs',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='2GB')
                dagman.add_job(job)

            if (data_gen_unlensed == 1) and (data_gen_train == 1):
                print('## Generating unlensed events QTs for training... ## ')
                arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts + '/train', start_idx_train_unlensed,  num_train_unlensed, train_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
                print(arguments)
                job = Job(
                    name= 'qts_train_unlensed',
                    executable=exec_file_loc + 'lensid_create_qts_unlensed_injs',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='2GB')
                dagman.add_job(job)

            if (data_gen_lensed == 1) and (data_gen_test == 1):
                print('## Generating lensed events QTs for testing... ## ')
                arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 2 -whitened %d ' %(base_out_dir + out_dir_qts + '/test', start_idx_test_lensed, num_test_lensed, test_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
                print(arguments)
                job = Job(
                    name= 'qts_test_lensed',
                    executable=exec_file_loc + 'lensid_create_qts_lensed_injs',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='2GB')
                dagman.add_job(job)



            if (data_gen_unlensed == 1) and (data_gen_test == 1):
                print('## Generating unlensed events QTs for testing... ## ')
                arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts + '/test', start_idx_test_unlensed,  num_test_unlensed, test_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
                print(arguments)
                job = Job(
                    name= 'qts_test_unlensed',
                    executable=exec_file_loc + 'lensid_create_qts_unlensed_injs',
                    submit=submit,
                    error=error,
                    output=output,
                    log=log,
                    arguments=arguments,
                    universe='vanilla',
                    getenv=True,
                    extra_lines=['accounting_group = ' + accounting_tag],
                    request_memory='2GB')
                dagman.add_job(job)
                
            # custom
        if (data_gen_lensed == 1) and (data_gen_custom == 1):

            print('## Generating input lensed events QTs... ## ')
            arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' %(base_out_dir + out_dir_qts , start_idx, num_injs, inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
            print(arguments)
            job = Job(
                name= 'custom_lensed',
                executable=exec_file_loc + 'lensid_create_qts_lensed_injs',
                submit=submit,
                error=error,
                output=output,
                log=log,
                arguments=arguments,
                universe='vanilla',
                getenv=True,
                extra_lines=['accounting_group = ' + accounting_tag],
                request_memory='2GB')
            dagman.add_job(job)

        if (data_gen_unlensed == 1) and (data_gen_custom == 1):
            print('## Generating unlensed events QTs for training... ## ')
            arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts , start_idx,  num_injs, inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
            print(arguments)
            job = Job(
                name= 'custom_unlensed',
                executable=exec_file_loc + 'lensid_create_qts_unlensed_injs',
                submit=submit,
                error=error,
                output=output,
                log=log,
                arguments=arguments,
                universe='vanilla',
                getenv=True,
                extra_lines=['accounting_group = ' + accounting_tag],
                request_memory='2GB')
            dagman.add_job(job)

    if data_gen_dfs == 1:

        if not os.path.exists(base_out_dir + out_dir_dfs):
            os.makedirs(base_out_dir + out_dir_dfs)
        if data_gen_custom ==0:
            if (data_gen_lensed == 1) and (data_gen_train == 1):
                print('## Generating lensed events dataframe for training... ## ')
                os.system(
                    'lensid_create_lensed_df -odir %s -outfile lensed.csv -start %d -n %d -infile %s' %
                    (base_out_dir +
                     out_dir_dfs +
                     '/train',
                     start_idx_train_lensed,
                     num_train_lensed,
                     train_lensed_inj_pars))        

            if (data_gen_unlensed == 1) and (data_gen_train == 1):
                print('## Generating unlensed events dataframe for training... ## ')
                os.system(
                    'lensid_create_unlensed_df -odir %s -outfile unlensed_half.csv -start %d -n %d -infile %s' %
                    (base_out_dir +
                     out_dir_dfs +
                     '/train',
                     start_idx_train_unlensed,
                     int(
                         num_train_unlensed /
                         2),
                        train_unlensed_inj_pars))
                os.system('lensid_create_unlensed_df -odir %s -outfile unlensed_second_half.csv -start %d -n %d -infile %s' %
                          (base_out_dir +
                           out_dir_dfs +
                           '/train', start_idx_train_unlensed +
                           int(num_train_unlensed /
                               2), int(num_train_unlensed /
                                       2), train_unlensed_inj_pars))

            if (data_gen_lensed == 1) and (data_gen_test == 1):
                print('## Generating lensed events dataframe for testing... ## ')
                os.system(
                    'lensid_create_lensed_df -odir %s -outfile lensed.csv -start %d -n %d -infile %s -mode 2' %
                    (base_out_dir +
                     out_dir_dfs +
                     '/test',
                     start_idx_test_lensed,
                     num_test_lensed,
                     test_lensed_inj_pars))
            if (data_gen_unlensed == 1) and (data_gen_test == 1):
                print('## Generating unlensed events dataframe for testing... ## ')
                os.system(
                    'lensid_create_unlensed_df -odir %s -outfile unlensed.csv -start %d -n %d -infile %s' %
                    (base_out_dir +
                     out_dir_dfs +
                     '/test',
                     start_idx_test_unlensed,
                     num_test_unlensed,
                     test_unlensed_inj_pars))
                #custom
        if (data_gen_lensed == 1) and (data_gen_custom == 1):
                print('## Generating lensed events dataframe for input pars... ## ')
                os.system(
                    'lensid_create_lensed_df -odir %s -outfile lensed.csv -start %d -n %d -infile %s -mode 2' %
                    (base_out_dir +
                     out_dir_dfs ,
                     start_idx,
                     num_injs,
                     inj_pars))
        if (data_gen_unlensed == 1) and (data_gen_custom == 1):
            print('## Generating unlensed events dataframe for input pars... ## ')
            os.system(
                'lensid_create_unlensed_df -odir %s -outfile unlensed.csv -start %d -n %d -infile %s' %
                (base_out_dir +
                 out_dir_dfs,
                 start_idx,
                 num_injs,
                 inj_pars))

    # train-bayestar_skymaps
    if data_gen_sky == 1:

        if not os.path.exists(base_out_dir + out_dir_sky):
            os.makedirs(base_out_dir + out_dir_sky)
            
        if data_gen_custom == 0 :

            if (data_gen_lensed == 1) and (data_gen_train == 1):
                print('## Generating lensed events skymaps for training... ## ')
                arguments = ' -o %s -s %d -n %d -i %s -p %s ' % (base_out_dir +
                out_dir_sky +
                '/train',
                start_idx_train_lensed,
                num_train_lensed,
                train_lensed_inj_pars,
                psd_xml)

                print(arguments)
                job = Job(
                    name= 'sky_train_lensed',
                    executable=exec_file_loc + 'lensid_create_bayestar_sky_lensed_injs.sh',
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


            if (data_gen_unlensed == 1) and (data_gen_train == 1):
                print('## Generating unlensed events skymaps for training... ## ')
                arguments ='-o %s -s %d -n %d -i %s -p %s' % (base_out_dir +
                out_dir_sky +
                '/train',
                start_idx_train_unlensed,
                num_train_unlensed,
                train_unlensed_inj_pars,
                psd_xml)

                job = Job(
                    name= 'sky_train_unlensed',
                    executable=exec_file_loc + 'lensid_create_bayestar_sky_unlensed_injs.sh',
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

            if (data_gen_lensed == 1) and (data_gen_test == 1):
                print('## Generating lensed events skymaps for testing... ## ')
                arguments = ' -o %s -s %d -n %d -i %s -p %s ' %(base_out_dir +
                out_dir_sky +
                '/test',
                start_idx_test_lensed,
                num_test_lensed,
                test_lensed_inj_pars,
                psd_xml)

                print(arguments)
                job = Job(
                    name= 'sky_test_lensed',
                    executable=exec_file_loc + 'lensid_create_bayestar_sky_lensed_injs.sh',
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


            if (data_gen_unlensed == 1) and (data_gen_test == 1):
                print('## Generating unlensed events skymaps for testing... ## ')
                arguments ='-o %s -s %d -n %d -i %s -p %s' % (base_out_dir +
                out_dir_sky +
                '/test',
                start_idx_test_unlensed,
                num_test_unlensed,
                test_unlensed_inj_pars,
                psd_xml)

                job = Job(
                    name= 'sky_test_unlensed',
                    executable=exec_file_loc + 'lensid_create_bayestar_sky_unlensed_injs.sh',
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
        if (data_gen_lensed == 1) and (data_gen_custom == 1):
            print('## Generating lensed events skymaps for the input injs... ## ')
            arguments = ' -o %s -s %d -n %d -i %s -p %s ' %(base_out_dir +
            out_dir_sky,
            start_idx,
            num_injs,
            inj_pars,
            psd_xml)

            print(arguments)
            job = Job(
                name= 'sky_test_lensed',
                executable=exec_file_loc + 'lensid_create_bayestar_sky_lensed_injs.sh',
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


        if (data_gen_unlensed == 1) and (data_gen_custom == 1):
            print('## Generating unlensed events skymaps for the input injs... ## ')
            arguments ='-o %s -s %d -n %d -i %s -p %s' % (base_out_dir +
            out_dir_sky,
            start_idx,
            num_injs,
            inj_pars,
            psd_xml)

            job = Job(
                name= 'sky_custom_unlensed',
                executable=exec_file_loc + 'lensid_create_bayestar_sky_unlensed_injs.sh',
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
