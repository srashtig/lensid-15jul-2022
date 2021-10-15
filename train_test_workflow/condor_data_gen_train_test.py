import os
from pycondor import Dagman, Job

base_out_dir = '/home/srashti.goyal/lensid_runs/ML_2p0_AnalyticalPsd_check'
exec_file_loc = '/home/srashti.goyal/.conda/envs/igwn-py37-hanabi/bin/'
submit_dag = 1
accounting_tag = 'ligo.prod.o3.cbc.testgr.tiger'
data_gen_dfs = 1
data_gen_sky = 1
data_gen_qts = 1
data_gen_train = 1
data_gen_test = 1
data_gen_lensed = 1
data_gen_unlensed = 1
data_gen_custom = 0

# train-lensed
train_lensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/training/dominik_plaw2_lensed_inj_params_include_pol_phi0_refined.npz'
start_idx_train_lensed = 0
num_train_lensed = 10 # 2813

# train-unlensed
start_idx_train_unlensed = 0
num_train_unlensed = 10 #1000
train_unlensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/training/analytical_psd_Dominik_powerlaw2_inj_samples_withsnr_refined.npz'

# test-lensed
test_lensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/haris-et-al/lensed_inj_data.npz'
start_idx_test_lensed = 0
num_test_lensed = 10 #300

# test-unlensed
test_unlensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/haris-et-al/unlensed_inj_data.npz'
start_idx_test_unlensed = 0
num_test_unlensed = 10 #1000

# qts
out_dir_qts = '/data/qts'
asd_dir_txt = '/home/srashti.goyal/lensid/data/PSDs/O3a_representative_psd'
whitened = 1
qmode = 2  # (1 -> q = (3,7 m1>60; 4,8 m1<60)  , 2 -> q = (3,30))
psd_mode = 1  # (1: analytical 2: from asd_dir_txt)

# dataframes
out_dir_dfs = '/data/dataframes'

# skymaps
out_dir_sky = '/data/bayestar_skymaps'
#psd_xml = '/home/srashti.goyal/lensid/data/PSDs/O3a_representative_psd/O3a_representative.xml'
psd_xml = '/home/srashti.goyal/lensid/data/PSDs/analytical_psd.xml'


# custom
inj_pars = ''
start_idx = 0
num_injs = 10

def main():
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
                arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' %(base_out_dir + out_dir_qts + '/train/lensed', start_idx_train_lensed, num_train_lensed, train_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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
                arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts + '/train/unlensed', start_idx_train_unlensed,  num_train_unlensed, train_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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
                arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' %(base_out_dir + out_dir_qts + '/test/lensed', start_idx_test_lensed, num_test_lensed, test_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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
                arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts + '/test/unlensed', start_idx_test_unlensed,  num_test_unlensed, test_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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
            arguments = '-odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' %(base_out_dir + out_dir_qts + '/lensed', start_idx, num_injs, inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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
            arguments = ' -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d ' % (base_out_dir + out_dir_qts + '/unlensed', start_idx,  num_injs, inj_pars, psd_mode, asd_dir_txt, qmode, whitened)
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