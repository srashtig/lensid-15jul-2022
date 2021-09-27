import os
base_out_dir = '/home/srashti.goyal/lensid_runs/ML_2p0'

data_gen_train = 1
data_gen_test = 1
data_gen_lensed = 1
data_gen_unlensed = 1
data_gen_dfs = 1
data_gen_sky = 1
data_gen_qts = 1
data_gen_custom = 0

#train-lensed
train_lensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/training/dominik_plaw2_lensed_inj_params_include_pol_phi0_refined.npz'
start_idx_train_lensed = 0
num_train_lensed = 2813

#train-unlensed
start_idx_train_unlensed = 0
num_train_unlensed =1000
train_unlensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/training/analytical_psd_Dominik_powerlaw2_inj_samples_withsnr_refined.npz'

#test-lensed
test_lensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/haris-et-al/lensed_inj_data.npz'
start_idx_test_lensed = 0 
num_test_lensed = 300

#test-unlensed
test_unlensed_inj_pars = '/home/srashti.goyal/lensid/data/injection_pars/haris-et-al/unlensed_inj_data.npz'
start_idx_test_unlensed = 0 
num_test_unlensed = 1004

#qts
out_dir_qts = '/data/qts'
asd_dir_txt = '/home/srashti.goyal/lensid/data/PSDs/O3a_representative_psd'
whitened = 1
qmode = 2 #(1 -> q = (3,7 m1>60; 4,8 m1<60)  , 2 -> q = (3,30))
psd_mode = 2  #(1: analytical 2: from asd_dir_txt)

#dataframes
out_dir_dfs = '/data/dataframes'

#skymaps
out_dir_sky = '/data/bayestar_skymaps'
psd_xml = '/home/srashti.goyal/lensid/data/PSDs/O3a_representative_psd/O3a_representative.xml'

#custom
inj_pars = ''
is_lensed = 0
start_idx = 0
num_injs = 10

if data_gen_qts == 1:
    
    if not os.path.exists(base_out_dir+out_dir_qts):
            os.makedirs(base_out_dir+out_dir_qts)

    if (data_gen_lensed == 1) and (data_gen_train == 1) :
        print('## Generating lensed events QTs for training... ## ')

        os.system('nohup lensid_create_qts_lensed_injs -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d >%s.out &'%(base_out_dir+out_dir_qts+'/train/lensed', start_idx_train_lensed, num_train_lensed, train_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened, base_out_dir+ out_dir_qts+'/train_lensed_qts' ))
        
    if (data_gen_unlensed == 1) and (data_gen_train == 1) :
        print('## Generating unlensed events QTs for training... ## ')

        os.system('nohup lensid_create_qts_unlensed_injs -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 1 -whitened %d >%s.out &'%(base_out_dir+out_dir_qts+'/train/unlensed', start_idx_train_unlensed, num_train_unlensed, train_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened, base_out_dir+ out_dir_qts+'/train_unlensed_qts' ))

    if (data_gen_lensed == 1) and (data_gen_test == 1) :
        print('## Generating lensed events QTs for testing... ## ')

        os.system('nohup lensid_create_qts_lensed_injs -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 2 -whitened %d >%s.out &'%(base_out_dir+out_dir_qts+'/test/lensed', start_idx_test_lensed, num_test_lensed, test_lensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened, base_out_dir+ out_dir_qts+'/test_lensed_qts' ))

    if (data_gen_unlensed == 1) and (data_gen_test == 1) :
        print('## Generating unlensed events QTs for testing... ## ')

        os.system('nohup lensid_create_qts_unlensed_injs -odir %s -start %d -n %d -infile  %s -psd_mode %d -asd_dir %s -qrange %d -mode 2 -whitened %d >%s.out &'%(base_out_dir+out_dir_qts+'/test/unlensed', start_idx_test_unlensed, num_test_unlensed, test_unlensed_inj_pars, psd_mode, asd_dir_txt, qmode, whitened, base_out_dir+ out_dir_qts+'/test_unlensed_qts' ))


if data_gen_dfs == 1:
    
    if not os.path.exists(base_out_dir+out_dir_dfs):
            os.makedirs(base_out_dir+out_dir_dfs)

    if (data_gen_lensed == 1) and (data_gen_train == 1) :
        print('## Generating lensed events dataframe for training... ## ')
        os.system('lensid_create_lensed_df -odir %s -outfile lensed.csv -start %d -n %d -infile %s'%(base_out_dir+out_dir_dfs+'/train', start_idx_train_lensed,num_train_lensed , train_lensed_inj_pars))
    if (data_gen_unlensed == 1) and (data_gen_train == 1) :
        print('## Generating unlensed events dataframe for training... ## ')
        os.system('lensid_create_unlensed_df -odir %s -outfile unlensed_half.csv -start %d -n %d -infile %s'%(base_out_dir+out_dir_dfs+'/train', start_idx_train_unlensed, int(num_train_unlensed/2) , train_unlensed_inj_pars))
        os.system('lensid_create_unlensed_df -odir %s -outfile unlensed_second_half.csv -start %d -n %d -infile %s'%(base_out_dir+out_dir_dfs+'/train', start_idx_train_unlensed+ int(num_train_unlensed/2), int(num_train_unlensed/2), train_unlensed_inj_pars))
        
    if (data_gen_lensed == 1) and (data_gen_test == 1) :
        print('## Generating lensed events dataframe for testing... ## ')
        os.system('lensid_create_lensed_df -odir %s -outfile lensed.csv -start %d -n %d -infile %s -mode 2'%(base_out_dir+out_dir_dfs+'/test', start_idx_test_lensed, num_test_lensed , test_lensed_inj_pars))
    if (data_gen_unlensed == 1) and (data_gen_test == 1) :
        print('## Generating unlensed events dataframe for testing... ## ')
        os.system('lensid_create_unlensed_df -odir %s -outfile unlensed.csv -start %d -n %d -infile %s'%(base_out_dir+out_dir_dfs+'/test', start_idx_test_unlensed, num_test_unlensed , test_unlensed_inj_pars))
        
    
#train-bayestar_skymaps
if data_gen_sky == 1:
    
    if not os.path.exists(base_out_dir+out_dir_sky):
            os.makedirs(base_out_dir+out_dir_sky)

    if (data_gen_lensed == 1) and (data_gen_train == 1) :
        print('## Generating lensed events skymaps for training... ## ')
        os.system('nohup lensid_create_bayestar_sky_lensed_injs.sh -o %s -s %d -n %d -i %s -p %s  >%s.out &'%(base_out_dir+out_dir_sky+'/train', start_idx_train_lensed, num_train_lensed, train_lensed_inj_pars, psd_xml, base_out_dir+ out_dir_sky+'/train_lensed_sky' ))
        
    if (data_gen_unlensed == 1) and (data_gen_train == 1) :
        print('## Generating unlensed events skymaps for training... ## ')
        os.system('nohup lensid_create_bayestar_sky_unlensed_injs.sh -o %s -s %d -n %d -i %s -p %s  >%s.out &'%(base_out_dir+out_dir_sky+'/train', start_idx_train_unlensed, num_train_unlensed, train_unlensed_inj_pars, psd_xml, base_out_dir+ out_dir_sky+'/train_unlensed_sky' ))
        
    if (data_gen_lensed == 1) and (data_gen_test == 1) :
        print('## Generating lensed events skymaps for testing... ## ')
        os.system('nohup lensid_create_bayestar_sky_lensed_injs.sh -o %s -s %d -n %d -i %s -p %s  >%s.out &'%(base_out_dir+out_dir_sky+'/test', start_idx_test_lensed, num_test_lensed, test_lensed_inj_pars, psd_xml, base_out_dir+ out_dir_sky+'/test_lensed_sky' ))
    
    if (data_gen_unlensed == 1) and (data_gen_test == 1) :
        print('## Generating unlensed events skymaps for testing... ## ')
        os.system('nohup lensid_create_bayestar_sky_unlensed_injs.sh -o %s -s %d -n %d -i %s -p %s  >%s.out &'%(base_out_dir+out_dir_sky+'/test', start_idx_test_unlensed, num_test_unlensed, test_unlensed_inj_pars, psd_xml, base_out_dir+ out_dir_sky+'/test_unlensed_sky' ))
        
#test-bayestar_skymaps

