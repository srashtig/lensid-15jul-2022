nohup python QTs_ML.py -infile O3a_psd/unlensed.csv -data_dir O3a_psd -outfile O3a_psd/unlensed_QTs.csv -dense_models_dir ../../saved_models/ >a.out &


nohup python QTs_ML.py -infile O3a_psd/unlensed.csv -data_dir O3a_psd -outfile O3a_psd/unlensed_QTs_kaggle.csv -dense_models_dir ../training_cv/dense_out/kaggle/ >a1.out &


nohup python skymaps_ML.py -infile O3a_psd/unlensed.csv -data_dir O3a_psd -outfile O3a_psd/unlensed_sky.csv >a2.out &

