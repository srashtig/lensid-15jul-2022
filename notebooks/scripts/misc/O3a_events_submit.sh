nohup python QTs_ML.py -infile O3a_events/filtered_mass_pairs.csv -data_dir gwtc-2 -outfile O3a_events/filtered_mass_pairs_QTs.csv -dense_models_dir ../../saved_models/ >a.out &

nohup python QTs_ML.py -infile O3a_events/filtered_mass_pairs.csv -data_dir gwtc-2 -outfile O3a_events/filtered_mass_pairs_QTs_kaggle.csv  -dense_models_dir ../training_cv/dense_out/kaggle/ >a1.out &

#nohup python skymaps_ML.py -infile O3a_events/filtered_sky_pairs.csv -data_dir gwtc-2 -outfile O3a_events/filtered_mass_pairs_sky.csv >a1.out &

#nohup python skymaps_ML.py -infile O3a_events/filtered_mass_pairs.csv -data_dir gwtc-2 -outfile O3a_events/filtered_mass_pairs_PE_sky.csv -pe_skymaps True >a2.out &

