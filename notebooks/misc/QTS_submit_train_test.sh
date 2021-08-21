tag="kaggle"
nohup python QTs_ML.py -infile train/lensed.csv -data_dir train -outfile train/lensed_QTs_${tag}.csv -dense_models_dir ../training_cv/dense_out/${tag}/ >a.out &

nohup python QTs_ML.py -infile train/unlensed_half.csv -data_dir train -outfile train/unlensed_half_QTs_${tag}.csv -dense_models_dir ../training_cv/dense_out/${tag}/ >a1.out &


nohup python QTs_ML.py -infile train/unlensed_second_half.csv -data_dir train -outfile train/unlensed_second_half_QTs_${tag}.csv -dense_models_dir ../training_cv/dense_out/${tag}/ >a2.out &

#nohup python QTs_ML.py -infile test/lensed.csv -data_dir test -outfile test/lensed_QTs_${tag}.csv -dense_models_dir ../training_cv/dense_out/${tag}/ >b1.out &

#nohup python QTs_ML.py -infile test/unlensed.csv -data_dir test -outfile test/unlensed_QTs_${tag}.csv -dense_models_dir ../training_cv/dense_out/${tag}/ >b2.out &

