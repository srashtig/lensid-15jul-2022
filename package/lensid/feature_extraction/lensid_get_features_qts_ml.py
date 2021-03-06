import sys
import argparse
import lensid.utils.ml_utils as ml
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for calculating 3-det QTs features using trained densenets and QTs, for given even pairs')
    parser.add_argument(
        '-infile',
        '--infile',
        help='input Dataframe path',
        default='train/lensed.csv')
    parser.add_argument(
        '-outfile',
        '--outfile',
        help='output Dataframe path ',
        default='train/lensed_QTs.csv')
    parser.add_argument(
        '-data_dir',
        '--data_dir',
        help='QTs images folder path',
        default='train')
    parser.add_argument(
        '-data_dir_0',
        '--data_dir_0',
        help='QTs images 0 folder path',
        default=None)
    parser.add_argument(
        '-data_dir_1',
        '--data_dir_1',
        help='QTs images 1 folder path',
        default=None)    

    parser.add_argument(
        '-start',
        '--start',
        type=int,
        help=' input DF start index',
        default=0)
    parser.add_argument('-n', '--n', type=int, help='no. of  pairs', default=0)
    parser.add_argument(
        '-whitened',
        '--whitened',
        type=int,
        help='1/0',
        default=0)
    parser.add_argument(
        '-dense_models_dir',
        '--dense_models_dir',
        help='trained densenets H1.h5, L1.h5, V1.h5 directory path ',
        required=1)

    parser.add_argument(
        '-model_id',
        '--model_id',
        help='model id to include in output DF columns',
        default=0)
    args = parser.parse_args()
    print('\n Arguments used:- \n')

    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))

    print(args.dense_models_dir)

    data_dir = args.data_dir
    n = args.n
    infile = args.infile
    start = args.start
    outfile = args.outfile
    dense_models_dir = args.dense_models_dir
    model_id =  args.model_id
    whitened = args.whitened
    data_dir_0 = args.data_dir_0
    data_dir_1 = args.data_dir_1
    _main(data_dir,n,infile,outfile,start, dense_models_dir, model_id, whitened,data_dir_0,data_dir_1)
def _main(data_dir,n,infile,outfile,start, dense_models_dir, model_id, whitened,data_dir_0=None,data_dir_1=None):
    
    dense_model_H1 = ml.load_model(dense_models_dir + 'H1.h5')
    dense_model_L1 = ml.load_model(dense_models_dir + 'L1.h5')
    dense_model_V1 = ml.load_model(dense_models_dir + 'V1.h5')
    if n == 0:
        df = pd.read_csv(infile, index_col=[0])[start:]
        print(len(df['img_0']), ' event pairs ')
    else:
        df = pd.read_csv(infile, index_col=[0])[
            start:start + n]

    dl = 1000
    l = len(df.img_0.values)
    dets = ['H1', 'L1', 'V1']
    models = [dense_model_H1, dense_model_L1, dense_model_V1]
    for m, det in enumerate(dets):
        model = models[m]
        df['dense_' + det + '_' + str(model_id)] = ''
        df['mean_overlap_qts_' + det], df['std_overlap_qts_' +
                                          det], df['lsq_overlap_qts_' + det] = '', '', ''
        for i in range(0, l, dl):
            if i + dl < l:
                print(i)
                X, y, missing_ids, df[i:i + dl] = ml.generate_resize_densenet_fm(df[i:i + dl]).DenseNet_input_matrix(
                    det=det, data_mode_dense="current", data_dir=data_dir, phenom=1, whitened=whitened, data_dir_0 = data_dir_0, data_dir_1 = data_dir_1 )
                df['dense_' +
                   det +
                   '_' +
                   str(model_id)].values[i:i +
                                              dl] = ml.Dense_predict(model, df[i:i +
                                                                               dl], X, missing_ids)[:, 0]
            else:
                X, y, missing_ids, df[i:l] = ml.generate_resize_densenet_fm(df[i:l]).DenseNet_input_matrix(
                    det=det, data_mode_dense="current", data_dir=data_dir, phenom=1, whitened=whitened, data_dir_0 = data_dir_0, data_dir_1 = data_dir_1)
                df['dense_' + det + '_' + str(model_id)].values[i:l] = ml.Dense_predict(
                    model, df[i:l], X, missing_ids)[:, 0]
    print(df.tail())
    df.to_csv(outfile)

if __name__ == '__main__':
    main()
