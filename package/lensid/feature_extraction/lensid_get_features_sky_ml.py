import sys
import lensid.utils.ml_utils as ml 
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for calculating features from the skymaps pairs(.npz)')
    parser.add_argument('-infile','--infile', help='input Dataframe path',default='train/lensed.csv')
    parser.add_argument('-outfile','--outfile', help='output Dataframe path ',default='train/lensed_sky.csv')
    parser.add_argument('-data_dir','--data_dir', help='sky .npz files folder path',default='train')

    parser.add_argument('-start','--start', type=int, help=' input DF start index',default=0)

    parser.add_argument('-n','--n', type=int, help='no. of  pairs',default=0)

    parser.add_argument('-pe_skymaps','--pe_skymaps',help='use PE skymaps True/False',default=False)
    args = parser.parse_args()

    data_dir = args.data_dir+'/'

    if args.n ==0:
        df = pd.read_csv(args.infile, index_col=[0])[args.start:]
        print(len(df['img_0']), ' event pairs ')
    else:
        df = pd.read_csv(args.infile, index_col=[0])[args.start:args.start+args.n]

    dl=1000
    l=len(df.img_0.values)

    if args.pe_skymaps==False:
        data_mode_xgb = 'current'
        df['bayestar_skymaps_blu']=''
        df['bayestar_skymaps_d2']=''
        df['bayestar_skymaps_d3']=''
        df['bayestar_skymaps_lsq']=''
    else:
        data_mode_xgb = 'pe'
        df['pe_skymaps_blu']=''
        df['pe_skymaps_d2']=''
        df['pe_skymaps_d3']=''
        df['pe_skymaps_lsq']=''

    for i in range(0,l,dl):
        if i + dl < l:
            print(i)
            features ,xgb_sky_labels,df[i:i+dl],missing_ids = ml.generate_skymaps_fm(df[i:i+dl]).XGBoost_input_matrix(data_mode_xgb=data_mode_xgb,data_dir=data_dir) 

        else:
            features ,xgb_sky_labels,df[i:l],missing_ids = ml.generate_skymaps_fm(df[i:l]).XGBoost_input_matrix(data_mode_xgb=data_mode_xgb,data_dir=data_dir) 

    print(df.tail())
    df.to_csv(args.outfile)




