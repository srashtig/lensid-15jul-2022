#! /home/srashti.goyal/.conda/envs/igwn-py37-hanabi/bin/python
import yaml
import matplotlib
import numpy as np
import matplotlib.pylab as plt
import os
import pandas as pd
import argparse
import lensid.utils.ml_utils as ml
import lensid.feature_extraction.lensid_get_features_qts_ml as qts_ml_features
import lensid.feature_extraction.lensid_get_features_sky_ml as sky_ml_features

import joblib
import warnings
warnings.filterwarnings('ignore')

# input: dataframe with pairs + QTs + Bayestar Skymaps + ML models.
# output: ML QTs predictions + ML Sky predictions + ML combined predictions
# process: Calc features/densenet predictions QTs--> XGboost QT  , Calc Features Skymaps --> XGboost Sky, Multiply.
# optional: Get False Positive Probabilities using distribution of ML
# statictic with background injections


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for computing ML predictions for a given event pairs in dataframe, their skymaps, Qtransforms and the trained ML models. Optionally computes the False Positive Probabilities given the background.')
    parser.add_argument(
        '-config',
        '--config',
        help='input CONFIG.yaml file',
        default='config_O3_events.yaml')
    args = parser.parse_args()

    def set_var(var_name, value):
        globals()[var_name] = value

    stream = open(args.config, 'r')
    dictionary = yaml.load_all(stream)

    for doc in dictionary:
        for key, value in doc.items():
            print(key + " : " + str(value))
            set_var(key, value)

    if not os.path.exists(odir):
        os.makedirs(odir)
    if not os.path.exists(odir + '/plots'):
        os.makedirs(odir + '/plots')
    if not os.path.exists(odir + '/dataframes'):
        os.makedirs(odir + '/dataframes')

    if calc_features_sky == 1:
        print('Calculating Sky features...')
        #    _main(data_dir,start, n,infile,outfile,pe_skymaps)

        sky_ml_features._main(data_dir_sky, 0, 0 , in_df, (odir + '/dataframes/ML_sky' + tag_sky + '.csv'), 0)

    if cal_features_qts == 1:
        #_main(data_dir, n, infile, outfile, start, dense_models_dir, model_id, whitened)

        print('Calculating Qtransform features...')
        qts_ml_features._main(data_dir_qts, 0, in_df, (odir + '/dataframes/ML_qts' + tag_qts + '.csv'),0 , dense_model_dir, 0, whitened)

    print('Calculating QTs ML predictions...')
    xgb_qts = joblib.load(xgboost_qt)

    df_qts = pd.read_csv(odir + '/dataframes/ML_qts' + tag_qts + '.csv')
    df_qts = ml.predict_xgboost_dense_qts(df_qts, xgb_qts)

    print('Calculating Sky ML predictions...')
    xgb_sky = joblib.load(xgboost_sky)
    df_sky = pd.read_csv(odir + '/dataframes/ML_sky' + tag_sky + '.csv')
    df_sky = ml.XGB_predict(df_sky, xgb_sky)

    print('Calculating combined ML predictions...')

    df_combined = pd.merge(df_qts, df_sky, on=["img_0", "img_1", "Lensing"])
    df_combined[col_dict['ML combined']] = df_combined[col_dict['ML QTs']
                                                       ] * df_combined[col_dict['ML sky']]

    plt.figure(figsize=(20, 5))
    bins = np.linspace(-11, 0, 30)
    plt.subplot(131)
    ml_stat = col_dict['ML combined']
    plt.hist(
        np.log10(
            df_combined[ml_stat]),
        bins=bins,
        label='ML combined',
        histtype='step',
        density=True,
        lw=2,
        ls='dashed')
    plt.legend()
    plt.ylabel('PDF')
    plt.xlabel('ML Statistic')
    plt.subplot(132)
    ml_stat = col_dict['ML sky']
    plt.hist(
        np.log10(
            df_combined[ml_stat]),
        bins=bins,
        label='ML sky ',
        histtype='step',
        density=True,
        lw=2,
        ls='dashed')
    plt.legend()
    plt.ylabel('PDF')
    plt.xlabel('ML Statistic')

    plt.subplot(133)
    bins = np.linspace(-6, 0, 20)
    ml_stat = col_dict['ML QTs']
    plt.hist(
        np.log10(
            df_combined[ml_stat]),
        bins=bins,
        label='ML qts ',
        histtype='step',
        density=True,
        lw=2,
        ls='dashed')
    plt.legend()
    plt.ylabel('PDF')
    plt.xlabel('ML Statistic')

    plt.savefig(odir + '/plots/ML_stat_dist.png')
    plt.show()

    if calc_fpp == 1:
        print('Calculating False Positive Probabilities...')
        df_injs = pd.read_csv(background_df)
        for stat in fpp_dict.keys():
            df_combined[col_dict[stat] + '_fpp'] = ml.get_fars(
                df_combined, col_dict[stat], df_injs, fpp_dict[stat])

        plt.figure(figsize=(7, 5))
        bins = np.linspace(-5, 0, 20)

        ml_stat = col_dict['ML combined']
        plt.hist(np.log10(df_combined[ml_stat + '_fpp']),
                 bins=bins,
                 label='ML combined',
                 histtype='step',
                 density=False,
                 lw=2,
                 ls='dashed')

        ml_stat = col_dict['ML sky']
        plt.hist(np.log10(df_combined[ml_stat + '_fpp']),
                 bins=bins,
                 label='ML sky ',
                 histtype='step',
                 density=False,
                 lw=2,
                 ls='dashed')

        ml_stat = col_dict['ML QTs']
        plt.hist(np.log10(df_combined[ml_stat + '_fpp']),
                 bins=bins,
                 label='ML qts ',
                 histtype='step',
                 density=False,
                 lw=2,
                 ls='dashed')

        plt.xlabel('FPP(log10)')
        plt.ylabel('Cumulative Counts')
        plt.yscale('log')
        plt.grid()
        plt.title('FPPs distribution')
        plt.legend()
        plt.savefig(odir + '/plots/FPP_dist.png')
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.scatter(df_combined[col_dict['ML QTs'] +
                                '_fpp'], df_combined[col_dict['ML sky'] +
                                                     '_fpp'], c=df_combined[col_dict['ML combined'] +
                                                                            '_fpp'], norm=matplotlib.colors.LogNorm())
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('ML QTs FPP')
        plt.ylabel('ML sky FPP')
        plt.grid()
        plt.colorbar(label='ML combined FPP')
        plt.savefig(odir + '/plots/FPP_scatter.png')

        plt.show()

    outfname = odir + '/dataframes/ML_combined' + tag_qts + tag_sky + '.csv'
    df_combined.to_csv(outfname)

    print('ML predictions saved to dataframe %s' % outfname)

    print(df_combined.tail())


if __name__ == "__main__":
    main()
