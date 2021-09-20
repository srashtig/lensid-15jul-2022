import lensid.utils.ml_utils as ml
import pandas as pd
import joblib
import warnings
import os
import argparse
import matplotlib.pylab as plt
from sklearn.metrics import plot_roc_curve
import numpy as np
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for training, testing and cross_validating XGBoost with Skymaps.')
    parser.add_argument('-df_dir_train','--df_dir_train', help='input train Dataframe with sky feautures path',default='/home/srashti.goyal/strong-lensing-ml/data/dataframes/train/')
    parser.add_argument('-df_dir_test','--df_dir_test', help='input test Dataframe with sky features path',default='/home/srashti.goyal/strong-lensing-ml/data/dataframes/test/')
    parser.add_argument('-odir','--odir', help='Output directory to save models dataframes and plots',default='out')
    parser.add_argument('-tag','--tag', help='Tag for the dataframes',default='')
    
    parser.add_argument('-compare_to_blu','--compare_to_blu', help='Compare the results with blu? True/False',type=bool,default=True)
    parser.add_argument('-path_to_blu','--path_to_blu',help = 'help path to haris et al blu directory',default = '/home/srashti.goyal/strong-lensing-ml/data/dataframes/haris_et_al/' )
    parser.add_argument('-train_size_lensed','--train_size_lensed',type=int,help='no. of lensed pairs to train on',default=2400)
    parser.add_argument('-cv_size_lensed','--cv_size_lensed',type=int,help='no. of lensed pairs to crossvalidate on',default=2400)
    parser.add_argument('-cv_splits','--cv_splits',type=int,help='no. of splits in crossvalidation',default=10)
    parser.add_argument('-scale_pos_weight','--scale_pos_weight',type=float,help='scale_pos_weight of XGBoost algorithm',default=0.01)

    parser.add_argument('-max_depth','--max_depth',type=int,help='max_depth of XGBoost algorithm',default=6)

    parser.add_argument('-n_estimators','--n_estimators',type=int,help='n_estimators of XGBoost algorithm',default=110)



    args = parser.parse_args()
    print('\n Arguments used:- \n')
    
    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))
    
    odir=args.odir

    df_dir_train=args.df_dir_train
    df_dir_test=args.df_dir_test
    blu_lensed = args.path_to_blu + 'Lensed_PE_blus.csv'
    blu_unlensed = args.path_to_blu  + 'Unlensed_PE_blus.csv'
    tag=args.tag
    if tag == 'None':
        tag=''
    print('\n Training...\n')
    df_lensed_sky = pd.read_csv(df_dir_train+'lensed_sky'+tag+'.csv',index_col=[0] )[:args.train_size_lensed]
    df_unlensed_sky_half = pd.read_csv(df_dir_train+'unlensed_half_sky'+tag+'.csv' ,index_col=[0])
    df_unlensed_sky_half = df_unlensed_sky_half.sample(frac = 1,random_state = 42).reset_index(drop = True)
    df_train_sky = pd.concat([df_lensed_sky,df_unlensed_sky_half],ignore_index = True)
    df_train_sky=df_train_sky.sample(frac = 1).reset_index(drop = True)


    xgboost_sky_model=ml.train_xgboost_sky(df_train_sky,scale_pos_weight=args.scale_pos_weight,max_depth=args.max_depth,n_estimators=args.n_estimators)

    if not os.path.exists(odir):
            os.makedirs(odir)
    if not os.path.exists(odir+'/plots'):
            os.makedirs(odir+'/plots')        
    if not os.path.exists(odir+'/models'):
            os.makedirs(odir+'/models')        
    if not os.path.exists(odir+'/dataframes'):
            os.makedirs(odir+'/dataframes')   

    joblib_file = odir+'/models/XGBsky_0'+tag+'.pkl'  
    joblib.dump(xgboost_sky_model, joblib_file)

    print('\n Validating...\n')


    df_lensed_sky = pd.read_csv(df_dir_train+'lensed_sky'+tag+'.csv',index_col=[0] )[args.train_size_lensed:]
    df_unlensed_sky_half = pd.read_csv(df_dir_train+'unlensed_second_half_sky'+tag+'.csv' ,index_col=[0])
    df_unlensed_sky_half = df_unlensed_sky_half.sample(frac = 1,random_state = 42).reset_index(drop = True)
    df_val_sky = pd.concat([df_lensed_sky,df_unlensed_sky_half],ignore_index = True)
    df_val_sky=df_val_sky.sample(frac = 1).reset_index(drop = True)


    df_val_sky=ml.XGB_predict(df_val_sky,xgboost_sky_model)

    fig=ml.plot_ROCs(df_val_sky,logy=True,cols=['bayestar_skymaps_blu','bayestar_skymaps_d2', 'bayestar_skymaps_d3', 'bayestar_skymaps_lsq',
           'xgb_pred_bayestar_skymaps'])

    plt.savefig(odir+'/plots'+'/validate-ROC-XGB_sky'+tag+'.png')


    print('\n Cross Validating...\n')

    df_lensed_sky = pd.read_csv(df_dir_train+'lensed_sky'+tag+'.csv',index_col=[0] )[:args.cv_size_lensed]
    df_unlensed_sky_half = pd.read_csv(df_dir_train+'unlensed_half_sky'+tag+'.csv' ,index_col=[0])
    df_unlensed_sky_second_half = pd.read_csv(df_dir_train+'unlensed_second_half_sky'+tag+'.csv' ,index_col=[0])
    df_cv_sky = pd.concat([df_lensed_sky,df_unlensed_sky_half],ignore_index = True)

    df_cv_sky=df_cv_sky.sample(frac = 1).reset_index(drop = True)


    cv = ml.StratifiedKFold(n_splits = args.cv_splits)


    plt.rcParams["figure.figsize"] = (10,10)

    tprs = []
    aucs = []

    mean_fpr = 10**np.linspace(-4,0,15)
    fig,ax = plt.subplots()
    cols=['bayestar_skymaps_blu','bayestar_skymaps_d2', 'bayestar_skymaps_d3', 'bayestar_skymaps_lsq']
    for i,(train_index, test_index) in enumerate(cv.split(df_cv_sky,df_cv_sky.Lensing.values)):
        xgboost_sky_model=ml.train_xgboost_sky(df_cv_sky.iloc[train_index],scale_pos_weight=args.scale_pos_weight,max_depth=args.max_depth,n_estimators=args.n_estimators)
        joblib_file = odir+"/models/XGBsky_"+str(i+1)+".pkl"  
        joblib.dump(xgboost_sky_model, joblib_file)
        X=np.c_[df_cv_sky.iloc[test_index][cols]]
        viz = plot_roc_curve(xgboost_sky_model,X,df_cv_sky.Lensing.values[test_index],name="ROC fold {}".format(i+1),alpha=0.3,lw=1,ax=ax)
        interp_tpr = np.interp(mean_fpr,viz.fpr,viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0,1],[0,1],linestyle="--",lw = 2,color="r",label="Chance",alpha=0.8)
    mean_tpr = np.mean(tprs,axis = 0)
    mean_tpr[-1]=1.0
    mean_auc = ml.auc(mean_fpr,mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr,mean_tpr,color='b',label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc , std_auc),lw = 2,alpha=.8)

    std_tpr = np.std(tprs,axis=0)
    tprs_upper = np.minimum(mean_tpr +std_tpr,1)
    tprs_lower = np.maximum(mean_tpr-std_tpr,0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=.2,label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[1e-4,1],ylim=[0,1.05],
              title = "ROC - XGBsky Error Bars",xscale='log')
    ax.legend(loc ="lower right")
    plt.grid()
    plt.savefig(odir+'/plots'+'/cv-ROC-XGB_sky'+tag+'.png')
    plt.show()


    print('\n Testing ...\n')
    df_lensed_sky = pd.read_csv(df_dir_test+'lensed_sky'+tag+'.csv',index_col=[0] )
    df_unlensed_sky = pd.read_csv(df_dir_test+'unlensed_sky'+tag+'.csv' ,index_col=[0])


    df_test_blu_lensed = pd.read_csv(blu_lensed,index_col=[0] )
    df_test_blu_unlensed = pd.read_csv(blu_unlensed,index_col=[0] )
    cols=['m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2, costheta_jn',
           'm1, m2, ra, sin_dec, costheta_jn', 'ra, sin_dec',
           '# m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2',
           'm1, m2, ra, sin_dec', 'm1, m2']
    df_lensed_sky=df_lensed_sky.join(df_test_blu_lensed[cols])
    df_unlensed_sky=df_unlensed_sky.join(df_test_blu_unlensed[cols])
    df_test_sky = pd.concat([df_lensed_sky,df_unlensed_sky],ignore_index = True)
    df_test_sky=df_test_sky.sample(frac = 1).reset_index(drop = True)


    xgboost_sky_model = joblib.load(odir+'/models/XGBsky_0'+tag+'.pkl')

    df_test_sky=ml.XGB_predict(df_test_sky,xgboost_sky_model)
    df_test_sky=df_test_sky.dropna()
    plt.rcParams["figure.figsize"] = (10,10)

    tprs = []
    aucs = []

    mean_fpr = 10**np.linspace(-5,0,20)

    fig,ax = plt.subplots()

    for i in range(1,11):
        xgb_sky_cv = joblib.load(odir+"/models/XGBsky_"+str(i)+".pkl")
        df = ml.XGB_predict(df_test_sky,xgb_sky_cv)
        df_test_sky['xgb_pred_bayestar_skymaps_' +str(i) ]=df['xgb_pred_bayestar_skymaps']

        false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test_sky.Lensing.values, df_test_sky['xgb_pred_bayestar_skymaps_' +str(i)])
        roc_auc = ml.auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate,true_positive_rate,alpha=0.3,lw=1)
        interp_tpr = np.interp(mean_fpr,false_positive_rate,true_positive_rate)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)



    mean_tpr = np.mean(tprs,axis = 0)
    mean_tpr[-1]=1.0
    mean_auc = ml.auc(mean_fpr,mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr,mean_tpr,label = r'Mean ROC  ML Skymaps (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc , std_auc),lw = 2,alpha=.8)

    std_tpr = np.std(tprs,axis=0)
    tprs_upper = np.minimum(mean_tpr +std_tpr,1)
    tprs_lower = np.maximum(mean_tpr-std_tpr,0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=.5)


    colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    if args.compare_to_blu == True:
        cols=['bayestar_skymaps_blu','ra, sin_dec']
        labels=['$B^L_U$ Bayestar skymaps','$B^L_U$ : RA DEC']
    else:
        cols=['bayestar_skymaps_blu']
        labels=['$B^L_U$ Bayestar skymaps']
        
    for i,col in enumerate(cols):
        false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test_sky.Lensing.values, df_test_sky[col])
        plt.plot(false_positive_rate,true_positive_rate,'-',label=labels[i],lw=2)

    ax.set(xlim=[2e-5,1],ylim=[2e-2,1.05],title = "ML Skymaps Testing",xscale='log',yscale='log')
    ax.grid()
    ax.legend(loc ="lower right")
    plt.savefig(odir+'/plots'+'/test-ROC-log-sky-xgb'+tag+'.png')
    plt.show()

    df_test_sky.to_csv(odir+'/dataframes/ML_sky.csv')
    
    if args.compare_to_blu ==True:
        print('\n Comparing to blu ...\n')

        df_test=df_test_sky
        ml_stat='xgb_pred_bayestar_skymaps'
        blu_stat= 'ra, sin_dec'
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.xlabel('ML statistic')
        bins=np.linspace(-6,0,30)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Unlensed', histtype='step',density=True)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Lensed', histtype='step',density=True)
        plt.legend()
        plt.subplot(132)
        plt.xlabel('BLU statistic')
        bins=np.linspace(-5,6,30)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Unlensed', histtype='step',density=True)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Lensed', histtype='step',density=True)
        plt.legend()
        plt.subplot(133)

        plt.xlabel('ML statistic')
        plt.ylabel('BLU statistic')
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.loglog(df[ml_stat],df[blu_stat],'x',label='Unlensed')
        df=df_test[df_test['Lensing'] == 1]
        plt.loglog(df[ml_stat],df[blu_stat],'+',label='Lensed')
        plt.legend()
        plt.savefig(odir+'/plots'+'/compare-blu-XGB_sky'+tag+'.png')

        plt.show()

        df_test[ml_stat+'_fpp']=ml.get_fars(df_test,ml_stat,df_test,ml_stat)
        df_test[blu_stat+'_fpp']=ml.get_fars(df_test,blu_stat,df_test,blu_stat)

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        bins=np.linspace(-5,0,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat+'_fpp']),bins=bins,label='ML Unlensed', histtype='step',density=True,color='C0',lw=4)
        plt.hist(np.log10(df[blu_stat+'_fpp']),bins=bins,label='BLU Unlensed', histtype='step',density=True,color='C0',lw=2)

        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[ml_stat+'_fpp']),bins=bins,label='ML Lensed', histtype='step',density=True,color='C1',lw=4)
        plt.hist(np.log10(df[blu_stat+'_fpp']),bins=bins,label='BLU Lensed', histtype='step',density=True,color='C1',lw=2)

        plt.legend()
        plt.xlabel('FPP(log10)')
        plt.ylabel('PDF')
        plt.grid()

        plt.subplot(122)
        df=df_test[df_test['Lensing'] == 0]
        plt.loglog(df[ml_stat+'_fpp'],df[blu_stat+'_fpp'],'+',label='Unlensed',color='C0')
        df=df_test[df_test['Lensing'] == 1]
        plt.loglog(df[ml_stat+'_fpp'],df[blu_stat+'_fpp'],'x',label='Lensed',color='C1')
        plt.legend()
        plt.loglog(10**bins,10**bins,'k--')
        plt.xlabel('ML')
        plt.ylabel('BLU')
        plt.xlim(1e-5,1)
        plt.ylim(1e-5,1)

        plt.grid()
        plt.suptitle('ML sky and BLU ra dec')
        plt.savefig(odir+'/plots'+'/compare-FPPs-blu-XGB_sky'+tag+'.png')
        plt.show()

        fig,rocs=ml.plot_ROCs(df_test_sky,cols=['xgb_pred_bayestar_skymaps','bayestar_skymaps_blu','ra, sin_dec'],\
                                                                                         labels=['ML skymaps','$B^L_U$ Bayestar skymaps','$B^L_U$ : RA DEC'],logy=True,ylim=1e-2)

        fpp_blu,eff_blu,thr_blu=rocs[blu_stat]
        fpp_ml,eff_ml,thr_ml=rocs[ml_stat]

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        plt.plot(fpp_blu,thr_blu,label='BLU')
        plt.xscale('log')
        plt.xlim(1e-5,1)
        plt.yscale('log')
        plt.ylim(np.percentile(thr_blu,10),np.max(thr_blu))
        plt.legend()
        plt.xlabel('FPP')
        plt.ylabel('Threshold')
        plt.grid()
        plt.subplot(122)
        plt.plot(fpp_ml,thr_ml,label='ML')
        plt.xscale('log')
        plt.xlim(1e-5,1)
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.xlabel('FPP')
        plt.ylabel('Threshold')
        plt.savefig(odir+'/plots'+'/compare-thresholds-blu-XGB_sky'+tag+'.png')
        plt.show()

        df_test_sky.to_csv(odir+'/dataframes/ML_sky'+tag+'.csv')
    
if __name__ == "__main__":
    main()