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
    parser = argparse.ArgumentParser(description='This is stand alone code for training, testing and cross_validating XGBoost with Qtransforms.')
    parser.add_argument('-df_dir_train','--df_dir_train', help='input train Dataframe with QT feautures path',default='/home/srashti.goyal/strong-lensing-ml/data/dataframes/train/')
    parser.add_argument('-df_dir_test','--df_dir_test', help='input test Dataframe with QT feautures path',default='/home/srashti.goyal/strong-lensing-ml/data/dataframes/test/')
    parser.add_argument('-odir','--odir', help='Output directory to save models dataframes and plots',default='out')
    parser.add_argument('-tag','--tag', help='Tag for the dataframes',default='_kaggle')
    
    parser.add_argument('-compare_to_blu','--compare_to_blu',type=int, help='Compare the results with blu? 1/0',default=1)
    parser.add_argument('-path_to_blu','--path_to_blu',help = 'help path to haris et al blu directory',default = '/home/srashti.goyal/strong-lensing-ml/data/dataframes/haris_et_al/' )
    parser.add_argument('-train_size_lensed','--train_size_lensed',type=int,help='no. of lensed pairs to train on',default=2400)
    parser.add_argument('-cv_size_lensed','--cv_size_lensed',type=int,help='no. of lensed pairs to crossvalidate on',default=2400)
    parser.add_argument('-cv_splits','--cv_splits',type=int,help='no. of splits in crossvalidation',default=10)
    parser.add_argument('-scale_pos_weight','--scale_pos_weight',type=float,help='scale_pos_weight of XGBoost algorithm',default=0.01)

    parser.add_argument('-max_depth','--max_depth',type=int,help='max_depth of XGBoost algorithm',default=6)

    parser.add_argument('-n_estimators','--n_estimators',type=int,help='n_estimators of XGBoost algorithm',default=135)



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
    df_lensed_qts = pd.read_csv(df_dir_train+'lensed_QTs'+tag+'.csv',index_col=[0] )[:args.train_size_lensed]
    df_unlensed_qts_half = pd.read_csv(df_dir_train+'unlensed_half_QTs'+tag+'.csv' ,index_col=[0])
    df_unlensed_qts_half = df_unlensed_qts_half.sample(frac = 1,random_state = 42).reset_index(drop = 1)
    df_train_qts = pd.concat([df_lensed_qts,df_unlensed_qts_half],ignore_index = 1)
    df_train_qts=df_train_qts.sample(frac = 1).reset_index(drop = 1)
    xgboost_dense_qts_model=ml.train_xgboost_dense_qts(df_train_qts,from_df=1,scale_pos_weight=args.scale_pos_weight,max_depth=args.max_depth,n_estimators=args.n_estimators)

    if not os.path.exists(odir):
            os.makedirs(odir)
    if not os.path.exists(odir+'/plots'):
            os.makedirs(odir+'/plots')        
    if not os.path.exists(odir+'/models'):
            os.makedirs(odir+'/models')        
    if not os.path.exists(odir+'/dataframes'):
            os.makedirs(odir+'/dataframes') 

    joblib_file = odir+'/models/XGBQT_0'+tag+'.pkl'  
    joblib.dump(xgboost_dense_qts_model, joblib_file)

    print('\n Validating...\n')

    df_lensed_qts = pd.read_csv(df_dir_train+'lensed_QTs'+tag+'.csv',index_col=[0] )[args.train_size_lensed:]
    df_unlensed_qts_half = pd.read_csv(df_dir_train+'unlensed_second_half_QTs'+tag+'.csv' ,index_col=[0])
    df_unlensed_qts_half = df_unlensed_qts_half.sample(frac = 1,random_state = 42).reset_index(drop = 1)
    df_val_qts = pd.concat([df_lensed_qts,df_unlensed_qts_half],ignore_index = 1)
    df_val_qts=df_val_qts.sample(frac = 1).reset_index(drop = 1)
    df_val_qts=ml.predict_xgboost_dense_qts(df_val_qts,xgboost_dense_qts_model)

    fig=ml.plot_ROCs(df_val_qts,logy=1,cols=['dense_H1_0', 'dense_L1_0', 'dense_V1_0','xgb_dense_QTS_0'])
    plt.savefig(odir+'/plots'+'/validate-ROC-XGB_QT'+tag+'.png')


    print('\n Cross Validating...\n')

    df_lensed_qts = pd.read_csv(df_dir_train+'lensed_QTs'+tag+'.csv',index_col=[0] )[:args.cv_size_lensed]
    df_unlensed_qts_half = pd.read_csv(df_dir_train+'unlensed_half_QTs'+tag+'.csv' ,index_col=[0])
    df_unlensed_qts_second_half = pd.read_csv(df_dir_train+'unlensed_second_half_QTs'+tag+'.csv' ,index_col=[0])
    df_cv_qts = pd.concat([df_lensed_qts,df_unlensed_qts_half],ignore_index = 1)

    df_cv_qts=df_cv_qts.sample(frac = 1).reset_index(drop = 1)


    df_cv_qts.tail()


    cv = ml.StratifiedKFold(n_splits = args.cv_splits)


    xgboost_dense_qts_models=[]
    plt.rcParams["figure.figsize"] = (10,10)

    tprs = []
    aucs = []

    mean_fpr = 10**np.linspace(-4,0,15)
    fig,ax = plt.subplots()
    cols=['dense_H1_0','dense_L1_0','dense_V1_0']
    for i,(train_index, test_index) in enumerate(cv.split(df_cv_qts,df_cv_qts.Lensing.values)):
        xgboost_dense_qts_model=ml.train_xgboost_dense_qts(df_cv_qts.iloc[train_index],from_df=1,n_estimators=args.n_estimators,max_depth = args.max_depth, scale_pos_weight=args.scale_pos_weight)
        joblib_file = odir+"/models/XGBQT_"+str(i+1)+tag+ ".pkl"  
        joblib.dump(xgboost_dense_qts_model, joblib_file)
        X=np.c_[df_cv_qts.iloc[test_index][cols]]
        viz = plot_roc_curve(xgboost_dense_qts_model,X,df_cv_qts.Lensing.values[test_index],name="ROC fold {}".format(i+1),alpha=0.3,lw=1,ax=ax)
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
              title = "ROC - XGBQT Error Bars",xscale='log')
    ax.legend(loc ="lower right")
    plt.grid()
    plt.savefig(odir+'/plots'+'/cv-ROC-XGB_QT'+tag+'.png')
    plt.show()


    print('\n Testing...\n')
    
    df_lensed_qts = pd.read_csv(df_dir_test+'lensed_QTs'+tag+'.csv',index_col=[0] )
    df_unlensed_qts = pd.read_csv(df_dir_test+'unlensed_QTs'+tag+'.csv' ,index_col=[0])

    if args.compare_to_blu ==1:
        df_test_blu_lensed = pd.read_csv(blu_lensed,index_col=[0] )
        df_test_blu_unlensed = pd.read_csv(blu_unlensed,index_col=[0] )
        cols=['m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2, costheta_jn',
               'm1, m2, ra, sin_dec, costheta_jn', 'ra, sin_dec',
               '# m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2',
               'm1, m2, ra, sin_dec', 'm1, m2']
        df_lensed_qts=df_lensed_qts.join(df_test_blu_lensed[cols])
        df_unlensed_qts=df_unlensed_qts.join(df_test_blu_unlensed[cols])
    df_test_qts = pd.concat([df_lensed_qts,df_unlensed_qts],ignore_index = 1)
    df_test_qts=df_test_qts.sample(frac = 1).reset_index(drop = 1)


    xgboost_dense_qts_model = joblib.load(odir+'/models/XGBQT_0'+tag+'.pkl')

    df_test_qts=ml.predict_xgboost_dense_qts(df_test_qts,xgboost_dense_qts_model)
    df_test_qts=df_test_qts.dropna()


    fig=ml.plot_ROCs(df_test_qts,cols=['dense_H1_0' ,'dense_L1_0','dense_V1_0','xgb_dense_QTS_0','m1, m2']                     ,labels=['ML H1 QTs', 'ML L1 QTs', 'ML V1 QTs','ML combined H1 L1 V1 QTS', '$B^L_U:$ $m_1 m_2$'],logy=1)


    plt.rcParams["figure.figsize"] = (10,10)

    tprs = []
    aucs = []

    mean_fpr = 10**np.linspace(-5,0,20)

    fig,ax = plt.subplots()
    cols=['dense_H1_0','dense_L1_0','dense_V1_0']

    for i in range(1,11):
        xgb_qt_cv = joblib.load(odir+"/models/XGBQT_"+str(i)+tag+".pkl")
        df = ml.predict_xgboost_dense_qts(df_test_qts,xgb_qt_cv)
        df_test_qts['xgb_dense_QTS_' +str(i) ]=df['xgb_dense_QTS_0']

        false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test_qts.Lensing.values, df_test_qts['xgb_dense_QTS_' +str(i)])
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
    ax.plot(mean_fpr,mean_tpr,label = r'Mean ROC Combined H1 L1 V1 QTs ML (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc , std_auc),lw = 2,alpha=.8)

    std_tpr = np.std(tprs,axis=0)
    tprs_upper = np.minimum(mean_tpr +std_tpr,1)
    tprs_lower = np.maximum(mean_tpr-std_tpr,0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=.5)


    colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    
    print('\n Comparing to blu...\n')

    if args.compare_to_blu==1:
        cols=['dense_H1_0' ,'dense_L1_0','dense_V1_0','m1, m2']

        labels=['ML H1 QTs', 'ML L1 QTs', 'ML V1 QTs', '$B^L_U:$ $m_1 m_2$']
    else:
        cols=['dense_H1_0' ,'dense_L1_0','dense_V1_0']

        labels=['ML H1 QTs', 'ML L1 QTs', 'ML V1 QTs']
        
    for i,col in enumerate(cols):
        false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test_qts.Lensing.values, df_test_qts[col])
        plt.plot(false_positive_rate,true_positive_rate,'-',label=labels[i],lw=1)

    #ax.set(xlim=[2e-5,1],ylim=[-0.05,1.05],title = "ML Skymaps for Haris et. al",xscale='log')

    ax.set(xlim=[2e-5,1],ylim=[5e-3,1],title = "ML QTs for Haris et. al",xscale='log',yscale='log')
    ax.grid()
    ax.legend(loc ="lower right")
    plt.savefig(odir+'/plots'+'/test-ROC-log-QTs-xgb'+tag+'.png')
    plt.show()
    df_test_qts.to_csv(odir+'/dataframes/ML_qts'+tag+'.csv')

    if args.compare_to_blu ==1:

        df_test=df_test_qts
        ml_stat='xgb_dense_QTS_0'
        blu_stat= 'm1, m2'
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.xlabel('ML statistic')
        bins=np.linspace(-6,0,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Unlensed', histtype='step',density=1)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Lensed', histtype='step',density=1)
        plt.legend()
        #plt.ylim(0,300)
        plt.subplot(132)
        plt.xlabel('BLU statistic')
        bins=np.linspace(-6,6,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Unlensed', histtype='step',density=1)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Lensed', histtype='step',density=1)
        plt.legend()
        #plt.ylim(0,300)
        plt.subplot(133)

        plt.xlabel('ML statistic')
        plt.ylabel('BLU statistic')
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.loglog(df[ml_stat],df[blu_stat],'x',label='Unlensed')
        df=df_test[df_test['Lensing'] == 1]
        plt.loglog(df[ml_stat],df[blu_stat],'+',label='Lensed')
        plt.legend()
        plt.savefig(odir+'/plots'+'/compare-blu-XGB_QT'+tag+'.png')

        plt.show()

        ml_stat='xgb_dense_QTS_0'
        blu_stat= 'm1, m2'
        df_test[ml_stat+'_fpp']=ml.get_fars(df_test,ml_stat,df_test,ml_stat)
        df_test[blu_stat+'_fpp']=ml.get_fars(df_test,blu_stat,df_test,blu_stat)

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        bins=np.linspace(-5,0,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat+'_fpp']),bins=bins,label='ML Unlensed', histtype='step',density=1,color='C0',lw=4)
        plt.hist(np.log10(df[blu_stat+'_fpp']),bins=bins,label='BLU Unlensed', histtype='step',density=1,color='C0',lw=2)

        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[ml_stat+'_fpp']),bins=bins,label='ML Lensed', histtype='step',density=1,color='C1',lw=4)
        plt.hist(np.log10(df[blu_stat+'_fpp']),bins=bins,label='BLU Lensed', histtype='step',density=1,color='C1',lw=2)

        plt.legend()
        plt.xlabel('FPP(log10)')
        plt.ylabel('PDF')
        plt.grid()

        plt.subplot(122)
        df=df_test[df_test['Lensing'] == 0]
        plt.loglog(df[ml_stat+'_fpp'],df[blu_stat+'_fpp'],'+',label='Unlensed',color='C0')
        df=df_test[df_test['Lensing'] == 1]
        plt.loglog(df[ml_stat+'_fpp'],df[blu_stat+'_fpp'],'x',label='Lensed',color='C1')
        plt.loglog(10**bins,10**bins,'k--')
        plt.legend()
        plt.xlabel('ML')
        plt.ylabel('BLU')
        plt.grid()
        plt.xlim(1e-5,1)
        plt.ylim(1e-5,1)
        plt.suptitle('ML QTs and BLU masses')
        plt.savefig(odir+'/plots'+'/compare-FPPs-blu-XGB_QT'+tag+'.png')
        plt.show()

        fig,rocs=ml.plot_ROCs(df_test_qts,cols=['dense_H1_0' ,'dense_L1_0','dense_V1_0','xgb_dense_QTS_0','m1, m2']                     ,labels=['ML H1 QTs', 'ML L1 QTs', 'ML V1 QTs','ML combined H1 L1 V1 QTS', '$B^L_U:$ $m_1 m_2$'],logy=1)



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
        plt.savefig(odir+'/plots'+'/compare-thresholds-blu-XGB_QT'+tag+'.png')
        plt.show()
        df_test_qts.to_csv(odir+'/dataframes/ML_qts'+tag+'.csv')

if __name__ == "__main__":
    main()
    