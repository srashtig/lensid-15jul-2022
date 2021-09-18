import lensid.utils.ml_utils as ml
import pandas as pd
import warnings
import os
import argparse
import numpy as np
from sklearn.metrics import plot_roc_curve
import matplotlib.pylab as plt
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='This is stand alone code for testing combined ML and comparing to blu')
    parser.add_argument('-indir','--indir', help='input Dataframe directory',default='out/dataframes')
    parser.add_argument('-odir','--odir', help='output directory',default='out')

    parser.add_argument('-tag_qts','--tag_qts', help='tag for reading ML with qtransforms predictions ',default='_kaggle')
    parser.add_argument('-tag_sky','--tag_sky', help='tag for reading MK with skymaps predictions ',default='')

    parser.add_argument('-compare_to_blu','--compare_to_blu', help='Compare the results with blu? True/False',type=bool,default=True)
    args = parser.parse_args()
    print('\n Arguments used:- \n')
    
    for arg in vars(args):
        print(arg, ': \t', getattr(args, arg))
    
    tag_sky=args.tag_sky
    tag_qts=args.tag_qts
    odir = args.odir
    df_sky = pd.read_csv(args.indir+'/ML_sky'+tag_sky+'.csv').drop(columns=['Unnamed: 0'])
    df_qts = pd.read_csv(args.indir+'/ML_qts'+tag_qts+'.csv').drop(columns=['Unnamed: 0'])

    if not os.path.exists(odir):
            os.makedirs(odir)
    if not os.path.exists(odir+'/plots'):
            os.makedirs(odir+'/plots')
    if not os.path.exists(odir+'/dataframes'):
            os.makedirs(odir+'/dataframes')        

    if args.compare_to_blu == True:
        df_test=pd.merge(df_sky, df_qts, on=['img_0', 'img_1','Lensing','m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2, costheta_jn',
       'm1, m2, ra, sin_dec, costheta_jn', 'ra, sin_dec',
       '# m1, m2, ra, sin_dec, a1, a2, costilt1, costilt2',
       'm1, m2, ra, sin_dec', 'm1, m2'])
    else: 
        df_test=pd.merge(df_sky, df_qts, on=['img_0', 'img_1','Lensing'])

    df_test['densnet_xgbsky_bayestar_mul_0'] = df_test['xgb_dense_QTS_0']*df_test['xgb_pred_bayestar_skymaps']



    plt.rcParams["figure.figsize"] = (10,7)

    tprs = []
    aucs = []

    mean_fpr = 10**np.linspace(-5,0,20)

    fig,ax = plt.subplots()

    for i in range(1,11):
        df_test['densnet_xgbsky_bayestar_mul_'+str(i)] = df_test['xgb_dense_QTS_'+str(i)]*df_test['xgb_pred_bayestar_skymaps_'+str(i)]

        false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test.Lensing.values, df_test['densnet_xgbsky_bayestar_mul_' +str(i)])
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
    ax.plot(mean_fpr,mean_tpr,label = 'Mean ROC ML combined',lw = 2,alpha=.8)
    print(r'Mean ROC  ML combined (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc , std_auc))
    std_tpr = np.std(tprs,axis=0)
    tprs_upper = np.minimum(mean_tpr +std_tpr,1)
    tprs_lower = np.maximum(mean_tpr-std_tpr,0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=.5)

    if args.compare_to_blu ==True: 
        colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        cols=['m1, m2, ra, sin_dec']
        labels=[r'$B^L_U$: $m_1,m_2$, $\alpha$, $\delta$']
        for i,col in enumerate(cols):
            false_positive_rate, true_positive_rate, thresholds = ml.roc_curve(df_test.Lensing.values, df_test[col])
            plt.plot(false_positive_rate,true_positive_rate,'-',label=labels[i],lw=2)

    ax.set(xlim=[2e-5,1],ylim=[9e-2,1.05],title = "ML Combined Testing",xscale='log',yscale='log')
    ax.grid()
    ax.legend(loc ="lower right")
    plt.savefig(odir+'/plots'+'/ROC-log-combined'+tag_qts+tag_sky+'.png')

    plt.show()


    df_test.to_csv(odir+'/dataframes/ML_combined'+tag_qts+tag_sky+'.csv')

    if args.compare_to_blu == True:
        ml_stat='densnet_xgbsky_bayestar_mul_0'
        blu_stat= 'm1, m2, ra, sin_dec'
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.xlabel('ML statistic')
        bins=np.linspace(-11,0,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Unlensed', histtype='step',density=True)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[ml_stat]),bins=bins,label='Lensed', histtype='step',density=True)
        plt.legend()
        #plt.ylim(0,300)
        plt.subplot(132)
        plt.xlabel('BLU statistic')
        bins=np.linspace(-5,6,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Unlensed', histtype='step',density=True)
        df=df_test[df_test['Lensing'] == 1]
        plt.hist(np.log10(df[blu_stat]),bins=bins,label='Lensed', histtype='step',density=True)
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
        plt.savefig(odir+'/plots'+'/compare-blu-ML_combined'+tag_qts+tag_sky+'.png')

        plt.show()




        df_test[ml_stat+'_fpp']=ml.get_fars(df_test,ml_stat,df_test,ml_stat)
        df_test[blu_stat+'_fpp']=ml.get_fars(df_test,blu_stat,df_test,blu_stat)
        plt.figure(figsize=(15,7))
        plt.subplot(121)
        bins=np.linspace(-5,0,30)
        #plt.ylim(-100,1e3)
        df=df_test[df_test['Lensing'] == 0]
        plt.hist(np.log10(df[ml_stat+'_fpp']),bins=bins,label='ML Unlensed', histtype='step',density=True,color='C0',lw=4)
        plt.hist(np.log10(df[blu_stat+'_fpp']),bins=bins,label='BLU Unlensed', histtype='step',density=True,color='C2',lw=2)

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
        plt.axvline(1e-2,color='r')
        plt.axhline(1e-2,color='r')

        plt.loglog(10**bins,10**bins,'k--')
        df=df_test[df_test['Lensing'] == 1]
        plt.loglog(df[ml_stat+'_fpp'],df[blu_stat+'_fpp'],'x',label='Lensed',color='C1')
        plt.legend()
        plt.xlabel('ML Log FPP')
        plt.ylabel('BLU Log FPP')
        plt.xlim(1e-5,1)
        plt.ylim(1e-5,1)
        plt.grid()
        plt.suptitle('ML combined and BLU m1,m2,ra,dec')
        plt.savefig(odir+'/plots'+'/compare-FPPs-blu-ML_combined'+tag_qts+tag_sky+'.png')

        plt.show()


        fig,rocs=ml.plot_ROCs(df_test,cols=['densnet_xgbsky_bayestar_mul_0','m1, m2, ra, sin_dec'],labels =['ML bayestar sky x ML QTs','$B^L_U$'],logy=True,ylim=7e-2)


        # In[13]:


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
        plt.savefig(odir+'/plots'+'/compare-thresholds-blu-ML_combined'+tag_qts+tag_sky+'.png')

        plt.show()
        df_test.to_csv(odir+'/dataframes/ML_combined'+tag_qts+tag_sky+'.csv')

if __name__ == "__main__":
    main()