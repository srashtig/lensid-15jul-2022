
import lensid.utils.ml_utils as ml
import pandas as pd
import numpy as np
import sys

# In[6]:


import lensid.train_test.train_densenets_qts as dense_train
data_dir_old = '/home/srashti.goyal/alice_data_lensid/qts/train/'
data_dir_ml2p0 = '/home/srashti.goyal/lensid_runs/ML_2p0_AnalyticalPsd/data/qts/train/'


# In[8]:


data_dir_ml1p0 = '/home/srashti.goyal/lensid_runs/ML_1p0_AnalyticalPsd/data/qts/train/'
data_dir_ml1p0_qmode2 = '/home/srashti.goyal/lensid_runs/ML_1p0_AnalyticalPsd_qmode_2/data/qts/train/'
data_dir_ml1p0_whitened = '/home/srashti.goyal/lensid_runs/ML_1p0_AnalyticalPsd_whitened/data/qts/train/'


# In[9]:

#mode = 'ml1p0' # old, 'ml1p0', 'ml2p0','ml1p0_qmode2','ml1p0_whiten'


# In[10]:


lensed_df = '/home/srashti.goyal/lensid_runs/ML_2p0_AnalyticalPsd/data/dataframes/train/lensed.csv'


# In[11]:


lensed_df_old = 'lensed_old.csv'


# In[12]:


unlensed_df = '/home/srashti.goyal/lensid_runs/ML_2p0_AnalyticalPsd/data/dataframes/train/unlensed_half.csv'


# In[13]:


size_lensed = 1400
size_unlensed = 1400
#det = 'H1'
epochs = 20

modes = ['ml1p0'  , 'ml1p0_whiten','ml2p0','ml1p0_qmode2','old']
lrs = [0.0001,0.0005, 0.001, 0.005, 0.01, 0.015]

#lr = 0.003


# In[14]:


for mode in modes:
    for lr in lrs:
        for det in ['L1','V1']:
            odir = 'optimising/'+mode+'_lr_max_'+ 'lr_'+str(lr)+'_epochs_'+str(epochs) +'_sizel_'+str(size_lensed)+'_sizeul_'+str(size_unlensed)+'_'

            orig_stdout = sys.stdout
            f = open(odir+det+'.txt', 'w')
            sys.stdout = f


            # In[ ]:


            # ml2p0
            if mode == 'ml2p0':
                dense_train._main(
                    odir,
                    data_dir_ml2p0,
                    lensed_df,
                    unlensed_df,
                    size_lensed,
                    size_unlensed,
                    det,
                    epochs,
                    lr,
                    1,
                )


            # In[15]:


            # ml1p0
            if mode == 'ml1p0':

                dense_train._main(
                    odir,
                    data_dir_ml1p0,
                    lensed_df,
                    unlensed_df,
                    size_lensed,
                    size_unlensed,
                    det,
                    epochs,
                    lr,
                    0,
                )
            # ml1p0_qmode2
            if mode == 'ml1p0_qmode2':

                dense_train._main(
                    odir,
                    data_dir_ml1p0_qmode2,
                    lensed_df,
                    unlensed_df,
                    size_lensed,
                    size_unlensed,
                    det,
                    epochs,
                    lr,
                    0,
                )
            # ml1p0
            if mode == 'ml1p0_whiten':

                dense_train._main(
                    odir,
                    data_dir_ml1p0_whitened,
                    lensed_df,
                    unlensed_df,
                    size_lensed,
                    size_unlensed,
                    det,
                    epochs,
                    lr,
                    1,
                )


            # In[ ]:


            #old
            if mode == 'old':

                dense_train._main(
                    odir,
                    data_dir_old,
                    lensed_df_old,
                    unlensed_df,
                    size_lensed,
                    size_unlensed,
                    det,
                    epochs,
                    lr,
                    0,
                )


            # In[ ]:


            print(odir)
            sys.stdout = orig_stdout
            f.close()