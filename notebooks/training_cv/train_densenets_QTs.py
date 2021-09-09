#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64"
#!export CUDA_HOME=/usr/local/cuda-10.1
from lensid.utils.ml_utils import *


# In[17]:


indir = 'train'

#data_dir = '../../data/qts/'  #ALICE

#df_dir='../../data/dataframes/train/' ##alice
df_dir = '/home/srashti.goyal/strong-lensing-ml/data/dataframes/' #CIT

data_dir = '/home/srashti.goyal/alice_data_lensid/qts/' + indir + '/' ##CIT

odir='dense_out/cit/'

if not os.path.exists(odir):
        os.makedirs(odir)

# # Load training dataframe

# In[11]:


df_lensed = pd.read_csv(df_dir+str(indir)+'/lensed.csv' )
df_lensed=df_lensed.drop(columns=['Unnamed: 0'])
df_lensed['img_0']=df_lensed['img_0'].values 
df_lensed['img_1']=df_lensed['img_1'].values 
df_lensed=df_lensed[:1400]
df_unlensed = pd.read_csv(df_dir+str(indir)+'/unlensed_half.csv' )
df_unlensed=df_unlensed.drop(columns=['Unnamed: 0'])
df_unlensed = df_unlensed.sample(frac = 1,random_state = 42).reset_index(drop = True)[:1400]
df_train = pd.concat([df_lensed,df_unlensed],ignore_index = True)
df_train=df_train.sample(frac = 1).reset_index(drop = True)


# In[12]:


df_train.tail()


# ## Load input feature matrix for Detector H1 from the Qtransforms
# 

# In[18]:


det='H1'
X , y,missing_ids, df_train =  generate_resize_densenet_fm(df_train).DenseNet_input_matrix(det = det,data_mode_dense="current",data_dir=data_dir,phenom=True)


# In[6]:


df_train.tail()


# dense_model_trained = train_densenet(X,y,det,20, 0.01) #20,0.01, .005
# dense_model_trained.save(odir+'det+'.h5')## Train and save the Densenet model
# 

# In[7]:


dense_model_trained = train_densenet(X,y,det,20, 0.01) #20,0.01, .005
dense_model_trained.save(odir+det+'.h5')


# # Det L1

# In[8]:
'''

del X
del y
det='L1'
X , y,missing_ids, df_train =  generate_resize_densenet_fm(df_train).DenseNet_input_matrix(det = det,data_mode_dense="current",data_dir=data_dir,phenom=True)


# In[9]:


dense_model_trained = train_densenet(X,y,det,20, 0.01)  
dense_model_trained.save(odir+det+'.h5')


# # Det V1

# In[10]:


del X
del y
det='V1'
X , y,missing_ids, df_train =  generate_resize_densenet_fm(df_train).DenseNet_input_matrix(det = det,data_mode_dense="current",data_dir=data_dir,phenom=True)


# In[11]:


dense_model_trained = train_densenet(X,y,det,20, 0.01) #20,0.01, .005
dense_model_trained.save(odir+det+'.h5')





'''