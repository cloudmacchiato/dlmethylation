#!/usr/bin/env python
# coding: utf-8

# In[73]:


import os, copy, torch, random, time, datetime
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from statistics import mean 
from imblearn.over_sampling import SMOTE
import shap
import pickle
from model import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# In[74]:


path_fn = "../../data/pathway_gobp_new.csv"
meth_fn = "../../data/beta.by.intensity.all.regions.csv"
label_fn = "../../data/label.csv"


# In[75]:


meth = pd.read_csv(meth_fn, header=0,index_col=0).T
pathway = pd.read_csv(path_fn, header=0,index_col=0)
pathway.index = pathway.index.str.split(':').str[0]
pheno = pd.read_csv(label_fn, header=0,index_col=0)
y_data = np.array(pheno['label']).reshape(-1,1)


# In[76]:


#pathway


# In[77]:


scaler = MinMaxScaler()
scaler = scaler.fit(meth)
x_data = scaler.transform(meth)


# In[78]:


random_seed = 0
all_cpgs = meth.columns.to_list()
all_path_cpgs = pathway.index.to_list()
num_cpgs = len(all_cpgs)
sel_feat_idx = {}
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = random_seed)
for fold, (train_index, test_index) in enumerate(kfold.split(x_data, y_data)):   
    print('****************************************************************************')
    print('Fold {} / {}'.format(fold + 1 , kfold.get_n_splits()))
    print('****************************************************************************')
    x_train_ = x_data[train_index]
    y_train_ = y_data[train_index] 
    x_test = x_data[test_index]  
    y_test = y_data[test_index] 
    x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, 
                                                      test_size=1/9, random_state = random_seed, stratify = y_train_)

    smote = SMOTE(random_state=random_seed)
    x_train, y_train = smote.fit_resample(x_train,y_train)
    y_train = y_train.reshape(-1,1)
    #print(x_train.shape,y_train.shape)
    mutual_info = list(mutual_info_classif(x_train, y_train))
    sel_feats_idx = sorted(range(len(mutual_info)), key=lambda i: mutual_info[i], reverse=True)[:50000]
    sel_cpgs = [all_cpgs[i] for i in sel_feats_idx]
    path_cpgs = list(set(all_path_cpgs).intersection(set(sel_cpgs)))
    sel_path_cpg_idx1 = [all_cpgs.index(value) for value in tqdm(path_cpgs)]
    sel_path_cpg_idx2 = [all_path_cpgs.index(value) for value in tqdm(path_cpgs)]
    all_sel_idx = sel_path_cpg_idx1 + [x + num_cpgs for x in sel_feats_idx]
    sel_feat_idx.update({f'fold{fold}': [all_sel_idx, sel_path_cpg_idx2]})
    
    dict_data = {"train": train_index, "test": test_index}
    with open(f"../../data/folds/fold_{fold+1}.pkl", 'wb') as file:
        pickle.dump(dict_data, file)


# In[79]:


with open("mi_feat_idx.pkl", 'wb') as file:
    pickle.dump(sel_feat_idx, file)

