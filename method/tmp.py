import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from boruta import BorutaPy

with open("mi_feat_idx_kegg.pkl", 'rb') as file:
    sel_feat_idx = pickle.load(file)
    
all_cpgs = list(pd.read_csv("../../data/beta.by.intensity.all.regions.csv").iloc[:,0])
all_path_cpgs = list(pd.read_csv("../../data/pathway_gobp_new.csv").iloc[:,0])

all_path_cpg = []
for i in all_path_cpgs:
    all_path_cpg.append(i.split(":")[0])
    
dic = {}
num_cpgs = len(all_cpgs)
for fold in range(0,5):
    sel_feats_idx = sel_feat_idx[f"fold{fold}"][2]
    sel_cpgs = [all_cpgs[i] for i in sel_feats_idx]
    path_cpgs = list(set(all_path_cpg).intersection(set(sel_cpgs)))
    sel_path_cpg_idx1 = [all_cpgs.index(value) for value in tqdm(path_cpgs)]
    sel_path_cpg_idx2 = [all_path_cpg.index(value) for value in tqdm(path_cpgs)]
    all_sel_idx = sel_path_cpg_idx1 + [x + num_cpgs for x in sel_feats_idx]
    dic.update({f'fold{fold}': [all_sel_idx, sel_path_cpg_idx2, sel_feats_idx]})
    
with open("mi_feat_idx_gobp.pkl", 'wb') as file:
    pickle.dump(dic, file)