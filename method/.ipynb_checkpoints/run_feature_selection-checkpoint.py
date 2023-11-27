import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dataPre import *

from feature_selection import *

fi = pd.read_csv("feature_importance_result.csv",index_col=0)

n_ls = []
means = [[],[],[],[],[],[]]
stds = [[],[],[],[],[],[]]
best_n = [0,0,0,0,0]
best = [0,0,0,0,0]
ls_names = ["AUC","Precision","Recall","Accuracy","F1"]
for n_features in range(0,20490,50):
    try:
        fsArgs = {}
        fsArgs['x_data'] = expression[:,fi.head(n_features).index.tolist()]
        fsArgs['y_data'] = status
        fsArgs['pathway_info'] = pathway_info[:,fi.head(n_features).index.tolist()]
        # trainArgs['num_fc_list'] = [32, 64, 128]
        # trainArgs['lr_list'] = [0.0001,0.0005,0.001]
        fsArgs['num_fc_list'] = [32]
        fsArgs['lr_list'] = [0.0001]
        fsArgs['device'] = '0'
        fsArgs['seed'] = 0
        fsArgs['pathway'] = pathway
        fsArgs['tissue'] = tissue
        
        fs = feature_selection(fsArgs)
        val_auc_ls, val_precision_ls, val_recall_ls, val_acc_ls, val_f1_ls = fs.kfold()
        n_ls.append(n_features)
        for i,ls in enumerate([val_auc_ls, val_precision_ls, val_recall_ls, val_acc_ls, val_f1_ls]):
            name = ls_names[i]
            m = mean(ls)
            if m > best[i]:
                best[i] = m
                best_n[i] = n_features
            std = np.std(ls)
            means[i].append(m)
            stds[i].append(std)
            plt.plot(n_ls, means[i], label = f'best n: {best_n[i]}, {name}: {round(best[i],3)}')
            for j in range(len(n_ls)):
                plt.fill_between([n_ls[j]], means[i][j] - stds[i][j], means[i][j] + stds[i][j], color='gray', alpha=0.2)
            plt.xlabel('Input Feature Number')
            plt.ylabel(f'Valid {name}')
            plt.title(f'Feature Selection for {name}')
            plt.legend()
            plt.savefig(f'./feature_selection/n_features_{name}.png',dpi=400)
            plt.close()
    except Exception as e:
        print("error:", n_features)