import os, copy, torch, random, time, datetime
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from statistics import mean 
from imblearn.over_sampling import SMOTE
import shap

from model import *
from utils import *

class feature_selection:
    def __init__(self, fsArgs):
   
        """
        args:
            x_data          : gene expression data
            y_data          : patient label
            pathway_info    : pathway matrics
            num_fc_list     : number of fully connected nodes 
            lr_list         : learning rate
            device          : GPU device
        Returns:
            10 folds' val_AUCs list
        """
        self.trainArgs = fsArgs
        self.seed_worker(fsArgs['seed'])
        os.environ["CUDA_VISIBLE_DEVICES"]=fsArgs['device']
        self.device = torch.device('cuda')
        directory = './feature_selection'
        if not os.path.exists(directory):
            os.makedirs(directory)

    def kfold(self):
        trainArgs = self.trainArgs
        x_data = trainArgs['x_data']
        input_dim = x_data.shape[1]
        y_data = trainArgs['y_data']
        print(f'Input data number: {x_data.shape[1]}')
        
        pathway_info = trainArgs['pathway_info'].to(self.device)
        num_pathway = pathway_info.shape[0]
        num_fc_list = trainArgs['num_fc_list']
        lr_list = trainArgs['lr_list']
        random_seed = trainArgs['seed']

        kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = random_seed)
        
        val_auc_ls = []
        val_acc_ls = []
        val_precision_ls = []
        val_recall_ls = []
        val_f1_ls = []
        
        for fold, (train_index, test_index) in enumerate(kfold.split(x_data, y_data)):   
            print(f'Fold {fold+1}')
            x_train_ = x_data[train_index]
            y_train_ = y_data[train_index] 
            x_test = x_data[test_index]  
            y_test = y_data[test_index] 
            x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, 
                                                              test_size=1/9, random_state = random_seed, stratify = y_train_)
            
            smote = SMOTE(random_state=random_seed)
            x_train, y_train = smote.fit_resample(x_train,y_train)
            y_train = y_train.reshape(-1,1)                              

            train_dataset = CustomDataset(x_train,y_train)
            val_dataset = CustomDataset(x_val,y_val)
            test_dataset = CustomDataset(x_test,y_test) 
                       
            train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
            val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False)
            test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False)            
                   
            best_val_auc = 0
            for  lr in lr_list:
                for num_fc in num_fc_list:
                    self.model = PINNet(input_dim,pathway_info,num_fc)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = 0)
                    self.criterion = nn.CrossEntropyLoss()
                    self.model = self.model.to(self.device)
                    early_stopping = EarlyStopping(patience=10, verbose = True, path = 'checkpoint_ES.pt')
                   
                    ##train 
                    for epoch in range(0, 200):
                        for batch_idx, samples in enumerate(train_loader):
                            _,_ = self.train_step(samples,training = True)
                        ##early stopping
                        y_prob, y_true = [],[]
                        for batch_idx, samples in enumerate(val_loader):
                            prob, true = self.train_step(samples,training = False)
                            # print(prob)
                            y_prob.extend(prob.detach().cpu().numpy())
                            y_true.extend(true.cpu().numpy())
                        
                        val_auc, _, _, _, _ = self.evalutaion(y_true,y_prob)
                        
                        # print(val_auc)
                        early_stopping(val_auc, self.model, epoch)
                        if early_stopping.early_stop:
                            break
                            
                    ##validation 
                    self.model = torch.load('checkpoint_ES.pt')
                    y_prob, y_true = [],[]
                    for batch_idx, samples in enumerate(val_loader):     
                        prob, true = self.train_step(samples,training = False)
                        y_prob.extend(prob.detach().cpu().numpy())
                        y_true.extend(true.cpu().numpy())

                    val_auc, val_precision, val_recall, val_f1, val_acc = self.evalutaion(y_true,y_prob)
                    print(">>[val] auc : {:.4f}, precision : {:.4f}, recall : {:.4f}, f1 : {:.4f}"
                          .format(val_auc, val_precision, val_recall, val_f1))                    
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc   
                    val_auc_ls.append(best_val_auc)
                    val_precision_ls.append(val_precision)
                    val_recall_ls.append(val_recall)
                    val_acc_ls.append(val_acc)
                    val_f1_ls.append(val_f1)
                    
        return val_auc_ls, val_precision_ls, val_recall_ls, val_acc_ls, val_f1_ls
    
    def train_step(self, batch_item, training):
        data,label = batch_item
        data = data.to(self.device)
        label = label.to(self.device)
        if training is True:
            self.model.train()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = self.model(data)
                true = torch.reshape(label,(-1,))
                loss = self.criterion(out,true)
                prob = out[:,1]                
                
            loss.backward()
            self.optimizer.step()
            return prob, true
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(data)
                # print(out)
                # print(data)
                true = torch.reshape(label,(-1,))
                prob = out[:,1]   
            return prob, true  
        
    def evalutaion(self, y_true, y_prob):
        np.seterr(divide='ignore', invalid='ignore')
        auc = roc_auc_score(y_true,y_prob)
        precision,recall,_ = precision_recall_curve(y_true,y_prob)
        f1 = (2*precision*recall)/(precision+recall)
        idx = np.nanargmax(f1)
        pr = precision[idx] 
        rc = recall[idx] 
        f1 = f1[idx] 
        threshold = 0.5
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        acc = accuracy_score(y_true, y_pred)
        return auc, pr, rc, f1, acc

    def seed_worker(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)