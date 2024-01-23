import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocessing1 (path_fn, gene_fn, label_fn):
    ### pathway datasets
    pathway = pd.read_csv(path_fn, header=0)       
    print(">> Pathway Data :",path_fn)
    pathway_info = pathway.iloc[:,1:]
    pathway_info = pathway_info.values
    pathway_info = np.transpose (pathway_info)
    pathway_info = torch.FloatTensor(pathway_info)
    print("pathway matrix shape : ",pathway_info.shape)
    print("num_pathway : ",pathway_info.shape[0])

    ### methylation datasets
    print(">> Methylation Gene-level Data:",gene_fn)
    data = pd.read_csv(gene_fn, header=0)

    expression = data.iloc[:,1:]
    features = data.iloc[:,0].tolist()
    expression = expression.values
    expression = np.transpose(expression)

    scaler = MinMaxScaler()
    scaler = scaler.fit(expression)
    expression = scaler.transform(expression)

    sample_dim = expression.shape[0]
    input_dim = expression.shape[1]

    #print dimension of sample and number of genes
    print("sample_dim : ",sample_dim)
    print("input_size (number of genes): ",input_dim)
    
    pheno = pd.read_csv(label_fn, index_col=0)
    status = np.array(pheno['label']).reshape(-1,1)

    patient = list(data.iloc[:,1:].columns.values.tolist()) 
    print("patient list : ",patient[0:6])
    print("feature list : ",features[0:6])
    
    return pathway_info, expression, status, features

def preprocessing2 (path_fn, meth_fn, label_fn):
    pathway = pd.read_csv(path_fn, header=0)       
    print(">> Pathway Data :",path_fn)
    pathway_info = pathway.iloc[:,1:]
    pathway_info = pathway_info.values
    pathway_info = np.transpose (pathway_info)
    pathway_info = torch.FloatTensor(pathway_info)
    print("pathway matrix shape : ",pathway_info.shape)
    print("num_pathway : ",pathway_info.shape[0])

    ## Methylation
    print(">> Methylation beta Data:",meth_fn)
    data_meth = pd.read_csv(meth_fn, header=0)
    methylation = data_meth.iloc[:,1:]
    features_meth = data_meth.iloc[:,0].tolist()
    methylation = methylation.values
    methylation = np.transpose(methylation)

    scaler = MinMaxScaler()
    scaler = scaler.fit(methylation)
    methylation = scaler.transform(methylation)

    sample_dim2 = methylation.shape[0]
    input_dim2 = methylation.shape[1]

    #print dimension of sample and number of genes
    print("sample_dim : ", sample_dim2)
    print("input_size (number of genes, methylation sites): ", input_dim2)

    pheno = pd.read_csv(label_fn, index_col=0)
    status = np.array(pheno['label']).reshape(-1,1)

    patient = list(data_meth.iloc[:,1:].columns.values.tolist()) 
    print("patient list : ",patient[0:6])
    print("feature list : ",features_meth[0:6])

    ### Gene-methylation datasets
    data_meth = data_meth.set_index(data_meth.columns[0])
    cpgs_kept = list(pathway[pathway.columns[0]].str.split(':', expand=True)[0])
    data = data_meth.loc[cpgs_kept,:]
    data = data.reset_index()
    expression = data.iloc[:,1:]
    features = data.iloc[:,0].tolist()
    expression = expression.values
    expression = np.transpose(expression)

    scaler = MinMaxScaler()
    scaler = scaler.fit(expression)
    expression = scaler.transform(expression)

    sample_dim = expression.shape[0]
    input_dim = expression.shape[1]

    #print dimension of sample and number of genes
    print("sample_dim : ",sample_dim)
    print("input_size (number of genes): ",input_dim)

    return pathway_info, expression, methylation, status, features, features_meth

