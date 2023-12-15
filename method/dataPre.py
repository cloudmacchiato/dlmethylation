import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocessing (path, tissue):
    ### pathway datasets
    if (path == "GO"):
        pathway = pd.read_csv("../../data/pathway_gobp.csv", header=0)
    elif (path == "KEGG"):
        pathway = pd.read_csv("../../data/pathway_kegg.csv", header=0)       
    print(">> Pathway Data :",path)

    pathway_info = pathway.iloc[:,1:]
    pathway_info = pathway_info.values
    pathway_info = np.transpose (pathway_info)
    pathway_info = torch.FloatTensor(pathway_info)
    print("pathway matrix shape : ",pathway_info.shape)
    print("num_pathway : ",pathway_info.shape[0])

    ### methylation datasets
    print(">> Methylation Data: gene.average.beta.by.intensity.csv")
    data = pd.read_csv("../../data/gene.average.beta.by.intensity.csv", header=0)

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
    
    pheno = pd.read_csv("../../data/label.csv",index_col=0)
    status = np.array(pheno['label']).reshape(-1,1)


    patient = list(data.iloc[:,1:].columns.values.tolist()) 
    print("patient list : ",patient[0:6])
    print("feature list : ",features[0:6])
    
    return pathway_info, expression, status, features


pathway = "KEGG"
tissue = "brca"
pathway_info, expression, status, features = preprocessing(pathway, tissue)

trainArgs = {}
trainArgs['x_data'] = expression
trainArgs['y_data'] = status
trainArgs['pathway_info'] = pathway_info
trainArgs['features'] = features
# trainArgs['num_fc_list'] = [32, 64, 128]
# trainArgs['lr_list'] = [0.0001,0.0005,0.001]
trainArgs['num_fc_list'] = [32]
trainArgs['lr_list'] = [0.0001]
trainArgs['device'] = '0'
trainArgs['seed'] = 0
trainArgs['pathway'] = pathway
trainArgs['tissue'] = tissue
trainArgs['filename'] = 'result.csv'
