import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from dataPre import *

from train import *

trainArgs = {}
trainArgs['x_data'] = expression
trainArgs['y_data'] = status
trainArgs['pathway_info'] = pathway_info
trainArgs['features'] = features
trainArgs['num_fc_list'] = [32]
trainArgs['lr_list'] = [0.001]
#trainArgs['num_fc_list'] = [32]
#trainArgs['lr_list'] = [0.0001]
trainArgs['device'] = '0'
trainArgs['seed'] = 0
trainArgs['pathway'] = pathway
trainArgs['tissue'] = tissue
trainArgs['filename'] = 'result.csv'

train = train_kfold2(trainArgs)
result = train.kfold()

result.to_csv("../../results/GOBP/result2.csv")