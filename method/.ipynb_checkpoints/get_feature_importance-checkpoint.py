import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from dataPre import *
from train import *

train = train_kfold1(trainArgs)
result,fi = train.kfold()

result.to_csv(trainArgs['filename'], mode='w')
fi['mean'] = fi.iloc[:, 1:10].mean(axis=1)
fi = fi.sort_values(by="mean", ascending=False)
fi.to_csv("../../results/feature_importance_result.csv")