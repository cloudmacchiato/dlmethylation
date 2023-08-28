import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from dataPre import *

from train import *
train = train_kfold2(trainArgs)
result = train.kfold()

result.to_csv("result2.csv")