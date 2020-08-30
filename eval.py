# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : eval.py
# Objective     :
# Created by    :
# Created on    : 08/27/2020
# Last modified : 08/27/2020 16:10
# Description   :
#   V1.0
# ************************************************************#

from data import datasets
from config import config
from models import default_model
from utils import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random

import os
import sys
from datetime import datetime

start = datetime.now()


# define training/evaluating/metrics
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim = True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float() / preds.shape[0]
    return acc, preds


def evaluate(model, device, iterator):
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            fx = model(x)

            acc, _ = calculate_accuracy(fx, y)

            epoch_acc += acc.item()
    return epoch_acc / len(iterator)


para = config.DefaultConfigs()

if os.path.exists("config.yaml"):
    para.parse("config.yaml", "eval")
sys.stdout = logger.Logger(para.records+datetime.strftime(start, '%Y%m%d')+'-evaluation.log')

device = torch.device("cuda:0" if torch.cuda.is_available() and para.gpus !="cpu"  else "cpu")
MODEL_PATH = para.checkmodel

# init data loading
NP = datasets.dataGroup()
NP.img_load("eval", "", "", para.test_data, para.batch_size)

classes = NP.test_data.classes

print("!########################################!")
print(f'[{start}]')
print(f'DEVICE: {device}')
print("initial evaluation...")
print(f'Number of testing examples: {len(NP.test_data)}')
print(f'classes: {classes}')

# init model
backbone_model = default_model.CNNModel(para.num_classes, device)
print(f'Network: {para.model_name}')

for models in os.listdir(MODEL_PATH):
    if models[-4:]==".pth":
        backbone_model.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, models), map_location = para.gpus))
        eval_acc = evaluate(backbone_model.model, device, NP.test_iterator)
        print(f'test model: {models} |accuracy: {eval_acc*100:05.2f}%')

end = datetime.now()
print("!########################################!")
print(f'| Duration: {end - start}')
