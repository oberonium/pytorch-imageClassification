# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : eval.py
# Objective     :
# Created by    :
# Created on    : 08/27/2020
# Last modified : 09/02/2020 10:57
# Description   :
#   V1.2 add image size
#        add specificity; pos acc; neg acc
#   V1.1 add confusion matrix; F1 score; precision
#        fixed gpu training model load into gpu/cpu bug
#   V1.0 basic function
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
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
    return acc, correct.item()


def evaluate(model, device, iterator):
    y_pred = []
    y_gt = []

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)
            _, preds = torch.max(fx, 1)

            y_pred.append(preds.view(-1).item())
            y_gt.append(y.view(-1).item())

    return y_pred, y_gt


para = config.DefaultConfigs()

if os.path.exists("config.yaml"):
    para.parse("config.yaml", "eval")
sys.stdout = logger.Logger(para.records + datetime.strftime(start, '%Y%m%d') + '-evaluation.log')

device = torch.device("cuda:0" if torch.cuda.is_available() and para.gpus != "cpu" else "cpu")
MODEL_PATH = para.checkmodel

# init data loading
NP = datasets.dataGroup(imgsize=(int(para.img_height), int(para.img_width)))
NP.img_load("eval", "", "", para.test_data, para.batch_size)

classes = NP.test_data.classes

print("!########################################!")
print(f'[{start}]')
print(f'DEVICE: {device}')
print("initial evaluation...")
print(f'Number of testing examples: {len(NP.test_data)}')
print(f'classes: {classes}')
print(f'image size: {para.img_height} {para.img_width}')
# init model
backbone_model = default_model.CNNModel(para.num_classes, device)
#backbone_model.model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
print(f'Network: {para.model_name}')

print("!########################################!")
for models in os.listdir(MODEL_PATH):
    if models[-4:] == ".pth":

        if device == "cpu":
            backbone_model.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, models), map_location = para.gpus))
        else:
            backbone_model.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, models)))

        pred_list, true_list = evaluate(backbone_model.model, device, NP.test_iterator)
        cm = confusion_matrix(true_list, pred_list)
        eval_acc = accuracy_score(true_list, pred_list)

        if len(classes) == 2:
            p = precision_score(true_list, pred_list, average = 'binary')
            r = recall_score(true_list, pred_list, average = 'binary')
            f1score = f1_score(true_list, pred_list, average = 'binary')
            clas_acc = cm.diagonal() / cm.sum(axis = 1)
            tn, fp, fn, tp = confusion_matrix(true_list, pred_list).ravel()
            speci = tn / (tn+fp)
            print(
                f'test model: {models} |accuracy: {eval_acc * 100:05.2f}% |precision: {p * 100:05.2f}% |specificity: {speci * 100:05.2f}% |recall: {r * 100:05.2f}% |f1: {f1score * 100:05.2f}% |{classes[0]}_acc: {clas_acc[0] * 100:05.2f}% |{classes[1]}_acc: {clas_acc[1] * 100:05.2f}% | tn;tp {tn/(tn+fn) * 100:05.2f}% {tp/(tp+fp) * 100:05.2f}%')
        else:
            print(f'test model: {models} |accuracy: {eval_acc * 100:05.2f}% ')

end = datetime.now()
print(f'| Duration: {end - start}')
