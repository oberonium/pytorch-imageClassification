# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : train.py
# Objective     : define training
# Created by    :
# Created on    : 08/25/2020
# Last modified : 09/01/2020 13:36
# Description   :
#   V1.3: add multiple optim methods
#   V1.2: add img size pre-processing.
#   V1.1: add save checkpoint; resume training from checkpoint
#         save whole model
#   V1.0: define training process
# ************************************************************#


from data import datasets
from config import config
from models import default_model
from utils import logger, optimizer

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


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        fx = model(x)

        loss = criterion(fx, y)
        acc, _ = calculate_accuracy(fx, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            fx = model(x)

            loss = criterion(fx, y)
            acc, _ = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


SEED = int(datetime.strftime(start, '%s')[:3:-1])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

para = config.DefaultConfigs()

if os.path.exists("config.yaml"):
    para.parse("config.yaml")

if not os.path.exists(para.records):
    os.makedirs(para.records)
sys.stdout = logger.Logger(para.records + datetime.strftime(start, '%Y%m%d') + '-' + para.model_name + '.log')

device = torch.device("cuda:0" if torch.cuda.is_available() and para.gpus != "cpu" else "cpu")
EPOCHS = para.epochs
SAVE_DIR = para.records
CHECKPOINT_PATH = para.checkpoint + datetime.strftime(start, '%Y%m%d') + '.pth'
BEST_MODEL_SAVE_PATH = para.best_model + '_model_' + datetime.strftime(start, '%Y%m%d') + '.pth'
BEST_MODEL_DICT_SAVE_PATH = para.best_model + '_dict_' + datetime.strftime(start, '%Y%m%d') + '.pth'

# init data loading
NP = datasets.dataGroup(imgsize=(int(para.img_height), int(para.img_width)))
NP.img_load("train", para.train_data, para.val_data, para.test_data, para.batch_size)

classes = NP.test_data.classes

print("!########################################!")
print(f'[{start}]')
print(f'DEVICE: {device}')
print("initial training...")
print(f'SEED: {SEED}')
print(f'Number of training examples: {len(NP.train_data)}')
print(f'Number of validation examples: {len(NP.valid_data)}')
print(f'Number of testing examples: {len(NP.test_data)}')
print(f'classes: {classes}')

# init model
backbone_model = default_model.CNNModel(para.num_classes, device)
optimizer = optimizer.init_optimizer(backbone_model.model, para.optimizer, lr = float(para.lr), weight_decay = para.lr_weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size = para.lr_decay_step, gamma = para.lr_decay)
criterion = nn.CrossEntropyLoss()
print(f'total epoch: {EPOCHS}')
print(f'batch size: {para.batch_size}')
print(f'optimizer: {para.optimizer}')
print(f'init learning rate: {para.lr}')
print(f'Network: {para.model_name}')
print(f'image size: {para.img_height} {para.img_width}')
print("!########################################!")
# training process

if para.resume == "resume":
    checkpoint = torch.load(CHECKPOINT_PATH)
    backbone_model.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_valid_loss = checkpoint['val_loss']
    print(f'Resume training from {CHECKPOINT_PATH} epoch {resume_ep}')
else:
    start_epoch = 0
    best_valid_loss = float('inf')
    print(f'start training')

for epoch in range(start_epoch, EPOCHS):
    train_loss, train_acc = train(backbone_model.model, device, NP.train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(backbone_model.model, device, NP.valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(backbone_model.model.state_dict(), BEST_MODEL_DICT_SAVE_PATH)
        torch.save(backbone_model.model, BEST_MODEL_SAVE_PATH)

    print(
        f'| Epoch: {epoch + 1:02} | lr: {optimizer.state_dict()["param_groups"][0]["lr"]} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:05.2f}% |')
    # update learning rate
    scheduler.step()
    torch.save({'epoch': epoch + 1,
                'model_state_dict': backbone_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_valid_loss,
                }, CHECKPOINT_PATH)

end = datetime.now()
print("!########################################!")
print(f'| Duration: {end - start}')
