# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : config.py
# Objective     : get configurations for train or inference
# Created by    :
# Created on    : 08/20/2020
# Last modified : 08/20/2020 15:05:09
# Description   :
#   V1.0 basic function
# ************************************************************#

import os
import yaml


class DefaultConfigs(object):
    def __init__(self):
        # Datasets
        self.train_data = "../data/train/"
        self.test_data = "../data/test/"
        self.val_data = "../data/val/"
        self.classes = ["neg", "pos"]
        self.num_classes = 2

        # models
        self.model_name = "resnet-18"
        self.records = "../checkpoints/"
        self.best_model = self.records + "bestModel.pth"
        self.checkpoint = self.records + "checkpoint"
        self.logs = "./log.txt"

        # hyper-parameters
        self.epochs = 40
        self.batch_size = 4
        self.img_height = 1244
        self.img_weight = 1024
        self.optimiser = "Adam"
        self.lr = 1e-3
        self.lr_decay = 1e-4
        self.lr_decay_step = 10
        self.lr_weight_decay = 0

        # hardware
        self.gpus = "1"
        self.mode = "train"

    def parse(self, yaml_path = 'config.yaml', mode = "train"):
        with open(yaml_path, 'rb') as f:
            params = yaml.load(f, Loader = yaml.FullLoader)

        self.train_data = params['train_data']
        self.test_data = params['lr_weight_decay']
        self.val_data = params['lr_weight_decay']

        # models
        self.model_name = params['model_name']


if __name__ == "__main__":
    if os.path.exists("config.yaml"):
        config = DefaultConfigs()
        config.parse()
    else:
        config = DefaultConfigs()
