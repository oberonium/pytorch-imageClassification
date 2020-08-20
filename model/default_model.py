# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : default_model.py
# Objective     : define cnn model structures
# Created by    :
# Created on    : 08/20/2020
# Last modified : 08/20/2020 15:00:48
# Description   :
#   V1.0 support resnet-18
# ************************************************************#
import torch
import torch.nn as nn
import torchvision.models as models


class CNNModel():
    def __init__(self, num_classes = 2, dev = "cuda"):
        device = torch.device(dev)

        self.model = models.resnet18(pretrained = True).to(device)

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.fc = nn.Linear(in_features = 512, out_features = num_classes).to(device)


if __name__ == "__main__":
    new_model = CNNModel(3)
    print(new_model.model)
