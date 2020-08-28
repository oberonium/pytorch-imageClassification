# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : datasets.py
# Objective     : define dataloading
# Created by    :
# Created on    : 08/25/2020
# Last modified : 08/28/2020 18:28
# Description   :
#   V1.0
# ************************************************************#

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class dataGroup():
    def __init__(self, imgsize=(78,65), norm_mean=(0.485, 0.456, 0.406), norm_dev=(0.229, 0.224, 0.225)):
        """
        define data directory
        """
        self.train_transforms = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_dev)
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_dev)
        ])

        self.train_data = ""
        self.valid_data = ""
        self.test_data = ""
        self.train_iterator = ""
        self.valid_iterator = ""
        self.test_iterator = ""

    def imgdata(self, images_dir):
        """
        images data prepared(if required)
        """
        pass

    def img_load(self, mode, train_dir, valid_dir, test_dir, BATCH_SIZE=1, TEST_BATCH_SIZE=1):
        if mode == "train":
            self.train_data = datasets.ImageFolder(train_dir, self.train_transforms)
            self.valid_data = datasets.ImageFolder(valid_dir, self.test_transforms)
            self.test_data = datasets.ImageFolder(test_dir, self.test_transforms)
            self.train_iterator = torch.utils.data.DataLoader(self.train_data, shuffle = True, batch_size = BATCH_SIZE)
            self.valid_iterator = torch.utils.data.DataLoader(self.valid_data, shuffle = True, batch_size = BATCH_SIZE)
            self.test_iterator = torch.utils.data.DataLoader(self.test_data, batch_size = TEST_BATCH_SIZE)
        if mode == "eval":
            self.test_data = datasets.ImageFolder(test_dir, self.test_transforms)
            self.test_iterator = torch.utils.data.DataLoader(self.test_data, batch_size = TEST_BATCH_SIZE)


if __name__ == "__main__":
    test = dataGroup()
    test.img_load("eval", "../images/testing")
    print(test.test_iterator)
