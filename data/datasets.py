# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : datasets.py
# Objective     : define dataloading
# Created by    :
# Created on    : 08/25/2020
# Last modified : 08/25/2020 13:40
# Description   :
#   V1.0
# ************************************************************#

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class dataGroup():
    def __init__(self, train_dir, valid_dir, test_dir, imgsize=(78,65), norm_mean=(0.485, 0.456, 0.406), norm_dev=(0.229, 0.224, 0.225)):
        """
        define data directory
        """
        train_transforms = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_dev)
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_dev)
        ])

        self.train_data = datasets.ImageFolder(train_dir, train_transforms)
        self.valid_data = datasets.ImageFolder(valid_dir, test_transforms)
        self.test_data = datasets.ImageFolder(test_dir, test_transforms)

        self.train_iterator = ""
        self.valid_iterator = ""
        self.test_iterator = ""

    def imgdata(self, images_dir):
        """
        images data prepared(if required)
        """
        pass

    def img_iterator(self, BATCH_SIZE=1, TEST_BATCH_SIZE=1):
        self.train_iterator = torch.utils.data.DataLoader(self.train_data, shuffle = True, batch_size = BATCH_SIZE)
        self.valid_iterator = torch.utils.data.DataLoader(self.valid_data, shuffle = True, batch_size = BATCH_SIZE)
        self.test_iterator = torch.utils.data.DataLoader(self.test_data, batch_size = TEST_BATCH_SIZE)


if __name__ == "__main__":
    test = dataGroup("../images/train", "../images/valid", "../images/testing")
    test.img_iterator(4)
    print(test.train_iterator)
