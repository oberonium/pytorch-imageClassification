# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : optimizer.py
# Objective     :
# Created by    :
# Created on    : 09/08/2020
# Last modified : 09/08/2020 11:07:16
# Description   :
#   V1.0 11 optimizer for torch 1.3
# ************************************************************#

import torch.optim as optim

def init_optimizer(model, opt="Adam", lr=1e-3, weight_decay=0.9):
    if opt == "SGD":
        return optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="ASGD":
        return optim.ASGD(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="Rprop":
        return optim.Rprop(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="Adagrad":
        return optim.Adagrad(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="Adadelta":
        return optim.Adadelta(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="RMSprop":
        return optim.RMSprop(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="Adam":
        return optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="Adamax":
        return optim.Adamax(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="AdamW":
        return optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="SparseAdam":
        return optim.SparseAdam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif opt =="LBFGS":
        return optim.LBFGS(model.parameters(), lr = lr, weight_decay = weight_decay)
    else:
        print("wrong optimizer!")
