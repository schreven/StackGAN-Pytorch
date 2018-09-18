#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:24:44 2018

@author: cyril-kubu
"""

import torch

import numpy as np
import torchfile
import pickle


from torch.utils.serialization import load_lua

orig_embeds = torchfile.load('/home/cyril-kubu/Documents/github_repos/StackGAN-Pytorch-schreven/data/coco/test/val_captions.t7')#, unknown_classes = True)
#tst = torch.load('/home/cyril-kubu/Documents/github_repos/StackGAN-Pytorch-schreven/data/coco/test/val_captions.t7')


new_raw_list = orig_embeds['raw_txt'][0:10]
new_feat_list = orig_embeds['fea_txt'][0:10]

#new_embeds = torchfile.hashable_uniq_dict()
new_embeds = dict()


new_embeds['raw_txt'] = new_raw_list
new_embeds['fea_txt'] = new_feat_list



#pickle.dump(new_embeds,'val_captions_custom.t7')
torch.save(new_embeds, 'val_captions_custom.t7')