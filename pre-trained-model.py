#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:11:49 2017

@author: akash
"""


import torch 
import torchvision.models as models
from torch.autograd import Variable
import time 

resnet = models.resnet18(pretrained = True)



img = torch.rand(3,224,224) 
img.unsqueeze_(0) # 3*224*224 --> 1*3*224*224
resnet.eval()     





a = time.time()
img = Variable(img,True)

for i in range(100):
    
    out = resnet(img)
    out.backward(out)
    print (time.time() - a)
    a = time.time()
    

