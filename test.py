import os

import cv2
import torch

from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from skimage import transform
import queue
import threading
from multiprocessing import Process, Pool
import pandas as pd
import random as r
import scipy as sc
import torch.nn as nn
from data_set import ImageSet
from torchvision import models
from torch.utils.data.dataloader import DataLoader
import torchvision as tv
from torchvision import transforms
from network import CSRNet




def validation(model,test_set):
    mse=0
    mae=0
    for i in range(0,182):
        img=test_set.__getitem__(i)
        '''
        img = Image.open(os.path.join(root_path,'test_data','images','IMG_{0}.jpg'.format(i+1))).convert('RGB')
        img=transform(img)
        img = img.cuda()
'''
        pet=model(img.unsqueeze(0)).data.cpu().numpy()
        '''
        print(pet.shape)
        plt.subplot(1,2,1)
        plt.imshow(img.data.cpu().numpy()[0][0])
        plt.subplot(1, 2, 2)
        plt.imshow(pet[0][0])
        plt.show()
        input()
        '''
        pet = np.sum(pet)

        ground_truth = pd.read_csv(os.path.join('origin','part_A','test_data', 'ground_truth/GT_IMG_{0}.csv'.format(i+1)), header=None)
        ground_truth = ground_truth.values.shape[0]

        gt=ground_truth
        dif=np.abs(pet-gt)
        mae+=dif
        mse+=dif*dif
    mae/=182
    mse=np.sqrt(mse/182)
    return mae,mse

root_path=os.path.join('origin','part_A')

vgg=tv.models.vgg16(pretrained=True)
model=CSRNet(vgg)
'''
state=torch.load(os.path.join('PartAmodel_best.pth.tar'))['state_dict']

dic={}
for (k1,v1),(k2,v2) in zip(model.state_dict().items(),state.items()):
    print(k1,k2)
    dic[k1]=v2
'''
#model.load_state_dict(dic)

model.load_state_dict(torch.load(os.path.join('model_best_ok.torch_model'))['net'])
model.cuda()
test_set=train_set=ImageSet(os.path.join(root_path,'test_data'),1,182, has_gt=False)

mae,mse=validation(model,test_set)

print(mae,mse)

