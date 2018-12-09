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

from data_set import ImageSet
from torch.utils.data.dataloader import DataLoader
import torchvision as tv
from network import CSRNet


def validation(model,test_set):
    mse=0
    mae=0
    for i in range(0,182):
        img,path=test_set.__getitem__(i)

        index=path.split('_')[-1].split('.')[0]
        img=img.view((-1,img.shape[0],img.shape[1],img.shape[2]))
        pet=model(img).data.cpu().numpy()
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

        ground_truth = pd.read_csv(os.path.join('origin','part_A','test_data', 'ground_truth/GT_IMG_{0}.csv'.format(index)), header=None)
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
model.cuda()
model.load_state_dict(torch.load(os.path.join('save','model_505_.torch_model')))
test_set=train_set=ImageSet(os.path.join(root_path,'test_data'), has_gt=False)

mae,mse=validation(model,test_set)

print(mae,mse)

