import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def npToTensor(x,is_cuda=True,requires_grad=False,dtype=torch.FloatTensor):
    if isinstance(x,torch.Tensor):
        t=x.type(dtype)
    else:
        t = torch.from_numpy(x).type(dtype)
    t.requires_grad=requires_grad
    if(is_cuda):
        t=t.cuda()
    return t
class ImageSet(Dataset):
    def __init__(self, path, has_gt=True):
        self.data_size=len
        self.img=[]
        self.is_train=has_gt
        if has_gt:
            self.gt=[]

        for root,dir,files in os.walk(os.path.join(path,'images')):
            for file in files:
                '''
                index=int(file.split('_')[1])
                pat=int(file.split('_')[2].split('patch')[1].split('.')[0])
                is_mir='_m' in file
                print(index,pat,is_mir)
                '''
                self.img.append(os.path.join(path,'images',file))
                if has_gt:
                    self.gt.append(os.path.join(path,'ground_truth',os.path.splitext(file)[0]+'.npy'))
        self.data_size=len(self.img)

    def read_item(self,img_path,gt_path=None):
        img=cv2.imread(img_path,0)
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #img=cv2.resize(img,(img.shape[1]*8,img.shape[0]*8))
        img = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=0)#convert to C H W
        if gt_path!=None:
            gt=np.load(gt_path)
            gt=gt.reshape((1,gt.shape[0],gt.shape[1]))
            return img,gt
        return img

    def __getitem__(self, item):
        if self.is_train:
            img,gt=self.read_item(self.img[item],self.gt[item])
            return npToTensor(img,requires_grad=True),torch.FloatTensor(gt).cuda()
        else:
            img= self.read_item(self.img[item],None)
            return npToTensor(img),self.img[item]

    def __len__(self):
        return self.data_size