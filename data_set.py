import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
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
    def __init__(self, path,sta_index,length, has_gt=True):
        self.data_size=length
        self.img=[]
        self.is_train=has_gt
        self.transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                   ])
        if has_gt:
            self.gt=[]
        for i in range(sta_index,sta_index+length):
            if has_gt:
                img,gt=self.read_item(os.path.join(path, 'images', 'IMG_{0}.JPG'.format(i)),
                                      os.path.join(path, 'ground_truth',  'IMG_{0}.npy'.format(i)))
                self.img.append(img)
                self.gt.append(gt)
                if 'test' not in path:
                    img, gt = self.read_item(os.path.join(path, 'images', 'IMG_{0}_m.JPG'.format(i)),
                                             os.path.join(path, 'ground_truth', 'IMG_{0}_m.npy'.format(i)))
                    self.img.append(img)
                    self.gt.append(gt)
            else:
                img= self.read_item(os.path.join(path, 'images', 'IMG_{0}.JPG'.format(i)))
                self.img.append(img)
                if 'test' not in path:
                    img= self.read_item(os.path.join(path, 'images', 'IMG_{0}_m.JPG'.format(i)))
                    self.img.append(img)

        self.data_size=len(self.img)

    def read_item(self,img_path,gt_path=None):
        img=Image.open(img_path).convert('RGB')
        #img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #img=img/255;
        #img=cv2.resize(img,(img.shape[1]*8,img.shape[0]*8))
        #print(img_path)
        #img = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=0)#convert to C H W
        #print(img.shape)
        if gt_path!=None:
            gt=np.load(gt_path)
            gt=gt.reshape((1,gt.shape[0],gt.shape[1]))
            return img,gt
        return img

    def __getitem__(self, item):
        if self.is_train:
            img,gt=(self.img[item],self.gt[item])
            return npToTensor(self.transform(img),requires_grad=True),torch.FloatTensor(gt).cuda()
        else:
            img= self.img[item]
            return npToTensor(self.transform(img))

    def __len__(self):
        return self.data_size