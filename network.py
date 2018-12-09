import torch
import torchvision as tv
import numpy as np
import torch.nn as nn
class CSRNet(nn.Module):
    def __init__(self,vgg):
        super(CSRNet, self).__init__()
        self.forwardEnd=nn.Sequential(*list(vgg.children())[0][:23])
        self.backEnd=self.initBackEnd()
        self.fuse=nn.Conv2d(64,1,1)
        torch.nn.init.normal_(self.fuse.weight, mean=0, std=0.01)

    def forward(self, x):
        x=self.forwardEnd(x)
        x=self.backEnd(x)
        x=self.fuse(x)
        return x



    def initBackEnd(self):
        layers=[]
        cfg=[512,512,512,256,128,64]
        in_channels=512
        for i in cfg:
            layers.append(self.makeConv(in_channels,i,3,2))
            in_channels=i
        return nn.Sequential(*layers)
    def makeConv(self,in_channels,out_channels,kernel_size,dilation):
        layers=[]
        padding=dilation
        conv=nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       padding=padding,
                       dilation=dilation)
        torch.nn.init.normal_(conv.weight, mean=0, std=0.01)
        layers.append(conv)
        relu=nn.ReLU(inplace=True)
        layers.append(relu)
        return nn.Sequential(*layers)

