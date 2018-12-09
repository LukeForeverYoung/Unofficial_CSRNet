import math
import os
import torch
import torchvision as tv
from network import CSRNet
from data_set import ImageSet
from torch.utils.data.dataloader import DataLoader
import numpy as np
def validation(model,validation):
    val_loder = DataLoader(validation, batch_size=1, shuffle=True)
    mse=0
    mae=0
    for img,gt in val_loder:
        pet = np.sum(model(img).data.cpu().numpy())
        gt=np.sum(gt.data.cpu().numpy())
        dif=np.abs(pet-gt)
        mae+=dif
        mse+=dif*dif
    mae/=len(val_loder.dataset)
    mse=np.sqrt(mse/len(val_loder.dataset))
    return mae,mse


lr=1e-6
print(lr)
vgg=tv.models.vgg16(pretrained=True)
model=CSRNet(vgg)
model.cuda()

optimizer=torch.optim.SGD(model.parameters(),lr,)
loss_fn=torch.nn.MSELoss()
data_size={'test_data':182*18,'train_data':(300-30)*18,'validation_data':30*18}
root_path=os.path.join('fixed_data','part_A')
train_set=ImageSet(os.path.join(root_path,'train_data'), has_gt=True)
validation_data=ImageSet(os.path.join(root_path,'validation_data'), has_gt=True)
train_loder=DataLoader(train_set,batch_size=1,shuffle=True)

#load check point
model_index=250
model.load_state_dict(torch.load(os.path.join('save','model_{0}_.torch_model'.format(model_index))))
epoch=model_index
bestMSE=math.inf
bestMAE=math.inf
bestModel=-1
stopFlag=0
while True:
    train_loss=0
    process=0

    for img,gt in train_loder:
        den=model(img)
        loss=loss_fn(den,gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        process+=1
        if process%240==0:
            print(process/len(train_loder.dataset)*100,'%')

    torch.save(model.state_dict(), os.path.join('save','model_{0}_.torch_model'.format(epoch)))
    epoch+=1
    mae,mse=validation(model,validation_data)

    stopFlag+=1
    if bestModel==-1 or bestMSE>mse:
        bestModel=epoch
        bestMSE=mse
        bestMAE=mae
        stopFlag=0
    print('epoch:', epoch, '\t\tloss:', train_loss, '\t\tBestModel:', bestModel,'\t\tBestMSE:',bestMSE)
    if stopFlag==30:
        print('Break')
        break