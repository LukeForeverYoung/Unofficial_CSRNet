import math
import os
import torch
import torchvision as tv
from network import CSRNet
from data_set import ImageSet
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd


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

lr=1e-7
print(lr)
vgg=tv.models.vgg16(pretrained=True)
model=CSRNet(vgg)
model.cuda()

optimizer=torch.optim.SGD(model.parameters(),lr,momentum=0.90,
                          weight_decay= 5*1e-4)
loss_fn=torch.nn.MSELoss(reduction='sum').cuda()

data_size={'test_data':182,'train_data':300,'validation_data':30*18}
root_path=os.path.join('fixed_data','part_A')
train_set=ImageSet(os.path.join(root_path,'train_data'),1,data_size['train_data'], has_gt=True)
test_set=ImageSet(os.path.join('origin','part_A','test_data'),1,182, has_gt=False)
train_loder=DataLoader(train_set,batch_size=1,shuffle=True)
#load check point

model_index=0
#model.load_state_dict(torch.load(os.path.join('save','model_best.torch_model'.format(model_index))))

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
        if process%10==0:
            print('{0:.2f}'.format(process/6) ,'%')

    if epoch%10==0:
        check_point = {'net': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
        torch.save(check_point, os.path.join('save','model_{0}_.torch_model'.format(epoch)))
    epoch+=1
    mae,mse=validation(model,test_set)

    stopFlag+=1
    if bestModel==-1 or bestMAE>mae:
        check_point={'net':model.state_dict(),'opt':optimizer.state_dict(),'epoch':epoch}
        torch.save(check_point, os.path.join('save', 'model_best.torch_model'))
        bestModel=epoch
        bestMSE=mse
        bestMAE=mae
        stopFlag=0
    print('epoch:', epoch, '\t\tloss:', train_loss, '\t\tBestModel:', bestModel,'\t\tBestMSE:',bestMSE,'\t\tBestMAE:',bestMAE)
    if stopFlag==60 and epoch>=400:
        print('Break')
        break
