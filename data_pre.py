import os
import shutil

import cv2

from PIL import Image
import scipy.io as sio
import  matplotlib.pyplot as plt
import numpy as np
import csv
import math
from skimage import transform
import queue
import threading
from multiprocessing import Process,Pool
import pandas as pd
import random as r
import scipy as sc
filter_size=20
beta=0.3
padding=int(filter_size/2)
k_close=3



def getGaussianFilter(size,sigma):
    mid=int(size/2)
    sum=0
    gaussian = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - mid) / np.square(sigma)  # 生成二维高斯分布矩阵
                                            + (np.square(j - mid) / np.square(sigma)))) / (2 * math.pi * sigma * sigma)
            sum = sum + gaussian[i, j]
    gaussian = gaussian / sum
    return gaussian


def distance(pointA,pointB):
    return math.sqrt((pointA[0]-pointB[0])*(pointA[0]-pointB[0])+(pointA[1]-pointB[1])*(pointA[1]-pointB[1]))


def kCloseMean(points,index,k):
    dMean=0
    stdp=points[index]
    points.remove(stdp)
    q=queue.PriorityQueue(k)
    for point in points:
        if(q.full()):
            q.get()
        q.put(distance(point,stdp)*-1)
    count = q.qsize()
    while not q.empty():
        dMean+=q.get()*-1
    if(count==0):
        return 1
    return dMean*1.0/count

def floor_div2(v,k):
    v=float(v)
    for i in range(k):
        v=math.floor(v/2)
    return v


def cut_patch(image,gt):
    #gt[i][0]->x gt[i][1]->y
    # 9 img
    (h,w,c) = image.shape
    nw = floor_div2(w,1)
    nh = floor_div2(h,1)
    imgs=[]
    points=[]
    for i in range(9):
        if i<4:
            x=i//2*nw
            y=i%2*nh
        else:
            x=r.randint(0,nw-1)
            y=r.randint(0,nh-1)
        imgs.append(image[y:y+nh,x:x+nw,1])
        point_patch=[]
        for point in gt:
            yy = int(point[1])
            xx = int(point[0])
            if y<=yy and yy<y+nh and x<=xx and xx<x+nw:
                yy-=y
                xx-=x
                point_patch.append([yy,xx])
        point_patch=np.array(point_patch)
        points.append(point_patch)
    return imgs,points

def generate_density(img,pts):
    [h, w] = [img.shape[0], img.shape[1]]
    density = np.zeros((h, w))
    if len(pts)==0:
        return density
    tree = sc.spatial.cKDTree(pts)
    distances, locations = tree.query(pts, k=4)
    p_size = len(pts)
    for j, pt in enumerate(pts):
        sub_den = np.zeros((h, w))
        sub_den[pt[0], pt[1]] = 1
        if p_size <= 1:
            sigma = np.average(np.array(pts.shape))
        else:
            # distances[0] is it self
            # beta=0.3
            # avgd=(d[1]+d[2]+d[3])/3*beta
            # so avgd=(d[1]+d[2]+d[3])*0.1
            d=0
            c=0
            for k in range(1,4):
                if distances[j][k]!=math.inf:
                    d+=distances[j][k]
                    c+=1
            sigma = d/c * 0.3

        density += sc.ndimage.filters.gaussian_filter(sub_den, sigma)
    return density


def solve(pName,dName,index):
    path = 'origin/' + pName + '/' + dName + '/'
    t_path='fixed_data/' + pName + '/' + dName + '/'
    print(os.path.join(path,'images/IMG_{0}.JPG'.format(index)))
    image = cv2.imread(os.path.join(path,'images/IMG_{0}.jpg'.format(index)))
    ground_truth=pd.read_csv(os.path.join(path,'ground_truth/GT_IMG_{0}.csv'.format(index)),header=None)
    ground_truth=ground_truth.values

    [imgs,pts]=cut_patch(image,ground_truth)
    os.makedirs(os.path.join(t_path, 'images/'), exist_ok=True)
    os.makedirs(os.path.join(t_path, 'ground_truth/'), exist_ok=True)
    for i in range(9):
        density=generate_density(imgs[i],pts[i])
        [h, w] = [imgs[i].shape[0], imgs[i].shape[1]]
        # imgs[i]=cv2.resize(imgs[i],(w*8,h*8)) #do it when training
        density = cv2.resize(density, (floor_div2(density.shape[1], 3), floor_div2(density.shape[0], 3)),interpolation=cv2.INTER_CUBIC) * 64

        cv2.imwrite(os.path.join(t_path,'images/IMG_{0}_patch{1}.JPG'.format(index,i)),imgs[i])
        np.save(os.path.join(t_path,'ground_truth/IMG_{0}_patch{1}.npy'.format(index,i)), density)

        imgs[i] = imgs[i][:, ::-1]
        density=density[:,::-1]
        cv2.imwrite(os.path.join(t_path, 'images/IMG_{0}_patch{1}_m.JPG'.format(index, i)), imgs[i])
        np.save(os.path.join(t_path, 'ground_truth/IMG_{0}_patch{1}_m.npy'.format(index, i)), density)
    print('ok',os.path.join(path,'images/IMG_{0}.JPG'.format(index)))
    '''
    with open(path + 'ground-truth/GT_IMG_' + str(index) + '.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        number = int(data[0][0])
        points = [[float(p[0]),float(p[1])] for p in data[1:]]
        H_map = np.zeros((h + padding * 2, w + padding * 2))#原密度图先加入padding
        #print(number)
        for j, point in enumerate(points):
            #print(j,point[1])
            y = int(point[1])
            x = int(point[0])
            sigma = beta * kCloseMean(points.copy(), j, k_close)

            #print(sigma)
            gaussian = getGaussianFilter(filter_size, sigma)
            #对于越界的区域,计算其有效值
            sum = 0.0
            for yy in range(filter_size):
                for xx in range(filter_size):
                    if (yy + y >= padding and yy + y < padding + h and xx + x >= padding and xx + x < padding + w):
                        sum = sum + gaussian[yy, xx]
            #处以有效值所占比例就可以让切割后的区域的积分维持在1,且中心依旧是label定位的点
            gaussian /= sum
            #print(w,h,nw,nh,H_map.shape)
            #print(y,x)
            H_map[y:y + filter_size, x:x + filter_size] += gaussian[:, :]
            # print(sigma)
            # print(gaussian)
            # print(H_map[y:y+filter_size,x:x+filter_size])
        H_map = H_map[padding:padding + h, padding:padding + w]
        change_rate=H_map.sum()
        H_map=transform.resize(H_map,(nh, nw),anti_aliasing=False,mode='constant')
        change_rate/=H_map.sum()
        H_map=H_map*change_rate
        #print(H_map.sum(),number,change_rate) #可以验证最终密度图的积分等于人数总数
        #plt.imshow(H_map)
        #plt.show()
        #保存numpy内容至二进制文件中
        np.save(path + 'ground-truth/Hot_IMG_' + str(index) + '.npy', H_map)
        print('ok', path + 'ground-truth/Hot_IMG_' + str(index) + '.npy')
    '''

def create_validation():
    path = os.path.join('fixed_data', 'part_A', 'train_data')
    print(path)
    img_path = os.path.join(path, 'images')
    gt_path = os.path.join(path, 'ground_truth')
    shutil.rmtree(os.path.join('fixed_data', 'part_A', 'validation_data'),ignore_errors=True)
    os.makedirs(os.path.join(os.path.join('fixed_data', 'part_A'), 'validation_data', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.join('fixed_data', 'part_A'), 'validation_data', 'ground_truth'), exist_ok=True)
    tar_img_path = os.path.join(os.path.join('fixed_data', 'part_A'), 'validation_data', 'images')
    tar_gt_path = os.path.join(os.path.join('fixed_data', 'part_A'), 'validation_data', 'ground_truth')

    for i in range(271, 301):
        for j in range(9):
            # imgs
            os.rename(os.path.join(img_path, 'IMG_{0}_patch{1}.jpg'.format(i, j)),
                      os.path.join(tar_img_path, 'IMG_{0}_patch{1}.jpg'.format(i, j)))
            os.rename(os.path.join(img_path, 'IMG_{0}_patch{1}_m.jpg'.format(i, j)),
                      os.path.join(tar_img_path, 'IMG_{0}_patch{1}_m.jpg'.format(i, j)))
            # gts
            os.rename(os.path.join(gt_path, 'IMG_{0}_patch{1}.npy'.format(i, j)),
                      os.path.join(tar_gt_path, 'IMG_{0}_patch{1}.npy'.format(i, j)))
            os.rename(os.path.join(gt_path, 'IMG_{0}_patch{1}_m.npy'.format(i, j)),
                      os.path.join(tar_gt_path, 'IMG_{0}_patch{1}_m.npy'.format(i, j)))

if __name__ =='__main__':#进程不能共用内存

    dirName = [['part_A', 'part_B'], ['train_data', 'test_data']]
    #solve('part_A', 'train_data', 204)

    imageNumber = [300, 182, 400, 316]
    dirIndex = 0
    process = Pool(6)
    for dName in dirName[1]:
        for i in range(1,imageNumber[dirIndex]+1):
                #print(i)
            process.apply_async(func=solve, args=(dirName[0][0], dName, i))
                #solve(pName,dName,i)
        dirIndex += 1
    process.close()
    process.join()

    create_validation()
    print("Finish")
