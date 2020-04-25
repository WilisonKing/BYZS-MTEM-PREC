#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import netCDF4
from netCDF4 import Dataset as D
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]='3,5,4'

BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=128
LR=0.001
EPOCH=200
Name='ResNet_prec.cosloss'
basic_day=3
train_step_print=30
test_step_print=4
seed=2
torch.cuda.manual_seed(seed)

fi=D('AI_DATA/data/TPQ365.nc')
fo=D('AI_DATA/data/prec.nc')

t1=fi.variables['hgt'][:]
t2=fi.variables['olr'][:]
t3=fi.variables['uwnd'][:]
t4=fi.variables['vwnd'][:]

t2=t2[:,np.newaxis,:,:]
t=torch.cat([torch.from_numpy(t1),
             torch.from_numpy(t2),
             torch.from_numpy(t3),
             torch.from_numpy(t4)],1)

def get_prec_mask():
    prec_path = "AI_DATA/data/cn_east_mask_150.nc"
    prec = D(prec_path)
    prec_mask = ~(prec['mask'][:].mask)[::-1]
    return prec_mask


t=get_prec_mask()*t
for i in range(10):
    mint = t[:, i, :, :].min()
    maxt = t[:, i, :, :].max()
    t[:, i, :, :] = (t[:, i, :, :] - mint) / (maxt - mint)
t=get_prec_mask()*t

l=fo.variables['PREC'][:]
l=get_prec_mask()*l
l[l<0]=0
minl = l.min()
maxl = l.max()
l = (l - minl) / (maxl - minl)
l=get_prec_mask()*l
constant_maxl=torch.from_numpy(np.array(maxl)).data.cuda()
print(constant_maxl)

traini=np.array(t[:365*30,:,:,:])
testi=np.array(t[365*30:365*37,:,:,:])
trainl=np.array(l[:245*30,:,:])
testl=np.array(l[245*30:,:,:])

class MyDataset(Dataset):
    def __init__(self, data, label,aver_label):
        self.data = data
        self.label = label
        self.aver_label=aver_label.reshape(-1,245,81,91)
        self.redata = data.reshape(-1,365,10,81,91)
        self.relabel = label.reshape(-1,245,81,91)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        for i in range(len(self.label)//245):
            if i-1<idx/245.0 and i+1>idx/245:
                year=i
                break
        datamerge = self.redata[year,57+idx-i*245:57+idx+basic_day-i*245,:,:,:].reshape(basic_day*10 , 81, 91)
        aver_l=np.mean(np.squeeze(self.aver_label[:,idx%245,:,:]),0)
        return datamerge, np.squeeze(self.label[idx,:,:]),aver_l

from torch.utils.data import DataLoader
train_data = MyDataset(data=traini,label=trainl,aver_label=trainl)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_data = MyDataset(data=testi,label=testl,aver_label=trainl)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False)

import torch.nn.functional as F
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Sigmoid(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(10*basic_day, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )
        self.layer1 = self.make_layer(ResidualBlock, 128,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out=self.conv2(out)
        return torch.squeeze(out)

def ResNet1():

    return ResNet(ResidualBlock)

net=nn.DataParallel(ResNet1()).cuda()




# import torchvision
# net = torchvision.models.resnet18(pretrained=True)

# net.conv1.in_channels = 30
# net.fc.out_features = 81 * 91

# net = nn.DataParallel(net).cuda()




def single(tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

def CosineSimilarity(v1, v2):
    cossim=0
    for i in range(len(v1)):
        vector1=v1[i,:,:].view(-1)
        vector2=v2[i,:,:].view(-1)
        cossim+=single(vector1, vector2)
    return 1-cossim.cuda()/len(v1)

optimizer = torch.optim.Adam(net.parameters(), lr=LR) 


from torch.autograd import Variable
for epoch in range(EPOCH): 
    net.train()
    running_loss=0.0

    for step,(b_x,b_y,aver_l) in enumerate(train_loader): 
        b_x, b_y, aver_l = b_x.cuda(), b_y.cuda(), aver_l.cuda()
        output = net(b_x.data.float()) 

        output=(output.float() * constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda()
        b_y= (b_y.float()* constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda()
        aver_l=(aver_l.float() * constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda()

        outlast=output-aver_l
        labellast=b_y-aver_l

        optimizer.zero_grad() 
        loss = CosineSimilarity(outlast, labellast)
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()

        if step % train_step_print == 0 and step != 0: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train_loss:',running_loss/train_step_print)
            running_loss =0.0

    net.eval()
    if epoch>-1:
        t_loss=0.0
        
        for k,(t_x,t_y,aver_l2) in enumerate(test_loader): 
            t_x,t_y,aver_l2 = t_x.cuda(), t_y.cuda(), aver_l2.cuda()
            output2 = net(t_x.float())

            output2.data=(output2.data.float() * constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda() #397.317
            t_y.data= (t_y.data.float()* constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda()
            aver_l2.data=(aver_l2.data.float() * constant_maxl) * torch.from_numpy(get_prec_mask().astype(np.float32)).cuda()
            
            outlast2=output2-aver_l2
            labellast2=t_y-aver_l2

            loss = CosineSimilarity(outlast2 , labellast2)
            t_loss += loss.item()

            if k % test_step_print == 0 and k != 0: 
                print( 'Test||','Epoch:', epoch, '|Step:', k, '|test_loss:',t_loss/test_step_print)
                t_loss=0.0
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))

predict=output2.data[-10:,:,:]
true=t_y.data[-10:,:,:]

def show(true, predict):
    plt.figure(figsize=(10, 5))
    # 纵横间隙
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    for i in range(10):
        plt.subplot(2, 10, i + 1)  # 2 row 10 clomn
        # auto resize the sub plot
        image1 = true[i, :, :]
        cbar = plt.colorbar(plt.imshow(image1, cmap='RdBu_r'),
                            orientation='horizontal')
        cbar.set_label('(mm)', fontsize=12)
        plt.title("True")

        plt.subplot(2, 10, i + 11)  # 2 row 10 clomn
        # auto resize the sub plot
        image2 = predict[i,:, :]
        cbar = plt.colorbar(plt.imshow(image2, cmap='RdBu_r'),
                            orientation='horizontal')
        cbar.set_label('(mm)', fontsize=12)
        plt.title("Pred")
    plt.savefig('figure.png')
show(true, predict)


