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

a=np.loadtxt('data/huanan.txt')
print(a[6944:14244,9].shape)
ot=a[6944:14244,9]

i1=D('data/h2.nc')
i2=D('data/h5.nc')
i3=D('data/h8.nc')
i4=D('data/o.nc')
i5=D('data/u2.nc')
i6=D('data/u5.nc')
i7=D('data/u8.nc')
i8=D('data/v2.nc')
i9=D('data/v5.nc')
i10=D('data/v8.nc')

t1=i1.variables['hgt'][:405]
t2=i2.variables['hgt'][:405]
t3=i3.variables['hgt'][:405]
t4=i4.variables['olr'][:405]
t5=i5.variables['uwnd'][:405]
t6=i6.variables['uwnd'][:405]
t7=i7.variables['uwnd'][:405]
t8=i8.variables['vwnd'][:405]
t9=i9.variables['vwnd'][:405]
t10=i10.variables['vwnd'][:405]
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)
print(t5.shape)
print(t6.shape)
print(t7.shape)
print(t8.shape)
print(t9.shape)
print(t10.shape)

t4=t4[:,np.newaxis,:,:]
print(t4.shape)
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)
print(t5.shape)
print(t6.shape)
print(t7.shape)
print(t8.shape)
print(t9.shape)
print(t10.shape)

t=torch.cat([torch.from_numpy(t1),
             torch.from_numpy(t2),
             torch.from_numpy(t3),
             torch.from_numpy(t4),
             torch.from_numpy(t5),
             torch.from_numpy(t6),
             torch.from_numpy(t7),
             torch.from_numpy(t8),
             torch.from_numpy(t9),
             torch.from_numpy(t10)], axis=1)
print(t.shape)


for i in range(10):
    mint = t[:, i, :, :].min()
    maxt = t[:, i, :, :].max()
    t[:, i, :, :] = (t[:, i, :, :] - mint) / (maxt - mint)
print(t.shape)

#traini=t[:6005,:,:,:]
#validi=t[6001:6605,:,:,:]
#testi=t[6505:,:,:,:]
#trainl=ot[:6001]
#validl=ot[6001:6601]
#testl=ot[6601:]

traini=np.array(t[:305,:,:,:])
validi=np.array(t[301:365,:,:,:])
testi=np.array(t[365:370,:,:,:])
trainl=np.array(ot[:301])
validl=np.array(ot[301:361])
testl=np.array(ot[361:366])

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        datamerge = self.data[idx:idx+5,:,:,:].reshape(-1, 50, 81, 91)
        return np.squeeze(datamerge), self.label[idx]

from torch.utils.data import DataLoader
train_data = MyDataset(data=traini,label=trainl)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
valid_data = MyDataset(data=validi,label=validl)
valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False)
test_data = MyDataset(data=testi,label=testl)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=60, kernel_size=8)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=10, kernel_size=8)
        self.fc1 = nn.Linear(2550, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=36)
        self.fc3 = nn.Linear(in_features=36, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=1)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]     
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

'''
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Residual,self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv1x1 = None
            
    def forward(self,x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x) 

        out = self.relu(o2 + x)
        return out

class ResNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=50,64,kernel_size=8,stride=2,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Residual(64,64),
            Residual(64,64),
            Residual(64,64),
        )

        self.conv3 = nn.Sequential(
            Residual(64,128,stride=2),
            Residual(128,128),
            Residual(128,128),
            Residual(128,128),
            Residual(128,128),
        )

        self.conv4 = nn.Sequential(
            Residual(128,256,stride=2),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
        )

        self.conv5 = nn.Sequential(
            Residual(256,512,stride=2),
            Residual(512,512),
            Residual(512,512),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        out = self.avg_pool(out)
        out = out.view((x.shape[0],-1))

        out = self.fc(out)

        return out
'''

optimizer = torch.optim.Adam(net.parameters(), lr=0.001) 
loss_function = nn.MSELoss() 

from torch.autograd import Variable 
for epoch in range(2): 
    for step,(x,y) in enumerate(train_loader): 
        b_x = Variable(x) 
        b_y = Variable(y)
        output = net(b_x) 
        loss = loss_function(output.reshape(-1), b_y)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        if step % 100 == 0: 
            for k,(m,n) in enumerate(valid_loader): 
                v_x = Variable(m) 
                v_y = Variable(n)
                output = net(v_x) 
                pre=output.reshape(-1)
                err=sum(pre)/sum(v_y)
                #err=sum(np.abs(pre[i]-v_y[i] for i in range(len(v_y)))/v_y)
                break
            #test_output = net(Variable(torch.from_numpy(testi.reshape(-1,50,81,91))))
            #error=sum(np.abs([(np.squeeze(test_output))[i] - testl[i] for i in range(len(testl))])/testl)
            print('Epoch:', epoch, '|Step:', step, '|train loss:%.4f'%loss.item(), '|test error:%.4f'%err) 
