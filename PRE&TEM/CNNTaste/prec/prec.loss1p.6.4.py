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
os.environ["CUDA_VISIBLE_DEVICES"]='4'

BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=128
LR=0.00015
EPOCH=300
Name='Conv2d_prec.loss1p.6.4'
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
                #print('idx:',idx,'\n','year:',year)
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
class Net(nn.Module):
    def __init__(self,input_dim=basic_day*10,conv_dim=128,linear_dim=30):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, conv_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, linear_dim, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(linear_dim, 1, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return np.squeeze(x)

net = Net().cuda()
print(net)

#class RMSSD(nn.Module):
#    def __init__(self, size_average=None, reduce=None, reduction='mean'):
#        super(RMSSD, self).__init__(size_average, reduce, reduction)
#
#    def forward(self, input, target):
#        return F.mse_loss(input, target, reduction=self.reduction)

optimizer = torch.optim.Adam(net.parameters(), lr=LR) 
loss_function = nn.MSELoss() 

class CosineSimilarity(nn.Module):
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
c_s=CosineSimilarity()

def cos_sim(v1, v2):
    cossim=0
    for i in range(len(v1)):
        vector1=torch.squeeze(v1[i,:,:]).view(-1,)
        vector2=torch.squeeze(v2[i,:,:]).view(-1,)
        cossim+=c_s(vector1, vector2)
    return cossim/len(v1)

from torch.autograd import Variable
for epoch in range(EPOCH): 
    running_loss=0.0
    running_cos=0.0
    for step,(x,y,aver_l) in enumerate(train_loader): 
        b_x = Variable(x,requires_grad=True).cuda() 
        b_y = Variable(y,requires_grad=True).cuda()
        aver_l=Variable(aver_l,requires_grad=True).cuda()
        output = net(b_x.float()) 
        output=((output.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        aver_l=((aver_l.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        b_y=((b_y.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        outlast=Variable(torch.from_numpy(output-aver_l),requires_grad=True).cuda()
        labellast=Variable(torch.from_numpy(b_y-aver_l)).cuda()
        optimizer.zero_grad() 
        loss = loss_function(outlast, labellast)
        loss = loss.requires_grad_()
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
        running_cos +=cos_sim(outlast, labellast)
        if step % train_step_print == train_step_print-1: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:',(running_loss/train_step_print),'|余弦相似度:%.6f'%(running_cos/train_step_print))
            running_loss =0.0
            running_cos=0.0
    if epoch>-1:
        t_loss=0.0
        t_cos=0.0
        for k,(m,n,aver_l2) in enumerate(test_loader): 
            t_x = Variable(m).cuda() 
            t_y = Variable(n).cuda()
            aver_l2=Variable(aver_l2).cuda()
            output2 = net(t_x.float()) 
            output2=((output2.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            aver_l2=((aver_l2.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            t_y=((t_y.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            outlast2=Variable(torch.from_numpy(output2-aver_l2),requires_grad=True).cuda()
            labellast2=Variable(torch.from_numpy(t_y-aver_l2)).cuda()
            loss = loss_function(outlast2 , labellast2)
            t_loss += loss.item()
            t_cos +=cos_sim(outlast2,labellast2)
            if k % test_step_print == test_step_print-1: 
                print( 'Test||','Epoch:', epoch, '|Step:', k, '|test_loss:',(t_loss/test_step_print),'|余弦相似度:%.6f'%(t_cos/test_step_print))
                t_loss=0.0
                t_cos=0.0
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))
