#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import netCDF4
from netCDF4 import Dataset as D
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'


# In[ ]:


BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=128
LR=0.0006
EPOCH=40
Name='Conv2d_prec6.0'
basic_day=3
seed=2
torch.cuda.manual_seed(seed)#为当前GPU设置随机种子 


# In[ ]:


fi=D('AI_DATA/data/TPQ365.nc')
fo=D('AI_DATA/data/prec.nc')


# In[ ]:


print(fi.variables.keys())


# In[ ]:


t1=fi.variables['hgt'][:]
t2=fi.variables['olr'][:]
t3=fi.variables['uwnd'][:]
t4=fi.variables['vwnd'][:]
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)


# In[ ]:


t2=t2[:,np.newaxis,:,:]
t=torch.cat([torch.from_numpy(t1),
             torch.from_numpy(t2),
             torch.from_numpy(t3),
             torch.from_numpy(t4)],1)
print(t.shape)


# In[ ]:


def get_prec_mask():
    prec_path = "AI_DATA/data/cn_east_mask_150.nc"
    prec = D(prec_path)
    prec_mask = ~(prec['mask'][:].mask)[::-1]
    return prec_mask


# In[ ]:


#t=torch.from_numpy(get_prec_mask())*t
#t=torch.from_numpy(get_prec_mask().astype(np.uint8))*t
#t=torch.from_numpy(get_prec_mask()).Byte()*t


# In[ ]:


for i in range(10):
    mint = t[:, i, :, :].min()
    maxt = t[:, i, :, :].max()
    t[:, i, :, :] = (t[:, i, :, :] - mint) / (maxt - mint)

t=t*get_prec_mask()


l=fo.variables['PREC'][:]

minl = l[:, :, :].min()
maxl = l[:, :, :].max()
l[:, :, :] = (l[:, :, :] - mint) / (maxt - mint)

l=l*get_prec_mask()
# In[ ]:


print(l.shape)


# In[ ]:


traini=np.array(t[:365*30,:,:,:])
testi=np.array(t[365*30:,:,:,:])
trainl=np.array(l[:245*30,:,:])
testl=np.array(l[245*30:,:,:])


# In[ ]:


print(traini.shape,testi.shape,trainl.shape,testl.shape)


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.redata = data.reshape(-1,365,10,81,91)
        self.relabel = label.reshape(-1,245,81,91)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        for i in range(len(self.label)//245):
            if i-1<idx/245.0 and i+1>idx/245:
                year=i
                break
        datamerge = self.redata[year,57+idx-i*245:57+idx+basic_day-i*245,:,:,:].reshape(-1,basic_day*10 , 81, 91)
        return np.squeeze(datamerge), self.relabel[year,idx-i*245,:,:]


# In[ ]:


from torch.utils.data import DataLoader
train_data = MyDataset(data=traini,label=trainl)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_data = MyDataset(data=testi,label=testl)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False)


# In[ ]:


import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=basic_day*10, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return np.squeeze(x)

net = Net().cuda()
print(net)


# In[ ]:


optimizer = torch.optim.Adam(net.parameters(), lr=LR) 
loss_function = nn.MSELoss() 


# In[ ]:


def cos_sim(v1, v2):
    vector1=v1.reshape(-1,)
    vector2=v2.reshape(-1,)
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


# In[ ]:


from torch.autograd import Variable # 获取变量
flag=0
for epoch in range(EPOCH): 
    running_loss=0.0
    running_cos=0.0
    for step,(x,y) in enumerate(train_loader): 
        if flag==1:
            break
        b_x = Variable(x).cuda() 
        b_y = Variable(y).cuda()
        output = net(b_x.float()) 
        optimizer.zero_grad() 
        loss = loss_function(output, b_y)
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
        #print(output.shape,b_y.shape)
        running_cos +=cos_sim(output,b_y)
        if step % 30 == 29: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:%.6f'%(running_loss/30.0),                  '|余弦相似度:%.6f'%(running_cos/30.0))
            cosmember=running_cos/30.0
            running_loss =0.0
            running_cos=0.0
    if epoch>EPOCH-5 or running_cos>0.80 or flag==1:
        flag=1
        t_loss=0.0
        t_cos=0.0
        for k,(m,n) in enumerate(test_loader): 
            t_x = Variable(m).cuda() 
            t_y = Variable(n).cuda()
            output2 = net(t_x.float()) 
            loss = loss_function(output2 , t_y)
            t_loss += loss.item()
            t_cos +=cos_sim(output2,t_y)
        print( '第%d轮结束后:'%epoch,'|test_loss:%.6f'%(t_loss/(k+1)),              '|余弦相似度:%.6f'%(t_cos/(k+1)))
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))


# In[ ]:




