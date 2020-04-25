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
import math
os.environ["CUDA_VISIBLE_DEVICES"]='2'


# In[ ]:


BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=128
LR=0.00015
EPOCH=300
Name='Conv2d1*1_prec6.5'
basic_day=3
train_step_print=30
test_step_print=4
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
# In[ ]:


print(l.shape)


# In[ ]:


traini=np.array(t[:365*30,:,:,:])
testi=np.array(t[365*30:365*37,:,:,:])
trainl=np.array(l[:245*30,:,:])
testl=np.array(l[245*30:,:,:])

#traini=np.array(t[:365*10,:,:,:])
#testi=np.array(t[365*10:365*13,:,:,:])
#trainl=np.array(l[:245*10,:,:])
#testl=np.array(l[245*10:245*13,:,:])
print(traini.shape,testi.shape,trainl.shape,testl.shape)


# In[ ]:


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

# In[ ]:


from torch.utils.data import DataLoader
train_data = MyDataset(data=traini,label=trainl,aver_label=trainl)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_data = MyDataset(data=testi,label=testl,aver_label=trainl)
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


#def cos_sim(v1, v2):
#    vector1=v1.reshape(-1,)
#    vector2=v2.reshape(-1,)
#    dot_product = 0.0
#    normA = 0.0
#    normB = 0.0
#    for a, b in zip(vector1, vector2):
#        dot_product += a * b
#        normA += a ** 2
#        normB += b ** 2
#    if normA == 0.0 or normB == 0.0:
#        return None
#    else:
#        return dot_product / ((normA * normB) ** 0.5)

#def cos_sim(v1, v2):
#    cossim=0
#    print('余弦相似开始')
#    for i in range(len(v1)):
#        vector1=torch.squeeze(v1[i,:,:]).view(-1,)
#        vector2=torch.squeeze(v2[i,:,:]).view(-1,)
#        dot_product = 0.0
#        normA = 0.0
#        normB = 0.0
#        for a, b in zip(vector1, vector2):
#            #if a<0 or b<0:
#            #    print('a:',a,'  b:',b)
#            #    exit()
#            dot_product += a * b
#            normA += a ** 2
#            normB += b ** 2
#        
#        cossim+=dot_product / ((normA * normB) ** 0.5)
#    print('余弦相似结束')
#    return cossim/len(v1)

class CosineSimilarity(nn.Module):
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
c_s=CosineSimilarity()

def cos_sim(v1, v2):
    cossim=0
    #print('余弦相似开始')
    for i in range(len(v1)):
        vector1=torch.squeeze(v1[i,:,:]).view(-1,)
        vector2=torch.squeeze(v2[i,:,:]).view(-1,)
#        dot_product = 0.0
#        normA = 0.0
#        normB = 0.0
#        for a, b in zip(vector1, vector2):
#            dot_product += a * b
#            normA += a ** 2
#            normB += b ** 2
#        cossim+=dot_product / ((normA * normB) ** 0.5)
        cossim+=c_s(vector1, vector2)
    #print('余弦相似结束')
    return cossim/len(v1)


# In[ ]:


from torch.autograd import Variable # 获取变量
flag=0
cosmember=0
for epoch in range(EPOCH): 
    #print('epoch=%d开始:'%epoch)
    running_loss=0.0
    running_cos=0.0
    #if cosmember>0.90:
    #    break
    for step,(x,y,aver_l) in enumerate(train_loader): 
        #print('step=%d已读取数据'%step)
        #if flag==1:
        #    break
        b_x = Variable(x, requires_grad=True).cuda() 
        b_y = Variable(y, requires_grad=True).cuda()
        aver_l=Variable(aver_l, requires_grad=True).cuda()
        output = net(b_x.float()) 
        optimizer.zero_grad() 
        output=((output.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        aver_l=((aver_l.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        b_y=((b_y.float().cpu().detach().numpy())*397.317)*get_prec_mask()
        loss = loss_function(Variable(torch.from_numpy(output-aver_l), requires_grad=True),Variable(torch.from_numpy(b_y-aver_l), requires_grad=True))
        #print('loss end:',loss)
        loss.backward() 
        loss.requires_grad=True
        optimizer.step() 
        running_loss += loss.item()
        #print(output.shape,b_y.shape)
        running_cos +=cos_sim(torch.from_numpy(output-aver_l),torch.from_numpy(b_y-aver_l))
        #print('running_cos end:',running_cos)
        if step % train_step_print == train_step_print-1: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:',(running_loss/train_step_print),'|余弦相似度:%.6f'%(running_cos/train_step_print))
            #cosmember=running_cos/step_print
            running_loss =0.0
            running_cos=0.0
    if epoch>-1:#EPOCH-10 or cosmember>0.85 or flag==1:
#        #print('进入测试(编号:%d):'%epoch)
#        flag=1
        t_loss=0.0
        t_cos=0.0
        for k,(m,n,aver_l2) in enumerate(test_loader): 
            #print('正在迭代测试  ','k=',k)
            t_x = Variable(m).cuda() 
            t_y = Variable(n).cuda()
            aver_l2=Variable(aver_l2).cuda()
            output2 = net(t_x.float()) 
            output2=((output2.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            aver_l2=((aver_l2.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            t_y=((t_y.float().cpu().detach().numpy())*397.317)*get_prec_mask()
            loss = loss_function(torch.from_numpy(output2-aver_l2),torch.from_numpy(t_y-aver_l2))
            t_loss += loss.item()
            t_cos +=cos_sim(torch.from_numpy(output2-aver_l2),torch.from_numpy(t_y-aver_l2))
            #print('迭代测试结束  ','k=',k)
            if k % test_step_print == test_step_print-1: 
                print( 'Test||','Epoch:', epoch, '|Step:', k, '|test_loss:',(t_loss/test_step_print),'|余弦相似度:%.6f'%(t_cos/test_step_print))
                t_loss=0.0
                t_cos=0.0
#print('测试结束(编号:%d):'%epoch)
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))


# In[ ]:




