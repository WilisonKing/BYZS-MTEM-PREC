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
os.environ["CUDA_VISIBLE_DEVICES"]='3'


# In[ ]:


BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=128
LR=0.1
EPOCH=1000
Name='Conv2d_tem7.2'
basic_day=3
step_print=35
seed=2
torch.cuda.manual_seed(seed)#为当前GPU设置随机种子 


# In[ ]:


fi=D('AI_DATA/data/TPQ365.nc')
fo=D('AI_DATA/data/tem.nc')


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


def get_tem_mask():
    tem_path = "AI_DATA/data/cn_east_mask_150.nc"
    tem = D(tem_path)
    tem_mask = ~(tem['mask'][:].mask)[::-1]
    return tem_mask


# In[ ]:


#t=torch.from_numpy(get_prec_mask())*t
#t=torch.from_numpy(get_prec_mask().astype(np.uint8))*t
#t=torch.from_numpy(get_prec_mask()).Byte()*t

# In[ ]:

t=get_tem_mask()*t
for i in range(10):
    mint = t[:, i, :, :].min()
    maxt = t[:, i, :, :].max()
    t[:, i, :, :] = (t[:, i, :, :] - mint) / (maxt - mint)
t=get_tem_mask()*t

# In[ ]:


l=fo.variables['TMAX'][:]
l=get_tem_mask()*l
l[l<-20]=-20.0
l[l>45]=45.0
minl = l.min()
maxl = l.max()
l= (l - minl) / (maxl - minl)
#print(minl,maxl)
l=get_tem_mask()*l
# In[ ]:


print(l.shape)


# In[ ]:


traini=np.array(t[:365*30,:,:,:])
testi=np.array(t[365*30:365*37,:,:,:])
trainl=np.array(l[:245*30,:,:])
testl=np.array(l[245*30:,:,:])


# In[ ]:


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


#optimizer = torch.optim.Adam(net.parameters(), lr=LR) 
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
loss_function = nn.MSELoss() 
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau \
(optimizer, mode='min', factor=0.1, patience=10,verbose=True, threshold=0.0001, \
threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


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
    #print(cossim,len(v1))
    return cossim/len(v1)

# In[ ]:

from torchvision import utils as vutils
def save_image_tensor(input_tensor, filename,minl,maxl):
    input_tensor=torch.squeeze(input_tensor)
    #assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    print(input_tensor.shape)
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    #print(input_tensor.type,torch.from_numpy(maxl-minl).type)
    #input_tensor=input_tensor*torch.from_numpy(np.array(maxl-minl))+torch.from_numpy(np.array(minl))
    vutils.save_image(input_tensor, filename)


from torch.autograd import Variable # 获取变量
flag=0
cosmember=0
for epoch in range(EPOCH): 
    #print('epoch=%d开始:'%epoch)
    running_loss=0.0
    running_cos=0.0
    #if cosmember>0.85:
    #    break
    for step,(x,y,aver_l) in enumerate(train_loader): 
        #print('step=%d已读取数据'%step)
        #if flag==1:
        #    break
        b_x = Variable(x).cuda() 
        b_y = Variable(y).cuda()
        aver_l=Variable(aver_l).cuda()
        output = net(b_x.float()) 
        optimizer.zero_grad() 
        loss = loss_function(output, b_y)
        #print('loss end:',loss)
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
        #print(output.shape,b_y.shape)
        #output=get_tem_mask()*((output.cpu().detach().numpy())*(maxl-minl)+minl)
        #aver_l=get_tem_mask()*((aver_l.cpu().detach().numpy())*(maxl-minl)+minl)
        #b_y=get_tem_mask()*((b_y.cpu().detach().numpy())*(maxl-minl)+minl)
        #print(output.float().cpu().detach().numpy().min(),output.float().cpu().detach().numpy().max())
        output=((output.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
        aver_l=((aver_l.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
        b_y=((b_y.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
        #print(output.min(),output.max())
        running_cos +=cos_sim(torch.from_numpy(output-aver_l),torch.from_numpy(b_y-aver_l))
        #print('running_cos end:',running_cos)
        if step % step_print == step_print-1: 
            #save_image_tensor(output[0,:,:], 'train_p_'+str(epoch)+'_'+str(step)+'.jpg',minl,maxl)
            #save_image_tensor(b_y[0,:,:], 'train_l_'+str(epoch)+'_'+str(step)+'.jpg',minl,maxl)
            #np.savetxt('train_p_'+str(epoch)+'_'+str(step)+'.txt',torch.squeeze(output[0,:,:]).cpu().detach().numpy())
            #np.savetxt('train_l_'+str(epoch)+'_'+str(step)+'.txt',torch.squeeze(b_y[0,:,:]).cpu().detach().numpy())
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:',(running_loss/step_print),'|余弦相似度:%.6f'%(running_cos/step_print))
            cosmember=running_cos/step_print
            running_loss =0.0
            running_cos=0.0
    scheduler.step(loss.item())
    if epoch>EPOCH-10 or cosmember>0.85 or flag==1:
        #print('进入测试(编号:%d):'%epoch)
        flag=1
        t_loss=0.0
        t_cos=0.0
        for k,(m,n,aver_l2) in enumerate(test_loader): 
            #print('正在迭代测试  ','k=',k)
            t_x = Variable(m).cuda() 
            t_y = Variable(n).cuda()
            aver_l2=Variable(aver_l2).cuda()
            output2 = net(t_x.float()) 
            loss = loss_function(output2 , t_y)
            t_loss += loss.item()
            #output2=get_tem_mask()*((output2.cpu().detach().numpy())*(maxl-minl)+minl)
            #aver_l2=get_tem_mask()*((aver_l2.cpu().detach().numpy())*(maxl-minl)+minl)
            #t_y=get_tem_mask()*((t_y.cpu().detach().numpy())*(maxl-minl)+minl)
            output2=((output2.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
            aver_l2=((aver_l2.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
            t_y=((t_y.float().cpu().detach().numpy())*65.0-20.0)*get_tem_mask()
            t_cos +=cos_sim(torch.from_numpy(output2-aver_l2),torch.from_numpy(t_y-aver_l2))
            #print('迭代测试结束  ','k=',k)
        print( '第%d轮结束后:'%epoch,'|test_loss:',(t_loss/(k+1)),'|余弦相似度:%.6f'%(t_cos/(k+1)))
        #print('测试结束(编号:%d):'%epoch)
        #save_image_tensor(output2[10,:,:], 'test_p_'+str(epoch)+'_'+str(step)+'.jpg',minl,maxl)
        #save_image_tensor(t_y[10,:,:], 'test_l_'+str(epoch)+'_'+str(step)+'.jpg',minl,maxl)
        #np.savetxt('test_p_'+str(epoch)+'_'+str(step)+'.txt',torch.squeeze(output2[0,:,:]).cpu().detach().numpy())
        #np.savetxt('test_l_'+str(epoch)+'_'+str(step)+'.txt',torch.squeeze(t_y[0,:,:]).cpu().detach().numpy())
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))


# In[ ]:




