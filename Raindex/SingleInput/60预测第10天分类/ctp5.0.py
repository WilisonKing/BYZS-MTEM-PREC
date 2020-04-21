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
#from pylab import mpl
#mpl.rcParams['font.sans-serif']=['SimHei']
os.environ["CUDA_VISIBLE_DEVICES"]='2'

BATCH_SIZE_TRAIN=256
BATCH_SIZE_TEST=128
LR=0.0006
EPOCH=40
Name='LeNet_ctp5.0'
delay=55
interval=9

seed=2


torch.cuda.manual_seed(seed)#为当前GPU设置随机种子 
#torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子


a=np.loadtxt('AI_DATA/data/huanan.txt')
print(a.shape)
ot=a[6579:,10]
print(ot.shape,ot)

for j in range(len(ot)):
    if ot[j]==100:
        ot[j]=2

i1=D('AI_DATA/data/daily_tp_1979-2019.nc')

t1=i1.variables['tp'][:13514]

print(t1.shape)

# In[6]:


t=t1[:,np.newaxis,:,:]
print(t.shape)

# In[7]:

mint = t[:, :, :, :].min()
maxt =t[:, :, :, :].max()
t[:, :, :, :] = (t[:, :, :, :] - mint) / (maxt - mint)
print(t.shape)

traini=np.array(t[:11005-interval,:,:,:])
#validi=np.array(t[301:365,:,:,:])
testi=np.array(t[11001-delay-interval:13514-interval,:,:,:])
trainl=np.array(ot[delay+interval:11001])
#validl=np.array(ot[301:361])
testl=np.array(ot[11001:])


# In[11]:


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        datamerge = self.data[idx:idx+5+delay,:,:,:].reshape(-1, 5+delay, 81, 91)
        return np.squeeze(datamerge), self.label[idx]


# In[12]:


from torch.utils.data import DataLoader
train_data = MyDataset(data=traini,label=trainl)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,drop_last=True)
#valid_data = MyDataset(data=validi,label=validl)
#valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False)
test_data = MyDataset(data=testi,label=testl)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False,drop_last=True)


# In[13]:


import torch.nn.functional as F
class NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5+delay,60,8)
        self.conv2 = nn.Conv2d(60,80,8)

        self.fc1 = nn.Linear(80*15*17,1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        #print('size', x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = NET().cuda()
print(net)


# In[14]:


optimizer = torch.optim.Adam(net.parameters(), lr=LR) 
loss_function = nn.CrossEntropyLoss() 


from torch.autograd import Variable # 获取变量
for epoch in range(EPOCH): 
    running_loss=0.0
    running_correct=0.0
    for step,(x,y) in enumerate(train_loader): 
        b_x = Variable(x).cuda() 
        b_y = Variable(y).cuda()
        output = net(b_x.float()) 
        _,pred = torch.max(output.data,1)
        optimizer.zero_grad() 
        loss = loss_function(output, b_y.long())
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
        running_correct += torch.sum(pred == b_y.data.long())
        if step % 30 == 29: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:%.6f'%(running_loss/30.0),'|accuracy rate:%.6f'%(running_correct.cpu().detach().numpy()/(30.00*BATCH_SIZE_TRAIN)))
            running_loss =0.0
            running_correct=0.0
    if epoch>EPOCH-5:
        t_loss=0.0
        t_correct=0.0
        for k,(m,n) in enumerate(test_loader): 
            t_x = Variable(m).cuda() 
            t_y = Variable(n).cuda()
            output2 = net(t_x.float()) 
            _,pred2 = torch.max(output2.data,1)
            loss = loss_function(output2 , t_y.long())
            t_loss += loss.item()
            t_correct += torch.sum(pred2 == t_y.data.long())
        print( '第%d轮结束后:'%epoch,'|test_loss:%.6f'%(t_loss/(k+1)),'|accuracy rate:%.6f'%(t_correct.cpu().detach().numpy()/((k+1)*BATCH_SIZE_TEST)))
print(str(Name)+'_BSTRAIN',str(BATCH_SIZE_TRAIN),'_BSTEST',str(BATCH_SIZE_TEST),'_LR',str(LR),'_EPOCH',str(EPOCH))

# In[18]:


#save net
PATH =str(Name)+'_BSTRAIN'+str(BATCH_SIZE_TRAIN)+'_BSTEST'+str(BATCH_SIZE_TEST)+'_LR'+str(LR)+'_EPOCH'+str(EPOCH)+'.pth'
torch.save(net.state_dict(), PATH)


# In[65]:


#t_loss = 0.0
#pre_all=torch.empty(0)
#for k,(m,n) in enumerate(test_loader): 
    #t_x = Variable(m).cuda() 
    #t_y = Variable(n).cuda()
    #output = net(t_x) 
    #pre=output.reshape(-1)
    #loss = loss_function(pre, t_y.float())
    #t_loss += loss.item()
    #pre_all=torch.cat([pre_all.cpu().float(),t_y.cpu().float()])
    #print( '|test_loss:',t_loss/(k+1),'|余弦相似:%.4f'%cos_sim(pre.cpu().detach().numpy(),t_y.cpu()))


#import matplotlib
#matplotlib.use('Agg')
#from matplotlib.pyplot #import plot,savefig

#plt.plot(pre_all.cpu().detach().numpy(),'-k',label='预测')
#plt.plot(testl,'-r',label='真实')
#plt.legend()
#savefig('MyFig.jpg')
#plt.rcParams['axes.unicode_minus'] = False

#print('所有预测的余弦相似%.4f'%cos_sim(pre_all.cpu().detach().numpy(),testl))