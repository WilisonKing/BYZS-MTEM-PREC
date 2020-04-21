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

BATCH_SIZE_TRAIN=128
BATCH_SIZE_TEST=128
LR=0.0006
EPOCH=30
Name='SimpleCNN_tp1.0'
delay=25

a=np.loadtxt('AI_DATA/data/huanan.txt')
print(a.shape)
ot=a[6579:,9]
print(ot.shape,ot)
# In[84]:


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

traini=np.array(t[:12005,:,:,:])
#validi=np.array(t[301:365,:,:,:])
testi=np.array(t[12001-delay:,:,:,:])
trainl=np.array(ot[delay:12001])
#validl=np.array(ot[301:361])
testl=np.array(ot[12001:])


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
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
#valid_data = MyDataset(data=validi,label=validl)
#valid_loader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False)
test_data = MyDataset(data=testi,label=testl)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TEST, shuffle=False)


# In[13]:


import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5+delay, out_channels=60, kernel_size=8)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=10, kernel_size=8)
        self.fc1 = nn.Linear(2550, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=36)
        self.fc3 = nn.Linear(in_features=36, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=1)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net().cuda()
print(net)


# In[14]:


optimizer = torch.optim.Adam(net.parameters(), lr=LR) 
loss_function = nn.MSELoss() 


# In[15]:


def cos_sim(vector1, vector2):
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


# In[17]:


from torch.autograd import Variable # 获取变量
for epoch in range(EPOCH): 
    running_loss = 0.0
    for step,(x,y) in enumerate(train_loader): 
        b_x = Variable(x).cuda() 
        b_y = Variable(y).cuda()
        output = net(b_x.float()) 
        loss = loss_function(output.reshape(-1), b_y.float())
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
        if step % 20 == 19: 
            print('Train||','Epoch:', epoch, '|Step:', step, '|train loss:',running_loss/20.0,'|余弦相似:%.4f:'%cos_sim(output.reshape(-1).cpu().detach().numpy(),b_y.cpu()))
            running_loss = 0.0
        if epoch>EPOCH-3:
             t_loss = 0.0
             cosim=0.0
             #pre_all=torch.empty(0)
             for k,(m,n) in enumerate(test_loader): 
                 t_x = Variable(m).cuda() 
                 t_y = Variable(n).cuda()
                 output = net(t_x.float()) 
                 pre=output.reshape(-1)
                 loss = loss_function(pre, t_y.float())
                 t_loss += loss.item()
                 cosim +=cos_sim(pre.cpu().detach().numpy(),t_y.cpu())
                 #pre_all=torch.cat([pre_all.cpu().float(),t_y.cpu().float()])
             print( '第%d轮结束后:'%epoch,'|average_test_loss:',t_loss/(k+1),'|平均余弦相似:',cosim/(k+1))
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