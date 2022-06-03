

# In[1]:


from __future__ import print_function
from builtins import range


# In[2]:


"""
SECTION 1 : Load and setup data for training

"""
import pandas as pd
import numpy as np


# In[3]:


# Data sets
IRIS_TRAINING = "train.txt"
IRIS_TEST = "test.txt"
train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
    dtype=float, delimiter=';')
test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
    dtype=float, delimiter=';')


# In[4]:


#split x and y (feature and target)
xtrain = train_data[:,:4000]
ytrain = train_data[:,4001]
print(ytrain)


# In[5]:




# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1457)


# In[7]:


#hyperparameters
hl1 = 20
hl2 = 20
#hl3 = 20
lr = 0.2
num_epoch = 1500


# In[8]:


#build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4000, hl1)
        self.fc3 = nn.Linear(hl1,hl2)
        self.fc2 = nn.Linear(hl2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
       
        x = self.fc2(x)
        return x
        
net = Net()


# In[9]:


#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)


# In[10]:


#train
for epoch in range(num_epoch):
    X = torch.Tensor(xtrain).float()
    Y = torch.Tensor(ytrain).long()

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    acc = 100 * torch.sum(Y==torch.max(out.data, 1)[1]).double() / len(Y)
    if (epoch % 50 == 1):
	    print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                   %(epoch+1, num_epoch, loss.item(), acc.item()))


# In[11]:


"""
SECTION 3 : Testing model
"""


# In[12]:


#split x and y (feature and target)
xtest = test_data[:,:4000]
ytest = test_data[:,4001]


# In[13]:


#get prediction
X = torch.Tensor(xtest).float()
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)


# In[14]:


#get accuration
print('Accuracy of testing %.4f %%' % (100 * torch.sum(Y==predicted).double() / len(Y)))

