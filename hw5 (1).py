#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import time
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# In[2]:


# Load the dataset
with np.load('mnist.npz') as data:
    xx = data['x_train']
    yy = data['y_train']

xx = xx.reshape(-1, 784) / 255.  # normalization of the data
yy = yy.astype(np.int64)

m = xx.shape[0]
n = xx.shape[1] + 1

X = np.concatenate((np.ones([m,1]),xx), axis=1)  # add one to each data point

cat = np.zeros([m,10])
for ind, num in enumerate(yy):
    cat[ind][num] = 1
Y = cat

# split the dataset into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=42)


# In[3]:


print('information of datasets:')
print(f'- X original: {X.shape[0]} x {X.shape[1]}')
print(f'- Y original: {Y.shape[0]} x {Y.shape[1]}')
print(f'- X Train: {x_train.shape[0]} x {x_train.shape[1]}')
print(f'- Y train: {y_train.shape[0]} x {y_train.shape[1]}')
print(f'- X Test: {x_test.shape[0]} x {x_test.shape[1]}')
print(f'- Y Test: {y_test.shape[0]} x {y_test.shape[1]}')


# In[5]:


NumSample = np.random.randint(0,59500)
plt.imshow(np.reshape(x_train[NumSample][1:], [28,28]), cmap='Greys')
print(np.argmax(y_train[NumSample]))


# In[6]:


dg1=7
dg2=8
num46 = sum(Y[:,dg1]) + sum(Y[:,dg2])
Y4 = Y[:,dg1];
Y6 = Y[:,dg2];
indx4 = np.where(Y4==1)[0];
indx6 = np.where(Y6==1)[0];

indx46 = np.concatenate((indx4,indx6))

X46 = X[indx46,1:]
Y46 = np.ones(len(indx46))
Y46[len(indx4):] =-1

print(X46)
print(Y46)

xdual_train, xdual_test, ydual_train, ydual_test = train_test_split(X46, Y46, test_size = 0.15, random_state=42)


# In[ ]:


# assemble matrix A in L = 1^T a - 0.5 a^T A a

#A = np.zeros([N,N])

#    print(i,N)
#    for j in range(i,N): # from i+1 to N
#        A[i,j] = np.dot(X46[i,:]*Y46[i],X46[j,:]*Y46[j])

#for j in range(N-1):
#    for i in range(N-1):
#        M[j,i] = A[i,j] + A[0,0]*Y46[i]*Y46[j]/Y46[0]**2 - A[0,j]*Y46[i]/Y46[0]


# In[ ]:


#xdual_train=np.array([[1,0],[0,0],[0,1]])
#ydual_train=np.array([-1,1])
#print(xdual_train)


# In[7]:


N = len(ydual_train)
XPY = xdual_train
for i in range(N):
    if ydual_train[i]==-1:
        XPY[i,:] =-1 * xdual_train[i,:]   

A = np.matmul(XPY,XPY.transpose())

AT = A.transpose()

YM = np.outer(ydual_train[1:],ydual_train[1:])
AY = np.outer(A[0,1:],ydual_train[1:])
YA = np.outer(ydual_train[1:],A[0,1:])


Y0S = ydual_train[0]**2
M = AT[1:,1:] + A[0,0]*YM/Y0S - AY/ydual_train[0] - YA/ydual_train[0]


b = np.zeros(N-1)
b = 1 - ydual_train[1:]/ydual_train[0]
#alpha = LA.solve(M,b)
#a1 =-sum(Y46[1:]*alpha)/Y46[0]
#alpha = np.insert(alpha,0,a1)#
#print(M)
#print(b)

aw = np.zeros(N)
for i in range(2,N):
    aw[i] = (1-ydual_train[i]/ydual_train[0])/(A[i,i] + A[0,0]*ydual_train[i]**2/ydual_train[0]**2 
                                               - 2*A[0,i]*ydual_train[i]/ydual_train[0])

aw[0] = -sum(ydual_train[1:]*aw[1:])/ydual_train[0]


# In[8]:


print(aw)
print(ydual_train)


# In[9]:


YA   = ydual_train*aw
wght = sum(xdual_train * YA[:,None])
#wght = sum(xdual_train * YA)


# In[10]:


b =sum(ydual_train - np.matmul(xdual_train,wght))/N
#b =sum(ydual_train - xdual_train*wght)/N
yfit = np.matmul(xdual_train,wght) + b
yfitsign = np.sign(yfit)
ydiff = yfitsign - np.asfarray(ydual_train)
wherediff = np.where(ydiff != 0)


# In[11]:


print(wherediff)
print(yfitsign[:15])
print(ydual_train[:15])
print(len(wherediff[0])/len(ydual_train))


# In[12]:


plt.imshow(np.reshape(xdual_train[9][:], [28,28]), cmap='Greys')


# In[13]:


print(wght.shape)


# In[14]:


mwght = max(wght)
wghtimg=np.reshape(wght, [28,28])


# In[15]:


wghtimg = wghtimg*225/mwght


# In[16]:


plt.imshow(wghtimg, cmap='Greys')


# In[17]:


X46=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[0,1],[0,2],[0,3],[0,4]])
Y46=np.array([-1,-1,-1,-1,-1,1,1,1,1])
N = 9
print(X46.shape)
print(Y46)

