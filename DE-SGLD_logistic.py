# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

size_w=6
N=100
sigma=1

# fully-connected network
'''
w=[[1/6,1/6,1/6,1/6,1/6,1/6],
   [1/6,1/6,1/6,1/6,1/6,1/6],
   [1/6,1/6,1/6,1/6,1/6,1/6],
   [1/6,1/6,1/6,1/6,1/6,1/6],
   [1/6,1/6,1/6,1/6,1/6,1/6],
   [1/6,1/6,1/6,1/6,1/6,1/6]]


# circular network

w=[[1/3,1/3,0,0,0,1/3],
   [1/3,1/3,1/3,0,0,0],
   [0,1/3,1/3,1/3,0,0],
   [0,0,1/3,1/3,1/3,0],
   [0,0,0,1/3,1/3,1/3],
   [1/3,0,0,0,1/3,1/3]]


# disconnected network
'''
w=[[1,0,0,0,0,0],
   [0,1,0,0,0,0],
   [0,0,1,0,0,0],
   [0,0,0,1,0,0],
   [0,0,0,0,1,0],
   [0,0,0,0,0,1]]


dim=3
k=0.0005
T=20
gamma=1
p=32


x=[]
np.random.seed(10)
for i in range(1000):
    x.append([-20+(20+20)*np.random.normal(),-10+np.random.normal()])
np.random.seed(11)
y=[1/(1+np.exp(-item[0]*1-1*item[1]+10)) for item in x]
for i in range(len(y)):
    temp=np.random.uniform(0,1)
    if temp<= y[i]:
        y[i]=1
    else:
        y[i]=0

x_all=np.array(x)
y_all=np.array(y)
x_all=x
y_all=y
x_all=np.insert(x_all, 0, 1, axis=1)
x=x_all

lam=10

'''
    Data spliting
'''

X_train1, x_trainRemain, y_train1, y_trainRemain = train_test_split(x, y, test_size=0.83333, random_state=42)
X_train2, x_trainRemain, y_train2, y_trainRemain = train_test_split(x_trainRemain, y_trainRemain, test_size=0.8, random_state=42)
X_train3, x_trainRemain, y_train3, y_trainRemain = train_test_split(x_trainRemain, y_trainRemain, test_size=0.75, random_state=42)
X_train4, x_trainRemain, y_train4, y_trainRemain = train_test_split(x_trainRemain, y_trainRemain, test_size=0.66666666, random_state=42)
X_train5, X_train6, y_train5, y_train6 = train_test_split(x_trainRemain, y_trainRemain, test_size=0.5, random_state=42)

x=[X_train1 , X_train2 , X_train3 , X_train4 , X_train5 , X_train6]
y=[y_train1 , y_train2 , y_train3 , y_train4 , y_train5 , y_train6]

'''
    Initialization
'''

beta=[]
for i in range(N):
    t=[]
    for j in range(size_w):
        temp=np.random.normal(0,sigma,dim)
        t.append(temp)
    beta.append(t)
beta=np.array(beta)
#beta = N * size_w * dim

history_all=[]       # T+1 * size_w * dim * N
beta_mean_all=[]     # T+1 * dim * N
for t in range(1):
    history=[]
    beta_mean=[]
    for i in range(size_w):
        temp=[]
        for d in range(dim):
            t=[]
            temp.append(t)
        history.append(temp)
    for d in range(dim):
        temp=[]
        beta_mean.append(temp)
    #history = size_w * dim * N
    #beta_mean = dim * N

    for i in range(size_w):
        for d in range(dim):
            for n in range(N):
                history[i][d].append(beta[n][i][d])
    for d in range(dim):
        for n in range(N):
            temp=0
            for i in range(size_w):
                temp=temp+1/size_w*history[i][d][n]
            beta_mean[d].append(temp)
    history_all.append(history)
    beta_mean_all.append(beta_mean)

def gradient(beta,x,y,dim,lam,p):
#     F=-y*log(h(beta,x))-(1-y)*log(1-h(beta,x))
#     gradient=(h(beta,x)-y)*x
    f=[]
    for i in range(dim):
        f.append(0)
    randomList=np.random.randint(0,len(y)-1,size=int(p))
    for item in randomList:
        h=1/(1+np.exp(-np.dot(beta,x[item])))
        f=f-np.dot((y[item]-h),x[item]) 
    f=f+np.dot(2/lam,beta)
    return f

'''
    Update
'''

for m in range(T):
    #step=k*1/(1*m+1)
    step=k
    for n in range(N):
        for i in range(size_w):
            g=gradient(beta[n][i],x[i],y[i],dim,lam,p)
            temp=np.zeros(dim)
            for j in range(len(beta[n])):
                temp=temp+w[i][j]*beta[n][j]
            noise=np.random.normal(0,1,dim)
            beta[n][i]=temp-step*g+math.sqrt(2*step)*noise

    history=[]
    beta_mean=[]
    for i in range(size_w):
        temp=[]
        for d in range(dim):
            t=[]
            temp.append(t)
        history.append(temp)
    for d in range(dim):
        temp=[]
        beta_mean.append(temp)
    #history = size_w * dim * N
    #beta_mean = dim * N

    for i in range(size_w):
        for d in range(dim):
            for n in range(N):
                history[i][d].append(beta[n][i][d])
    for d in range(dim):
        for n in range(N):
            temp=0
            for i in range(size_w):
                temp=temp+1/size_w*history[i][d][n]
            beta_mean[d].append(temp)
    history_all.append(history)
    beta_mean_all.append(beta_mean)
    
'''
    Calculate the accuracy
'''

mis_class=[]
for m in range(T+1):
    mis_class.append([])
for t in range(T+1):
    #mis_class.append([])
    for n in range(N):
        temp0=0
        for i in range(len(x_all)):
            z=1/(1+np.exp(-np.dot(np.transpose(history_all[t][1])[n],x_all[i])))
            if z>=0.5:
                z=1
            else:
                z=0
            if y_all[i]!=z:
                temp0=temp0+1
        mis_class[t].append(1-temp0/len(x_all))
        
result_acc=[]
result_std=[]
for m in range(T+1):
    result_acc.append(np.mean(mis_class[m]))
    result_std.append(np.std(mis_class[m]))
result_acc=np.array(result_acc)
result_std=np.array(result_std)
result_up=result_acc+result_std
result_down=result_acc-result_std

'''
    Plot the figures
'''

index=[]
for i in range(T+1):
    index.append(i)
figsize = 8.8,6.6
figure, ax = plt.subplots(figsize=figsize)
plt.plot(result_acc,'b-',linewidth=3)
plt.fill_between(index,result_up,result_down,alpha=0.5)
plt.ylabel('Accuracy',fontsize=25)
plt.xlabel(r'Iterations $k$',fontsize=25)
plt.legend(['Mean of accuracy',r'Accuracy $\pm$ Std'], loc='lower right',fontsize=25)
plt.ylim(top=1,bottom=0)
plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.grid()
plt.show()
