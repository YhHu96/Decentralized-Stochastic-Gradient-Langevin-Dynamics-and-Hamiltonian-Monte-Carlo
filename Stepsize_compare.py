# -*- coding: utf-8 -*-
import numpy as np
import math
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

size_w=6
N=1000
dim=2
sigma_sample=1

def generate_cir(N):
    x=np.zeros([N,N])
    for i in range(N-2):
        x[i][i]=1/3
        x[i][i+1]=1/3
        x[i][i+2]=1/3
    x[N-2][N-2]=1/3
    x[N-2][N-1]=1/3
    x[N-2][0]=1/3
    x[N-1][N-1]=1/3
    x[N-1][0]=1/3
    x[N-1][1]=1/3
    return x

def generate_ful(N):
    x=np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            x[i][j]=1/N
    return x

# fully-connected network
w=generate_ful(size_w)

# circular network
#w=generate_cir(size_w)

# disconnected network
#w=np.eye(size_w)


k=[0.005,0.01,0.02]     # step size list
T=350
lam=10
p=25                    # batch size
sigma=lam*np.eye(dim)

'''
    Data generation
'''

x=[]
#np.random.seed(10)
for i in range(5000):
    x.append([np.random.random()*1])
#np.random.seed(11)
y=[item[0]*3-0.5+np.random.random() for item in x]

'''
    Data spliting
'''

x_all=np.array(x)
y_all=np.array(y)
x_all=np.insert(x_all, 0, 1, axis=1)
cov_pri=lam*np.eye(dim)

avg_post=np.dot(np.linalg.inv(np.linalg.inv(cov_pri)+np.dot(np.transpose(x_all),x_all)/(sigma_sample**2)),(np.dot(np.transpose(x_all),y_all)/(sigma_sample**2)))
cov_post=np.linalg.inv(np.linalg.inv(cov_pri)+np.dot(np.transpose(x_all),x_all)/(sigma_sample**2))

x=np.split(x_all,100)
y=np.split(y_all,100)

def gradient(beta,x,y,dim,lam,p):
    f=[]
    for i in range(dim):
        f.append(0)
    f=np.array(f)
    randomList=np.random.randint(0,len(y)-1,size=int(p))
    for i in randomList:
        f=f-np.dot((y[i]-np.dot(beta,x[i])),x[i])
    f=f+np.dot(1/lam,beta)
    return f

'''
    Update
'''

w2=[]
for s in range(3):
    step=k[s]
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
        
    
    
    for m in range(T):
        for n in range(N):
            for i in range(size_w):
                g=gradient(beta[n][i],x[i],y[i],dim,lam,p)
                temp=np.zeros(dim)
                for j in range(len(beta[n])):
                    temp=temp+w[i][j]*beta[n][j]
                #gradient_noise=np.random.normal(0,1,dim)
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
    Calculate the W2 distance for agent 1
    '''

    w2dis=[]
    for t in range(T+1):
        d=0
        i=0
        avg_temp=[]
        avg_temp.append(np.mean(history_all[t][i][0]))
        avg_temp.append(np.mean(history_all[t][i][1]))
        avg_temp=np.array(avg_temp)
        cov_temp=np.cov(history_all[t][i])
        '''
        avg_temp=[]
        avg_temp.append(np.mean(beta_mean_all[t][0]))
        avg_temp.append(np.mean(beta_mean_all[t][1]))
        avg_temp=np.array(avg_temp)
        cov_temp=np.cov(beta_mean_all[t])
        '''
        d=np.linalg.norm(avg_post-avg_temp)*np.linalg.norm(avg_post-avg_temp)
        d=d+np.trace(cov_post+cov_temp-2*sqrtm(np.dot(np.dot(sqrtm(cov_temp),cov_post),sqrtm(cov_temp))))
        w2dis.append(np.log(math.sqrt(abs(d))))
    w2.append(w2dis)
    
'''
    Plot the figures
'''

for i in range(len(w2)):
    w2[i]=np.array(w2[i])

figsize = 8.8,6.6
figure, ax = plt.subplots(figsize=figsize)
plt.plot(w2[0],linewidth=3)
plt.plot(w2[1],linewidth=3)
plt.plot(w2[2],linewidth=3)
plt.ylabel('log(W2 Distance)',fontsize=25)
plt.xlabel(r'Iterations $k$',fontsize=25)
plt.legend([r'Step $\eta=0.005$',r'step $\eta=0.01$',r'step $\eta=0.02$'], loc='upper right',fontsize=25)
plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.grid()
plt.show()