# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:06:52 2021

@author: sturm
"""
import numpy as np
import matplotlib.pyplot as plt

#model = 'DE-SGLD-telescope-'
model = 'DE-SGHMC-telescope-'
data_dir = model + 'dis-'

result_up = np.load(data_dir+'train-up.npy')
result_down = np.load(data_dir+'train-down.npy')

result_acc = (result_up+result_down)/2

result_up2 = np.load(data_dir+'test-up.npy')
result_down2 = np.load(model +'down-test-down.npy')
result_acc2 = (result_up2+result_down2)/2


T=20
index=[]
for i in range(T+1):
    index.append(i)
figsize = 8.8,6.6
figure, ax = plt.subplots(figsize=figsize)
plt.plot(result_acc,'b-',linewidth=3)
plt.fill_between(index,result_up,result_down,alpha=0.5)
plt.ylabel('Train accuracy',fontsize=25)
plt.xlabel(r'Iterations $k$',fontsize=25)
plt.legend(['Mean of accuracy',r'Accuracy $\pm$ Std'], loc='lower right',fontsize=25)
plt.ylim(top=0.85,bottom=0)
plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.grid()
plt.show()

figsize = 8.8,6.6
figure, ax = plt.subplots(figsize=figsize)
plt.plot(result_acc2,'b-',linewidth=3)
plt.fill_between(index,result_up2,result_down2,alpha=0.5)
plt.ylabel('Test accuracy',fontsize=25)
plt.xlabel(r'Iterations $k$',fontsize=25)
plt.legend(['Mean of accuracy',r'Accuracy $\pm$ Std'], loc='lower right',fontsize=25)
plt.ylim(top=0.85,bottom=0)
plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.grid()
plt.show()