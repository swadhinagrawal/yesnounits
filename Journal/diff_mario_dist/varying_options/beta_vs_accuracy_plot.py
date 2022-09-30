# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import guvectorize, float64
from math import sqrt,exp
from celluloid import Camera
import pickle as pkl


beta = [i for i in range(1,8)]

fig1, ax1 = plt.subplots()
num_opts = [2,5,8,10]
thrs = [0.3,0.5,0.8,1]
memory = range(2,22,2)
hellinger_dist = []
std_dev = []
colors = ['mediumslateblue','teal','goldenrod','tomato']
for i in range(1,len(beta)+1):
    hellinger_dist_n = []
    std_dev_n = []
    path = os.getcwd() + '/beta' + str(i) +'/'
    hell_dis_opts = pkl.load(open(path+'hell_dis_opts.pickle', 'rb'))

    hell_dis_opts = np.array(hell_dis_opts)

    for n in range(hell_dis_opts.shape[0]):
        hell_dis_thr_last = np.sum(hell_dis_opts[n,:,-1],axis=0)/hell_dis_opts.shape[1]
        std_dev_ = 0
        for runs in range(hell_dis_opts.shape[1]):
            std_dev_ = std_dev_ + ((hell_dis_opts[n,runs,-1] - hell_dis_thr_last)**2)/hell_dis_opts.shape[1]
        std_dev_n.append(np.sqrt(std_dev_))
        hellinger_dist_n.append(hell_dis_thr_last)
    std_dev.append(std_dev_n)
    hellinger_dist.append(hellinger_dist_n)

hellinger_dist = np.array(hellinger_dist)
std_dev = np.array(std_dev)
for n in range(hell_dis_opts.shape[0]):
    ax1.fill_between(beta,np.array(hellinger_dist[:,n])-np.array(std_dev[:,n]),np.array(hellinger_dist[:,n])+np.array(std_dev[:,n]),color=colors[n],alpha=0.1)
    ax1.plot(beta,np.array(hellinger_dist[:,n]),c=colors[n],label='Number of options: '+str(num_opts[n]))

ax1.set_xlabel(r"$\beta$",fontsize = 18)
ax1.set_ylabel("Hellinger distance",fontsize = 18)
ax1.legend()
fig1.savefig('accuracy_beta_n.png')
plt.show()



beta = [i for i in range(1,8)]

fig1, ax1 = plt.subplots()
num_opts = [2,5,8,10]
thrs = [0.3,0.5,0.8,1]
memory = range(2,22,2)
success_rates = []
std_dev = []
colors = ['mediumslateblue','teal','goldenrod','tomato']
for i in range(1,len(beta)+1):
    success_rates_n = []
    std_dev_n = []
    path = os.getcwd() + '/beta' + str(i) +'/results/'
    data_files = np.array([])
    for i in range(len(num_opts)):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(i))] == str(i) and len(f)>10]))),axis=0)
    data_files = data_files.reshape((4,20,50))
    for n in range(len(num_opts)):
        count = 0
        for runs in range(data_files.shape[2]):
            data = pd.read_csv(path+data_files[n,4,runs])
            if data.iloc[-1]['$x_{max}$ opt No.'] == data.iloc[-1]['$CDM$ opt No.']:
                count += 1
        success_rate = count/data_files.shape[2]
        success_rates_n.append(success_rate)

    success_rates.append(success_rates_n)

success_rates = np.array(success_rates)
for n in range(hell_dis_opts.shape[0]):
    ax1.plot(beta,np.array(success_rates[:,n]),c=colors[n],label='Number of options: '+str(num_opts[n]))
    # ax1.plot(beta,np.array(hellinger_dist[:,n]),c=colors[n],label='Number of options: '+str(num_opts[n]))

ax1.set_xlabel(r"$\beta$",fontsize = 18)
ax1.set_ylabel("Success rate",fontsize = 18)
ax1.legend()
fig1.savefig('success_rate_beta_n.png')
plt.show()



