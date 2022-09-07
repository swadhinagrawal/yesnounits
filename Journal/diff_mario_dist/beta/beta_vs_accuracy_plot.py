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


beta = [i for i in range(1,6)]

fig1, ax1 = plt.subplots()

thrs = [0.3,0.5,0.8,1]
memory = range(2,22,2)
hellinger_dist = []
std_dev = []
for i in range(1,6):
    path = os.getcwd() + '/beta' + str(i) +'/'
    hell_dis_thr = pkl.load(open(path+str(0)+'hell_dis_thr.pickle', 'rb'))
    hell_dis_mem = pkl.load(open(path+str(0)+'hell_dis_mem.pickle', 'rb'))

    hell_dis_thr = np.array(hell_dis_thr)

    for th in range(1):
        for mems in range(4,5):
            hell_dis_thr_last = np.sum(hell_dis_thr[th,mems*20:(mems+1)*20,-1],axis=0)/20
            std_dev_ = 0
            for elem in hell_dis_thr[th,mems*20:(mems+1)*20,-1]:
                std_dev_ = std_dev_ + ((elem - hell_dis_thr_last)**2)/20
            std_dev.append(np.sqrt(std_dev_))
            hellinger_dist.append(hell_dis_thr_last)

ax1.fill_between(beta,np.array(hellinger_dist)-np.array(std_dev),np.array(hellinger_dist)+np.array(std_dev),color='black',alpha=0.2)
ax1.plot(beta,np.array(hellinger_dist),c='black',label='Threshold step: '+str(thrs[0])+'; '+'Memory size: '+str(memory[4]))

ax1.set_xlabel(r"$\beta$",fontsize = 18)
ax1.set_ylabel("Hellinger distance",fontsize = 18)
ax1.legend()
fig1.savefig('accuracy_beta.png')
plt.show()

beta = [i for i in range(1,6)]
fig1, ax1 = plt.subplots()
thrs = [0.3,0.5,0.8,1]
memory = range(2,22,2)
success_rates = []
std_dev = []
run_no = 0
for i in range(1,6):
    path = os.getcwd() + '/beta' + str(i) +'/results/'

    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    data_files = data_files.reshape((8,10,20))
    
    for th in range(1):
        for mems in range(4,5):
            count = 0
            for runs in range(data_files.shape[2]):
                data = pd.read_csv(path+data_files[th,mems,runs])
                if data.iloc[-1]['$x_{max}$ opt No.'] == data.iloc[-1]['$CDM$ opt No.']:
                    count += 1
            success_rate = count/data_files.shape[2]
            success_rates.append(success_rate)

ax1.plot(beta,np.array(success_rates),c='black',label='Threshold step: '+str(thrs[0])+'; '+'Memory size: '+str(memory[4]))

ax1.set_xlabel(r"$\beta$",fontsize = 18)
ax1.set_ylabel("Success rate",fontsize = 18)
ax1.legend()
fig1.savefig('success_rate_beta.png')
plt.show()
