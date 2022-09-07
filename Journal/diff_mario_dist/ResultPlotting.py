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


class Prediction:
    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
    def gaussian(x,mu,sigma,result):
        '''
        Inputs: base, list of means, list of variance (in case of bimodal distributions) 
        Output: list of Normal PDF values
        '''
        n = x.shape[0]
        m = mu.shape[0]
        for j in range(x.shape[0]):         #   For all base values we need the PDF values
            f = 0.0
            for i in range(len(mu)):        #   We calculate the PDF value due to each modal and superimpose them to 
                #   get PDF of bimodal distribution (We can later also make it weighted by adding weights list for each peak)
                k = 1/(sqrt(2*np.pi)*sigma[i])
                f += k*exp(-((x[j]-mu[i])**2)/(2*(sigma[i]**2)))
            result[j] = f

    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
    def uniform(x,mu,sigma,result):
        '''
        Inputs: base, list of means, list of variance (in case of bimodal distributions) 
        Output: list of Uniform PDF values
        '''
        n = x.shape[0]
        m = mu.shape[0]
        for j in range(n):      #   For all base values we need the PDF values
            f = 0.0
            for i in range(m):  #   We calculate the PDF value due to each modal and superimpose them to 
                #   get PDF of bimodal distribution (We can later also make it weighted by adding weights list for each peak)
                a = (mu[i] - np.sqrt(3)*sigma[i])   #   Lower bound of PDF
                b = (mu[i] + np.sqrt(3)*sigma[i])   #   Upper bound of PDF
                if x[j]<=b and x[j]>=a:
                    f += 1/abs(b-a)
            result[j] = f

    @staticmethod
    def ICPDF(area,mu,step,x,pdf):
        '''
        Inputs: Area where we need base of PDF, list of mean of PDF, resolution of PDF, base of PDF, PDF function values
        Output: The base where cummulative area under the PDF is as desired
        '''
        if len(mu)>1:                           #   If the PDF is bimodal
            dummy_area = 0.5                    #   We make a dummy area variable to start at a known point closest to the area where we need the base
            x_ = (mu[0]+mu[1])/2.0              #   The base where dummy area under the PDF is true is the mean of means
        else:
            dummy_area =0.5
            x_ = mu[0]                          #   If the PDF is unimodal, the 50% area is at the mean of PDF

        count = np.argmin(np.abs(x-x_))         #   We find the index of element in our base list closest to the known base point for dummy area
        
        while abs(dummy_area-area)>0.001:       #   While the distance between dummy area and the desired area is not small enough, we keep searching the point 
            if dummy_area>area:                 #   If desired area is smaller than the dummy
                count -= 1                      #   We decrease the index count in base list
                dummy_area -= pdf[count]*step   #   Hence, decrease the area under the PDF accordingly
                x_ -= step                      #   Also, move the search point location towards left
            elif area>dummy_area:               #   Otherwise, if desired area > dummy
                count += 1                      #   We increment the index of base list
                dummy_area += pdf[count]*step   #   Hence, add area to existing dummy
                x_ += step                      #   Also, move the search point rightward
        return x_

def Hellinger_distance(pdf1,pdf2,bins):
    pdf1 = np.sqrt(np.array(pdf1))
    pdf2 = np.sqrt(np.array(pdf2))
    norm = np.linalg.norm((pdf1-pdf2)*(bins[1]-bins[0]))/np.sqrt(2)
    return norm

def PDF(pdf_func,mu,sigma,bins):
    pdf =  pdf_func(bins,mu,sigma)                          #   PDF function values at each base values (array)
    normalized_pdf = pdf/(np.sum(pdf)*(bins[1]-bins[0]))                #   Normalized PDF
    return normalized_pdf

def init(ax):
    line1, = ax.plot([],[],c='red')
    line2, = ax.plot([],[],c='black')
    dist1, = ax.plot([],[],c='black')
    dist2, = ax.plot([],[],c='blue')
    line1.set_data([], [])
    line2.set_data([], [])
    dist1.set_data([], [])
    dist2.set_data([], [])
    return line1, line2, dist1, dist2

def animate(i,line1,line2,dist1,dist2,storage,icpdf,bins,pdf_h):
    line1.set_data([storage[i][0],storage[i][0]], [0,max(storage[i][1])])
    line2.set_data([icpdf,icpdf],[0,max(storage[i][1])])
    dist1.set_data(bins,pdf_h)
    dist2.set_data(bins,storage[i][1])
    return line1, line2, dist1, dist2

plt.ion()
thr_mem_anim_some_averaged = 1
if thr_mem_anim_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()
    run_no = 24
    params = pd.read_csv(path+str(run_no)+'.csv')
    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    data_files = data_files.reshape((8,10,20))

    start = 0.0
    end = 25.0
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
    hell_dis_thr = []
    hell_dis_mem = []
    for short_var in range(data_files.shape[0]):
        hell_dists_2d_array = []
        for long_var in range(data_files.shape[1]):
            for runs in range(data_files.shape[2]):
                hell_dis = []
                data = pd.read_csv(path+data_files[short_var,long_var,runs])
                fig, ax = plt.subplots()
                camera = Camera(fig)

                if short_var < 4:
                    for t in range(-1,1500,50):
                        if t<0:
                            t = 0
                        thresholds = [data.iloc[t][str(i)] for i in range(250)]
                        distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                        mean = np.sum(thresholds)/len(thresholds)
                        if len(distribution)>len(bins):
                            distribution = distribution[1:]
                        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                        distribution = distribution/distribution_area

                        hell_dis.append(Hellinger_distance(pdf_x1,distribution,bins))

                        ax.plot([mean,mean],[0,max(distribution)],c='blue')
                        # ax.plot([icpdf,icpdf],[0,max(pdf_h1)],c='green')
                        ax.plot([10,10],[0,max(pdf_x)],c='red')
                        # ax.plot(bins,pdf_h,c='black')
                        # ax.plot(x,pdf_h1,c='green')
                        ax.plot(x,pdf_x,c='red')
                        ax.plot(bins,distribution,c='blue')
                        
                        plt.pause(0.0001)
                        camera.snap()

                        # ax.clear()
                        # plt.clf()

                else:
                    for t in range(-1,1500,50):
                        if t<0:
                            t = 0
                        thresholds = [data.iloc[t][str(i)] for i in range(250)]
                        distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                        mean = np.sum(thresholds)/len(thresholds)
                        if len(distribution)>len(bins):
                            distribution = distribution[1:]
                        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                        distribution = distribution/distribution_area
                        hell_dis.append(Hellinger_distance(pdf_x1,distribution,bins))

                        ax.plot([mean,mean],[0,max(distribution)],c='blue')
                        # ax.plot([icpdf,icpdf],[0,max(pdf_h1)],c='green')
                        ax.plot([10,10],[0,max(pdf_x)],c='red')
                        # ax.plot(bins,pdf_h,c='black')
                        # ax.plot(x,pdf_h1,c='green')
                        ax.plot(x,pdf_x,c='red')
                        ax.plot(bins,distribution,c='blue')
                        
                        plt.pause(0.0001)
                        camera.snap()
                        # ax.clear()
                        # plt.clf()

                animation = camera.animate()
                animation.save(path+data_files[short_var,long_var,runs][:-4]+'.gif',writer='imagemagick')
                
                hell_dists_2d_array.append(hell_dis)
                plt.close()
        if short_var<4:
            hell_dis_thr.append(hell_dists_2d_array)
        else:
            hell_dis_mem.append(hell_dists_2d_array)
                    
    plt.ioff()
    pkl.dump(hell_dis_thr, open(str(run_no)+'hell_dis_thr.pickle', 'wb'))
    pkl.dump(hell_dis_mem, open(str(run_no)+'hell_dis_mem.pickle', 'wb'))

    t = np.arange(-1,1500,50)
    t[0] = 0

    fig1, ax1 = plt.subplots()
    hell_dis_thr = np.array(hell_dis_thr)
    
    thrs = [0.1,0.4,0.8,1.0]
    colors = ['green','blue','brown','red']
    for th in range(len(thrs)):
        hell_dis_thr_i = np.sum(hell_dis_thr[th,:,:,:],axis=1)/20
        ax1.plot(t,np.array(hell_dis_thr_i[1]),c=colors[th],label=thrs[th])

        # for mem in range(len(hell_dis_thr_i)):
        #     ax1.plot(t,np.array(hell_dis_thr_i[mem]),c=colors[th],label=thrs[th])

    fig2, ax2 = plt.subplots()
    hell_dis_mem = np.array(hell_dis_mem)
    
    mems = [2,10,20,40]
    for m in range(len(mems)):
        hell_dis_mem_i = np.sum(hell_dis_mem[m,:,:,:],axis=1)/20
        ax2.plot(t,np.array(hell_dis_mem_i[1]),c=colors[m],label=mems[m])

        # for mem in range(len(hell_dis_thr_i)):
        #     ax1.plot(t,np.array(hell_dis_thr_i[mem]),c=colors[th],label=thrs[th])

    ax1.set_xlabel("Time",fontsize = 18)
    ax1.set_ylabel("Hellinger distance",fontsize = 18)
    ax1.set_title("For varying thresholds steps, memory size = 10")
    ax1.legend()
    ax2.set_xlabel("Time",fontsize = 18)
    ax2.set_ylabel("Hellinger distance",fontsize = 18)
    ax2.set_title("For varying memory sizes, threshold step = 0.3")
    ax2.legend()
    plt.show()

plt.ion()
speed_accuracy_tradeoff_some_averaged = 1
if speed_accuracy_tradeoff_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()
    run_no = 0
    params = pd.read_csv(path+str(run_no)+'.csv')
    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    data_files = data_files.reshape((8,20))

    start = 0.0
    end = 25.0
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
    hell_dis_thr = []
    hell_dis_mem = []
    for var in range(data_files.shape[0]):
        for runs in range(data_files.shape[1]):
            hell_dis = []
            if var < 4:
                data = pd.read_csv(path+data_files[var,runs])
                fig, ax = plt.subplots()
                camera = Camera(fig)

                for t in range(-1,1000,50):
                    if t<0:
                        t = 0
                    thresholds = [data.iloc[t][str(i)] for i in range(250)]
                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]
                    distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                    distribution = distribution/distribution_area

                    hell_dis.append(Hellinger_distance(pdf_x1,distribution,bins))

                    ax.plot([mean,mean],[0,max(distribution)],c='blue')
                    # ax.plot([icpdf,icpdf],[0,max(pdf_h1)],c='green')
                    ax.plot([10,10],[0,max(pdf_x)],c='red')
                    # ax.plot(bins,pdf_h,c='black')
                    # ax.plot(x,pdf_h1,c='green')
                    ax.plot(x,pdf_x,c='red')
                    ax.plot(bins,distribution,c='blue')
                    
                    plt.pause(0.0001)
                    camera.snap()

                    # ax.clear()
                    # plt.clf()

                animation = camera.animate()
                animation.save(path+data_files[var,runs][:-4]+'.gif',writer='imagemagick')
                
                hell_dis_thr.append(hell_dis)
                plt.close()

            else:
                data = pd.read_csv(path+data_files[var,runs])
                fig, ax = plt.subplots()
                camera = Camera(fig)
                for t in range(-1,1000,50):
                    if t<0:
                        t = 0
                    thresholds = [data.iloc[t][str(i)] for i in range(250)]
                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]
                    distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                    distribution = distribution/distribution_area
                    hell_dis.append(Hellinger_distance(pdf_x1,distribution,bins))

                    ax.plot([mean,mean],[0,max(distribution)],c='blue')
                    # ax.plot([icpdf,icpdf],[0,max(pdf_h1)],c='green')
                    ax.plot([10,10],[0,max(pdf_x)],c='red')
                    # ax.plot(bins,pdf_h,c='black')
                    # ax.plot(x,pdf_h1,c='green')
                    ax.plot(x,pdf_x,c='red')
                    ax.plot(bins,distribution,c='blue')
                    
                    plt.pause(0.0001)
                    camera.snap()
                    # ax.clear()
                    # plt.clf()
                animation = camera.animate()
                animation.save(path+data_files[var,runs][:-4]+'.gif',writer='imagemagick')
                
                hell_dis_mem.append(hell_dis)
                plt.close()
                
    plt.ioff()
    t = np.arange(-1,1000,50)
    t[0] = 0

    fig1, ax1 = plt.subplots()
    hell_dis_thr = np.array(hell_dis_thr)
    
    thrs = [0.1,0.4,0.8,1.0]
    colors = ['green','blue','brown','red']
    for i in range(0,len(hell_dis_thr),20):
        hell_dis_thr_i = np.sum(hell_dis_thr[i:i+20],axis=0)/20
        i = int(i/20)
        if i==0:
            ax1.plot(t,np.array(hell_dis_thr_i),c=colors[i],label=thrs[i])
        elif i==len(thrs)-1:
            ax1.plot(t,np.array(hell_dis_thr_i),c=colors[i],label=thrs[i])
        else:
            ax1.plot(t,np.array(hell_dis_thr_i),c=colors[i],label=thrs[i])

    fig2, ax2 = plt.subplots()
    hell_dis_mem = np.array(hell_dis_mem)
    
    mems = [2,10,20,40]
    for i in range(0,len(hell_dis_mem),20):
        hell_dis_mem_i = np.sum(hell_dis_mem[i:i+20],axis=0)/20
        i = int(i/20)
        if i==0:
            ax2.plot(t,np.array(hell_dis_mem_i),c=colors[i],label=mems[i])
        elif i==len(mems)-1:
            ax2.plot(t,np.array(hell_dis_mem_i),c=colors[i],label=mems[i])
        else:
            ax2.plot(t,np.array(hell_dis_mem_i),c=colors[i],label=mems[i])

    ax1.set_xlabel("Time",fontsize = 18)
    ax1.set_ylabel("Hellinger distance",fontsize = 18)
    ax1.set_title("For varying thresholds steps, memory size = 2")
    ax1.legend()
    ax2.set_xlabel("Time",fontsize = 18)
    ax2.set_ylabel("Hellinger distance",fontsize = 18)
    ax2.set_title("For varying memory sizes, threshold step = 0.3")
    ax2.legend()
    plt.show()
