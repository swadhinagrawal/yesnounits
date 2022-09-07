# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import guvectorize, float64
from math import sqrt,exp
from celluloid import Camera

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


# plt.ion()
hell_vs_mem_thr = 0
if hell_vs_mem_thr:
    path = os.getcwd() + "/results/"
    prd = Prediction()
    run_no = 19
    params = pd.read_csv(path+str(run_no)+'.csv')
    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    print(data_files)
    start = 0.0
    end = 30.0
    x = np.arange(start,end,0.0001)
    pdf_x,area = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,30)
    icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    sig_h = (0.07*np.log10(1)+0.57)*1
    pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    hellinger_dis = np.zeros((100,19))
    memory_size = []
    thresholds_step = []
    for j in range(0,100):
        for d in range(j*19,(j+1)*19):
            # param_columns = list(params.columns)

            data = pd.read_csv(path+data_files[d])
            # data_columns = list(data.columns)
            thresholds = [data.iloc[-1][str(i)] for i in range(250)]
            distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
            if len(distribution)>len(bins):
                distribution = distribution[1:]
            mean = np.sum(thresholds)/len(thresholds)

            distribution_area = (bins[1]-bins[0])*np.sum(distribution)

            if distribution_area!=0:
                distribution = distribution/distribution_area

            hellinger_dis[int(j),int((d%19))] = Hellinger_distance(pdf_h,distribution)
            memory_size.append(data.iloc[-1][-2])
            thresholds_step.append(data.iloc[-1][-1])
    #         plt.plot([mean,mean],[0,max(distribution)],c='red')
    #         plt.plot([icpdf,icpdf],[0,max(distribution)],c='black')
    #         plt.plot(bins,pdf_h,c='black')
    #         # plt.plot(x,pdf_h1,c='brown')
    #         plt.plot(bins,distribution,c='blue')
    #         plt.pause(0.001)
    #         plt.clf()
    # plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(np.unique(memory_size),np.sum(hellinger_dis,axis=0)/100)
    plt.xlabel("Memory size",fontsize = 18)
    plt.ylabel("Average Hellinger distance",fontsize = 18)
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(np.unique(thresholds_step),np.sum(hellinger_dis,axis=1)/int(len(range(2,40,2))))
    plt.xlabel("Change in threshold step",fontsize = 18)
    plt.ylabel("Average Hellinger distance",fontsize = 18)
    plt.show()

# plt.ion()
speed_accuracy_tradeoff = 0
if speed_accuracy_tradeoff:
    path = os.getcwd() + "/results/"
    prd = Prediction()
    run_no = 18
    params = pd.read_csv(path+str(run_no)+'.csv')
    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    print(data_files)
    shakiness_speed_accuracy = []
    avg_hel_dis = np.zeros(int(len(range(2,40,2))/2))

    start = 0.0
    end = 30.0
    x = np.arange(start,end,0.0001)
    pdf_x,area = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,30)
    icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    sig_h = (0.07*np.log10(1)+0.57)*1
    pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    for j in range(0,100,10):
        for d in [data_files[i] for i in range(j*len(range(2,40,2)),(j+1)*len(range(2,40,2)),2)]:
            print("In")
            # param_columns = list(params.columns)

            data = pd.read_csv(path+d)
            # data_columns = list(data.columns)
            if j==30 and d == data_files[(j)*len(range(2,40,2))]:
                print("in if")
                hell_dis_accurate = []
                for t in range(0,1500,50):
                    thresholds = [data.iloc[t][str(i)] for i in range(250)]
                    # print(thresholds)
                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    # print(distribution)
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]
                    distribution_area = (bins[1]-bins[0])*np.sum(distribution)
                    if distribution_area!=0:
                        distribution = distribution/distribution_area
                    
                    hell_dis_accurate.append(Hellinger_distance(pdf_h,distribution))
                shakiness_speed_accuracy.append(hell_dis_accurate)
            
            elif j==90 and d == data_files[(j)*len(range(2,40,2))]:
                print("in elif")
                hell_dis_fast = []
                for t in range(0,1500,50):
                    thresholds = [data.iloc[t][str(i)] for i in range(250)]

                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]
                    distribution_area = (bins[1]-bins[0])*np.sum(distribution)
                    if distribution_area!=0:
                        distribution = distribution/distribution_area
                    
                    hell_dis_fast.append(Hellinger_distance(pdf_h,distribution))
                shakiness_speed_accuracy.append(hell_dis_fast)
            else:
                print("in else")
                hell_dis = []
                for t in range(0,1500,50):
                    thresholds = [data.iloc[t][str(i)] for i in range(250)]

                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]

                    distribution_area = (bins[1]-bins[0])*np.sum(distribution)
                    if distribution_area!=0:
                        distribution = distribution/distribution_area
                    
                    hell_dis.append(Hellinger_distance(pdf_h,distribution))
                shakiness_speed_accuracy.append(hell_dis)
    #         plt.plot([mean,mean],[0,max(distribution)],c='red')
    #         plt.plot([icpdf,icpdf],[0,max(distribution)],c='black')
    #         plt.plot(bins,pdf_h,c='black')
    #         # plt.plot(x,pdf_h1,c='brown')
    #         plt.plot(bins,distribution,c='blue')
    #         plt.pause(0.01)
    #         plt.clf()
    # plt.ioff()
    avg_speed_accuracy = np.zeros_like(shakiness_speed_accuracy[0])
    
    for i in shakiness_speed_accuracy:
        plt.plot(range(0,1500,50),np.array(i),alpha= 0.1,c='blue')
        avg_speed_accuracy += np.array(i)
    avg_speed_accuracy = avg_speed_accuracy/len(shakiness_speed_accuracy)
    plt.plot(range(0,1500,50),hell_dis_accurate,c='green')
    plt.plot(range(0,1500,50),hell_dis_fast,c='red')
    plt.plot(range(0,1500,50),avg_speed_accuracy,c = 'orange')
    plt.xlabel("Time",fontsize = 18)
    plt.ylabel("Hellinger distance",fontsize = 18)
    plt.show()

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

# plt.ion()
speed_accuracy_tradeoff_some = 0
if speed_accuracy_tradeoff_some:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()
    run_no = 18
    params = pd.read_csv(path+str(run_no)+'.csv')
    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    data_files = data_files.reshape((100,19))

    start = -50.0
    end = 50.0
    x = np.arange(start,end,0.0001)
    pdf_x,area = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    sig_h = (0.07*np.log10(1)+0.57)*1
    pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)

    shakiness_speed_accuracy = []
    hell_dis_thr = []
    hell_dis_mem = []
    for th in range(data_files.shape[0]):
        for mem in range(data_files.shape[1]):
            # hell_dis = []
            # if th in [0,25,30,99] and mem == 0:
            #     data = pd.read_csv(path+data_files[th,mem])
            #     fig, ax = plt.subplots()
            #     camera = Camera(fig)

            #     for t in range(-1,1500,50):
            #         if t<0:
            #             t = 0
            #         thresholds = [data.iloc[t][str(i)] for i in range(250)]
            #         distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
            #         mean = np.sum(thresholds)/len(thresholds)
            #         if len(distribution)>len(bins):
            #             distribution = distribution[1:]
            #         distribution_area = (bins[1]-bins[0])*np.sum(distribution)
            #         if distribution_area!=0:
            #             distribution = distribution/distribution_area
            #         hell_dis.append(Hellinger_distance(pdf_h,distribution))

            #         ax.plot([mean,mean],[0,max(pdf_h)],c='red')
            #         ax.plot([icpdf,icpdf],[0,max(pdf_h)],c='black')
            #         ax.plot(bins,pdf_h,c='black')
            #         # plt.plot(x,pdf_h1,c='brown')
            #         ax.plot(bins,distribution,c='blue')
                    
            #         plt.pause(0.001)
            #         camera.snap()

            #         # ax.clear()
            #         # plt.clf()

            #     animation = camera.animate()
            #     animation.save(path+'a.gif',writer='imagemagick')
                
            #     hell_dis_thr.append(hell_dis)

            # hell_dis = []
            # if th == 25 and mem in [0,8,14,18]:
            #     data = pd.read_csv(path+data_files[th,mem])
            #     fig, ax = plt.subplots()
            #     camera = Camera(fig)
            #     for t in range(-1,1500,50):
            #         if t<0:
            #             t = 0
            #         thresholds = [data.iloc[t-1][str(i)] for i in range(250)]
            #         distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
            #         mean = np.sum(thresholds)/len(thresholds)
            #         if len(distribution)>len(bins):
            #             distribution = distribution[1:]
            #         distribution_area = (bins[1]-bins[0])*np.sum(distribution)
            #         if distribution_area!=0:
            #             distribution = distribution/distribution_area
            #         hell_dis.append(Hellinger_distance(pdf_h,distribution))

            #         ax.plot([mean,mean],[0,max(pdf_h)],c='red')
            #         ax.plot([icpdf,icpdf],[0,max(pdf_h)],c='black')
            #         ax.plot(bins,pdf_h,c='black')
            #         # plt.plot(x,pdf_h1,c='brown')
            #         ax.plot(bins,distribution,c='blue')
                    
            #         plt.pause(0.001)
            #         camera.snap()
            #         # ax.clear()
            #         # plt.clf()

            #     animation = camera.animate()
            #     animation.save(path+'b.gif',writer='imagemagick')
                
            #     hell_dis_mem.append(hell_dis)
            hell_dis_ = []
            
            if th%10==0 and mem%2==0 and th>20:
                data = pd.read_csv(path+data_files[th,mem])
                for t in range(-1,1500,10):
                    if t<0:
                        t = 0
                    thresholds = [data.iloc[t-1][str(i)] for i in range(250)]
                    distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                    mean = np.sum(thresholds)/len(thresholds)
                    if len(distribution)>len(bins):
                        distribution = distribution[1:]
                    distribution_area = (bins[1]-bins[0])*np.sum(distribution)
                    if distribution_area!=0:
                        distribution = distribution/distribution_area
                    hell_dis_.append(Hellinger_distance(pdf_h,distribution))
                    plt.plot([mean,mean],[0,max(distribution)],c='red')
                    plt.plot([icpdf,icpdf],[0,max(pdf_h)],c='black')
                    plt.plot(bins,pdf_h,c='black')
                    # plt.plot(x,pdf_h1,c='brown')
                    plt.plot(bins,distribution,c='blue',label='threshold step:'+str((th/100)+0.01)+';memory size:'+str(2*mem + 2))
                    plt.legend(loc='upper left')
                    plt.pause(0.0001)
                    plt.clf()
                shakiness_speed_accuracy.append(hell_dis_)
                

            
    plt.ioff()
    t = np.arange(-1,1500,50)
    t[0] = 0
    avg_speed_accuracy = np.zeros_like(shakiness_speed_accuracy[0])
    fig, ax = plt.subplots()
    for i in shakiness_speed_accuracy:
        ax.plot(t,np.array(i),alpha= 0.1,c='blue')
        avg_speed_accuracy += np.array(i)
    avg_speed_accuracy = avg_speed_accuracy/len(shakiness_speed_accuracy)
    ax.plot(t,np.array(avg_speed_accuracy),c='orange')
    # fig1, ax1 = plt.subplots()
    # for i in range(len(hell_dis_thr)):
    #     if i==0:
    #         ax1.plot(t,np.array(hell_dis_thr[i]),c='green')
    #     elif i==len(hell_dis_thr)-1:
    #         ax1.plot(t,np.array(hell_dis_thr[i]),c='red')
    #     else:
    #         ax1.plot(t,np.array(hell_dis_thr[i]),linewidth=2*i,c='blue')

    # fig2, ax2 = plt.subplots()
    # for i in range(len(hell_dis_mem)):
    #     if i==0:
    #         ax2.plot(t,np.array(hell_dis_mem[i]),c='green')
    #     elif i==len(hell_dis_mem)-1:
    #         ax2.plot(t,np.array(hell_dis_mem[i]),c='red')
    #     else:
    #         ax2.plot(t,np.array(hell_dis_mem[i]),linewidth=2*i,c='blue')

    ax.set_xlabel("Time",fontsize = 18)
    ax.set_ylabel("Hellinger distance",fontsize = 18)
    # ax1.set_xlabel("Time",fontsize = 18)
    # ax1.set_ylabel("Hellinger distance",fontsize = 18)
    # ax1.set_title("For varying thresholds step (0.01,0.26,0.31,1.0), memory size = 2")
    # ax2.set_xlabel("Time",fontsize = 18)
    # ax2.set_ylabel("Hellinger distance",fontsize = 18)
    # ax2.set_title("For varying memory size (2,18,30,38), threshold step = 0.26")
    plt.show()

plt.ion()
speed_accuracy_tradeoff_some_averaged = 1
if speed_accuracy_tradeoff_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()
    run_no = 23
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

# # plt.ion()
# hell_vs_mem_thr = 1
# if hell_vs_mem_thr:
#     path = os.getcwd() + "/results/"
#     prd = Prediction()
#     run_no = 2
#     params = pd.read_csv(path+str(run_no)+'.csv')
#     data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
#     print(data_files)
#     start = 0.0
#     end = 30.0
#     x = np.arange(start,end,0.0001)
#     pdf_x,area = PDF(prd.gaussian,[10,10],[1,1],x)
#     bins = np.linspace(start,end,30)
#     icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
#     sig_h = (0.07*np.log10(1)+0.57)*1
#     pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
#     hellinger_dis = np.zeros((10,18))
#     memory_size = []
#     thresholds_step = []
#     for j in range(10):
#         for d in range(j*18,(j+1)*18):
#             param_columns = list(params.columns)

#             data = pd.read_csv(path+data_files[d])
#             data_columns = list(data.columns)
#             thresholds = [data.iloc[-1][str(i)] for i in range(250)]
#             distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
#             if len(distribution)>len(bins):
#                 distribution = distribution[1:]
#             mean = np.sum(thresholds)/len(thresholds)

#             distribution_area = 0
#             for i in range(len(bins)-1):
#                 distribution_area += (bins[i+1]-bins[i])*distribution[i]
#             if distribution_area!=0:
#                 distribution = distribution/distribution_area

#             hellinger_dis[j,int(d%18)] = Hellinger_distance(pdf_h,distribution)
#             memory_size.append(data.iloc[-1][-2])
#             thresholds_step.append(data.iloc[-1][-1])
#     #         plt.plot([mean,mean],[0,max(distribution)],c='red')
#     #         plt.plot([icpdf,icpdf],[0,max(distribution)],c='black')
#     #         plt.plot(bins,pdf_h,c='black')
#     #         # plt.plot(x,pdf_h1,c='brown')
#     #         plt.plot(bins,distribution,c='blue')
#     #         plt.pause(0.001)
#     #         plt.clf()
#     # plt.ioff()
#     plt.plot(np.unique(memory_size),np.sum(hellinger_dis,axis=0)/10)
#     plt.xlabel("Memory size",fontsize = 18)
#     plt.ylabel("Average Hellinger distance",fontsize = 18)
#     plt.show()
#     plt.plot(np.unique(thresholds_step),np.sum(hellinger_dis,axis=1)/len(range(2,20)))
#     plt.xlabel("Change in threshold step",fontsize = 18)
#     plt.ylabel("Average Hellinger distance",fontsize = 18)
#     plt.show()

# # plt.ion()
# speed_accuracy_tradeoff = 1
# if speed_accuracy_tradeoff:
#     path = os.getcwd() + "/results/"
#     prd = Prediction()
#     run_no = 2
#     params = pd.read_csv(path+str(run_no)+'.csv')
#     data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
#     print(data_files)
#     shakiness_speed_accuracy = []
#     avg_hel_dis = np.zeros(len(range(2,40,2)))

#     start = 0.0
#     end = 30.0
#     x = np.arange(start,end,0.0001)
#     pdf_x,area = PDF(prd.gaussian,[10,10],[1,1],x)
#     bins = np.linspace(start,end,30)
#     icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
#     sig_h = (0.07*np.log10(1)+0.57)*1
#     pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
#     pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
#     for j in range(10):
#         for d in data_files[j*len(range(2,20)):(j+1)*len(range(2,20))]:
#             param_columns = list(params.columns)

#             data = pd.read_csv(path+d)
#             data_columns = list(data.columns)
#             if j==0 and d == data_files[(j)*len(range(2,20))]:
#                 hell_dis_accurate = []
#                 for t in range(200):
#                     thresholds = [data.iloc[t][str(i)] for i in range(250)]

#                     distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
#                     mean = np.sum(thresholds)/len(thresholds)
#                     if len(distribution)>len(bins):
#                         distribution = distribution[1:]
#                     distribution_area = 0
#                     for i in range(len(bins)-1):
#                         distribution_area += (bins[i+1]-bins[i])*distribution[i]
#                     if distribution_area!=0:
#                         distribution = distribution/distribution_area
                    
#                     hell_dis_accurate.append(Hellinger_distance(pdf_h,distribution))
#                 shakiness_speed_accuracy.append(hell_dis_accurate)
            
#             elif j==1 and d == data_files[(j)*len(range(2,20))]:
#                 hell_dis_fast = []
#                 for t in range(200):
#                     thresholds = [data.iloc[t][str(i)] for i in range(250)]

#                     distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
#                     mean = np.sum(thresholds)/len(thresholds)
#                     if len(distribution)>len(bins):
#                         distribution = distribution[1:]
#                     distribution_area = 0
#                     for i in range(len(bins)-1):
#                         distribution_area += (bins[i+1]-bins[i])*distribution[i]
#                     if distribution_area!=0:
#                         distribution = distribution/distribution_area
                    
#                     hell_dis_fast.append(Hellinger_distance(pdf_h,distribution))
#                 shakiness_speed_accuracy.append(hell_dis_fast)
#             else:
#                 hell_dis = []
#                 for t in range(200):
#                     thresholds = [data.iloc[t][str(i)] for i in range(250)]

#                     distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
#                     mean = np.sum(thresholds)/len(thresholds)
#                     if len(distribution)>len(bins):
#                         distribution = distribution[1:]

#                     distribution_area = 0
#                     for i in range(len(bins)-1):
#                         distribution_area += (bins[i+1]-bins[i])*distribution[i]
#                     if distribution_area!=0:
#                         distribution = distribution/distribution_area
                    
#                     hell_dis.append(Hellinger_distance(pdf_h,distribution))
#                 shakiness_speed_accuracy.append(hell_dis)
#     #         plt.plot([mean,mean],[0,max(distribution)],c='red')
#     #         plt.plot([icpdf,icpdf],[0,max(distribution)],c='black')
#     #         plt.plot(bins,pdf_h,c='black')
#     #         # plt.plot(x,pdf_h1,c='brown')
#     #         plt.plot(bins,distribution,c='blue')
#     #         plt.pause(0.0001)
#     #         plt.clf()
#     # plt.ioff()
#     avg_speed_accuracy = np.zeros_like(shakiness_speed_accuracy[0])
    
#     for i in shakiness_speed_accuracy:
#         plt.plot(range(200),np.array(i),alpha= 0.1,c='blue')
#         avg_speed_accuracy += np.array(i)
#     avg_speed_accuracy = avg_speed_accuracy/len(shakiness_speed_accuracy)
#     plt.plot(range(200),hell_dis_accurate,c='green')
#     plt.plot(range(200),hell_dis_fast,c='red')
#     plt.plot(range(200),avg_speed_accuracy,c = 'orange')
#     plt.xlabel("Time",fontsize = 18)
#     plt.ylabel("Hellinger distance",fontsize = 18)
#     plt.show()


# if __name__=="__main__":
#     path = os.getcwd() + "/results/"
#     prd = Prediction()
#     run_no = 2
# 	# file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
#     params = pd.read_csv(path+str(run_no)+'.csv')
#     data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
    
#     for j in range(len(np.arange(0.1,1.1,0.1))):
#         hellinger_dis = []
#         memory_length = []
#         for d in data_files[j*len(range(2,20)):(j+1)*len(range(2,20))]:
#             param_columns = list(params.columns)

#             data = pd.read_csv(path+d)
#             data_columns = list(data.columns)
            
            
# #            plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\mu_{h_1}$'])
# #            plt.xlabel("Time",fontsize = 18)
# #            plt.ylabel("$\mu_{h_1}$",fontsize = 18)
# #            plt.show()

# #            plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\sigma_{h_1}$'])
# #            plt.xlabel("Time",fontsize = 18)
# #            plt.ylabel("$\sigma_{h_1}$",fontsize = 18)
# #            plt.show()

#             thresholds = [i for i in data.iloc[-1][6:]]
#             bins = np.linspace(min(thresholds),max(thresholds),50)
#             distribution = np.bincount(np.digitize(thresholds, bins))
#             mean = np.sum(thresholds)/len(thresholds)

#             pdf_h,area = PDF(prd.gaussian,[data.iloc[-1][0],data.iloc[-1][0]],[data.iloc[-1][1],data.iloc[-1][1]],bins)

# #            plt.plot(bins,distribution[1:])
# #            plt.plot([mean,mean],[0,max(distribution)],c='red')
#             distribution_area = 0
#             for i in range(len(bins)):
#                 distribution_area += bins[i]*distribution[i]
#             distribution = distribution/distribution_area
#             hellinger_dis.append(Hellinger_distance(pdf_h,distribution[1:]))
#             memory_length.append(data.iloc[-1][-2])

# #            plt.xlabel("$h_{i}$",fontsize = 18)
# #            plt.ylabel("Frequency",fontsize = 18)
# #            plt.show()

#         plt.plot(memory_length,hellinger_dis)
#         plt.xlabel("Memory size",fontsize = 18)
#         plt.ylabel("Hellinger distance",fontsize = 18)
#         plt.show()
        
    ##########################################################################
#    path = os.getcwd() + "/results/"
#    prd = Prediction()
#    run_no = 2
#	# file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
#    params = pd.read_csv(path+str(run_no)+'.csv')
#    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))
#    
##    for j in range(len(np.arange(0.1,1.1,0.1))):
##        hellinger_dis = []
##        memory_length = []
##        for d in data_files[j*len(range(2,20)):(j+1)*len(range(2,20))]:
#    for k in range(0,18):
#        hellinger_dis = []
#        memory_length = []
#        for j in range(10):
#            
#            d = data_files[k + j*18]
#            
#            param_columns = list(params.columns)

#            data = pd.read_csv(path+d)
#            data_columns = list(data.columns)
#            
#            
##            plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\mu_{h_1}$'])
##            plt.xlabel("Time",fontsize = 18)
##            plt.ylabel("$\mu_{h_1}$",fontsize = 18)
##            plt.show()

##            plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\sigma_{h_1}$'])
##            plt.xlabel("Time",fontsize = 18)
##            plt.ylabel("$\sigma_{h_1}$",fontsize = 18)
##            plt.show()

#            thresholds = [i for i in data.iloc[-1][6:]]
#            bins = np.linspace(min(thresholds),max(thresholds),50)
#            distribution = np.bincount(np.digitize(thresholds, bins))
#            mean = np.sum(thresholds)/len(thresholds)

#            pdf_h,area = PDF(prd.gaussian,[data.iloc[-1][0],data.iloc[-1][0]],[data.iloc[-1][1],data.iloc[-1][1]],bins)

##            plt.plot(bins,distribution[1:])
##            plt.plot([mean,mean],[0,max(distribution)],c='red')
#            distribution_area = 0
#            for i in range(len(bins)):
#                distribution_area += bins[i]*distribution[i]
#            distribution = distribution/distribution_area
#            hellinger_dis.append(Hellinger_distance(pdf_h,distribution[1:]))
#  #          memory_length.append(data.iloc[-1][-2])
#            memory_length.append(data.iloc[-1][-1])
##            plt.xlabel("$h_{i}$",fontsize = 18)
##            plt.ylabel("Frequency",fontsize = 18)
##            plt.show()
#        print(hellinger_dis,memory_length)
#        plt.plot(memory_length,hellinger_dis)
#        plt.xlabel("Change in threshold step",fontsize = 18)
#        plt.ylabel("Hellinger distance",fontsize = 18)
#        plt.show()
    #############################################################################
#    path = os.getcwd() + "/results/"
#    prd = Prediction()
#    n =  [2,3,5,8,10,15,20,30,40,80,100]
#    acceptance_rates = []
#    data_files = []
#    for run in range(4,15):
#        run_no = run
#        # file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
#        params = pd.read_csv(path+str(run_no)+'.csv')
#        data_files += list(np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10])))
#    for j in range(len(n)):
#        d = data_files[j*18*10+1]
#        data = pd.read_csv(path+d)
#        data_columns = list(data.columns)
#        acceptance_rates.append(np.sum(data['Response_1'][-int(data.iloc[-1][-2]):])/data.iloc[-1][-2])
#    plt.plot(n,acceptance_rates)
#    plt.xlabel("Number of options",fontsize = 18)
#    plt.ylabel("acceptance rate",fontsize = 18)
#    plt.show()
