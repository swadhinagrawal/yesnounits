# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import guvectorize, float64
from numba.typed import List
from math import sqrt,exp
import pickle as pkl
from matplotlib import rc,rcParams
import random

def parallel(func,inputs):
	batch_size = len(inputs)
	inps = [(i,) for i in inputs]
	output = []
	for i in range(0,len(inputs),batch_size):
		opt_var = []
		with Pool(20) as processor:#,ray_address="auto") as p:
			opt_var = processor.starmap(func,inps[i:i+batch_size])

		output += list(opt_var)
	return output

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
    normalized_pdf = pdf/(np.sum(pdf)*(bins[1]-bins[0]))
    ar = np.sum(pdf)*(bins[1]-bins[0])                #   Normalized PDF
    return normalized_pdf,ar

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

# Fig responseTevolution
animations = 0
if animations:
    rc('font', weight='bold',size=18)
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    path = os.getcwd() + "/animation_results/"
    
    prd = Prediction()

    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f]))

    start = 0.0
    end = 25.0
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    
    hell_dis_thr = []
    hell_dis_mem = []
    def dis(distribution):
        if len(distribution)>len(bins):
            distribution = distribution[1:]
        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
        distribution = distribution/distribution_area
        return distribution
    print(data_files)
    for d in data_files:
        
        data = pd.read_csv(path+d)
        fig, ax = plt.subplots()
        ax.axes.get_yaxis().set_visible(False)

        thresholds_s = [data.iloc[0][str(i)] for i in range(250)]
        thresholds_e = [data.iloc[-1][str(i)] for i in range(250)]
        distribution_s = np.bincount(np.digitize(thresholds_s, bins),minlength=len(bins))
        distribution_e = np.bincount(np.digitize(thresholds_e, bins),minlength=len(bins))
        mean_s = np.sum(thresholds_s)/len(thresholds_s)
        mean_e = np.sum(thresholds_e)/len(thresholds_e)
        distribution_s = dis(distribution_s)
        distribution_e = dis(distribution_e)

        if 'G' in d:
            pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
            pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
            pdf_h = PDF(prd.gaussian,[10,10],[1,1],x)
            ax.plot([10,10],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')
        elif 'K' in d:
            pdf_x = PDF(prd.gaussian,[10,15],[1,1],x)
            pdf_x1 = PDF(prd.gaussian,[10,15],[1,1],bins)
            pdf_h = PDF(prd.gaussian,[12.5,12.5],[1,1],x)
            ax.plot([12.5,12.5],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')
        elif 'U' in d:
            pdf_x = PDF(prd.uniform,[10,10],[1,1],x)
            pdf_x1 = PDF(prd.uniform,[10,10],[1,1],bins)
            pdf_h = PDF(prd.gaussian,[10,10],[1,1],x)
            ax.plot([10,10],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

        

        hell_dis_s = Hellinger_distance(pdf_x1,distribution_s,bins)
        hell_dis_e = Hellinger_distance(pdf_x1,distribution_e,bins)
        ax.plot(x,-pdf_x,color='orange',linewidth = 2,label=r'$D_q$')
        
        for i in range(len(bins)):
            if i == 0:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=2,label='initial '+r'$D_h$')
            else:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=3)
        ax.plot([mean_e,mean_e],[0,max(distribution_e)],c='red',linewidth=3,label=r'$\mu_h$')
        ax.plot(x,pdf_h,c='green',linewidth=3,label=r'$D_h$',linestyle='--')
        
        ax.set_xticks(np.arange(0, 25, 5))
        ax.legend(prop=dict(weight='bold',size=14),frameon=False)
        plt.tight_layout()
        # plt.savefig(path+d[0]+'_s.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        plt.show()

# Fig responseTevolution
animations_evol = 0
if animations_evol:
    rc('font', weight='bold',size=18)
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    path = os.getcwd() + "/animation_results/"
    
    prd = Prediction()

    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and len(f)>5]))

    start = 13.0
    end = 21.0
    x = np.arange(start,end,0.0001)
    pdf_x,area = PDF(prd.gaussian,[17,17],[1,1],x)
    bins = np.linspace(start,end,100)
    
    
    
    hell_dis_thr = []
    hell_dis_mem = []
    def dis(distribution):
        if len(distribution)>len(bins):
            distribution = distribution[1:]
        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
        distribution = distribution/distribution_area
        return distribution
    print(data_files)
    for d in data_files:
        
        data = pd.read_csv(path+d)
        fig, ax = plt.subplots()
        ax.axes.get_yaxis().set_visible(False)

        
        if 'G' in d:
            pdf_x,area = PDF(prd.gaussian,[17,17],[1,1],x)
            pdf_x1,_ = PDF(prd.gaussian,[17,17],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17,17],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

            slices = []
            mid_slices=[]
            mean_x = List([17,17])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            for i in range(1,5,1):
                ESM = prd.ICPDF(float(i)/5,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))
            sig_h = (0.07*np.log10(5)+0.57)*1
            mu_h = prd.ICPDF(1-(1/5),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            color = ['#5a1911','#2b8a92','#cf4964','#e7e482','#c8ffd3']
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])

        elif 'K' in d:
            start = 10.0
            end = 25.0
            x = np.arange(start,end,0.0001)
            bins = np.linspace(start,end,100)
            pdf_x,area = PDF(prd.gaussian,[14.5,19.5],[1,1],x)
            pdf_x1,_ = PDF(prd.gaussian,[14.5,19.5],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17.5,17.5],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

            slices = []
            mid_slices=[]
            mean_x = List([14.5,19.5])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            sig_h = (0.05*np.log10(5)+0.58)*1
            mu_h = prd.ICPDF(1-(1/5),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            for i in range(1,5,1):
                ESM = prd.ICPDF(float(i)/5,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))

            color = ['#5a1911','#2b8a92','#cf4964','#e7e482','#c8ffd3']
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
        elif 'U' in d:
            pdf_x,area = PDF(prd.uniform,[17,17],[1,1],x)
            pdf_x1,_ = PDF(prd.uniform,[17,17],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17,17],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

        
            slices = []
            mid_slices=[]
            mean_x = List([17,17])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            sig_h = (-0.07*np.log10(5)+0.67)*1
            mu_h = prd.ICPDF(1-(1/5),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            for i in range(1,5,1):
                ESM = prd.ICPDF(float(i)/5,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))

            color = ['#5a1911','#2b8a92','#cf4964','#e7e482','#c8ffd3']
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])

        thresholds_s = [data.iloc[0][str(i)] for i in range(250)]
        thresholds_e = [data.iloc[-1][str(i)] for i in range(250)]
        distribution_s = np.bincount(np.digitize(thresholds_s, bins),minlength=len(bins))
        distribution_e = np.bincount(np.digitize(thresholds_e, bins),minlength=len(bins))
        mean_s = np.sum(thresholds_s)/len(thresholds_s)
        mean_e = np.sum(thresholds_e)/len(thresholds_e)
        distribution_s = dis(distribution_s)
        distribution_e = dis(distribution_e)
        distribution_e = distribution_e/max(distribution_e)


        hell_dis_s = Hellinger_distance(pdf_x1,distribution_s,bins)
        hell_dis_e = Hellinger_distance(pdf_x1,distribution_e,bins)
        ax.plot(x,-pdf_x,color='orange',linewidth = 2,label=r'$D_q$')
        
        for i in range(len(bins)):
            if i == 0:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=2,label='initial '+r'$D_h$')
            else:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=3)
        ax.plot([mean_e,mean_e],[0,max(distribution_e)],c='red',linewidth=3,label=r'$\mu_h$')
        ax.plot(x,pdf_h,c='green',linewidth=3,label=r'$D_h$',linestyle='--')
        
        ax.set_xticks(np.arange(start, end, 2))
        ax.legend(prop=dict(weight='bold',size=14),frameon=False)
        plt.tight_layout()
        plt.savefig(path+d[:2]+'_e.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        plt.show()


# Fig responseTevolution
animations_evol_100 = 0
if animations_evol_100:
    rc('font', weight='bold',size=18)
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    path = os.getcwd() + "/animation_results/"
    
    prd = Prediction()

    data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and len(f)==6]))

    start = 13.0
    end = 25.0
    x = np.arange(start,end,0.0001)
    pdf_x,area = PDF(prd.gaussian,[17,17],[1,1],x)
    bins = np.linspace(start,end,100)
    
    
    
    hell_dis_thr = []
    hell_dis_mem = []
    def dis(distribution):
        if len(distribution)>len(bins):
            distribution = distribution[1:]
        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
        distribution = distribution/distribution_area
        return distribution
    print(data_files)
    for d in data_files:
        
        data = pd.read_csv(path+d)
        fig, ax = plt.subplots()
        ax.axes.get_yaxis().set_visible(False)

        
        if 'G' in d:
            pdf_x,area = PDF(prd.gaussian,[17,17],[1,1],x)
            pdf_x1,_ = PDF(prd.gaussian,[17,17],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17,17],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

            slices = []
            mid_slices=[]
            mean_x = List([17,17])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            for i in range(1,100,1):
                ESM = prd.ICPDF(float(i)/100,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))
            sig_h = (0.07*np.log10(100)+0.57)*1
            mu_h = prd.ICPDF(1-(1/100),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			for i in range(100)]
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])

        elif 'K' in d:
            start = 10.0
            end = 25.0
            x = np.arange(start,end,0.0001)
            bins = np.linspace(start,end,100)
            pdf_x,area = PDF(prd.gaussian,[14.5,19.5],[1,1],x)
            pdf_x1,_ = PDF(prd.gaussian,[14.5,19.5],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17.5,17.5],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

            slices = []
            mid_slices=[]
            mean_x = List([14.5,19.5])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            sig_h = (0.05*np.log10(100)+0.58)*1
            mu_h = prd.ICPDF(1-(1/100),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            for i in range(1,100,1):
                ESM = prd.ICPDF(float(i)/100,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))

            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			for i in range(100)]
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.gaussian(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
        elif 'U' in d:
            pdf_x,area = PDF(prd.uniform,[17,17],[1,1],x)
            pdf_x1,_ = PDF(prd.uniform,[17,17],[1,1],bins)
            # pdf_h,_ = PDF(prd.gaussian,[17,17],[1,1],x)
            ax.plot([17,17],[0,-max(pdf_x)],color='orange',linewidth = 2,linestyle='--',label=r'$\mu_q$')

        
            slices = []
            mid_slices=[]
            mean_x = List([17,17])
            sigma_x = List([1,1])
            start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
            stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
            step = 0.0001
            # x = np.arange(start,stop,step)
            sig_h = (-0.07*np.log10(100)+0.67)*1
            mu_h = prd.ICPDF(1-(1/100),mean_x,step,x,pdf_x)
            pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
            for i in range(1,100,1):
                ESM = prd.ICPDF(float(i)/100,mean_x,step,x,pdf_x)
                slices.append(np.round(ESM,decimals=3))

            # color = ['#5a1911','#2b8a92','#cf4964','#e7e482','#c8ffd3']
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			for i in range(100)]
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  prd.uniform(x1,mean_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    ax.fill_between(x1,0,-pdf1,facecolor=color[i])

        thresholds_s = [data.iloc[0][str(i)] for i in range(5000)]
        thresholds_e = [data.iloc[-1][str(i)] for i in range(5000)]
        distribution_s = np.bincount(np.digitize(thresholds_s, bins),minlength=len(bins))
        distribution_e = np.bincount(np.digitize(thresholds_e, bins),minlength=len(bins))
        mean_s = np.sum(thresholds_s)/len(thresholds_s)
        mean_e = np.sum(thresholds_e)/len(thresholds_e)
        distribution_s = dis(distribution_s)
        distribution_e = dis(distribution_e)
        distribution_e = distribution_e/max(distribution_e)


        hell_dis_s = Hellinger_distance(pdf_x1,distribution_s,bins)
        hell_dis_e = Hellinger_distance(pdf_x1,distribution_e,bins)
        ax.plot(x,-pdf_x,color='orange',linewidth = 2,label=r'$D_q$')
        
        for i in range(len(bins)):
            if i == 0:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=2,label='initial '+r'$D_h$')
            else:
                ax.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=3)
        ax.plot([mean_e,mean_e],[0,max(distribution_e)],c='red',linewidth=3,label=r'$\mu_h$')
        ax.plot(x,pdf_h,c='green',linewidth=3,label=r'$D_h$',linestyle='--')
        
        ax.set_xticks(np.arange(start, end, 2))
        ax.legend(prop=dict(weight='bold',size=14),frameon=False)
        plt.tight_layout()
        plt.savefig(path+d[:2]+'_e.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        plt.show()

# Fig RT distribution
RT_distribution = 0
if RT_distribution:
    rc('font', weight='bold',size=18)
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    # data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and len(f)>10]))
    data_files = []
    for i in range(66,77):#(95,190):#(110,205)
        # params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    start = 0.0
    end = 1.0
    x = np.arange(start,end,0.0001)
    bins = np.linspace(start,end,100)
    
    hell_dis_thr = []
    hell_dis_mem = []
    def dis(distribution,bins):
        if len(distribution)>len(bins):
            distribution = distribution[1:]
        distribution_area = np.sum(distribution)*(bins[1]-bins[0])
        distribution = distribution/distribution_area
        return distribution
    print(data_files)
    num_opts = [2,3,5,8,10,15,20,30,40,80,100]#[5,40,100]
    count = 0
    mean_mu_hs = []
    fig1, ax1 = plt.subplots()
    for d in data_files[:]:
        n = num_opts[count%len(num_opts)]
        count += 1
        data = pd.read_csv(path+d)
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax.axes.get_yaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        start = -0.2
        end = 1.0
        x = np.arange(start,end,0.0001)
        bins = np.linspace(start,end,100)
        # rt_s = [data.iloc[0]['tr_'+str(i)] for i in range(50*n)]
        rt_e = [data.iloc[-1]['tr_'+str(i)] for i in range(50*n)]
        # rtdistribution_s = np.bincount(np.digitize(rt_s, bins),minlength=len(bins))
        rtdistribution_e = np.bincount(np.digitize(rt_e, bins),minlength=len(bins))
        # rtmean_s = np.sum(rt_s)/len(rt_s)
        rtmean_e = np.sum(rt_e)/len(rt_e)
        # rtdistribution_s = dis(rtdistribution_s)
        rtdistribution_e = dis(rtdistribution_e,bins)
        rtdistribution_e = rtdistribution_e/max(rtdistribution_e)

        for i in range(len(bins)):
            if i == 0:
                ax.plot([bins[i],bins[i]],[0,rtdistribution_e[i]],c='blue',linewidth=2,label=r'$D_{t_r}$ for '+str(n)+' opts, '+r'$\delta t_r$ '+str(data.iloc[0]['rt_step']))
            else:
                ax.plot([bins[i],bins[i]],[0,rtdistribution_e[i]],c='blue',linewidth=3)
        ax.plot([rtmean_e,rtmean_e],[0,max(rtdistribution_e)],c='red',linewidth=3,label=r'$\mu_{t_r}$')


        start = 0.0
        end = 25.0
        x = np.arange(start,end,0.0001)
        bins = np.linspace(start,end,100)
        # thresholds_s = [data.iloc[0]['h_'+str(i)] for i in range(50*n)]
        thresholds_e = [data.iloc[-1]['h_'+str(i)] for i in range(50*n)]
        # distribution_s = np.bincount(np.digitize(thresholds_s, bins),minlength=len(bins))
        distribution_e = np.bincount(np.digitize(thresholds_e, bins),minlength=len(bins))
        # mean_s = np.sum(thresholds_s)/len(thresholds_s)
        mean_e = np.sum(thresholds_e)/len(thresholds_e)
        # distribution_s = dis(distribution_s)
        distribution_e = dis(distribution_e,bins)
        distribution_e = distribution_e/max(distribution_e)
        
        for i in range(len(bins)):
            if i == 0:
                ax2.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=2,label=r'$D_{h}$ for '+str(n)+' opts, '+r'$t_r$ '+str(data.iloc[0]['rt_step']))
            else:
                ax2.plot([bins[i],bins[i]],[0,distribution_e[i]],c='blue',linewidth=3)
        ax2.plot([mean_e,mean_e],[0,max(distribution_e)],c='red',linewidth=3,label=r'$\mu_{h}$')
        mean_x = List([17,17])
        sigma_x = List([1,1])
        step = 0.0001
        sig_h = (0.07*np.log10(n)+0.57)*1
        pdf_x,_ = PDF(prd.gaussian,mean_x,sigma_x,x)
        mu_h = prd.ICPDF(1-(1/n),mean_x,step,x,pdf_x)
        pdf_h,_ = PDF(prd.gaussian,[mu_h,mu_h],[sig_h,sig_h],x)
        ax2.plot(x,pdf_h,c='green',linewidth=3,label=r'$D_h$',linestyle='--')
        
        ax.legend(prop=dict(weight='bold',size=14),frameon=False)
        plt.tight_layout()
        
        mean_mu_hs.append(np.sum(data.iloc[::100]['$\mu_{h_1}$'])/len(data.iloc[::100]['$\mu_{h_1}$']))
        if count%len(num_opts) == 0:
            print(mean_mu_hs)
            ax1.plot(num_opts,mean_mu_hs,label='tr: '+str(data.iloc[0]['rt_step']))
            ref_data = []
            start = 0.0
            end = 25.0
            x = np.arange(start,end,0.0001)
            pdf_x,_ = PDF(prd.gaussian,[17,17],[1,1],x)
            for n in num_opts:
                icpdf = prd.ICPDF(1-1/n,[17,17],0.0001,x,pdf_x)
                ref_data.append(icpdf)
            ax1.plot(num_opts,ref_data,label='Reference')
            ax1.set_xlabel(r"$n$",fontsize = 18)
            ax1.set_ylabel(r"$\mu_{h}$",fontsize = 18)
            ax1.legend()
            fig1.savefig(path+d[:-4]+'_mun.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
            # plt.show()
            mean_mu_hs = []
            fig1, ax1 = plt.subplots()
        ax2.legend(prop=dict(weight='bold',size=14),frameon=False)
        fig.savefig(path+d[:-4]+'_rt.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        fig2.savefig(path+d[:-4]+'_thr.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    
    # plt.show()

success_plot = 1
if success_plot:
    num_opts = np.array([2,3,5,8,10,15,20,30,40,80,100])
    path = os.getcwd() +'/results/'

    data_files = np.array([])
    for i in range(66,77):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)
    
    mean_mu_hs = []
    success_rates = np.zeros(len(num_opts))
    for row in range(len(data_files)):
        data = pd.read_csv(path+data_files[row])
        count = 0
        for i in range(len(data)):
            if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                count += 1
        success_rates[row] = count/data.shape[0]

    fig1, ax1 = plt.subplots()

    # data_files = np.array([])
    # for i in range(66,77):
    #     params = pd.read_csv(path+str(i)+'.csv')
    #     data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)
    
    # mean_mu_hs = []
    # success_rates_without_update = np.zeros(len(num_opts))
    # for row in range(len(data_files)):
    #     data = pd.read_csv(path+data_files[row])
    #     count = 0
    #     for i in range(len(data)):
    #         if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
    #             count += 1
    #     success_rates_without_update[row] = count/data.shape[0]
    ax1.plot(num_opts,success_rates,label=r'$N_{t_r}$'+': updated')
    # ax1.plot(num_opts,success_rates_without_update,label='Mario: not updated')
    ax1.set_xlabel(r"$n$",fontsize = 18)
    ax1.set_ylabel("Success rate",fontsize = 18)
    ax1.legend()
    # fig1.savefig(path+'n_succ_mario_mux_17.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()