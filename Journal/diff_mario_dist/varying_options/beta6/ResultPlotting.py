# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


from cmath import asin
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

# plt.ion()
thr_mem_anim_some_averaged = 1
if thr_mem_anim_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    
    data_files = np.array([])
    for i in range(4):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(i))] == str(i) and len(f)>10]))),axis=0)
    data_files = data_files.reshape((4,20,50))

    start = 0.0
    end = 25.0
    num_opts = [2,5,8,10]
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
    hell_dis_opts = []
    for short_var in range(data_files.shape[0]):
        for long_var in range(data_files.shape[1]):
            if long_var == 4:
                hell_dis_runs = []
                for runs in range(data_files.shape[2]):
                    hell_dis = []
                    data = pd.read_csv(path+data_files[short_var,long_var,runs])
                    fig, ax = plt.subplots()
                    camera = Camera(fig)

                    if short_var < 4:
                        for t in range(-1,1000,50):
                            if t<0:
                                t = 0
                            thresholds = [data.iloc[t][str(i)] for i in range(int(50*num_opts[short_var]))]
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
                        for t in range(-1,1000,50):
                            if t<0:
                                t = 0
                            thresholds = [data.iloc[t][str(i)] for i in range(int(50*num_opts[short_var]))]
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
                    
                    plt.close()
                    hell_dis_runs.append(hell_dis)
        hell_dis_opts.append(hell_dis_runs)
                    
    plt.ioff()
    pkl.dump(hell_dis_opts, open('hell_dis_opts.pickle', 'wb'))