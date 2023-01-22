# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import guvectorize, float64
from math import sqrt,exp
import pickle as pkl

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
    normalized_pdf = pdf/(np.sum(pdf)*(bins[1]-bins[0]))                #   Normalized PDF
    return normalized_pdf

# plt.ion()
thr_mem_anim_some_averaged = 0
if thr_mem_anim_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])

    data_files = np.array([])

    for i in range(88,99):#(110,205)
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    
    data_files = data_files.reshape((betas.shape[0],num_opts.shape[0]))

    start = 0.0
    end = 25.0
    num_opts = [2,3,5,8,10]
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[17,17],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[17,17],[1,1],bins)
    hell_dis = np.zeros((len(betas),len(num_opts),1000))
    success_rates = np.zeros((len(betas),len(num_opts)))
    for row in range(data_files.shape[0]):
        for col in range(data_files.shape[1]):
            data = pd.read_csv(path+data_files[row,col])
            count = 0
            for i in range(len(data)):
                thresholds = [data.iloc[i][str(j)] for j in range(int(50*num_opts[col]))]
                distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                mean = np.sum(thresholds)/len(thresholds)
                if len(distribution)>len(bins):
                    distribution = distribution[1:]
                distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                distribution = distribution/distribution_area

                hell_dis[row,col,i] = Hellinger_distance(pdf_x1,distribution,bins)

                if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                    count += 1
            success_rates[row,col] = count/data.shape[0]

    pkl.dump(hell_dis, open('hell_dis_beta_3.pickle', 'wb'))
    pkl.dump(success_rates, open('success_rates_beta_3.pickle', 'wb'))

# plt.ion()
parallel_thr_mem_anim_some_averaged = 0
if parallel_thr_mem_anim_some_averaged:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])

    data_files = np.array([])

    for i in range(88,99):#(110,205)
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    
    data_files = data_files.reshape((betas.shape[0],num_opts.shape[0]))

    start = 0.0
    end = 25.0
    num_opts = [2,3,5,8,10]
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[17,17],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[17,17],[1,1],bins)
    hell_dis = np.zeros((len(betas),len(num_opts),1000))
    success_rates = np.zeros((len(betas),len(num_opts)))
    for row in range(data_files.shape[0]):
        for col in range(data_files.shape[1]):
            data = pd.read_csv(path+data_files[row,col])
            count = 0
            for i in range(len(data)):
                thresholds = [data.iloc[i][str(j)] for j in range(int(50*num_opts[col]))]
                distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
                mean = np.sum(thresholds)/len(thresholds)
                if len(distribution)>len(bins):
                    distribution = distribution[1:]
                distribution_area = np.sum(distribution)*(bins[1]-bins[0])
                distribution = distribution/distribution_area

                hell_dis[row,col,i] = Hellinger_distance(pdf_x1,distribution,bins)

                if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                    count += 1
            success_rates[row,col] = count/data.shape[0]

    pkl.dump(hell_dis, open('hell_dis_beta_3.pickle', 'wb'))
    pkl.dump(success_rates, open('success_rates_beta_3.pickle', 'wb'))



thr_mem_anim_some_averaged_gumbel = 0
if thr_mem_anim_some_averaged_gumbel:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])

    data_files = np.array([])

    for i in range(105,110):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    start = 0.0
    end = 25.0
    num_opts = [2,3,5,8,10]
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
    hell_dis = np.zeros((len(num_opts),1000))
    success_rates = np.zeros(len(num_opts))

    for col in range(data_files.shape[0]):
        data = pd.read_csv(path+data_files[col])
        count = 0
        for i in range(len(data)):
            thresholds = [data.iloc[i][str(j)] for j in range(int(50*num_opts[col]))]
            distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
            mean = np.sum(thresholds)/len(thresholds)
            if len(distribution)>len(bins):
                distribution = distribution[1:]
            distribution_area = np.sum(distribution)*(bins[1]-bins[0])
            distribution = distribution/distribution_area

            hell_dis[col,i] = Hellinger_distance(pdf_x1,distribution,bins)

            if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                count += 1
        success_rates[col] = count/data.shape[0]

    pkl.dump(hell_dis, open('hell_dis_gumbel.pickle', 'wb'))
    pkl.dump(success_rates, open('success_rates_gumbel.pickle', 'wb'))


thr_mem_anim_some_averaged_u = 0
if thr_mem_anim_some_averaged_u:
    path = os.getcwd() + "/results/"
    
    prd = Prediction()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])

    data_files = np.array([])

    for i in range(100,105):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    start = 0.0
    end = 25.0
    num_opts = [2,3,5,8,10]
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.gaussian,[10,10],[1,1],x)
    bins = np.linspace(start,end,100)
    # icpdf = prd.ICPDF(1-1/5.0,[10,10],0.0001,x,pdf_x)
    # sig_h = (0.07*np.log10(1)+0.57)*1
    # pdf_h,area = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],bins)
    # pdf_h1,_ = PDF(prd.gaussian,[icpdf,icpdf],[sig_h,sig_h],x)
    pdf_x1 = PDF(prd.gaussian,[10,10],[1,1],bins)
    hell_dis = np.zeros((len(num_opts),1000))
    success_rates = np.zeros(len(num_opts))

    for col in range(data_files.shape[0]):
        data = pd.read_csv(path+data_files[col])
        count = 0
        for i in range(len(data)):
            thresholds = [data.iloc[i][str(j)] for j in range(int(50*num_opts[col]))]
            distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
            mean = np.sum(thresholds)/len(thresholds)
            if len(distribution)>len(bins):
                distribution = distribution[1:]
            distribution_area = np.sum(distribution)*(bins[1]-bins[0])
            distribution = distribution/distribution_area

            hell_dis[col,i] = Hellinger_distance(pdf_x1,distribution,bins)

            if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                count += 1
        success_rates[col] = count/data.shape[0]

    pkl.dump(hell_dis, open('hell_dis_u.pickle', 'wb'))
    pkl.dump(success_rates, open('success_rates_u.pickle', 'wb'))

plot = 0
if plot:
    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])
    # fig1, ax1 = plt.subplots()

    # hellinger_dist = np.zeros((len(betas),len(num_opts)))
    # std_dev = np.zeros((len(betas),len(num_opts)))
    # for b in range(len(betas)):
    #     path = os.getcwd() +'/'
    #     hell_dis = np.array(pkl.load(open('hell_dis_beta_2.pickle', 'rb')))

    #     for n in range(len(num_opts)):
    #         hell_dis_avg = np.sum(hell_dis[b,n,:])/len(hell_dis[b,n,:])
    #         std_dev_ = 0
    #         for elem in hell_dis[b,n,:]:
    #             std_dev_ = std_dev_ + ((elem - hell_dis_avg)**2)/len(hell_dis[b,n,:])
    #         std_dev[b,n] = np.sqrt(std_dev_)
    #         hellinger_dist[b,n] = hell_dis_avg


    # colors = ['mediumslateblue','darkolivegreen','teal','goldenrod','tomato']
    # for n in range(len(num_opts)):
    #     ax1.fill_between(betas,np.array(hellinger_dist[:,n])-np.array(std_dev[:,n]),np.array(hellinger_dist[:,n])+np.array(std_dev[:,n]),color=colors[n],alpha=0.2)
    #     ax1.plot(betas,np.array(hellinger_dist[:,n]),c=colors[n],label='Mario: Beta; n: '+ str(num_opts[n]))

    # path = os.getcwd() +'/'
    # hell_dis = np.array(pkl.load(open('hell_dis_gumbel.pickle', 'rb')))
    # hellinger_dist = np.zeros(len(num_opts))
    # std_dev = np.zeros(len(num_opts))
    # for n in range(len(num_opts)):
    #     hell_dis_avg = np.sum(hell_dis[n,:])/len(hell_dis[n,:])
    #     std_dev_ = 0
    #     for elem in hell_dis[n,:]:
    #         std_dev_ = std_dev_ + ((elem - hell_dis_avg)**2)/len(hell_dis[n,:])
    #     std_dev[n] = np.sqrt(std_dev_)
    #     hellinger_dist[n] = hell_dis_avg


    # colors = ['mediumslateblue','darkolivegreen','teal','goldenrod','tomato']
    # for n in range(len(num_opts)):
    #     ax1.fill_between(betas,np.array([hellinger_dist[n] for i in range(len(betas))])-np.array([std_dev[n] for i in range(len(betas))]),np.array([hellinger_dist[n] for i in range(len(betas))])+np.array([std_dev[n] for i in range(len(betas))]),color=colors[n],alpha=0.2)
    #     ax1.plot(betas,np.array([hellinger_dist[n] for i in range(len(betas))]),c=colors[n],linestyle='-.',label='Mario: Gumbel; n: '+ str(num_opts[n]))


    # path = os.getcwd() +'/'
    # hell_dis = np.array(pkl.load(open('hell_dis_u.pickle', 'rb')))
    # hellinger_dist = np.zeros(len(num_opts))
    # std_dev = np.zeros(len(num_opts))
    # for n in range(len(num_opts)):
    #     hell_dis_avg = np.sum(hell_dis[n,:])/len(hell_dis[n,:])
    #     std_dev_ = 0
    #     for elem in hell_dis[n,:]:
    #         std_dev_ = std_dev_ + ((elem - hell_dis_avg)**2)/len(hell_dis[n,:])
    #     std_dev[n] = np.sqrt(std_dev_)
    #     hellinger_dist[n] = hell_dis_avg

    colors = ['mediumslateblue','darkolivegreen','teal','goldenrod','tomato']
    # for n in range(len(num_opts)):
    #     ax1.fill_between(betas,np.array([hellinger_dist[n] for i in range(len(betas))])-np.array([std_dev[n] for i in range(len(betas))]),np.array([hellinger_dist[n] for i in range(len(betas))])+np.array([std_dev[n] for i in range(len(betas))]),color=colors[n],alpha=0.2)
    #     ax1.plot(betas,np.array([hellinger_dist[n] for i in range(len(betas))]),c=colors[n],linestyle='--',label='Mario: Uniform; n: '+ str(num_opts[n]))


    # ax1.set_xlabel(r"$\beta$",fontsize = 18)
    # ax1.set_ylabel("Hellinger distance",fontsize = 18)
    # ax1.legend()
    # fig1.savefig('accuracy_beta_1.png')
    # plt.show()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])
    fig1, ax1 = plt.subplots()

    path = os.getcwd() +'/'
    success_rates = np.array(pkl.load(open('success_rates_beta_3.pickle', 'rb')))

    for n in range(len(num_opts)):
        ax1.plot(betas,np.array(success_rates[:,n]),c=colors[n],label='Mario: Beta; n: '+ str(num_opts[n]))

    # path = os.getcwd() +'/'
    # success_rates = np.array(pkl.load(open('success_rates_gumbel.pickle', 'rb')))

    # for n in range(len(num_opts)):
    #     ax1.plot(betas,np.array([success_rates[n] for i in range(len(betas))]),c=colors[n],linestyle='-.',label='Mario: Gumbel; n: '+ str(num_opts[n]))

    # path = os.getcwd() +'/'
    # success_rates = np.array(pkl.load(open('success_rates_u.pickle', 'rb')))

    # for n in range(len(num_opts)):
    #     ax1.plot(betas,np.array([success_rates[n] for i in range(len(betas))]),c=colors[n],linestyle='--',label='Mario: Uniform; n: '+ str(num_opts[n]))

    ax1.set_xlabel(r"$\beta$",fontsize = 18)
    ax1.set_ylabel("Success rate",fontsize = 18)
    ax1.legend()
    fig1.savefig('success_rate_beta_3.png')
    plt.show()

    betas = np.arange(1,20)
    num_opts = np.array([2,3,5,8,10])
    fig1, ax1 = plt.subplots()

    path = os.getcwd() +'/'
    success_rates = np.array(pkl.load(open('success_rates_beta_3.pickle', 'rb')))

    best_betas = []
    for n in range(len(num_opts)):
        where = np.where(success_rates[:,n]==max(success_rates[:,n]))
        best_betas.append(betas[where])

    ax1.plot(num_opts,np.array(best_betas),c='black',label='Mario: Beta')

    ax1.set_xlabel(r"$n$",fontsize = 18)
    ax1.set_ylabel("Best Beta",fontsize = 18)
    ax1.legend()
    fig1.savefig('n_beta_3.png')
    plt.show()


beta_muh = 1
if beta_muh:
    path = os.getcwd() + "/results/"#"/results_before_ANTS/"
    
    prd = Prediction()

    # betas = np.arange(1,20)
    num_opts = [2,3,5,8,10,15,20,30,40,80,100]
    data_files = np.array([])

    for i in range(121,132):#(95,190):#(110,205)
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)

    
    # data_files = data_files.reshape(num_opts.shape[0]))
    mean_mu_hs = []
    
    for row in range(len(data_files)):
        data = pd.read_csv(path+data_files[row])
        # count = 0
        # for i in range(len(data)):
        #     thresholds = [data.iloc[i][str(j)] for j in range(int(50*num_opts[col]))]
        #     distribution = np.bincount(np.digitize(thresholds, bins),minlength=len(bins))
        #     mean = np.sum(thresholds)/len(thresholds)
        #     if len(distribution)>len(bins):
        #         distribution = distribution[1:]
        #     distribution_area = np.sum(distribution)*(bins[1]-bins[0])
        #     distribution = distribution/distribution_area

        #     hell_dis[row,col,i] = Hellinger_distance(pdf_x1,distribution,bins)

        #     if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
        #         count += 1
        # success_rates[row,col] = count/data.shape[0]
        mean_mu_hs.append(np.sum(data.iloc[::100]['$\mu_{h_1}$'])/len(data.iloc[::100]['$\mu_{h_1}$']))
    print(mean_mu_hs)
    # mu_means = np.sum(mean_mu_hs,axis=1)
    # xlog_data = np.log(betas)
    # curve = np.polyfit(xlog_data, mean_mu_hs, 2) 
    fig1, ax1 = plt.subplots()
    # ax1.plot(betas,mean_mu_hs,c='black',label='Mario: Beta')

    # y = curve[0]*(np.log(np.array(betas)))**2 + curve[1]*np.log(np.array(betas)) + curve[2]
    # ax1.plot(betas,y,label='Fitted: '+str(curve))
    ax1.plot(num_opts,mean_mu_hs,label='Mario: Beta')

    # data = [17,17.44,17.842,18.15,18.282,18.499,18.645,18.834,18.96,19.241,19.326] # For mu_q = 17
    # data = [10,10.432,10.842,11.15,11.282,11.502,11.645,11.834,11.96,12.241,12.326] # For mu_q = 10

    ref_data = []
    prd = Prediction()
    start = 0.0
    end = 30.0
    x = np.arange(start,end,0.0001)
    pdf_x = PDF(prd.uniform,[17,17],[1,1],x)
    for n in num_opts:
        icpdf = prd.ICPDF(1-1/n,[17,17],0.0001,x,pdf_x)
        ref_data.append(icpdf)
    ax1.plot(num_opts,ref_data,label='Reference')
    ax1.set_xlabel(r"$n$",fontsize = 18)
    ax1.set_ylabel(r"$\mu_{h}$",fontsize = 18)
    ax1.legend()
    fig1.savefig('clipped_n_mu_h_17_2_my_0p001to0p1_step_0p2_uniform_taylor_update.png')
    plt.show()
    # pkl.dump(hell_dis, open('hell_dis_beta_3.pickle', 'wb'))
    # pkl.dump(success_rates, open('success_rates_beta_3.pickle', 'wb'))


plot_test2 = 0
if plot_test2:
    num_opts = np.array([2,3,5,8,10,15,20,30,40,80,100])
    path = os.getcwd() +'/results/'

    data_files = np.array([])
    for i in range(22,33):
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

    data_files = np.array([])
    for i in range(33,44):
        params = pd.read_csv(path+str(i)+'.csv')
        data_files = np.concatenate((data_files,np.array([f for f in np.sort(os.listdir(path)) if '.csv' in f and f[:len(str(i)+'_')] == str(i)+'_' and len(f)>10])),axis=0)
    
    mean_mu_hs = []
    success_rates_without_update = np.zeros(len(num_opts))
    for row in range(len(data_files)):
        data = pd.read_csv(path+data_files[row])
        count = 0
        for i in range(len(data)):
            if data.iloc[i]['$x_{max}$ opt No.'] == data.iloc[i]['$CDM$ opt No.']:
                count += 1
        success_rates_without_update[row] = count/data.shape[0]
    ax1.plot(num_opts,success_rates,label='Mario: updated')
    ax1.plot(num_opts,success_rates_without_update,label='Mario: not updated')
    ax1.set_xlabel(r"$n$",fontsize = 18)
    ax1.set_ylabel("Success rate",fontsize = 18)
    ax1.legend()
    fig1.savefig('n_succ_mario_mux_17.png')
    plt.show()
