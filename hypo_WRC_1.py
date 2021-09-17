#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import classexploration as yn
from multiprocessing import Pool
import pandas as pd
import os
import requests
import json
import random_number_generator as rng
from numba.typed import List
import faulthandler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# import ray
#from ray.util.multiprocessing import Pool

#ray.init(address='auto', _redis_password='5241590000000000')
path = os.getcwd() + "/results/"

WRC_normal = 0
pval_WRC_normal = 0
pval_WRC_bimodal_x_gaussian_h = 0
pval_WRC_uniform_x_gaussian_h = 0
bimodal_x_normal_h = 0
bimodal_x_normal_h_sigma = 0
uniform_x_uniform_h = 0
uniform_x_uniform_h_sigma = 0
uniform_x_normal_h = 0
uniform_x_normal_h_sigma = 0
normal_x_normal_h = 0
normal_x_normal_h_1 = 0
normal_x_normal_h_sigma = 0

wf = yn.workFlow()
vis = yn.Visualization()


def parallel(func,a,b,batch_size,save_string,columns_name,continuation = False,do=False,mu_x=None,n=None):
    step = 0.0001
    prd = yn.Prediction()
    if continuation==False:    
        f = open(path+save_string+'.csv','a')
        f_path = path+save_string+'.csv'
        columns = pd.DataFrame(data=np.array([columns_name]))
        columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

        inp = []
        if do == True:
            for i in a:
                sigma_x_1 = delta_sigma + i
                sigma_x_2 = delta_sigma + i
                sigma_x1 = List([sigma_x_1,sigma_x_2])
                start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
                stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
                dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
                pdf =  prd.gaussian(dis_x,mu_x,sigma_x1)
                area = (np.sum(pdf)*step)
                pdf_x = np.multiply(pdf,1/area)
                mean_esmes2m = prd.ICPDF(1-(1/int(n)),mu_x,stop1,step,dis_x,pdf_x)
                for j in b:
                    mu_h_1 = mean_esmes2m
                    mu_h_2 = mean_esmes2m
                    inp.append((i,j,mu_h_1,mu_h_2))
        else:
            for i in a:
                for j in b:
                    inp.append((i,j))
    else:
        # f_path = path+str(int(save_string[0])-1)+save_string[1:]+'.csv'
        # f1 = pd.read_csv(path+str(int(save_string[0])-1)+save_string[1:]+'.csv')
        # f_path = path+save_string[0]+str(int(save_string[1])-1)+save_string[2:]+'.csv'
        f_path = path+save_string+'.csv'
        f1 = pd.read_csv(f_path)
        ai = f1.iloc[-1,0]
        bi = f1.iloc[-1,1]
        ii = np.where(a == ai)[0][0]
        inp = []
        if do == True:
            for i in a[ii+1:]:
                sigma_x_1 = delta_sigma + i
                sigma_x_2 = delta_sigma + i
                sigma_x1 = List([sigma_x_1,sigma_x_2])
                start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
                stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
                dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
                pdf =  prd.uniform(dis_x,mu_x,sigma_x1)
                area = (np.sum(pdf)*step)
                pdf_x = np.multiply(pdf,1/area)
                mean_esmes2m = prd.ICPDF(1-(1/int(n)),mu_x,stop1,step,dis_x,pdf_x)
                for j in b:
                    mu_h_1 = mean_esmes2m
                    mu_h_2 = mean_esmes2m
                    inp.append((i,j,mu_h_1,mu_h_2))
        else:
            for i in a[ii+1:]:
                for j in b:
                    inp.append((i,j))
    opt_var = []
    progress = 0
    for i in range(0,len(inp),batch_size):
        with Pool(20) as p:#,ray_address="auto") as p:
            opt_var = p.starmap(func,inp[i:i+batch_size])
        out = pd.DataFrame(data=opt_var,columns=columns_name)
        out.to_csv(f_path,mode = 'a',header = False, index=False)
        progress +=1
        print("\r Percent of input processed : {}%".format(np.round(100*progress*batch_size/len(inp)),decimals=1), end="")

def save_data(save_string,continuation):
    check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))
    count = 0
    for i in check:
        if count==i:
            count+=1
    
    if continuation==False:
        save_string = str(count)+save_string
        f1 = open(path+str(count),'w+')
        return save_string,f1
    else:
        save_string = str(count-1)+save_string
        f1 = None
        return save_string,f1

def pushbullet_message(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = 'o.YlTBKuQWnkOUsCP9ZxzWC9pvFNz1G0mi'
    # resp = requests.post('https://api.pushbullet.com/v2/pushes', 
    #                      data=json.dumps(msg),
    #                      headers={'Authorization': 'Bearer ' + TOKEN,
    #                               'Content-Type': 'application/json'})
    # if resp.status_code != 200:
    #     raise Exception('Error',resp.status_code)
    # else:
    #     print ('Message sent')

def plot_slopes(mum_slopes,mum,num_opts,ax,distribution,color):
    # color = ['red','green','blue','purple','violet','brown','black','orange']
    marker=['o','s','*','+','D','x']
    for i in range(len(num_opts)):
        x = []
        y = []
        for j in range(len(mum)):
            x.append(mum[j])
            y.append(mum_slopes[j][i][0])
        ax.plot(x,y,linewidth=0.1*num_opts[i],color = color,marker=marker[i],label=str(num_opts[i])+distribution)

def plot_3d(mum_slopes,mum,num_opts,ax=None,distribution=None,color="orange"):
    x = []
    y = []
    dz = []
    trans = []
    for i in range(len(mum_slopes)):
        for j in range(len(mum_slopes[0])):
            x.append(i)
            y.append(j)
            dz.append(mum_slopes[i,j,0])
            trans.append(mum_slopes[i,j,0]*np.random.uniform(0,1))
    dx = np.ones_like(x)*0.3
    dy = np.ones_like(y)*0.3
    z = np.zeros_like(dz)
    x = np.array(x)-0.3*distribution
    y = np.array(y)
    for i in range(len(dz)):
        ax1.bar3d(x[i], y[i], z[i], dx[i], dy[i], dz[i],alpha=trans[i],color=color)

    ax1.set_xlabel(r'$\mu_m$')
    ax1.set_ylabel('Number of options')
    ax1.set_zlabel('slope and HARS')
    

def plot_HARS(mum_slopes,mum,num_opts,ax,distribution,color):
    # color = ['red','green','blue','purple','violet','brown','black','orange']
    marker=['o','s','*','+','D','x']
    for i in range(len(num_opts)):
        x = []
        y = []
        for j in range(len(mum)):
            x.append(mum[j])
            y.append(mum_slopes[j][i][1])
        ax.plot(x,y,linewidth=0.1*num_opts[i],color = color,marker=marker[i],label=str(num_opts[i])+distribution)
    
if WRC_normal==1:
    mu_m = [i for i in range(500,1000)]
    sigma_m = [i for i in range(0,180)]
    batch_size = len(mu_m)
    runs = 200
    continuation = False
    save_string = "0WRC"
    # save_string = save_data(save_string,continuation)
    mu_h_1 = 0
    sigma_h_1 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    number_of_options = 10

    def mumf(mum,sigm,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0,avg_correct_ranking = 0):
        loop = 0
        while loop <= runs:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,\
                distribution_h=rng.threshold_n,mu_h=[mu_h_1,mu_h_1],sigma_h=[sigma_h_1,sigma_h_1],mu_x=[mu_x_1,mu_x_1],sigma_x=[sigma_x_1,sigma_x_1],\
                err_type=0,number_of_options=number_of_options,mu_m=[mum,mum],sigma_m=[sigm,sigm])
            flag = 0
            for i in yes_test:
                for j in i:
                    if j[0][0]== np.nan or j[1]<0:
                        flag = 1
                        break
            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                avg_pval += max_rat_pval[0][1]
            else:
                avg_pval += 1

            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n

            if incrt_w_n == 0:
                avg_correct_ranking +=1
            if success == 1:
                count += 1
            loop+=1

        v = {"$\mu_{m}$": mum,"$\sigma_{m}$":sigm, "success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_with_no':avg_incrtness/runs, 'Wrong_ranking_cost_without_no_proportion':avg_incrtness_w_n/runs,'Rate_of_correct_ranking':avg_correct_ranking/runs}
        return v
    
    # parallel(mumf,mu_m,sigma_m,batch_size=batch_size,save_string=save_string,columns_name=["$\mu_{m}$","$\sigma_{m}$", "success_rate",'avg_pvalue','Wrong_ranking_cost_with_no', 'Wrong_ranking_cost_without_no_proportion','Rate_of_correct_ranking'],continuation=continuation)

    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_with_no',file_name=save_string+'.csv',save_plot=save_string+'with_no',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options,gaussian=0)

    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_="Rate_of_correct_ranking",file_name=save_string+'.csv',save_plot=save_string+'RCR',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options,gaussian=0)

    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_="Wrong_ranking_cost_without_no_proportion",file_name=save_string+'.csv',save_plot=save_string+'without_no',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options,gaussian=0)

    message = 'wrong_ranking_cost_contour' + ' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)

if pval_WRC_normal ==1:
    runs = 500
    continuation = False

    mu_h_1 = 0
    sigma_h_1 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    save_string = 'gxgh_Pval_2D_sx='+str(np.round(sigma_x_1/sigma_h_1,decimals=1))+'sh('+str(sigma_h_1)+')'
    save_string = save_data(save_string,continuation)
    mu_m = [i for i in range(50,1500,20)]
    number_of_options = [2,5,10]
    batch_size = len(mu_m)
    sigma_m_1 = 0

    def mumf(nop,mum,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0):
        loop = 0
        while loop<=runs:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,\
                distribution_h=rng.threshold_n,mu_h=List([mu_h_1,mu_h_1]),sigma_h=List([sigma_h_1,sigma_h_1]),mu_x=List([mu_x_1,mu_x_1]),sigma_x=List([sigma_x_1,sigma_x_1]),\
                err_type=0,number_of_options=nop,mu_m=List([mum,mum]),sigma_m=List([sigma_m_1,sigma_m_1]))
            flag = 0
            for i in yes_test:
                for j in i:
                    if j[0][0]== np.nan or j[1]<0:
                        flag = 1
                        break
            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                avg_pval += max_rat_pval[0][1]
            else:
                avg_pval += 1

            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
            loop += 1

        output = {"nop":nop,"$\mu_{m}$": mum,"success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_without_no':avg_incrtness_w_n/runs, 'Wrong_ranking_cost_with_no_proportion':avg_incrtness/runs}
        return output

    parallel(mumf,number_of_options,mu_m,columns_name=["nop","$\mu_{m}$","success_rate",'avg_pvalue','Wrong_ranking_cost_without_no', 'Wrong_ranking_cost_with_no_proportion'],batch_size=batch_size,save_string=save_string)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='avg_pvalue',file_name=save_string+'.csv',save_plot=save_string+'without_no_Pval',plot_type='line',num_of_opts=number_of_options)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options)
    # vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',z1_var_='Wrong_ranking_cost_with_no_proportion',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options+number_of_options)

if pval_WRC_bimodal_x_gaussian_h ==1:
    runs = 1000
    continuation = False
    save_string = "Pval_2D_bimodal_x_gaussian_h"
    save_string = save_data(save_string,continuation)
    mu_h_1 = 0
    sigma_h_1 = 1
    mu_h_2 = 0
    sigma_h_2 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    mu_x_2 = 3
    sigma_x_2 = 1
    mu_m = [i for i in range(500,2000,20)]
    number_of_options = [2,5,10,20]
    batch_size = len(mu_m)
    sigma_m_1 = 170
    sigma_m_2 = 170
    def mumf(nop,mum,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0):
        loop = 0
        while loop<=runs:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,\
                distribution_h=rng.threshold_n,mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2],\
                err_type=0,number_of_options=nop,mu_m=[mum,mum],sigma_m=[sigma_m_1,sigma_m_2])
            flag = 0
            for i in yes_test:
                for j in i:
                    if j[0][0]== np.nan or j[1]<0:
                        flag = 1
                        break
            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                avg_pval += max_rat_pval[0][1]
            else:
                avg_pval += 1

            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
            loop += 1

        output = {"nop":nop,"$\mu_{m}$": mum,"success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_without_no':avg_incrtness_w_n/runs, 'Wrong_ranking_cost_with_no_proportion':avg_incrtness/runs}
        return output

    # parallel(mumf,number_of_options,mu_m,columns_name=["nop","$\mu_{m}$","success_rate",'avg_pvalue','Wrong_ranking_cost_without_no', 'Wrong_ranking_cost_with_no_proportion'],batch_size=batch_size,save_string=save_string)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='avg_pvalue',file_name=save_string+'.csv',save_plot=save_string+'without_no_Pval',plot_type='line',num_of_opts=number_of_options)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',z1_var_='Wrong_ranking_cost_with_no_proportion',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options+number_of_options)

    message =' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)

if pval_WRC_uniform_x_gaussian_h ==1:
    runs = 1000
    continuation = False
    save_string = "Pval_2D_uniform_x_gaussian_h"
    save_string = save_data(save_string,continuation)
    mu_h_1 = 0
    sigma_h_1 = 1
    mu_h_2 = 0
    sigma_h_2 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    mu_x_2 = 0
    sigma_x_2 = 1
    mu_m = [i for i in range(500,2000,20)]
    number_of_options = [2,5,10,20]
    batch_size = len(mu_m)
    sigma_m_1 = 170
    sigma_m_2 = 170
    def mumf(nop,mum,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0):
        loop = 0
        while loop<=runs:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,\
                distribution_h=rng.threshold_n,mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mu_x_1 - np.sqrt(3)*sigma_x_1,mu_x_2  - np.sqrt(3)*sigma_x_2],sigma_x=[mu_x_1 + np.sqrt(3)*sigma_x_1,mu_x_2  + np.sqrt(3)*sigma_x_2],\
                err_type=0,number_of_options=nop,mu_m=[mum,mum],sigma_m=[sigma_m_1,sigma_m_2])
            flag = 0
            for i in yes_test:
                for j in i:
                    if j[0][0]== np.nan or j[1]<0:
                        flag = 1
                        break
            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                avg_pval += max_rat_pval[0][1]
            else:
                avg_pval += 1

            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
            loop += 1

        output = {"nop":nop,"$\mu_{m}$": mum,"success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_without_no':avg_incrtness_w_n/runs, 'Wrong_ranking_cost_with_no_proportion':avg_incrtness/runs}
        return output

    parallel(mumf,number_of_options,mu_m,columns_name=["nop","$\mu_{m}$","success_rate",'avg_pvalue','Wrong_ranking_cost_without_no', 'Wrong_ranking_cost_with_no_proportion'],batch_size=batch_size,save_string=save_string)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='avg_pvalue',file_name=save_string+'.csv',save_plot=save_string+'without_no_Pval',plot_type='line',num_of_opts=number_of_options)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',z1_var_='Wrong_ranking_cost_with_no_proportion',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options+number_of_options)

    message =' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)

if bimodal_x_normal_h==1:
    continuation = False
    mum = [10,50,100,200,500]
    number_of_opts = [2,5,10,20]
    cnt = 125
    mum_slopes_bxgh = []
    for i in mum:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_h_1 = 1
        sigma_h_2=1
        sigma_x_1=sigma_h_1
        sigma_x_2=sigma_h_1
        runs = 500
        batch_size = 50
        delta_mu = 5
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes_bxgh = []
        for nop in number_of_opts:
            number_of_options = nop
            save_string = str(cnt)+'bxgh_sx=_sh_mu_h_vs_mu_x1_mu_x2_delta_mu_vs_RCD_nop_'+str(nop) # str(cnt)+
            # save_string,param = save_data(save_string,continuation)
            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("sigma_x_1 : "+str(sigma_x_1)+"\n")
            #     param.write("sigma_x_2 : "+str(sigma_x_2)+"\n")
            #     param.write("sigma_h_1 : "+str(sigma_h_1)+"\n")
            #     param.write("sigma_h_2 : "+str(sigma_h_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
            #     param.write("delta_mu : "+str(delta_mu)+"\n")

            def mux1muh1(muh,mux):
                mux1 = mux
                mux2 = delta_mu + mux
                muh1 = muh
                muh2 = muh
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                        mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
                        mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
                    if success == 1:
                        count += 1
                mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}
                return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h),continuation=continuation)
            continuation = False
            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=1,uniform=0,mu_m=mu_m_1)

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1

if bimodal_x_normal_h_sigma==1:
    continuation = False
    mum_bxgh = [10,50,100,200,500]
    file_num = 0
    mum_slopes_bxgh = []
    number_of_opts_bxgh = [2,5,10,20,100]
    for i in mum_bxgh:
        mu_m_1 = i
        sigma_m_1 = 0
        mu_m_2 = i
        sigma_m_2 = 0

        runs = 500
        batch_size = 10
        delta_sigma = 0
        delta_mu = 5
        mu_x_1 = 5
        mu_x_2 = 5 + delta_mu
        # mu_h_1 = (mu_x_1+mu_x_2)/2
        # mu_h_2 = mu_h_1
        mu_x = List([mu_x_1,mu_x_2])
        sigma_x = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        sigma_h = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        
        step = 0.0001
        num_slopes = []
        for nop in number_of_opts_bxgh:
            number_of_options = nop
            save_string = str(file_num)+'bxgh_mx_mh_sigma_h_vs_sigma_x1_sigma_x2_vs_RCD_nop_'+str(nop) # str(file_num)+
            # save_string,param = save_data(save_string,continuation)

            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("mu_x_1 : "+str(mu_x_1)+"\n")
            #     param.write("mu_x_2 : "+str(mu_x_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
                # param.write("mu_h_1 : "+str(mu_h_1)+"\n")
                # param.write("mu_h_2 : "+str(mu_h_2)+"\n")

            
            def sigx1sigh1(sigma_h,sigma_x):
                sigma_x_1 = delta_sigma + sigma_x
                sigma_x_2 = delta_sigma + sigma_x
                sigma_h_1 = sigma_h
                sigma_h_2 = sigma_h
                
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                        mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2],err_type=0,number_of_options=number_of_options,\
                        mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                    if success == 1:
                        count += 1
                mu_va = {'$\sigma_{h_1}$':sigma_h_1,'$\sigma_{h_2}$':sigma_h_2,'$\sigma_{x_1}$': sigma_x_1,'$\sigma_{x_2}$': sigma_x_2,"success_rate":count/runs}
                return mu_va

            # parallel(sigx1sigh1,sigma_h,sigma_x,columns_name=['$\sigma_{h_1}$','$\sigma_{h_2}$','$\sigma_{x_1}$','$\sigma_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(sigma_h),continuation=continuation,do=True,mu_x=mu_x,n=number_of_options)
            # continuation=False
            # step = 0.0001
            # prd = yn.Prediction()
            # min_sig_h=[]
            # for i in sigma_x:
            #     sigma_x_1 = delta_sigma + i
            #     sigma_x_2 = delta_sigma + i
            #     sigma_x1 = List([sigma_x_1,sigma_x_2])
            #     start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
            #     stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
            #     dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
            #     pdf =  prd.gaussian(dis_x,mu_x,sigma_x1)
            #     area = (np.sum(pdf)*step)
            #     pdf_x = np.multiply(pdf,1/area)
            #     mean_esmes2m = prd.ICPDF(1-(1/number_of_options),mu_x,stop1,step,dis_x,pdf_x)
            #     es2m = prd.ICPDF(1-(5/(3*number_of_options)),mu_x,stop1,step,dis_x,pdf_x)
            #     min_sig_h.append(mean_esmes2m-es2m)

            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1)#,min_sig_h=min_sig_h
            num_slopes.append([slope,hars])

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            file_num += 1
            print(file_num)
        mum_slopes_bxgh.append(num_slopes)

bimodal_x_normal_h_sigma_1=1
if bimodal_x_normal_h_sigma_1==1:
    continuation = False
    mum_bxgh = [10,50,100,200,500]
    file_num = 0
    mum_slopes_bxgh = []
    number_of_opts_bxgh = [8,15,30,40]
    for i in mum_bxgh:
        mu_m_1 = i
        sigma_m_1 = 0
        mu_m_2 = i
        sigma_m_2 = 0

        runs = 500
        batch_size = 10
        delta_sigma = 0
        delta_mu = 5
        mu_x_1 = 5
        mu_x_2 = 5 + delta_mu
        mu_h_1 = (mu_x_1+mu_x_2)/2
        mu_h_2 = mu_h_1
        mu_x = List([mu_x_1,mu_x_2])
        sigma_x = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        sigma_h = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        
        step = 0.0001
        num_slopes = []
        for nop in number_of_opts_bxgh:
            number_of_options = nop
            save_string = 'bxgh_mx_mh_sigma_h_vs_sigma_x1_sigma_x2_vs_RCD_nop_'+str(nop) # str(file_num)+
            save_string,param = save_data(save_string,continuation)

            if isinstance(param,type(None))==False:
                param.write("mu_m_1 : "+str(mu_m_1)+"\n")
                param.write("mu_m_2 : "+str(mu_m_2)+"\n")
                param.write("mu_x_1 : "+str(mu_x_1)+"\n")
                param.write("mu_x_2 : "+str(mu_x_2)+"\n")
                param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
                param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
                param.write("nop : "+str(number_of_options)+"\n")
                param.write("mu_h_1 : "+str(mu_h_1)+"\n")
                param.write("mu_h_2 : "+str(mu_h_2)+"\n")

            
            def sigx1sigh1(sigma_h,sigma_x):
                sigma_x_1 = delta_sigma + sigma_x
                sigma_x_2 = delta_sigma + sigma_x
                sigma_h_1 = sigma_h
                sigma_h_2 = sigma_h
                
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                        mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2],err_type=0,number_of_options=number_of_options,\
                        mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                    if success == 1:
                        count += 1
                mu_va = {'$\sigma_{h_1}$':sigma_h_1,'$\sigma_{h_2}$':sigma_h_2,'$\sigma_{x_1}$': sigma_x_1,'$\sigma_{x_2}$': sigma_x_2,"success_rate":count/runs}
                return mu_va

            parallel(sigx1sigh1,sigma_h,sigma_x,columns_name=['$\sigma_{h_1}$','$\sigma_{h_2}$','$\sigma_{x_1}$','$\sigma_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(sigma_h),continuation=continuation)#,do=True,mu_x=mu_x,n=number_of_options)
            # continuation=False
            # step = 0.0001
            # prd = yn.Prediction()
            # min_sig_h=[]
            # for i in sigma_x:
            #     sigma_x_1 = delta_sigma + i
            #     sigma_x_2 = delta_sigma + i
            #     sigma_x1 = List([sigma_x_1,sigma_x_2])
            #     start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
            #     stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
            #     dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
            #     pdf =  prd.gaussian(dis_x,mu_x,sigma_x1)
            #     area = (np.sum(pdf)*step)
            #     pdf_x = np.multiply(pdf,1/area)
            #     mean_esmes2m = prd.ICPDF(1-(1/number_of_options),mu_x,stop1,step,dis_x,pdf_x)
            #     es2m = prd.ICPDF(1-(5/(3*number_of_options)),mu_x,stop1,step,dis_x,pdf_x)
            #     min_sig_h.append(mean_esmes2m-es2m)

            # [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1)#,min_sig_h=min_sig_h
            # num_slopes.append([slope,hars])

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            file_num += 1
            print(file_num)
        # mum_slopes_bxgh.append(num_slopes)

if uniform_x_uniform_h==1:
    continuation = False
    number_of_opts = [10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    sigma_h_1 = 1
    sigma_h_2=1
    sigma_x_1=sigma_h_1
    sigma_x_2=sigma_h_1
    low_x_1 = -np.sqrt(3)*sigma_x_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
    high_x_1 = np.sqrt(3)*sigma_x_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
    low_h_1 = -np.sqrt(3)*sigma_h_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
    high_h_1 = np.sqrt(3)*sigma_h_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
    runs = 500
    batch_size = 50
    delta_mu = 0
    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    cnt = 13
    for nop in number_of_opts:
        number_of_options = nop
        save_string = str(cnt)+'uxuh_mu_h_vs_mu_x_vs_RCD_nop' + str(nop) #str(cnt)+
        # save_string = save_data(save_string,continuation)

        def mux1muh1(muh,mux):
            mux1 = mux + low_x_1
            sigmax1 = mux + high_x_1
            muh1 = muh + low_h_1
            sigmah1 = muh + high_h_1
            count = 0
            for k in range(runs):
                success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_u,\
                    mu_h=[muh1,muh1],sigma_h=[sigmah1,sigmah1],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
                    mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                if success == 1:
                    count += 1
            mu_va = {'$\mu_{h_1}$':muh,'$\mu_{h_2}$':muh,'$\mu_{x_1}$': mux,'$\mu_{x_2}$': mux,"success_rate":count/runs}
            return mu_va

        # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=0,uniform=1)

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)
        cnt += 1

if uniform_x_uniform_h_sigma==1:
    continuation = False
    number_of_opts = [2,5,10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    mu_h_1 = 5
    mu_h_2 = 5
    mu_x_1 = mu_h_1
    mu_x_2 = mu_h_1
    runs = 500
    batch_size = 50
    delta_mu = 0
    sigma_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    cnt = 14
    for nop in number_of_opts:
        number_of_options = nop
        save_string = str(cnt)+'uxuh_sigma_h_vs_sigma_x_vs_RCD_nop'+str(nop) # str(cnt)+
        # save_string = save_data(save_string,continuation)

        def sigmax1sigmah1(sigh,sigx):
            mux1 = mu_x_1 - np.sqrt(3)*sigx
            sigmax1 = mu_x_1 + np.sqrt(3)*sigx
            muh1 = mu_h_1 - np.sqrt(3)*sigh
            sigmah1 = mu_h_1 + np.sqrt(3)*sigh
            count = 0
            for k in range(runs):
                success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_u,\
                    mu_h=[muh1,muh1],sigma_h=[sigmah1,sigmah1],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
                    mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                if success == 1:
                    count += 1
            mu_va = {'$\sigma_{h_1}$':sigh,'$\sigma_{h_2}$':sigh,'$\sigma_{x_1}$': sigx,'$\sigma_{x_2}$': sigx,"success_rate":count/runs}
            return mu_va

        # parallel(sigmax1sigmah1,sigma_h,sigma_x,columns_name=['$\sigma_{h_1}$','$\sigma_{h_2}$','$\sigma_{x_1}$','$\sigma_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(sigma_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',delta_mu=delta_mu,gaussian=0,uniform=0)

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)
        cnt += 1

if uniform_x_normal_h==1:
    continuation = False
    cnt = 173
    mum = [100,200,500]
    for i in mum:
        number_of_opts = [2,5,10,20]
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_h_1 = 1
        sigma_h_2=1
        sigma_x_1=sigma_h_1
        sigma_x_2=sigma_h_1
        low_x_1 = -np.sqrt(3)*sigma_x_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
        high_x_1 = np.sqrt(3)*sigma_x_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
        runs = 500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        
        for nop in number_of_opts:
            number_of_options = nop
            save_string = str(cnt)+'uxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop)#str(cnt)+
            # save_string,param = save_data(save_string,continuation)
            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("sigma_x_1 : "+str(sigma_x_1)+"\n")
            #     param.write("sigma_x_2 : "+str(sigma_x_2)+"\n")
            #     param.write("sigma_h_1 : "+str(sigma_h_1)+"\n")
            #     param.write("sigma_h_2 : "+str(sigma_h_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
            #     param.write("delta_mu : "+str(delta_mu)+"\n")

            def mux1muh1(muh,mux):
                mux1 = mux + low_x_1
                sigmax1 = mux + high_x_1
                muh1 = muh
                muh2 = muh
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_n,\
                        mu_h=[muh1,muh2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
                        mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                    if success == 1:
                        count += 1
                mu_va = {'$\mu_{h_1}$':muh,'$\mu_{h_2}$':muh,'$\mu_{x_1}$': mux,'$\mu_{x_2}$': mux,"success_rate":count/runs}
                return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=0,uniform=1,mu_m=mu_m_1)

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1

if uniform_x_normal_h_sigma==1:
    continuation = False
    mum_uxgh = [10,50,100,200,500]
    number_of_opts_uxgh = [2,5,10,20]
    cnt = 45
    mum_slopes_uxgh = []
    for i in mum_uxgh:
        
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        
        mu_x_1 = 7.5
        mu_x_2 = 7.5
        # mu_h_1 = mu_x_1
        # mu_h_2 = mu_x_1
        mu_x = List([mu_x_1,mu_x_2])
        runs = 500
        batch_size = 50
        delta_sigma = 0
        sigma_x = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        sigma_h = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        num_slopes = []
        for nop in number_of_opts_uxgh:
            number_of_options = nop
            save_string = str(cnt)+'uxgh_sigma_h_vs_sigma_x_vs_RCD_nop'+str(nop) # str(cnt)+
            # save_string,param = save_data(save_string,continuation)
            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("mu_x_1 : "+str(mu_x_1)+"\n")
            #     param.write("mu_x_2 : "+str(mu_x_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
                # param.write("mu_h_1 : "+str(mu_h_1)+"\n")
                # param.write("mu_h_2 : "+str(mu_h_2)+"\n")

            def sigmax1sigmah1(sigh,sigx):
                mux1 = mu_x_1 - np.sqrt(3)*sigx
                sigmax1 = mu_x_1 + np.sqrt(3)*sigx
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_n,\
                        mu_h=[mu_h_1,mu_h_2],sigma_h=[sigh,sigh],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
                        mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                    if success == 1:
                        count += 1
                mu_va = {'$\sigma_{h_1}$':sigh,'$\sigma_{h_2}$':sigh,'$\sigma_{x_1}$': sigx,'$\sigma_{x_2}$': sigx,"success_rate":count/runs}
                return mu_va

            # parallel(sigmax1sigmah1,sigma_h,sigma_x,columns_name=['$\sigma_{h_1}$','$\sigma_{h_2}$','$\sigma_{x_1}$','$\sigma_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(sigma_h),do=True,mu_x=mu_x,n=number_of_options)
            # step = 0.0001
            # prd = yn.Prediction()
            # min_sig_h=[]
            # for i in sigma_x:
            #     sigma_x_1 = delta_sigma + i
            #     sigma_x_2 = delta_sigma + i
            #     sigma_x1 = List([sigma_x_1,sigma_x_2])
            #     start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
            #     stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
            #     dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
            #     pdf =  prd.uniform(dis_x,mu_x,sigma_x1)
            #     area = (np.sum(pdf)*step)
            #     pdf_x = np.multiply(pdf,1/area)
            #     mean_esmes2m = prd.ICPDF(1-(1/number_of_options),mu_x,stop1,step,dis_x,pdf_x)
            #     es2m = prd.ICPDF(1-(5/(3*number_of_options)),mu_x,stop1,step,dis_x,pdf_x)
            #     min_sig_h.append(mean_esmes2m-es2m)
            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',delta_mu=delta_sigma,gaussian=0,uniform=0,mu_m=mu_m_1)#,min_sig_h=min_sig_h
            num_slopes.append([slope,hars])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
            print(cnt)
        mum_slopes_uxgh.append(num_slopes)


if normal_x_normal_h==1:
    continuation = False
    mum = [10,50,100,200,500]
    cnt = 145
    for i in mum:
        number_of_opts = [2,5,10,20]
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        
        # sigma_h_1=1
        # sigma_h_2=1
        # sigma_x_1 = 3*sigma_h_1
        # sigma_x_2= 3*sigma_h_1
        sigma_x_1 = 1
        sigma_x_2= 1
        sigma_h_1=sigma_x_1
        sigma_h_2=sigma_x_1
        runs = 500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        
        for nop in number_of_opts:
            number_of_options = nop
            save_string =  str(cnt)+'gxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop) # str(cnt)+
            # save_string,param = save_data(save_string,continuation)
            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("sigma_x_1 : "+str(sigma_x_1)+"\n")
            #     param.write("sigma_x_2 : "+str(sigma_x_2)+"\n")
            #     param.write("sigma_h_1 : "+str(sigma_h_1)+"\n")
            #     param.write("sigma_h_2 : "+str(sigma_h_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
            #     param.write("delta_mu : "+str(delta_mu)+"\n")

            def mux1muh1(muh,mux,avg_pval = 0):
                # avg_incrtness_w_n = np.zeros((number_of_options,5*number_of_options))
                mux1 = mux
                mux2 = delta_mu + mux
                muh1 = muh
                muh2 = muh
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                        mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
                        mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
                    if success == 1:
                        count += 1

                    # flag = 0
                    # for i in yes_test:
                    #     for j in i:
                    #         if j[0][0]== np.nan or j[1]<0:
                    #             flag = 1
                    #             break
                    # if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                    #     avg_pval += max_rat_pval[0][1]
                    # else:
                    #     avg_pval += 1

                    # avg_incrtness_w_n += np.concatenate((incrt_w_n[0],incrt_w_n[1],incrt_w_n[2]*pval_mat,incrt_w_n[3],incrt_w_n[4]),axis=1)

                mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}#{'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs,'Average p-value':avg_pval/runs,'Wrong_ranking_cost_without_no_proportion':avg_incrtness_w_n/runs}
                return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate",'Average p-value','Wrong_ranking_cost_without_no_proportion'],save_string=save_string,batch_size=3*len(mu_h))

            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,mu_m=mu_m_1)

            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'Pval',x_var_='$\mu_{x_1}$',y_var_='Average p-value',num_of_opts=number_of_opts,plot_type='line')
            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'WRC',x_var_='$\mu_{x_1}$',y_var_='Wrong_ranking_cost_without_no_proportion',num_of_opts=number_of_opts,plot_type='line')

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1

if normal_x_normal_h_1==1:
    continuation = False
    number_of_opts = [2,5,10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    
    sigma_h_1=1
    sigma_h_2=1
    sigma_x_1 = 3*sigma_h_1
    sigma_x_2= 3*sigma_h_1
    # sigma_x_1 = 1
    # sigma_x_2= 1
    # sigma_h_1=sigma_x_1
    # sigma_h_2=sigma_x_1
    runs = 500
    batch_size = 50
    delta_mu = 0
    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    cnt = 39
    for nop in number_of_opts:
        number_of_options = nop
        save_string = str(cnt)+'gxgh_sx='+str(np.round(sigma_x_1/sigma_h_1,decimals=1))+'_sh('+str(sigma_h_1)+')_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop) # str(cnt)+
        # save_string = save_data(save_string,continuation)

        def mux1muh1(muh,mux,avg_pval = 0,avg_incrtness_w_n = 0):
            mux1 = mux
            mux2 = delta_mu + mux
            muh1 = muh
            muh2 = muh
            count = 0
            for k in range(runs):
                success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                    mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
                    mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
                if success == 1:
                    count += 1

                flag = 0
                for i in yes_test:
                    for j in i:
                        if j[0][0]== np.nan or j[1]<0:
                            flag = 1
                            break
                if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                    avg_pval += max_rat_pval[0][1]
                else:
                    avg_pval += 1
                avg_incrtness_w_n += incrt_w_n

            mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs,'Average p-value':avg_pval/runs,'Wrong_ranking_cost_without_no_proportion':avg_incrtness_w_n/runs}
            return mu_va

        # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate",'Average p-value','Wrong_ranking_cost_without_no_proportion'],save_string=save_string,batch_size=3*len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)
        
        # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'Pval',x_var_='$\mu_{x_1}$',y_var_='Average p-value',num_of_opts=nop,plot_type='line')
        # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'WRC',x_var_='$\mu_{x_1}$',y_var_='Wrong_ranking_cost_without_no_proportion',num_of_opts=nop,plot_type='line')

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)
        cnt += 1

if normal_x_normal_h_sigma==1:
    continuation = False
    mum_gxgh = [10,50,100,200,500]
    number_of_opts_gxgh = [2,5,10,20]
    file_num = 25
    mum_slopes_gxgh = []
    for i in mum_gxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        mu_x_1=7.5
        mu_x_2=7.5
        # mu_h_1 = mu_x_1
        # mu_h_2= mu_x_1
        mu_x = List([mu_x_1,mu_x_2])
        runs = 500
        batch_size = 50
        delta_sigma = 0
        sigma_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes = []
        for nop in number_of_opts_gxgh:
            number_of_options = nop
            save_string = str(file_num)+'gxgh_mx=mh_sigma_h_vs_sigma_x_vs_RCD_nop_'+str(nop) # str(file_num)+
            # save_string,param = save_data(save_string,continuation)
            # if isinstance(param,type(None))==False:
            #     param.write("mu_m_1 : "+str(mu_m_1)+"\n")
            #     param.write("mu_m_2 : "+str(mu_m_2)+"\n")
            #     param.write("mu_x_1 : "+str(mu_x_1)+"\n")
            #     param.write("mu_x_2 : "+str(mu_x_2)+"\n")
            #     param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
            #     param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
            #     param.write("nop : "+str(number_of_options)+"\n")
            #     param.write("mu_h_1 : "+str(mu_h_1)+"\n")
            #     param.write("mu_h_2 : "+str(mu_h_2)+"\n")

            def sigx1sigh1(sigma_h,sigma_x):
                sigma_x_1 = sigma_x
                sigma_x_2 = delta_sigma + sigma_x
                sigma_h_1 = sigma_h
                sigma_h_2 = sigma_h
                count = 0
                for k in range(runs):
                    success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                        mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2],err_type=0,number_of_options=number_of_options,\
                        mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                    if success == 1:
                        count += 1
                mu_va = {'$\sigma_{h_1}$':sigma_h_1,'$\sigma_{h_2}$':sigma_h_2,'$\sigma_{x_1}$': sigma_x_1,'$\sigma_{x_2}$': sigma_x_2,"success_rate":count/runs}
                return mu_va

            # parallel(sigx1sigh1,sigma_h,sigma_x,columns_name=['$\sigma_{h_1}$','$\sigma_{h_2}$','$\sigma_{x_1}$','$\sigma_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(sigma_h),continuation=continuation,do=True,mu_x=mu_x,n=number_of_options)

            # step = 0.0001
            # prd = yn.Prediction()
            # min_sig_h=[]
            # for i in sigma_x:
            #     sigma_x_1 = delta_sigma + i
            #     sigma_x_2 = delta_sigma + i
            #     sigma_x1 = List([sigma_x_1,sigma_x_2])
            #     start1 = np.sum(mu_x)/2 - sigma_x_1-sigma_x_2-45
            #     stop1 = np.sum(mu_x)/2 +sigma_x_1+sigma_x_1+45
                
            #     dis_x = np.round_(np.arange(start1,stop1,step),decimals=4)
            #     pdf =  prd.gaussian(dis_x,mu_x,sigma_x1)
            #     area = (np.sum(pdf)*step)
            #     pdf_x = np.multiply(pdf,1/area)
            #     mean_esmes2m = prd.ICPDF(1-(1/number_of_options),mu_x,stop1,step,dis_x,pdf_x)
            #     es2m = prd.ICPDF(1-(5/(3*number_of_options)),mu_x,stop1,step,dis_x,pdf_x)
            #     min_sig_h.append(mean_esmes2m-es2m)
            [slope,intercept,hars] = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1)#,min_sig_h=min_sig_h)
            num_slopes.append([slope,hars])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            file_num += 1
            print(file_num)
        mum_slopes_gxgh.append(num_slopes)

slopes_HARS = 0
if slopes_HARS ==1:
    fig, ax = plt.subplots()
    plot_slopes(mum_slopes_bxgh,mum_bxgh,number_of_opts_bxgh,ax,'bxgh',color='red')
    plot_slopes(mum_slopes_uxgh,mum_uxgh,number_of_opts_uxgh,ax,'uxgh',color='green')
    plot_slopes(mum_slopes_gxgh,mum_gxgh,number_of_opts_gxgh,ax,'gxgh',color='blue')
    plt.legend()
    plt.xlabel(r'$\mu_m$')
    plt.ylabel('Slope of best fit')
    plt.show()

    fig, ax = plt.subplots()
    plot_HARS(mum_slopes_bxgh,mum_bxgh,number_of_opts_bxgh,ax,'bxgh',color='red')
    plot_HARS(mum_slopes_uxgh,mum_uxgh,number_of_opts_uxgh,ax,'uxgh',color='green')
    plot_HARS(mum_slopes_gxgh,mum_gxgh,number_of_opts_gxgh,ax,'gxgh',color='blue')
    plt.legend()
    plt.xlabel(r'$\mu_m$')
    plt.ylabel('HARS of best fit')
    plt.show()

figures3d = 0
if figures3d:
    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    plot_3d(np.array(mum_slopes_bxgh),mum_bxgh,number_of_opts_bxgh,ax=ax1,distribution=1,color="slateblue")
    plot_3d(np.array(mum_slopes_uxgh),mum_uxgh,number_of_opts_uxgh,ax=ax1,distribution=2,color="lightseagreen")
    plot_3d(np.array(mum_slopes_gxgh),mum_gxgh,number_of_opts_gxgh,ax=ax1,distribution=3,color="coral")
    plt.legend()
    plt.show()

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))

    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

visua = 0
prd = yn.Prediction()
if visua == 1:
    step = 0.0001
    
    number_of_options = [100]

    fig,axs = plt.subplots()

    # fig.tight_layout(pad=0.5)
    dist = prd.uniform
    # delta_mu = 4
    mu_x = List([10])
    sigma_x = List([1])
    low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])#,mu_x[1]-np.sqrt(3)*sigma_x[1]])
    high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])#,mu_x[1]+np.sqrt(3)*sigma_x[1]])
    start1 = np.sum(mu_x)/len(mu_x) - np.sum(sigma_x)-5
    stop1 = np.sum(mu_x)/len(mu_x) + np.sum(sigma_x)+5

    dis_x = np.round(np.arange(start1,stop1,step),decimals=4)
    pdf =  dist(dis_x,mu_x,sigma_x)
    area = (np.sum(pdf)*step)
    pdf_x = np.multiply(pdf,1/area)

    
    for nop in range(len(number_of_options)):
        mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,stop1,step,dis_x,pdf_x)
        axs.axvline(mean_esmes2m,0,500,color='red',label = r'$\mu_{ESM,ES2M}$',linewidth = 0.5)

        mu_h = List([mean_esmes2m])
        # mu_h = mu_x
        sigma_h = List([1/3])
        
        start = np.sum(mu_h)/len(mu_h) - np.sum(sigma_h)-5
        stop = np.sum(mu_h)/len(mu_h) + np.sum(sigma_h)+5

        dis_h = np.round(np.arange(start,stop,step),decimals=4)
        pdf =  prd.gaussian(dis_h,mu_h,sigma_h)
        area = (np.sum(pdf)*step)
        pdf_h = np.multiply(pdf,1/area)


        axs.plot(dis_x,pdf_x,color='black')
        
        axs.invert_yaxis()

        slices = []
        mid_slices=[]
        for i in range(1,number_of_options[nop],1):
            ESM = prd.ICPDF(float(i)/number_of_options[nop],mu_x,stop,step,dis_x,pdf_x)
            slices.append(np.round(ESM,decimals=3))
        for i in range(1,2*number_of_options[nop],1):
            if i%2!=0:
                mid_slices.append(np.round(prd.ICPDF((i/(2*number_of_options[nop])),mu_x,stop,step,dis_x,pdf_x),decimals=1))

        number_of_colors = number_of_options[nop]

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]

        for i in range(len(slices)+1):
            if i!=0 and i!=len(slices):
                x1 = np.arange(slices[i-1],slices[i],0.0001)
                pdf1 =  dist(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                axs.fill_between(x1,0,pdf1,facecolor=color[i])
            elif i==0:
                x1 = np.arange(start,slices[i],0.0001)
                pdf1 =  dist(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                axs.fill_between(x1,0,pdf1,facecolor=color[i])
            elif i==len(slices):
                x1 = np.arange(slices[-1],stop,0.0001)
                pdf1 =  dist(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                axs.fill_between(x1,0,pdf1,facecolor=color[i])

        for i in range(3):
            ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
            # ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
            axs.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            axs.scatter(options_quality,(0.3-i*0.1)*np.ones_like(options_quality),s=3,edgecolor = 'black')
            axs.text(start1+2,(0.3-i*0.1)-0.01,'trial '+str(i))
            axs.text(start1+2,0.03,r'$\mu_x$')
            
        axs1 = axs.twinx()
        axs1.plot(dis_h,pdf_h,color='maroon')
        for i in range(3):
            units = rng.threshold_n(m_units=number_of_options[nop]*100,mu_h=mu_h,sigma_h=sigma_h)
            axs1.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            axs1.scatter(units,(0.3-i*0.1)*np.ones_like(units),s=1,edgecolor = 'black')
            axs1.text(start1+2,(0.3-i*0.1)+0.01,'trial '+str(i))
            axs.text(start1+2,-0.01,r'$\mu_h$')
        
        axs.legend(loc='upper right')

        axs.title.set_text("Number of samples drawn = "+str(number_of_options[nop]))
        align_yaxis(axs1,0.0,axs,0.0)
        axs.set_yticks([])
        axs1.set_yticks([])

    plt.show()

# if __name__=="__main__":
#     plt.style.use('ggplot')
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, projection='3d')
#     co = ["slateblue","lightseagreen","coral"]
#     for c in range(len(co)):
#         plot_3d(np.random.random((5,5,2)),[10,50,100,200,500],[2,5,10,20],ax=ax1,color=co[c],distribution=c+1)
#     plt.show()