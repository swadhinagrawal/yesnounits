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

path = os.getcwd() + "/results/"

confidence = 0.02                           #   Confidence for distinguishing qualities

WRC_normal = 0
pval_WRC_normal = 0
bimodal_x_normal_h = 1
mu_x_vs_mu_h_vs_RCD = 0
uniform_x_normal_h = 0
uniform_x_uniform_h = 0

wf = yn.workFlow()
vis = yn.Visualization()

def parallel(func,a,b,batch_size,save_string,columns_name,continuation = False):

    if continuation==False:    
        f = open(path+save_string+'.csv','a')
        f_path = path+save_string+'.csv'
        columns = pd.DataFrame(data=np.array([columns_name]))
        columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

        inp = []
        for i in a:
            for j in b:
                inp.append((i,j))
    else:
        f_path = path+str(int(save_string[0])-1)+save_string[1:]+'.csv'
        f1 = pd.read_csv(path+str(int(save_string[0])-1)+save_string[1:]+'.csv')
        ai = f1.iloc[-1,0]
        bi = f1.iloc[-1,1]
        ii = np.where(a == ai)[0][0]
        jj = np.where(b == bi)[0][0]
        inp = []
        for i in a[ii+1:]:
            for j in b:
                inp.append((i,j))

    opt_var = []
    progress = 0
    for i in range(0,len(inp),batch_size):
        with Pool(12) as p:#,ray_address="auto") as p:
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
    save_string = str(count)+save_string
    if continuation==False:
        f1 = open(path+str(count),'w')
    return save_string

def pushbullet_message(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = 'o.YlTBKuQWnkOUsCP9ZxzWC9pvFNz1G0mi'
    resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error',resp.status_code)
    else:
        print ('Message sent')

if WRC_normal==1:
    mu_m = [i for i in range(500,1000)]
    sigma_m = [i for i in range(0,180)]
    batch_size = len(mu_m)
    runs = 200
    continuation = False
    save_string = "WRC"
    save_string = save_data(save_string,continuation)
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
    
    parallel(mumf,mu_m,sigma_m,batch_size=batch_size,save_string=save_string,columns_name=["$\mu_{m}$","$\sigma_{m}$", "success_rate",'avg_pvalue','Wrong_ranking_cost_with_no', 'Wrong_ranking_cost_without_no_proportion','Rate_of_correct_ranking'],continuation=continuation)

    message = 'wrong_ranking_cost_contour' + ' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)


    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_with_no',file_name=save_string+'.csv',save_plot=save_string+'with_no',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options)

    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_="Rate_of_correct_ranking",file_name=save_string+'.csv',save_plot=save_string+'RCR',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options)

    vis.data_visualize(x_var_='$\sigma_{m}$',y_var_="$\mu_{m}$",z_var_="Wrong_ranking_cost_without_no_proportion",file_name=save_string+'.csv',save_plot=save_string+'without_no',plot_type='graphics',cbar_orien='vertical',num_of_opts=number_of_options)

if pval_WRC_normal ==1:
    runs = 1000
    continuation = False
    save_string = "Pval_2D"
    save_string = save_data(save_string,continuation)
    mu_h_1 = 0
    sigma_h_1 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    mu_m = [i for i in range(500,2000,20)]
    number_of_options = [2,5,10,20]
    batch_size = len(mu_m)
    sigma_m_1 = 170

    def mumf(nop,mum,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0):
        loop = 0
        while loop<=runs:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,\
                distribution_h=rng.threshold_n,mu_h=[mu_h_1,mu_h_1],sigma_h=[sigma_h_1,sigma_h_1],mu_x=[mu_x_1,mu_x_1],sigma_x=[sigma_x_1,sigma_x_1],\
                err_type=0,number_of_options=nop,mu_m=[mum,mum],sigma_m=[sigma_m_1,sigma_m_1])
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

        output = {"nop":nop,"$\mu_{m}$": mum,"success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_without_no':avg_incrtness/runs, 'Wrong_ranking_cost_with_no_proportion':avg_incrtness_w_n/runs}
        return output

    parallel(mumf,number_of_options,mu_m,columns_name=["nop","$\mu_{m}$","success_rate",'avg_pvalue','Wrong_ranking_cost_without_no', 'Wrong_ranking_cost_with_no_proportion'],batch_size=batch_size,save_string=save_string)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='avg_pvalue',file_name=save_string+'.csv',save_plot=save_string+'without_no_Pval',plot_type='line',num_of_opts=number_of_options)

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',z1_var_='Wrong_ranking_cost_with_no_proportion',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options+number_of_options)

if bimodal_x_normal_h==1:
    continuation = False
    number_of_opts = [2]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    sigma_h_1 = 1
    sigma_h_2=1
    sigma_x_1=1
    sigma_x_2=1
    runs = 500
    batch_size = 50
    delta_mu = 5
    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    for nop in number_of_opts:
        number_of_options = nop
        save_string ='3delta_mu_5_mu_h_vs_mu_x1_mu_x2_vs_RCDnop2' #'delta_mu_'+str(delta_mu)+'_mu_h_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop)
        # save_string = save_data(save_string,continuation)

        def mux1muh1(muh,mux):
            mux1 = mux
            mux2 = delta_mu + mux
            muh1 = muh
            muh2 = muh
            count = 0
            for k in range(runs):
                success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                    mu_h=[muh1,muh2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mux1,mux2],sigma_x=[sigma_x_1,sigma_x_2],err_type=0,number_of_options=number_of_options,\
                    mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                if success == 1:
                    count += 1
            mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}
            return mu_va

        # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=[number_of_options,number_of_options+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)

if uniform_x_normal_h==1:
    continuation = False
    number_of_opts = [10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    sigma_h_1 = 1
    sigma_h_2=1
    sigma_x_1=1
    sigma_x_2=1
    low_x_1 = -np.sqrt(3)*sigma_x_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
    high_x_1 = np.sqrt(3)*sigma_x_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
    runs = 500
    batch_size = 50
    delta_mu = 0
    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    for nop in number_of_opts:
        number_of_options = nop
        save_string = '19uniform_normal_mu_h_vs_mu_x1_mu_x2_vs_RCDnop10'#'uniform_normal'+'_mu_h_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop)
        # save_string = save_data(save_string,continuation)

        def mux1muh1(muh,mux):
            mux1 = mux + low_x_1
            sigmax1 = mux + high_x_1
            muh1 = muh
            muh2 = muh
            count = 0
            for k in range(runs):
                success,incrt,incrt_w_n,yes_test,max_rat_pval = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_n,\
                    mu_h=[muh1,muh2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
                    mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
                if success == 1:
                    count += 1
            mu_va = {'$\mu_{h_1}$':muh,'$\mu_{h_2}$':muh,'$\mu_{x_1}$': mux,'$\mu_{x_2}$': mux,"success_rate":count/runs}
            return mu_va

        # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=[number_of_options,number_of_options+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)

if uniform_x_uniform_h==1:
    continuation = False
    number_of_opts = [2,5,10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    sigma_h_1 = 1
    sigma_h_2=1
    sigma_x_1=1
    sigma_x_2=1
    low_x_1 = -np.sqrt(3)*sigma_x_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
    high_x_1 = np.sqrt(3)*sigma_x_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
    low_h_1 = -np.sqrt(3)*sigma_h_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
    high_h_1 = np.sqrt(3)*sigma_h_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
    runs = 500
    batch_size = 50
    delta_mu = 5
    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
    for nop in number_of_opts:
        number_of_options = nop
        save_string = 'uniform_uniform'+'_mu_h_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop)
        save_string = save_data(save_string,continuation)

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

        parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=[number_of_options,number_of_options+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)

        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)



check_gaussian = 0
if check_gaussian == 1:
    # [_,p_x,p_fx] = finding_gaussian_base(1.000000000,5,1)
    # [_,n_x,n_fx] = finding_gaussian_base(0.0000000,5,1)
    # [_,intermediate_x,intermediate_fx] = finding_gaussian_base(0.1,5,1)
    # x = [n_x[i] for i in range(len(n_x)-1,-1,-1)]+p_x
    # fx = [n_fx[i] for i in range(len(n_fx)-1,-1,-1)]+p_fx
    # f,ax = plt.subplots()
    # plt.plot(x,fx)
    # x = np.array(x)[:x.index(intermediate_x[-1])]
    # fx = np.array(fx)[:fx.index(intermediate_fx[-1])]
    # ax.fill_between(x,0,fx,facecolor='orange')
    # plt.show()
    number_of_options = [2,5,10,20,100]
    f,ax = plt.subplots(5,5)
    for nop in number_of_options:
        for l in range(5):
            options_quality = quality(number_of_options=nop,x_type=x_type,mu_x_1=5,sigma_x_1=1,mu_x_2=8,sigma_x_2=1)
            options_quality = options_quality.Dx
            y = [1 for i in range(len(options_quality))]
            [_,p1_x,p1_fx] = wf.finding_gaussian_base(1.000000000,5,1)
            [_,n1_x,n1_fx] = wf.finding_gaussian_base(0.0000000,5,1)
            [_,p2_x,p2_fx] = wf.finding_gaussian_base(1.000000000,8,1)
            [_,n2_x,n2_fx] = wf.finding_gaussian_base(0.0000000,8,1)
            x = [n1_x[i] for i in range(len(n1_x)-1,-1,-1)]+p1_x
            fx = [n1_fx[i] for i in range(len(n1_fx)-1,-1,-1)]+p1_fx 
            ax[number_of_options.index(nop),l].plot(x,fx)
            x = [n2_x[i] for i in range(len(n2_x)-1,-1,-1)]+p2_x
            fx = [n2_fx[i] for i in range(len(n2_fx)-1,-1,-1)]+p2_fx
            ax[number_of_options.index(nop),l].plot(x,fx)
            for i in range(len(y)):
                ax[number_of_options.index(nop),l].plot([options_quality[i],options_quality[i]],[0,1])
    plt.show()