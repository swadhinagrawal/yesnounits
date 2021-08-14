#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

from matplotlib import markers
import numpy as np
import classexploration as yn
from multiprocessing import Pool
import pandas as pd
import os
import requests
import json
import random_number_generator as rng
from numba.typed import List
import matplotlib.pyplot as plt

path = os.getcwd() + "/results/"

confidence = 0.02                           #   Confidence for distinguishing qualities

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
normal_x_normal_h = 1
normal_x_normal_h_sigma = 0

wf = yn.workFlow()
vis = yn.Visualization()

def parallel(func,a,b,batch_size,save_string,columns_name,continuation = False):
    ties = []
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
        with Pool(18) as p:#,ray_address="auto") as p:
            opt_var = p.starmap(func,inp[i:i+batch_size])
        out = pd.DataFrame(data=opt_var,columns=columns_name)
        out.to_csv(f_path,mode = 'a',header = False, index=False)
        progress +=1
        print("\r Percent of input processed : {}%".format(np.round(100*progress*batch_size/len(inp)),decimals=1), end="")
        ties.append(np.array(pd.DataFrame(data=opt_var,columns=['ties'])))
    return ties

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
    # resp = requests.post('https://api.pushbullet.com/v2/pushes', 
    #                      data=json.dumps(msg),
    #                      headers={'Authorization': 'Bearer ' + TOKEN,
    #                               'Content-Type': 'application/json'})
    # if resp.status_code != 200:
    #     raise Exception('Error',resp.status_code)
    # else:
    #     print ('Message sent')

if pval_WRC_normal ==1:
    runs = 500
    continuation = False

    mu_h_1 = 0
    sigma_h_1 = 1
    mu_x_1 = 0
    sigma_x_1 = 1
    save_string = 'gxgh_Pval_2D_sx='+str(np.round(sigma_x_1/sigma_h_1,decimals=1))+'sh('+str(sigma_h_1)+')'
    save_string = save_data(save_string,continuation)
    mu_m = [i for i in range(50,1500,200)]
    number_of_options = [2,5,10]
    batch_size = len(mu_m)
    sigma_m_1 = 0

    def mumf(nop,mum,count = 0,avg_pval = 0,avg_incrtness = 0,avg_incrtness_w_n = 0):
        loop = 0
        ties1 = np.zeros((nop,2))
        while loop<=runs:
            success,incrt_w_n,max_rat_pval,ties = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,\
                distribution_h=rng.threshold_n,mu_h=List([mu_h_1,mu_h_1]),sigma_h=List([sigma_h_1,sigma_h_1]),mu_x=List([mu_x_1,mu_x_1]),sigma_x=List([sigma_x_1,sigma_x_1]),\
                err_type=0,number_of_options=nop,mu_m=List([mum,mum]),sigma_m=List([sigma_m_1,sigma_m_1]))

            avg_pval += max_rat_pval

            # avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
            loop += 1
            ties1 += ties
        ties1 = np.array(ties1/runs)
        output = {"nop":nop,"$\mu_{m}$": mum,"success_rate":count/runs,'avg_pvalue':avg_pval/runs,'Wrong_ranking_cost_without_no':avg_incrtness_w_n/runs,'ties':ties1} #, 'Wrong_ranking_cost_with_no_proportion':avg_incrtness/runs
        return output

    ties = parallel(mumf,number_of_options,mu_m,columns_name=["nop","$\mu_{m}$","success_rate",'avg_pvalue','Wrong_ranking_cost_without_no', 'Wrong_ranking_cost_with_no_proportion'],batch_size=batch_size,save_string=save_string)

    for i in range(len(np.array(ties))):
        for ti in range(len(np.array(ties)[i])):
            plt.plot(np.array(ties)[i][0][0][:,0],np.array(ties)[i][ti][0][:,1],label = str(mu_m[ti]),marker='o')
    plt.xlabel('Option ranks',fontsize = 14)
    plt.ylabel('Average P-value',fontsize = 14)
    plt.legend(title = r"$\mu_m$")
    plt.show()

    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='avg_pvalue',file_name=save_string+'.csv',save_plot=save_string+'without_no_Pval',plot_type='line',num_of_opts=number_of_options)
    vis.data_visualize(y_var_="nop",x_var_="$\mu_{m}$",z_var_='Wrong_ranking_cost_without_no',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=number_of_options)

if normal_x_normal_h==1:
    continuation = False
    number_of_opts = [2,5,10]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    
    
    sigma_x_1 = 1
    sigma_x_2= 1
    sigma_h_1=sigma_x_1
    sigma_h_2=sigma_x_1
    runs = 500
    batch_size = 50
    delta_mu = 0
    mu_x = [np.round(i*0.1,decimals=1) for i in range(10)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(5)]
    cnt = 50
    for nop in number_of_opts:
        number_of_options = nop
        save_string = 'gxgh_sx='+str(np.round(sigma_x_1/sigma_h_1,decimals=1))+'_sh('+str(sigma_h_1)+')_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop) # str(cnt)+
        save_string = save_data(save_string,continuation)

        def mux1muh1(muh,mux,avg_pval = 0,avg_incrtness_w_n = 0):
            mux1 = mux
            mux2 = delta_mu + mux
            muh1 = muh
            muh2 = muh
            count = 0
            ties1 = np.zeros((number_of_options,2))
            for k in range(runs):
                success,incrt_w_n,max_rat_pval,ties = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
                    mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
                    mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
                if success == 1:
                    count += 1
                avg_pval += max_rat_pval
                avg_incrtness_w_n += incrt_w_n
                ties1 += ties
            ties1 = np.array(ties1/runs)
            mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs,'Average p-value':avg_pval/runs,'Wrong_ranking_cost_without_no_proportion':avg_incrtness_w_n/runs,'ties':ties1}
            return mu_va

        ties = parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate",'Average p-value','Wrong_ranking_cost_without_no_proportion'],save_string=save_string,batch_size=3*len(mu_h))
        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)
    
        plt.plot(np.array(ties)[0][0][0][:,0],np.array(ties)[0][0][0][:,1],label= str(mu_m_1),marker='o')
        plt.xlabel('Option ranks',fontsize = 14)
        plt.ylabel('Average P-value',fontsize = 14)
        plt.legend(title = r"$\mu_m$")
        plt.show()
        
        vis.data_visualize(y_var_="$\mu_{x_1}$",x_var_="$\mu_{x_1}$",z_var_='Average p-value',file_name=save_string+'.csv',save_plot=save_string+'Pval',plot_type='line',num_of_opts=[nop])
        vis.data_visualize(y_var_="$\mu_{x_1}$",x_var_="$\mu_{x_1}$",z_var_='Wrong_ranking_cost_without_no_proportion',file_name=save_string+'.csv',save_plot=save_string+'WRC',plot_type='line',num_of_opts=[nop])
    
        message = str(nop)+' number of options simulation finished'
        pushbullet_message('Python Code','Results out! '+message)
        cnt += 1
