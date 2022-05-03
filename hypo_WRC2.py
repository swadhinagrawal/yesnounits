#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import classexploration2 as yn
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
import random
from matplotlib.legend_handler import HandlerTuple

# import ray
#from ray.util.multiprocessing import Pool

#ray.init(address='auto', _redis_password='5241590000000000')
# path = os.getcwd() + "/results1/"
# path = os.getcwd() + "/results_new/"
# path = os.getcwd() + "/results_sigma/"
# path = os.getcwd() + "/mu_h=mu_x_s_h_vs_s_q/"
# path1 = os.getcwd() + "/mu_h=mu_h_pred_s_h_vs_sq/"
# path = os.getcwd() + "/results_paper/"
# path = os.getcwd() + "/results_compare_best_two/"
# path = os.getcwd() + "/sigma_sigma/"
# path1 = os.getcwd() + "/sig_mu_imp/"
# path = os.getcwd() + "/F7/"

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
bimodal_x_normal_h_plotter = 0
uniform_x_normal_h_plotter = 0
normal_x_normal_h_plotter = 0
normal_x_normal_h_sigma_plotter = 0
Fig1 = 0
bimodal_x_normal_h_vary_sig = 0
uniform_x_normal_h_vary_sig = 0
normal_x_normal_h_vary_sig = 0
normal_x_normal_h_compare_best_two = 0

wf = yn.workFlow()
vis = yn.Visualization()
# fontproperties = {'weight' : 'bold', 'size' : 40}
fontproperties = {'weight' : 'bold', 'size' : 18}
prd = yn.Prediction()


# initials = "mu_h=mu_x"
# initials = "mu_h=mu_h_pred"
# initials = "sig_q_vs_mu_q_vs_mu_h"
# initials = "mu_q_vs_mu_h"
# initials = 'sigma_h=sigma_x'
# initials = 'sigma_h_vs_sigma_x_mu_h_pred'
# initials = 'sigma_h_vs_sigma_x_mu_x'
# initials = 'sigma_h_vs_sigma_x_mu_x_1_'
# initials2 = 'HARS_mu_h_pred_1_'
# initials = 'HARS_sigma_h_pred_1_'
# initials1 = 'sigma_h_vs_sigma_x_mu_h_pred_1_'

# initials = 'Hars_Dq_'
# initials = 'Hars_sh*_'
# initials = 'Hars_mh*_'
# initials = 'Hars_Dh*_'
# initials1 = 'Hars_Dq_'
# initials2 = 'Hars_mh*_'
# initials3 = 'Hars_sh*_'

# ylab2 = r'$\bf \mu_h\;vs\;\mu_q$'
# ylab1 = r'$\bf \sigma_h\;vs\;\sigma_q$'
# # bxgh_string = initials+'_bxgh_new_optimization'
# # uxgh_string = initials+'_uxgh_new_optimization'
# # gxgh_string = initials+'_gxgh_new_optimization'
# bxgh_string = initials+'_bxgh_full'
# uxgh_string = initials+'_uxgh_full'
# gxgh_string = initials+'_gxgh_full'

# bxgh_string1 = initials1+'_bxgh_full'
# uxgh_string1 = initials1+'_uxgh_full'
# gxgh_string1 = initials1+'_gxgh_full'

# bxgh_string2 = initials2+'_bxgh_full'
# uxgh_string2 = initials2+'_uxgh_full'
# gxgh_string2 = initials2+'_gxgh_full'

# bxgh_string3 = initials2+'_bxgh_full'
# uxgh_string3 = initials2+'_uxgh_full'
# gxgh_string3 = initials2+'_gxgh_full'
# bxgh_string = initials+'_bxgh_1'
# uxgh_string = initials+'_uxgh_1'
# gxgh_string = initials+'_gxgh_1'

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

def save_data(save_string,continuation=False):
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

def read_slopes_HARS_1(f_path,columns_name):
    op = pd.read_csv(f_path)
    opt_var = []
    for j in range(len(op['$\mu_{m}$'])):
        a = {}
        for i in op:
            a[str(i)] = op[str(i)][j]
        opt_var.append(a)

    data1 = [[],[],[],[],[],[],[],[]]
    for i in opt_var:
        for j in range(len(columns_name)):
            data1[j].append(i[columns_name[j]])
    
    x = np.array(data1[0])
    y = np.array(data1[1])
    z = np.array(data1[2])
    dx = np.array(data1[3])
    dy = np.array(data1[4])
    dz = np.array(data1[5])
    trans = np.array(data1[6])
    inte = np.array(data1[7])
    return x,y,z,dx,dy,dz,trans,inte

def read_slopes_HARS(f_path,columns_name):
    op = pd.read_csv(f_path)
    opt_var = []
    for j in range(len(op['$\mu_{m}$'])):
        a = {}
        for i in op:
            a[str(i)] = op[str(i)][j]
        opt_var.append(a)

    data1 = [[],[],[],[],[],[],[],[]]
    for i in opt_var:
        for j in range(len(columns_name)):
            data1[j].append(i[columns_name[j]])
    
    x = np.array(data1[0])
    y = np.array(data1[1])
    z = np.array(data1[2])
    dx = np.array(data1[3])
    dy = np.array(data1[4])
    dz = np.array(data1[5])
    trans = np.array(data1[6])
    inte = np.array(data1[7])
    return x,y,z,dx,dy,dz,trans,inte


def save_plot_data_1(mum_slopes,mum,num_opts,save_string=None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    f = open(path+save_string+'.csv','a')
    columns = pd.DataFrame(data=np.array([columns_name]))
    columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)
    x = []
    y = []
    dz = []
    trans = []
    dx = []
    dy = []
    intercept = []
    for i in range(len(mum_slopes)):
        for j in range(len(mum_slopes[0])):
            if i==0:
                dx.append(0.3*mum[i])
            else:
                dx.append(0.3*(mum[i]-mum[i-1]))
            x.append(mum[i])
            if j==0:
                dy.append(0.3*num_opts[j])
            else:
                dy.append(0.3*(num_opts[j]-num_opts[j-1]))
            y.append(num_opts[j])
            dz.append(mum_slopes[i,j,0])
            trans.append(mum_slopes[i,j,1])
            intercept.append(mum_slopes[i,j,2])
            # trans.append(1)
    z = np.zeros_like(dz)
    
    y = np.array(y)
    data = []
    for i in range(len(dz)):
        data.append({'$\mu_{m}$':x[i],'n':y[i],'z':z[i],'dx':dx[i],'dy':dy[i],'Slope of bestfit':dz[i],'HARS':trans[i],"Intercept":intercept[i]})
    out = pd.DataFrame(data=data,columns=columns_name)
    out.to_csv(f_path,mode = 'a',header = False, index=False)

def save_plot_data(analyzed_data,mum,num_opts,sig_q=None,save_string=None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','$\sigma_{q}$','slope_fit','y_intercept_fit','fit '+ r'$\mu_{HARS}$','slope*','y_intercept*','* '+ r'$\mu_{HARS}$']
    f = open(path+save_string+'.csv','a')
    columns = pd.DataFrame(data=np.array([columns_name]))
    columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

    data = []
    for i in range(len(analyzed_data)):
        for j in range(len(analyzed_data[0])):
            if isinstance(sig_q,type(None)):
                data.append({'$\mu_{m}$':mum[i],'n':num_opts[j],'$\sigma_{q}$':None,'slope_fit':analyzed_data[i][j][0][0][0],'y_intercept_fit':analyzed_data[i][j][0][0][1],'fit '+ r'$\mu_{HARS}$':analyzed_data[i][j][0][1],'slope*':analyzed_data[i][j][1][0][0],'y_intercept*':analyzed_data[i][j][1][0][1],'* '+ r'$\mu_{HARS}$':analyzed_data[i][j][1][1]})
            else:
                data.append({'$\mu_{m}$':mum[i],'n':num_opts[j],'$\sigma_{q}$':sig_q[j],'slope_fit':analyzed_data[i][j][0][0][0],'y_intercept_fit':analyzed_data[i][j][0][0][1],'fit '+ r'$\mu_{HARS}$':analyzed_data[i][j][0][1],'slope*':analyzed_data[i][j][1][0][0],'y_intercept*':analyzed_data[i][j][1][0][1],'* '+ r'$\mu_{HARS}$':analyzed_data[i][j][1][1]})
    out = pd.DataFrame(data=data,columns=columns_name)
    out.to_csv(f_path,mode = 'a',header = False, index=False)

def save_fitting_data(save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','$\sigma_{q}$','slope_fit','y_intercept_fit','fit '+ r'$\mu_{HARS}$','slope*','y_intercept*','* '+ r'$\mu_{HARS}$']
    f = open(f_path,'a')
    columns = pd.DataFrame(data=np.array([columns_name]))
    columns.to_csv(f_path,mode='a',header=False,index=False)
    return columns_name,f_path
 

def plot_3d(save_string=None,ax=None,distribution=None,color="orange",one=0,ylab = None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if one:
        num_opts = np.unique(y)
        x = np.array(x)[:int(len(num_opts))]
        y = np.array(y)[:int(len(num_opts))]
        z = np.array(z)[:int(len(num_opts))]
        dx = np.array(dx)[:int(len(num_opts))]
        dy = np.array(dy)[:int(len(num_opts))]
        dz = np.array(dz)[:int(len(num_opts))]
        trans = np.array(trans)[:int(len(num_opts))]

    
    x1 = np.array(x)-np.array(dx)*distribution
    for i in range(len(dz)):
        bars = ax.bar3d(x1[i], y[i], z[i], dx[i], dy[i], dz[i],alpha=trans[i],color=color,shade=True)
    
    y1 = np.unique(y)
    x2 = np.unique(x1)
    for i in range(len(x2)):
        x3 = [x2[i] for j in range(len(y1))]
        # linear_fit = np.polyfit(y1,np.log(dz[i*len(y1):(i+1)*len(y1)]),2)
        # ax.plot(x3,y1,np.exp(np.array(y1)*np.array(y1)*linear_fit[0]+np.array(y1)*linear_fit[1] + linear_fit[2]),linewidth=1,color = "black")
        # linear_fit = np.polyfit(y1,dz[i*len(y1):(i+1)*len(y1)],1)
        # ax.plot(x3,y1,np.array(y1)*linear_fit[0] + linear_fit[1],linewidth=1,color = "black")

    ax.set_xlabel(r'$\mu_m$')
    ax.set_ylabel('Number of options')
    ax.set_zlabel('slope  in '+ylab1)
    num_opts = np.unique(y)
    mum = np.unique(x)
    plt.xticks(mum)
    plt.yticks(num_opts)

def plot_2d(save_string=None,ax=None,distribution=None,color="orange",one=0,ylab = None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept'] # dz = slope of the best fit
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if one:
        num_opts = np.unique(y)
        x = np.array(x)[:int(len(num_opts))]
        y = np.array(y)[:int(len(num_opts))]
        z = np.array(z)[:int(len(num_opts))]
        dx = np.array(dx)[:int(len(num_opts))]
        dy = np.array(dy)[:int(len(num_opts))]
        dz = np.array(dz)[:int(len(num_opts))]
        trans = np.array(trans)[:int(len(num_opts))]

    
    x1 = np.array(x)-np.array(dx)*distribution
    for i in range(len(dz)):
        ax.bar((0.03*(distribution-1)+np.log10(y[i])),dz[i],alpha=trans[i],color=color,width=0.03)

    
    # y1 = np.unique(y)
    # x2 = np.unique(x1)
    # for i in range(len(x2)):
    #     x3 = [x2[i] for j in range(len(y1))]
    #     # linear_fit = np.polyfit(y1,np.log(dz[i*len(y1):(i+1)*len(y1)]),2)
    #     # ax.plot(x3,y1,np.exp(np.array(y1)*np.array(y1)*linear_fit[0]+np.array(y1)*linear_fit[1] + linear_fit[2]),linewidth=1,color = "black")
    #     # linear_fit = np.polyfit(y1,dz[i*len(y1):(i+1)*len(y1)],1)
    #     # ax.plot(x3,y1,np.array(y1)*linear_fit[0] + linear_fit[1],linewidth=1,color = "black")
    num_opts = np.unique(y)
    mum = np.unique(x)

    xf = np.zeros(len(num_opts))
    yf = np.zeros(len(num_opts))

    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(dz[i*len(num_opts)+j])
        xf += np.array(x1)
        yf += np.array(y1)
    xf /= len(mum)
    yf /= len(mum)

    log_fit = np.polyfit(np.log10(xf),yf,1)
    ax.plot(0.03*(distribution-1)+np.log10(xf),(np.log10(xf)*log_fit[0]+log_fit[1]),linewidth=1,color = color)
    

    ax.set_xlabel('Number of options (n) (in log scale)',fontproperties)
    ax.set_ylabel('slope  in '+ylab1,fontproperties)
    num_opts = np.unique(y)
    num_opts1 = [np.log10(i) for i in num_opts]
    mum = np.unique(x)
    ax.tick_params(labelsize=10)
    ax.set_xticks(num_opts1)
    ax.set_xticklabels(num_opts)
    ax.set_title(r'$\bf \mu_m$'+' = '+str(x[0]),fontproperties,color=(0.3,0.3,0.3,1))
    return log_fit

def plot_2d_slope_error(save_string=None,save_string1=None,ax=None,distribution=None,color="orange",one=0,ylab = None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if one:
        num_opts = np.unique(y)
        x = np.array(x)[:int(len(num_opts))]
        y = np.array(y)[:int(len(num_opts))]
        z = np.array(z)[:int(len(num_opts))]
        dx = np.array(dx)[:int(len(num_opts))]
        dy = np.array(dy)[:int(len(num_opts))]
        dz = np.array(dz)[:int(len(num_opts))]
        trans = np.array(trans)[:int(len(num_opts))]
    
    f_path1 = path+save_string1+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x_1,y_1,z_1,dx_1,dy_1,dz_1,trans_1,inte_1 = read_slopes_HARS(f_path1,columns_name)
    
    if one:
        num_opts = np.unique(y)
        x_1 = np.array(x_1)[:int(len(num_opts))]
        y_1 = np.array(y_1)[:int(len(num_opts))]
        z_1 = np.array(z_1)[:int(len(num_opts))]
        dx_1 = np.array(dx_1)[:int(len(num_opts))]
        dy_1 = np.array(dy_1)[:int(len(num_opts))]
        dz_1 = np.array(dz_1)[:int(len(num_opts))]
        trans_1 = np.array(trans_1)[:int(len(num_opts))]

    
    for i in range(len(dz_1)):
        ax.bar((0.3*(distribution-1)+y[i]),(dz[i] - dz_1[i]),alpha=1-(trans[i]-trans_1[i]),color=color,width=0.3)

    ax.set_xlabel('Number of options (n)',fontproperties)
    ax.set_ylabel(r'$\bf \Delta$'+' slope  in '+ylab1,fontproperties)
    num_opts = np.unique(y)
    mum = np.unique(x)
    plt.title(r'$\bf \mu_m$'+' = '+str(x[0]),fontproperties,color=(0.3,0.3,0.3,1))
    plt.xticks(num_opts,fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')

def plot_3d_inter(save_string=None,ax=None,distribution=None,color="orange",one=0,ylab = None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
        
    if one:
        num_opts = np.unique(y)
        x = np.array(x)[:int(len(num_opts))]
        y = np.array(y)[:int(len(num_opts))]
        z = np.array(z)[:int(len(num_opts))]
        dx = np.array(dx)[:int(len(num_opts))]
        dy = np.array(dy)[:int(len(num_opts))]
        dz = np.array(dz)[:int(len(num_opts))]
        trans = np.array(trans)[:int(len(num_opts))]
        inte = np.array(inte)[:int(len(num_opts))]
        
    x1 = np.array(x)-np.array(dx)*distribution
    for i in range(len(dz)):
        ax.bar3d(x1[i], y[i], z[i], dx[i], dy[i], inte[i],alpha=trans[i],color=color,shade=True)
    
    y1 = np.unique(y)
    x2 = np.unique(x1)
    for i in range(len(x2)):
        x3 = [x2[i] for j in range(len(y1))]
        # linear_fit = np.polyfit(y1,np.log(dz[i*len(y1):(i+1)*len(y1)]),2)
        # ax.plot(x3,y1,np.exp(np.array(y1)*np.array(y1)*linear_fit[0]+np.array(y1)*linear_fit[1] + linear_fit[2]),linewidth=1,color = "black")
        # linear_fit = np.polyfit(y1,dz[i*len(y1):(i+1)*len(y1)],1)
        # ax.plot(x3,y1,np.array(y1)*linear_fit[0] + linear_fit[1],linewidth=1,color = "black")

    ax.set_xlabel(r'$\mu_m$')
    ax.set_ylabel('Number of options')
    ax.set_zlabel('Intercept in '+ylab1)
    num_opts = np.unique(y)
    mum = np.unique(x)
    plt.xticks(mum)
    plt.yticks(num_opts)

def plot_2d_inter(save_string=None,ax=None,distribution=None,color="orange",one=0,ylab = None,dist = None):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
        
    if one:
        num_opts = np.unique(y)
        x = np.array(x)[:int(len(num_opts))]
        y = np.array(y)[:int(len(num_opts))]
        z = np.array(z)[:int(len(num_opts))]
        dx = np.array(dx)[:int(len(num_opts))]
        dy = np.array(dy)[:int(len(num_opts))]
        dz = np.array(dz)[:int(len(num_opts))]
        trans = np.array(trans)[:int(len(num_opts))]
        inte = np.array(inte)[:int(len(num_opts))]
        
    x1 = np.array(x)-np.array(dx)*distribution
    for i in range(len(dz)):
        ax.bar((0.03*(distribution-1)+np.log10(y[i])),inte[i],alpha=trans[i],color=color,width=0.03)
    
    # y1 = np.unique(y)
    # x2 = np.unique(x1)
    # for i in range(len(x2)):
    #     x3 = [x2[i] for j in range(len(y1))]
    #     # linear_fit = np.polyfit(y1,np.log(dz[i*len(y1):(i+1)*len(y1)]),2)
    #     # ax.plot(x3,y1,np.exp(np.array(y1)*np.array(y1)*linear_fit[0]+np.array(y1)*linear_fit[1] + linear_fit[2]),linewidth=1,color = "black")
    #     # linear_fit = np.polyfit(y1,dz[i*len(y1):(i+1)*len(y1)],1)
    #     # ax.plot(x3,y1,np.array(y1)*linear_fit[0] + linear_fit[1],linewidth=1,color = "black")
    
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(num_opts))
    yf = np.zeros(len(num_opts))

    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(inte[i*len(num_opts)+j])
        xf += np.array(x1)
        yf += np.array(y1)
    xf /= len(mum)
    yf /= len(mum)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'['+str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2))+']')
    # ax.text(20,min(inte),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    ax.plot(0.03*(distribution-1)+np.log10(xf),(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=1,color = color)
    
    ax.set_xlabel('Number of options (n) (in log scale)',fontproperties)
    ax.set_ylabel('Intercept in '+ylab1,fontproperties)

    plt.title(r'$\bf \mu_m$'+' = '+str(x[0]),fontproperties,color=(0.3,0.3,0.3,1))
    num_opts1 = [np.log10(i) for i in num_opts]
    ax.set_xticks(num_opts1)
    ax.set_xticklabels(num_opts)
    ax.set_title(r'$\bf \mu_m$'+' = '+str(x[0]),fontproperties,color=(0.3,0.3,0.3,1))
    return log_fit
    
def plot_slopes(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'
    num_opts = np.unique(y)
    mum = np.unique(x)
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    xf = np.zeros(len(mum))
    yf = np.zeros(len(mum))
    for i in range(len(num_opts)):
        x1 = []
        y1 = []
        for j in range(len(mum)):
            x1.append(mum[j])
            y1.append(dz[i+j*len(num_opts)])
        xf += np.array(x1)
        yf += np.array(y1)
        ax.scatter(x1,y1,color = color,s=10,marker=marker[i])
    xf /= len(num_opts)
    yf /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*np.round(linear_fit[0],decimals=2) + np.round(linear_fit[1],decimals=2),linewidth=1,color = color,label=distribution + '['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # ax.text(400,min(dz),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))

    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(20,min(dz),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=1,color = color)
    
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    return log_fit

def plot_inter(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(mum))
    yf = np.zeros(len(mum))
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    for i in range(len(num_opts)):
        x1 = []
        y1 = []
        for j in range(len(mum)):
            x1.append(mum[j])
            y1.append(inte[i+j*len(num_opts)])
        xf += np.array(x1)
        yf += np.array(y1)
        ax.scatter(x1,y1,color = color,marker=marker[i],s=10)#,label=str(num_opts[i])+'_'+distribution)
    xf /= len(num_opts)
    yf /= len(num_opts)
    linear_fit = np.polyfit(xf,yf,1)
    ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color)#,label=distribution+'['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # ax.text(400,min(inte),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
    # ax.plot(xf,np.exp(np.array(xf)*exp_fit[0]+exp_fit[1]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    return linear_fit

def plot_slopes_n(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(num_opts))
    yf = np.zeros(len(num_opts))
    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(dz[i*len(num_opts)+j])
        xf += np.array(x1)
        yf += np.array(y1)
        # ax.plot(x1,y1,linewidth=1,color = color,marker=marker[i],label=str(mum[i])+'_'+distribution)
        ax.scatter(x1,y1,color = color,s=10,marker=marker[i])#,label=str(mum[i])+'_'+distribution)
        # linear_fit = np.polyfit(x1,y1,1)
        # ax.plot(x1,np.array(x1)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,marker=marker[i],label=str(mum[i])+'_'+distribution,linestyle='-.')
        # linear_fit = np.polyfit(x1,np.log(np.array(y1)),2)
        # ax.plot(x1,np.exp(np.array(x1)*np.array(x1)*linear_fit[0]+np.array(x1)*linear_fit[1] + linear_fit[2]),linewidth=1,color = color,marker=marker[i],label=str(mum[i])+'_'+distribution,linestyle='-.')
    xf /= len(mum)
    yf /= len(mum)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color)#,label=distribution+'[ %.2fn %+.2f'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # linear_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*linear_fit[0]+np.array(xf)*linear_fit[1] + linear_fit[2]),linewidth=1,color = color,label=distribution)
    # ax.text(20,min(dz),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))

    if distribution==2:
        exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
        ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color)
    else:
        exp_fit = np.polyfit(np.log10(xf),yf,2)
        ax.plot(xf,(np.log10(xf)*np.log10(xf)*exp_fit[0]+np.log10(xf)*exp_fit[1]+exp_fit[2]),linewidth=1,color = color)
    
    return exp_fit

def plot_inte_n(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(num_opts))
    yf = np.zeros(len(num_opts))

    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(inte[i*len(num_opts)+j])
        xf += np.array(x1)
        yf += np.array(y1)

        ax.scatter(x1,y1,color = color,marker=marker[i],s=10)#,label=str(mum[i])+'_'+distribution)
    xf /= len(mum)
    yf /= len(mum)
    # ax.plot(xf,yf,linewidth=1,color = 'black')#,label=distribution+'['+r'$\bf e^{%.2fn^{2} %+.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2),np.round(exp_fit[3],decimals=2))+']')
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'['+str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2))+']')
    # ax.text(20,min(inte),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),3)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*np.array(xf)*exp_fit[0] + np.array(xf)*np.array(xf)*exp_fit[1] + np.array(xf)*exp_fit[2] + exp_fit[3]),linewidth=1,color = color)
    exp_fit = np.polyfit(1/np.array(xf),np.array(yf),3)
    ax.plot(xf,(1/(np.array(xf)*np.array(xf)*np.array(xf)))*exp_fit[0] + (1/(np.array(xf)*np.array(xf)))*exp_fit[1] + (1/(np.array(xf)))*exp_fit[2] + exp_fit[3],linewidth=1,color = color)

    return exp_fit
    
def plot_HARS(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(mum))
    yf = np.zeros(len(mum))
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    for i in range(len(num_opts)):
        x1 = []
        y1 = []
        for j in range(len(mum)):
            x1.append(mum[j])
            y1.append(trans[i+j*len(num_opts)])
        xf += np.array(x1)
        yf += np.array(y1)

        ax.scatter(x1,y1,color = color,marker=marker[i],s=10)#,label=str(num_opts[i])+'_'+distribution)
    xf /= len(num_opts)
    yf /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'[ %.2fx %+.2f'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(400,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=1,color = color)#,label=distribution+'['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    return log_fit

def plot_HARS_n(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    marker=['o','s','*','D','X','p','d','v','^']
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'
    num_opts = np.unique(y)
    mum = np.unique(x)
    xf = np.zeros(len(num_opts))
    yf = np.zeros(len(num_opts))
    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(trans[i*len(num_opts)+j])
        xf += np.array(x1)
        yf += np.array(y1)
        ax.scatter(x1,y1,color = color,marker=marker[i],s=10)#,label=str(mum[i])+'_'+distribution)
    xf /= len(mum)
    yf /= len(mum)
    exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color)#,label=distribution+'['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(20,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))

    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    return exp_fit

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def genericPlotter1(ax,distribution,color,save_string,function,fit_type,first_variable,vs_n = 0,predicted = 0,plot_against_sig=0):
    f_path = path+save_string+'.csv'
    file = pd.read_csv(f_path)
    mum = np.unique(file['$\mu_{m}$'])
    n = np.unique(file['n'])
    
    
    if plot_against_sig:
        sig_q = np.unique(file['$\sigma_{q}$'])
        n = sig_q
        slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
        slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    else:
        slope_fit = np.array(file['slope_fit']).reshape((len(n),len(mum)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(n),len(mum)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(n),len(mum)))
        slope_predicted = np.array(file['slope*']).reshape((len(n),len(mum)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(n),len(mum)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(n),len(mum))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    

    if function=="slope_fit":
        variable = slope_fit
    elif function == "y_intercept_fit":
        variable = y_intercept_fit
    elif function == "mu_hars_fit":
        variable = mu_hars_fit
    elif function == "slope_predicted":
        variable = slope_predicted
    elif function == "y_intercept_predicted":
        variable = y_intercept_predicted
    elif function == "mu_hars_predicted":
        variable = mu_hars_predicted
    elif function == 'distances':
        variable = y_intercept_predicted - y_intercept_fit
    
    if predicted ==1:
        predicted_intercepts = []
        predicted_slope = []
        if distribution==1:
            mu = List([0,5])
            sigma = List([1,1])
        else:
            mu = List([0,0])
            sigma = List([1,1])
        
        if distribution==1:
            mu1 = List([5,10])
            sigma1 = List([1,1])
        else:
            mu1 = List([5,5])
            sigma1 = List([1,1])
        
        if distribution==2:
            distribution_fn = prd.uniform
        else:
            distribution_fn = prd.gaussian

        step = 0.0001                                           #   Resolution of the PDF
        start = np.sum(mu)/len(mu) - np.sum(sigma)-5            #   Start point of PDF (left tail end)
        stop = np.sum(mu)/len(mu) + np.sum(sigma)+5              #   End point of PDF (right tail end)
        dis_x = np.round(np.arange(start,stop,step),decimals=4)   #   Base axis of the PDF
        pdf =  distribution_fn(dis_x,mu,sigma)                          #   PDF function values at each base values (array)
        area = (np.sum(pdf)*step)                               #   Area under the PDF
        pdf = np.multiply(pdf,1/area)                #   Normalized PDF

        start1 = np.sum(mu1)/len(mu1) - np.sum(sigma1)-5            #   Start point of PDF (left tail end)
        stop1 = np.sum(mu1)/len(mu1) + np.sum(sigma1)+5              #   End point of PDF (right tail end)
        dis_x1 = np.round(np.arange(start1,stop1,step),decimals=4)   #   Base axis of the PDF
        pdf1 =  distribution_fn(dis_x1,mu1,sigma1)                          #   PDF function values at each base values (array)
        area1 = (np.sum(pdf1)*step)                               #   Area under the PDF
        pdf1 = np.multiply(pdf1,1/area1)                #   Normalized PDF
        
        for i in n:
            _1 = prd.ICPDF(1-(1/i),mu,stop,step,dis_x,pdf)
            _2 = prd.ICPDF(1-(1/i),mu1,stop1,step,dis_x1,pdf1)
            if distribution==1:
                predicted_intercepts.append(_1-2.5)
            else:
                predicted_intercepts.append(_1)
            if 'slope' in function:
                predicted_slope.append((_2 - _1)/5)
        if 'intercept' in function:
            ax.plot(n,predicted_intercepts,linewidth=3,color = color)
        elif 'slope' in function:
            ax.plot(n,predicted_slope,linewidth=3,color = color)
    if predicted ==2:
        predicted_intercepts = []
        
        for i in n[1:]:
            if distribution==1:
                mu = List([0,5])
                sigma = List([i,i])
            else:
                mu = List([0,0])
                sigma = List([i,i])
            
            if distribution==2:
                distribution_fn = prd.uniform
            else:
                distribution_fn = prd.gaussian

            step = 0.0001                                           #   Resolution of the PDF
            start = np.sum(mu)/len(mu) - np.sum(sigma)-5            #   Start point of PDF (left tail end)
            stop = np.sum(mu)/len(mu) + np.sum(sigma)+5              #   End point of PDF (right tail end)
            dis_x = np.round(np.arange(start,stop,step),decimals=4)   #   Base axis of the PDF
            pdf =  distribution_fn(dis_x,mu,sigma)                          #   PDF function values at each base values (array)
            area = (np.sum(pdf)*step)                               #   Area under the PDF
            pdf = np.multiply(pdf,1/area)                #   Normalized PDF
        
            _1 = prd.ICPDF(1-(1/5),mu,stop,step,dis_x,pdf)
            if distribution==1:
                predicted_intercepts.append(_1-2.5)
            else:
                predicted_intercepts.append(_1)
        # if function == 'y_intercept_fit' and vs_n==1 and predicted ==1 :
        ax.plot(n[1:],predicted_intercepts,linewidth=3,color = color)

    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'

    if first_variable == "n":
        first_var = n
        second_var = mum
    else:
        first_var = mum
        second_var = n
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    xf = np.zeros(len(second_var))
    yf = np.zeros(len(second_var))

    for i in range(len(first_var)):
        x1 = []
        y1 = []
        for j in range(len(second_var)):
            x1.append(second_var[j])
            # index = i+j*len(n)
            if vs_n:
                if distribution==1 and function == "y_intercept_fit" and not plot_against_sig:
                    y1.append(variable[j,i]-2.5)
                elif distribution==1 and plot_against_sig:
                    y1.append(variable[i,j]-2.5)
                else:
                    if not plot_against_sig:
                        y1.append(variable[j,i])
                    else:
                        y1.append(variable[i,j])
                # index = i*len(n)+j
            # y1.append(variable[i,j])
            else:
                y1.append(variable[i,j])

        xf += np.array(x1)
        yf += np.array(y1)
        ax.scatter(x1,y1,s=40,marker=marker[i], facecolors='none', edgecolors=color)
    xf /= len(first_var)
    yf /= len(first_var)
    # ax.plot(xf,yf, color=color,alpha = 0.3,linewidth=4)
    # ax.plot(xf,yf, color=color,linewidth=8)

    if fit_type == "log":
        log_fit = np.polyfit(np.log10(xf),yf,1)
        ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=2,color = color)
        return_var = log_fit
    elif fit_type == "linear":
        linear_fit = np.polyfit(xf,yf,1)
        ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=2,color = color)
        return_var = linear_fit
    elif fit_type == "exp":
        if distribution==2:
            exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
            ax.plot(xf,np.exp(np.array(xf)*exp_fit[0] + exp_fit[1]),linewidth=2,color = color)
        else:
            exp_fit = np.polyfit(np.log10(xf),yf,1)
            ax.plot(xf,(np.log10(xf)*exp_fit[0]+exp_fit[1]),linewidth=2,color = color)
        return_var = exp_fit
    elif fit_type == "n_exp":
        exp_fit = np.polyfit(1/np.array(xf),np.array(yf),3)
        ax.plot(xf,(1/(np.array(xf)*np.array(xf)*np.array(xf)))*exp_fit[0] + (1/(np.array(xf)*np.array(xf)))*exp_fit[1] + (1/(np.array(xf)))*exp_fit[2] + exp_fit[3],linewidth=2,color = color,linestyle='-.')
        return_var = exp_fit
    else:
        return_var = None
    
    
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    # ax.set_ylim([0, 1])
    return return_var

def genericPlotter(ax,distribution,color,f_path,function,fit_type,first_variable,vs_n = 0,predicted = 0,plot_against_sig=0,point_facecolor='none',line_style='-.',f_path1=None,marker_line=None):
    file = pd.read_csv(f_path)
    mum = np.unique(file['$\mu_{m}$'])
    n = np.unique(file['n'])
    
    if plot_against_sig:
        sig_q = np.unique(file['$\sigma_{q}$'])
        n = sig_q
        slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
        slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    else:
        slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
        slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if not isinstance(f_path1,type(None)):
        file = pd.read_csv(f_path1)

        if plot_against_sig:
            sig_q = np.unique(file['$\sigma_{q}$'])
            n = sig_q
            slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
            y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
            mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
            slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
            y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
            mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
        else:
            slope_fit1 = np.array(file['slope_fit']).reshape((len(mum),len(n)))
            y_intercept_fit1 = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
            mu_hars_fit1 = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
            slope_predicted1 = np.array(file['slope*']).reshape((len(mum),len(n)))
            y_intercept_predicted1 = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
            mu_hars_predicted1 = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if function=="slope_fit":
        variable = slope_fit
    elif function == "y_intercept_fit":
        variable = y_intercept_fit
    elif function == "mu_hars_fit":
        variable = mu_hars_fit
    elif function == "slope_predicted":
        variable = slope_predicted
    elif function == "y_intercept_predicted":
        variable = y_intercept_predicted
    elif function == "mu_hars_predicted":
        variable = mu_hars_predicted
    elif function == 'distances':
        variable = abs(y_intercept_predicted - y_intercept_fit)
    elif function == 'hars_diff':
        variable = mu_hars_fit1-mu_hars_fit

    
    if predicted ==1:
        predicted_intercepts = y_intercept_predicted
        predicted_slope = slope_predicted

        if distribution==1:
            predicted_intercepts -= 2.5

        if 'intercept' in function:
            ax.plot(n,predicted_intercepts[0,:],linewidth=3,color = color)
        elif 'slope' in function:
            ax.plot(n,predicted_slope[0,:],linewidth=3,color = color)
            
    if predicted ==2:
        predicted_intercepts = y_intercept_predicted[0,:]
        if distribution==1:
            predicted_intercepts -= 2.5

        ax.plot(n[1:],predicted_intercepts[1:],linewidth=3,color = color)

    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'

    if first_variable == "n":
        first_var = n
        second_var = mum
    else:
        first_var = mum
        second_var = n
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    xf = np.zeros(len(second_var))
    yf = np.zeros(len(second_var))

    for i in range(len(first_var)):
        # if first_var[i]==2 or first_var[i]==10 or first_var[i]==100: 
        x1 = []
        y1 = []
        for j in range(len(second_var)):
            x1.append(second_var[j])
            # index = i+j*len(n)
            if vs_n:
                if distribution==1 and function == "y_intercept_fit" and not plot_against_sig:
                    # y1.append(variable[i,j])
                    y1.append(variable[i,j]-2.5)
                elif distribution==1 and plot_against_sig:
                    y1.append(variable[i,j]-2.5)
                else:
                    if not plot_against_sig:
                        y1.append(variable[i,j])
                    else:
                        y1.append(variable[i,j])
                # index = i*len(n)+j
            # y1.append(variable[i,j])
            else:
                if distribution==1 and function == "y_intercept_fit" and not plot_against_sig:
                    y1.append(variable[j,i]-2.5)
                    # y1.append(variable[j,i])
                else:
                    y1.append(variable[j,i])
                    # y1.append(variable[i,j])

        xf += np.array(x1)
        yf += np.array(y1)
        ax.scatter(x1,y1,s=50,marker=marker[i], facecolors=point_facecolor, edgecolors=color)
            
    xf /= len(first_var)
    yf /= len(first_var)
    # ax.plot(xf,yf, color=color,alpha = 0.5,linewidth=6)
    # ax.plot(xf,yf, color=color,linewidth=8)
    # ax.plot(xf,yf, color=color,linewidth=8,linestyle=line_style)
    ax.plot(xf,yf, color=color,linewidth=3,linestyle=line_style,marker=marker_line)

    if fit_type == "log":
        log_fit = np.polyfit(np.log10(xf),yf,1)
        ax.plot(xf,(np.log10(xf)*log_fit[0]+log_fit[1]),linewidth=3,color = color,linestyle=line_style)
        return_var = log_fit
    elif fit_type == "linear":
        linear_fit = np.polyfit(xf,yf,1)
        ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=3,color = color)
        return_var = linear_fit
    elif fit_type == "exp":
        if distribution==2:
            exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
            ax.plot(xf,np.exp(np.array(xf)*exp_fit[0] + exp_fit[1]),linewidth=3,color = color,linestyle=line_style)
        else:
            exp_fit = np.polyfit(np.log10(xf),yf,1)
            ax.plot(xf,(np.log10(xf)*exp_fit[0]+exp_fit[1]),linewidth=3,color = color,linestyle=line_style)
        return_var = exp_fit
    elif fit_type == "n_exp":
        exp_fit = np.polyfit(1/np.array(xf),np.array(yf),3)
        ax.plot(xf,(1/(np.array(xf)*np.array(xf)*np.array(xf)))*exp_fit[0] + (1/(np.array(xf)*np.array(xf)))*exp_fit[1] + (1/(np.array(xf)))*exp_fit[2] + exp_fit[3],linewidth=3,color = color,linestyle=line_style)
        return_var = exp_fit
    else:
        return_var = None
    
    # ax.set_yticks(np.arange(0,2,0.5))
    # plt.xticks(fontsize=32,fontweight='bold')
    # plt.yticks(fontsize=32,fontweight='bold')
    # plt.xticks(fontsize=18,fontweight='bold')
    # plt.yticks(fontsize=18,fontweight='bold')
    plt.xticks(fontsize=22,fontweight='bold')
    plt.yticks(fontsize=22,fontweight='bold')
    # ax.set_ylim([0, 2])
    return return_var

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
    col_name,file = save_fitting_data(save_string=bxgh_string)
    continuation = False
    mum_bxgh = [10,50,100,200,500]
    number_of_opts_bxgh = [2,5,8,10,15,20,30,40,80,100]#[2,5,8,10,15,20,30,40]
    cnt = 0#389#0
    mum_slopes_bxgh = []
    for i in mum_bxgh:
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
        for nop in number_of_opts_bxgh:
            number_of_options = nop
            save_string = str(cnt)+'bxgh_mu_h_vs_mu_x_nop_'+str(nop) #str(cnt)+'bxgh_sx=_sh_mu_h_vs_mu_x1_mu_x2_delta_mu_vs_RCD_nop_'+str(nop) #str(cnt)+'bxgh_mu_h_vs_mu_x_nop_'+str(nop) # str(cnt)+
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

            # def mux1muh1(muh,mux):
            #     mux1 = mux
            #     mux2 = delta_mu + mux
            #     muh1 = muh
            #     muh2 = muh
            #     count = 0
            #     for k in range(runs):
            #         success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
            #             mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
            #             mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
            #         if success == 1:
            #             count += 1
            #     mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}
            #     return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h),continuation=continuation)
            continuation = False
            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=1,uniform=0,mu_m=mu_m_1,smoothened=0)
            print([hrcc,predicted_hrcc])
            num_slopes_bxgh.append([hrcc,predicted_hrcc])
            # message = str(nop)+' number of options simulation finished'
            # pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
            values = [i,nop,sigma_x_1,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc[0][0],predicted_hrcc[0][1],predicted_hrcc[1]]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)
            plt.close()
        cnt += 0
        mum_slopes_bxgh.append(num_slopes_bxgh)
        

if bimodal_x_normal_h_vary_sig==1:
    continuation = False
    mum_bxgh = [10,50,100,200,500]
    number_of_options = 5
    sig_q = [np.round(i,decimals=1) for i in range(15)]
    # number_of_opts_bxgh = [2,5,8,10,15,20,30,40,80,100]#[2,5,8,10,15,20,30,40]
    cnt = 76#0
    mum_slopes_bxgh = []
    for i in mum_bxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_h_1 = 1
        sigma_h_2=1
        
        runs = 500
        batch_size = 50
        delta_mu = 5
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        sig_slopes_bxgh = []
        for sig in sig_q:
            sigma_x_1=sig
            sigma_x_2=sig
            save_string = str(cnt)+'bxgh_mu_h_vs_mu_x_sig_q_'+str(sig) #str(cnt)+'bxgh_sx=_sh_mu_h_vs_mu_x1_mu_x2_delta_mu_vs_RCD_nop_'+str(nop) # str(cnt)+
            
            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_options,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=1,uniform=0,mu_m=mu_m_1,smoothened=1)
            sig_slopes_bxgh.append([hrcc,predicted_hrcc])

            cnt += 1
        cnt += 0
        mum_slopes_bxgh.append(sig_slopes_bxgh)


if bimodal_x_normal_h_plotter==1:
    mum_bxgh = [100]#[10,50,100,200,500]
    number_of_opts_bxgh = [2,10,30]#[2,5,8,10,15,20,30,40]
    cnt = 20#389#405:100
    mum_slopes_bxgh = []
    d_bxgh = []
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(1,3,figsize=(30,10),sharey=True)
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']
    for i in mum_bxgh:
        d_bxgh_m = []
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
        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        for nop in range(len(number_of_opts_bxgh)):
            save_string = str(cnt)+'bxgh_mu_h_vs_mu_x_nop_'+str(number_of_opts_bxgh[nop])#str(cnt)+'bxgh_sx=_sh_mu_h_vs_mu_x1_mu_x2_delta_mu_vs_RCD_nop_'+str(number_of_opts_bxgh[nop]) #str(cnt)+'bxgh_mu_h_vs_mu_x_nop_'+str(number_of_opts_bxgh[nop])# str(cnt)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot="Joined_"+save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_opts_bxgh[nop],line_labels=number_of_opts_bxgh[nop],z_var_='success_rate',plot_type='paper',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=1,uniform=0,mu_m=mu_m_1)
            hrcc[0][1] -= 2.5
            num_slopes_bxgh.append([hrcc,predicted_hrcc])
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax[nop].pcolormesh(np.array(b)+2.5,a,z,shading='auto')
            axes_data.append([a,b,z])
            cnt += 3
            ax[len(ax)-nop-1].set_aspect('equal', 'box')
            # if isinstance(cbar,type(None))==True:
            #     # cax = fig.add_axes([ax[2].get_position().x1+0.12,ax[2].get_position().y0+0.19,0.02,ax[2].get_position().height*0.51])
            #     # cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
            #     cax = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1-0.1,(ax[2].get_position().x1 - ax[0].get_position().x0)*1.16,0.02])
            #     cbar = fig.colorbar(cs,orientation="horizontal",cax=cax)
            #     cax.set_title(z_name,y=1.2,fontsize=18,fontweight='bold')
            font = {"fontsize":18,"fontweight":'bold'}
            #     cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
            #     cbar.ax.tick_params(labelsize=18)
            #     cbar.minorticks_on()
            #     cs.set_clim(0,1)
            #     # cbar.ax.set_aspect(30)
            ax[nop].set_xlim(min(np.array(b)+2.5),15)#max(np.array(b)+2.5)
            ax[nop].set_ylim(2.5,max(a))
            ax[nop].set_yticks(np.arange(2.5,max(a),1.5))
            ax[nop].set_xlabel('Mean option quality $\mu_{q}$',fontsize = 18)
            ax[0].set_ylabel('Mean response threshold $\mu_{h}$',fontsize = 18)
            ax[nop].tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax[nop].minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            ax[nop].set_title(title,font,y=-0.26,color=(0,0,0,1))
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            predicted_hrcc = prd.Hrcc_predict(delta_mu,'$\mu_{x_1}$',b,a,z,sigma_x_1,sigma_x_2,line_labels,prd.gaussian,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
            predicted_hrcc[0][1] -= 2.5
            d = np.round(abs(predicted_hrcc[0][1]-intercept)/ np.sqrt(slope**2 +1),decimals=2)
            d_bxgh_m.append(d)
            delta_slope = np.round(abs(predicted_hrcc[0][0]-slope),decimals=2)
            intercepts = [intercept,predicted_hrcc[0][1]]
            vis.graphicPlot_paper(a,np.array(b)+2.5,x_name,y_name,z_name,title,save_name,z_var,fig, ax[nop],cbar,[slope,intercept],hars,[predicted_hrcc[0]],[predicted_hrcc[1]],d,delta_slope,intercepts,linestyle=line_style[nop])
        d_bxgh.append(d_bxgh_m)
        mum_slopes_bxgh.append(num_slopes_bxgh)
        lines_1, labels_1 = ax[0].get_legend_handles_labels()
        lines_2, labels_2 = ax[1].get_legend_handles_labels()
        lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                # z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                # z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable])#,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.02)
            return points
        plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.02)
        point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        mum_slopes_bxgh.append(num_slopes_bxgh)

if bimodal_x_normal_h_sigma==1:
    col_name,file = save_fitting_data(save_string=bxgh_string)
    continuation = False
    mum_bxgh = [10,50,100,200,500]
    file_num = 0#450#150#560#325#150#195#0
    mum_slopes_bxgh = []
    number_of_opts_bxgh = [2,5,8,10,15,20,30,40,80,100]
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
        mu_x_2 = mu_x_1 + delta_mu
        mu_h_1 = (mu_x_1+mu_x_2)/2
        mu_h_2 = mu_h_1
        mu_x = List([mu_x_1,mu_x_2])
        sigma_x = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [np.round(0.1+i,decimals=1) for i in range(8)]
        sigma_h = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [2]
        # sigma_h = [2]
        
        step = 0.0001
        num_slopes = []
        for nop in number_of_opts_bxgh:
            fig, ax = plt.subplots()
            # sigma_h = [(0.05*np.log10(nop)+0.58)*sigma_x[0]]
            number_of_options = nop
            save_string = str(file_num)+'bxgh_sigma_h_vs_sigma_x_nop_'+str(nop)#str(file_num)+'D_h*_bxgh_sigma_h_vs_sigma_x_nop_'+str(nop)#str(file_num)+'bxgh_mx_mh_sigma_h_vs_sigma_x1_sigma_x2_vs_RCD_nop_'+str(nop) # str(file_num)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=0)#,min_sig_h=min_sig_h
            # hrcc = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=0)#,min_sig_h=min_sig_h
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            num_slopes.append([hrcc,predicted_hrcc])
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax.pcolormesh(b,a,z,shading='auto')

            ax.set_aspect('equal', 'box')
            cbar = fig.colorbar(cs,orientation='vertical')
            z_name = "Average rate of success"
            cbar.set_label(z_name,fontsize=18,fontweight='bold')
            font = {"fontsize":18,"fontweight":'bold'}
            cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
            cbar.ax.tick_params(labelsize=18)
            cbar.minorticks_on()
            cs.set_clim(0,1)
            ax.set_aspect('equal', 'box')

            ax.set_xlim(min(b),max(b))
            ax.set_ylim(min(a),max(a))
            ax.set_yticks(np.arange(min(a),max(a),1.5))
            # ax[nop].axes.get_xaxis().set_visible(False)
            ax.set_xlabel("Options qualities' std. dev. $\sigma_{q}$",fontsize = 18)
            ax.set_ylabel("Response thresholds' std. dev. $\sigma_{h}$",fontsize = 18)
            ax.tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax.minorticks_on()

            z_best_fit = [hrcc[0][0]*bb+hrcc[0][1] for bb in b]
            ax.plot(b,z_best_fit,color = 'red',linewidth=4,linestyle='--')
            
            lines_1, labels_1 = ax.get_legend_handles_labels()

            lines = lines_1 #+ lines_2 + lines_3
            labels = labels_1 #+ labels_2 + labels_3
            
            plt.savefig(save_name[:-3]+'png',format = "png",bbox_inches="tight",pad_inches=0.05)

            file_num += 1
            values = [i,nop,None,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc,predicted_hrcc,predicted_hrcc]
            # values = [i,nop,None,None,None,hrcc,None,None,None]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)
            plt.close()
        file_num += 0
        mum_slopes_bxgh.append(num_slopes)

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
    col_name,file = save_fitting_data(save_string=uxgh_string)
    continuation = False
    cnt = 98#429#98#429
    mum_uxgh = [10,50,100,200,500]
    number_of_opts_uxgh = [2,5,8,10,15,20,30,40,80,100]
    mum_slopes_uxgh = []
    for i in mum_uxgh:
        
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
        runs = 100#500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes_uxgh = []
        for nop in number_of_opts_uxgh:
            number_of_options = nop
            save_string = str(cnt)+'uxgh_mu_h_vs_mu_x_nop_'+str(nop)#str(cnt)+'uxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop)# str(cnt)+'uxgh_mu_h_vs_mu_x_nop_'+str(nop)#str(cnt)+
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

            # def mux1muh1(muh,mux):
            #     mux1 = mux + low_x_1
            #     sigmax1 = mux + high_x_1
            #     muh1 = muh
            #     muh2 = muh
            #     count = 0
            #     for k in range(runs):
            #         success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_u,distribution_h=rng.threshold_n,\
            #             mu_h=[muh1,muh2],sigma_h=[sigma_h_1,sigma_h_2],mu_x=[mux1,mux1],sigma_x=[sigmax1,sigmax1],err_type=0,number_of_options=number_of_options,\
            #             mu_m=[mu_m_1,mu_m_2],sigma_m=[sigma_m_1,sigma_m_2])
            #         if success == 1:
            #             count += 1
            #     mu_va = {'$\mu_{h_1}$':muh,'$\mu_{h_2}$':muh,'$\mu_{x_1}$': mux,'$\mu_{x_2}$': mux,"success_rate":count/runs}
            #     return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h  = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=0,uniform=1,mu_m=mu_m_1,smoothened=0)
            print([hrcc,predicted_hrcc])
            num_slopes_uxgh.append([hrcc,predicted_hrcc])
            # message = str(nop)+' number of options simulation finished'
            # pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
            values = [i,nop,sigma_x_1,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc[0][0],predicted_hrcc[0][1],predicted_hrcc[1]]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)

        cnt += 0
        mum_slopes_uxgh.append(num_slopes_uxgh)

if uniform_x_normal_h_vary_sig==1:
    cnt = 151#98#429
    mum_uxgh = [10,50,100,200,500]
    number_of_options = 5
    sig_q = [np.round(i,decimals=1) for i in range(15)]
    mum_slopes_uxgh = []
    for i in mum_uxgh:  
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_h_1 = 1
        sigma_h_2=1
        runs = 100#500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        sig_slopes_uxgh = []
        for sig in sig_q:
            sigma_x_1=sig
            sigma_x_2=sig
            low_x_1 = -np.sqrt(3)*sigma_x_1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
            high_x_1 = np.sqrt(3)*sigma_x_1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
        
            save_string = str(cnt)+'uxgh_mu_h_vs_mu_x_sig_q_'+str(sig)#str(cnt)+'uxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop)#str(cnt)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h  = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_options,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=0,uniform=1,mu_m=mu_m_1,smoothened=1)
            sig_slopes_uxgh.append([hrcc,predicted_hrcc])

            cnt += 1
        cnt += 0
        mum_slopes_uxgh.append(sig_slopes_uxgh)

if uniform_x_normal_h_plotter==1:
    mum_bxgh = [100]#[10,50,100,200,500]
    number_of_opts_bxgh = [2,10,30]#[2,5,8,10,15,20,30,40]
    cnt = 118#429#445
    mum_slopes_bxgh = []
    d_uxgh = []
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(1,3,figsize=(30,10),sharey=True)
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']
    for i in mum_bxgh:
        d_uxgh_m = []
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
        num_slopes_bxgh = []
        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        for nop in range(len(number_of_opts_bxgh)):
            save_string = str(cnt)+'uxgh_mu_h_vs_mu_x_nop_'+str(number_of_opts_bxgh[nop])#str(cnt)+'uxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(number_of_opts_bxgh[nop]) #  str(cnt)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot="Joined_"+save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_opts_bxgh[nop],line_labels=number_of_opts_bxgh[nop],z_var_='success_rate',plot_type='paper',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=0,uniform=1,mu_m=mu_m_1)
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            num_slopes_bxgh.append([slope,hars,intercept])
            
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax[nop].pcolormesh(b,a,z,shading='auto')
            axes_data.append([a,b,z])
            cnt += 3
            ax[len(ax)-nop-1].set_aspect('equal', 'box')
            # if isinstance(cbar,type(None))==True:
            #     # cax = fig.add_axes([ax[2].get_position().x1+0.12,ax[2].get_position().y0+0.19,0.02,ax[2].get_position().height*0.51])
            #     # cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
            #     cax = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1-0.1,(ax[2].get_position().x1 - ax[0].get_position().x0)*1.16,0.02])
            #     cbar = fig.colorbar(cs,orientation="horizontal",cax=cax)
            #     cax.set_title(z_name,y=1.2,fontsize=18,fontweight='bold')
            #     font = {"fontsize":18,"fontweight":'bold'}
            #     cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
            #     cbar.ax.tick_params(labelsize=18)
            #     cbar.minorticks_on()
            #     cs.set_clim(0,1)
            #     # cbar.ax.set_aspect(30)
            ax[nop].set_xlim(2.5,max(b))
            ax[nop].set_ylim(2.5,max(a))
            ax[nop].set_yticks(np.arange(2.5,max(a),1.5))
            ax[nop].axes.get_xaxis().set_visible(False)
            # ax.set_xlabel('Mean option quality $\mu_{x}$',fontsize = 18)
            ax[0].set_ylabel('Mean response threshold $\mu_{h}$',fontsize = 18)
            ax[nop].tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax[nop].minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            # ax.set_title(title,font,y=-0.25,color=(0.3,0.3,0.3,1))
            predicted_hrcc = prd.Hrcc_predict(delta_mu,'$\mu_{x_1}$',b,a,z,sigma_x_1,sigma_x_2,line_labels,prd.uniform,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
            d = np.round(abs(predicted_hrcc[0][1]-intercept)/ np.sqrt(slope**2 +1),decimals=2)
            d_uxgh_m.append(d)
            delta_slope = np.round(abs(predicted_hrcc[0][0]-slope),decimals=2)
            intercepts = [intercept,predicted_hrcc[0][1]]
        
            vis.graphicPlot_paper(a,b,x_name,y_name,z_name,title,save_name,z_var,fig, ax[nop],cbar,[slope,intercept],hars,[predicted_hrcc[0]],[predicted_hrcc[1]],d,delta_slope,intercepts,linestyle=line_style[nop])
        d_uxgh.append(d_uxgh_m)
        lines_1, labels_1 = ax[0].get_legend_handles_labels()
        lines_2, labels_2 = ax[1].get_legend_handles_labels()
        lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                # z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                # z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable])#,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.02)
            return points
        plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.02)
        point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        mum_slopes_bxgh.append(num_slopes_bxgh)

if uniform_x_normal_h_sigma==1:
    col_name,file = save_fitting_data(save_string=uxgh_string)
    continuation = False
    mum_uxgh = [10,50,100,200,500]
    number_of_opts_uxgh = [2,5,8,10,15,20,30,40,80,100]
    cnt = 50#500#200#460#375#200#260#65
    mum_slopes_uxgh = []
    for i in mum_uxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        
        mu_x_1 = 5
        mu_x_2 = 5
        mu_h_1 = mu_x_1
        mu_h_2 = mu_x_1
        mu_x = List([mu_x_1,mu_x_2])
        runs = 500
        batch_size = 50
        delta_sigma = 0
        sigma_x = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [np.round(0.1+i,decimals=1) for i in range(8)]
        sigma_h = [np.round(0.1+i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [2]
        # sigma_h = [2]
        num_slopes = []
        for nop in number_of_opts_uxgh:
            # sigma_h = [(-0.07*np.log10(nop)+0.67)*sigma_x[0]]
            number_of_options = nop
            save_string = str(cnt)+'uxgh_sigma_h_vs_sigma_x_nop'+str(nop)#str(cnt)+'D_h*_uxgh_sigma_h_vs_sigma_x_nop'+str(nop)#str(cnt)+'uxgh_sigma_h_vs_sigma_x_vs_RCD_nop'+str(nop) # str(cnt)+
            fig, ax = plt.subplots()

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',delta_mu=delta_sigma,gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=0)#,min_sig_h=min_sig_h
            # hrcc = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=0)#,min_sig_h=min_sig_h
            # num_slopes.append([hrcc,predicted_hrcc])
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax.pcolormesh(b,a,z,shading='auto')

            ax.set_aspect('equal', 'box')
            cbar = fig.colorbar(cs,orientation='vertical')
            z_name = "Average rate of success"
            cbar.set_label(z_name,fontsize=18,fontweight='bold')
            font = {"fontsize":18,"fontweight":'bold'}
            cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
            cbar.ax.tick_params(labelsize=18)
            cbar.minorticks_on()
            cs.set_clim(0,1)
            ax.set_aspect('equal', 'box')

            ax.set_xlim(min(b),max(b))
            ax.set_ylim(min(a),max(a))
            ax.set_yticks(np.arange(min(a),max(a),1.5))
            # ax[nop].axes.get_xaxis().set_visible(False)
            ax.set_xlabel("Options qualities' std. dev. $\sigma_{q}$",fontsize = 18)
            ax.set_ylabel("Response thresholds' std. dev. $\sigma_{h}$",fontsize = 18)
            ax.tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax.minorticks_on()

            z_best_fit = [hrcc[0][0]*bb+hrcc[0][1] for bb in b]
            ax.plot(b,z_best_fit,color = 'red',linewidth=4,linestyle='--')
            
            lines_1, labels_1 = ax.get_legend_handles_labels()

            lines = lines_1 #+ lines_2 + lines_3
            labels = labels_1 #+ labels_2 + labels_3
            
            plt.savefig(save_name[:-3]+'png',format = "png",bbox_inches="tight",pad_inches=0.05)

            cnt += 1
            values = [i,nop,None,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc,predicted_hrcc,predicted_hrcc]
            # values = [i,nop,None,None,None,hrcc,None,None,None]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)
            plt.close()
            # print(cnt)
        cnt += 0
        mum_slopes_uxgh.append(num_slopes)

if normal_x_normal_h==1:
    col_name,file = save_fitting_data(save_string=gxgh_string)
    continuation = False
    mum_gxgh = [10,50,100,200,500]
    cnt = 148#469#148#469
    number_of_opts_gxgh = [2,5,8,10,15,20,30,40,80,100]
    mum_slopes_gxgh = []
    for i in mum_gxgh:
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
        num_slopes_gxgh = []
        for nop in number_of_opts_gxgh:
            number_of_options = nop
            save_string = str(cnt)+'gxgh_mu_h_vs_mu_x_nop_'+str(nop) #str(cnt)+'gxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(nop) # str(cnt)+'gxgh_mu_h_vs_mu_x_nop_'+str(nop) # str(cnt)+
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

            # def mux1muh1(muh,mux,avg_pval = 0):
            #     # avg_incrtness_w_n = np.zeros((number_of_options,5*number_of_options))
            #     mux1 = mux
            #     mux2 = delta_mu + mux
            #     muh1 = muh
            #     muh2 = muh
            #     count = 0
            #     for k in range(runs):
            #         success,incrt,incrt_w_n,yes_test,max_rat_pval,pval_mat = wf.multi_run(distribution_m=rng.units_n,distribution_x=rng.dx_n,distribution_h=rng.threshold_n,\
            #             mu_h=List([muh1,muh2]),sigma_h=List([sigma_h_1,sigma_h_2]),mu_x=List([mux1,mux2]),sigma_x=List([sigma_x_1,sigma_x_2]),err_type=0,number_of_options=number_of_options,\
            #             mu_m=List([mu_m_1,mu_m_2]),sigma_m=List([sigma_m_1,sigma_m_2]))
            #         if success == 1:
            #             count += 1

            #         # flag = 0
            #         # for i in yes_test:
            #         #     for j in i:
            #         #         if j[0][0]== np.nan or j[1]<0:
            #         #             flag = 1
            #         #             break
            #         # if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
            #         #     avg_pval += max_rat_pval[0][1]
            #         # else:
            #         #     avg_pval += 1

            #         # avg_incrtness_w_n += np.concatenate((incrt_w_n[0],incrt_w_n[1],incrt_w_n[2]*pval_mat,incrt_w_n[3],incrt_w_n[4]),axis=1)

            #     mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}#{'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs,'Average p-value':avg_pval/runs,'Wrong_ranking_cost_without_no_proportion':avg_incrtness_w_n/runs}
            #     return mu_va

            # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate",'Average p-value','Wrong_ranking_cost_without_no_proportion'],save_string=save_string,batch_size=3*len(mu_h))

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,mu_m=mu_m_1,smoothened=0)
            print([hrcc,predicted_hrcc])
            num_slopes_gxgh.append([hrcc,predicted_hrcc])
            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'Pval',x_var_='$\mu_{x_1}$',y_var_='Average p-value',num_of_opts=number_of_opts,plot_type='line')
            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'WRC',x_var_='$\mu_{x_1}$',y_var_='Wrong_ranking_cost_without_no_proportion',num_of_opts=number_of_opts,plot_type='line')

            # message = str(nop)+' number of options simulation finished'
            # pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
            values = [i,nop,sigma_x_1,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc[0][0],predicted_hrcc[0][1],predicted_hrcc[1]]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)
        cnt += 0
        mum_slopes_gxgh.append(num_slopes_gxgh)

if normal_x_normal_h_compare_best_two==1:
    continuation = False
    mum_gxgh = [100]
    cnt = 23#469
    number_of_opts_gxgh = [10]
    mum_slopes_gxgh = []

    fig, ax = plt.subplots()
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']

    for i in mum_gxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_x_1 = 1
        sigma_x_2= 1
        sigma_h_1=sigma_x_1
        sigma_h_2=sigma_x_1
        runs = 500
        batch_size = 50
        delta_mu = 0
        mu_x = range(1,8)
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes_gxgh = []

        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        
        for nop in number_of_opts_gxgh:
            save_string =  str(cnt)+'gxgh_mu_h_vs_mu_x_nop_'+str(nop) # str(cnt)+

            vote_diff = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=nop,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,mu_m=mu_m_1,smoothened=1,vote_compare=1)
            
            vote_diff = np.array(vote_diff).reshape(len(mu_h),len(mu_x))
            cs = ax.pcolormesh(np.array(mu_x),mu_h,vote_diff,shading='auto')

            ax.set_aspect('equal', 'box')
            if isinstance(cbar,type(None))==True:
                # cax = fig.add_axes([ax[2].get_position().x1+0.12,ax[2].get_position().y0+0.19,0.02,ax[2].get_position().height*0.51])
                # cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
                # cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1-0.1,(ax.get_position().x1 - ax.get_position().x0)*1.16,0.02])
                cbar = fig.colorbar(cs,orientation="vertical")
                # cax.set_title(z_name,y=1.2,fontsize=18,fontweight='bold')
                font = {"fontsize":18,"fontweight":'bold'}
                # cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
                cbar.ax.tick_params(labelsize=18)
                cbar.minorticks_on()
                # cs.set_clim(0,1)
                # cbar.ax.set_aspect(30)
            ax.set_xlim(min(np.array(mu_x)),max(np.array(mu_x)))#max(np.array(b)+2.5)
            ax.set_ylim(min(mu_h),max(mu_h))
            ax.set_xlabel('Mean option quality $\mu_{q}$',fontsize = 18)
            ax.set_ylabel('Mean response threshold $\mu_{h}$',fontsize = 18)
            ax.tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax.minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            # ax.set_title(title,font,y=-0.26,color=(0,0,0,1))
            predicted_hrcc = prd.Hrcc_predict(delta_mu,'$\mu_{x_1}$',mu_h,mu_x,vote_diff,sigma_x_1,sigma_x_2,nop,prd.gaussian,prd.ICPDF,1.0-(1.0/(nop)),prd.z_extractor,prd.optimization,nop)
            ESM = [predicted_hrcc[0][0]*bb+predicted_hrcc[0][1] for bb in mu_x]
            ax.plot(mu_x,ESM,color = 'black',linestyle=':',linewidth=4)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        # lines_2, labels_2 = ax[1].get_legend_handles_labels()
        # lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 #+ lines_2 + lines_3
        labels = labels_1 #+ labels_2 + labels_3
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                # z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                # z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable])#,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            # plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
            return points
        # plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
        # point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        cnt += 0
        mum_slopes_gxgh.append(num_slopes_gxgh)


if normal_x_normal_h_vary_sig==1:
    mum_gxgh = [10,50,100,200,500]
    number_of_options = 5
    sig_q = [np.round(i,decimals=1) for i in range(15)]
    cnt = 0#0
    mum_slopes_gxgh = []
    for i in mum_bxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        sigma_h_1 = 1
        sigma_h_2=1
        
        runs = 100
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        sig_slopes_bxgh = []
        for sig in sig_q:
            sigma_x_1=sig
            sigma_x_2=sig
            save_string = str(cnt)+'gxgh_mu_h_vs_mu_x_sig_q_'+str(sig) #str(cnt)+'bxgh_sx=_sh_mu_h_vs_mu_x1_mu_x2_delta_mu_vs_RCD_nop_'+str(nop) # str(cnt)+
            
            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_options,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,gaussian=1,uniform=0,mu_m=mu_m_1,smoothened=1)
            sig_slopes_bxgh.append([hrcc,predicted_hrcc])

            cnt += 1
        cnt += 0
        mum_slopes_gxgh.append(sig_slopes_bxgh)


if normal_x_normal_h_sigma_plotter==1:
    mum_bxgh = [100]
    number_of_opts_gxgh = [2,15,80]
    cnt = 270#118#351#mu_h=mu_h* #156#mu_h=mu_x
    mum_slopes_gxgh = []
    num_slopes_gxgh = []
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']
    for i in mum_bxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        
        mu_x_1 = 7.5
        mu_x_2 = 7.5
        mu_h_1 = mu_x_1
        mu_h_2 = mu_x_2
        runs = 500
        batch_size = 50
        delta_mu = 0
        sigma_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        for nop in range(len(number_of_opts_gxgh)):
            save_string = str(cnt)+'gxgh_sigma_h_vs_sigma_x_nop_'+str(number_of_opts_gxgh[nop])#str(cnt)+'gxgh_mx=mh_sigma_h_vs_sigma_x_vs_RCD_nop_'+str(number_of_opts_gxgh[nop]) # str(cnt)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot="Joined"+save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_opts_gxgh[nop],line_labels=number_of_opts_gxgh[nop],z_var_='success_rate',plot_type='paper',delta_mu=delta_mu,mu_m=mu_m_1)
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            num_slopes_gxgh.append([slope,hars,intercept])
            
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax[nop].pcolormesh(b,a,z,shading='auto')
            axes_data.append([a,b,z])
            cnt += 4
            ax[len(ax)-nop-1].set_aspect('equal', 'box')
            if isinstance(cbar,type(None))==True:
                cax = fig.add_axes([ax[2].get_position().x1+0.05,ax[2].get_position().y0+0.19,0.02,ax[2].get_position().height*0.51])
                cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
                cbar.set_label(z_name,fontsize=18,fontweight='bold')
                font = {"fontsize":18,"fontweight":'bold'}
                cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
                cbar.ax.tick_params(labelsize=18)
                cbar.minorticks_on()
                cs.set_clim(0,1)
                cbar.ax.set_aspect(30)
            ax[nop].set_xlim(min(b),max(b))
            ax[nop].set_ylim(min(a),max(a))
            # ax.axes.get_xaxis().set_visible(False)
            ax[nop].set_xlabel("Qualities' std. dev. $\sigma_{q}$",fontsize = 20,fontweight='bold')
            ax[nop].set_ylabel("Thresholds' std. dev. $\sigma_{h}$",fontsize = 20,fontweight='bold')
            # ax[2].set_ylabel("Thresholds' std. dev. $\sigma_{h}$",fontsize = 20,fontweight='bold')
            ax[nop].set_xticks(np.arange(0,15,5))
            ax[nop].set_yticks(np.arange(0,15,5))
            ax[nop].tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax[nop].minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            ax[nop].set_title('n = '+str(number_of_opts_gxgh[nop]),font,y=0.9,color=(0,0,0,1))

            intercepts = [intercept]
            vis.graphicPlot_paper(a,b,x_name,y_name,z_name,title,save_name,z_var,fig, ax[nop],cbar,[slope,intercept],hars,linestyle=line_style[nop],intercepts=intercepts)
        lines_1, labels_1 = ax[0].get_legend_handles_labels()
        lines_2, labels_2 = ax[1].get_legend_handles_labels()
        lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.0)
            return points
        plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.0)
        point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        mum_slopes_gxgh.append(num_slopes_gxgh)

if normal_x_normal_h_plotter==1:
    mum_bxgh = [100]#[10,50,100,200,500]
    number_of_opts_gxgh = [2,10,30]#[2,5,8,10,15,20,30,40]
    cnt = 168#485#469
    mum_slopes_gxgh = []
    num_slopes_gxgh = []
    d_gxgh = []
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(1,3,figsize=(30,10),sharey=True)
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']
    for i in mum_bxgh:
        d_gxgh_m = []
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
        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        for nop in range(len(number_of_opts_gxgh)):
            save_string = str(cnt)+'gxgh_mu_h_vs_mu_x_nop_'+str(number_of_opts_gxgh[nop]) #str(cnt)+'gxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(number_of_opts_gxgh[nop]) # str(cnt)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot="Joined"+save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_opts_gxgh[nop],line_labels=number_of_opts_gxgh[nop],z_var_='success_rate',plot_type='paper',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,mu_m=mu_m_1)
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            num_slopes_gxgh.append([slope,hars,intercept])
            
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax[nop].pcolormesh(b,a,z,shading='auto')
            axes_data.append([a,b,z])
            cnt += 3
            ax[len(ax)-nop-1].set_aspect('equal', 'box')
            if isinstance(cbar,type(None))==True:
                # cax = fig.add_axes([ax[2].get_position().x1+0.12,ax[2].get_position().y0+0.19,0.02,ax[2].get_position().height*0.51])
                # cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
                cax = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1-0.1,(ax[2].get_position().x1 - ax[0].get_position().x0)*1.16,0.02])
                
                cbar = fig.colorbar(cs,orientation="horizontal",cax=cax)
                cax.set_title(z_name,y=1.2,fontsize=18,fontweight='bold')
                
                font = {"fontsize":18,"fontweight":'bold'}
                cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
                cbar.ax.tick_params(labelsize=18)
                
                cbar.minorticks_on()
                cs.set_clim(0,1)
                # cbar.ax.set_aspect(30)
            ax[nop].set_xlim(2.5,max(b))
            ax[nop].set_ylim(2.5,max(a))
            ax[nop].set_yticks(np.arange(2.5,max(a),1.5))
            ax[nop].axes.get_xaxis().set_visible(False)
            ax[nop].set_xlabel('Mean option quality $\mu_{x}$',fontsize = 18)
            ax[0].set_ylabel('Mean response threshold $\mu_{h}$',fontsize = 18)
            ax[nop].tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax[nop].minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            # ax.set_title(title,font,y=-0.25,color=(0.3,0.3,0.3,1))
            predicted_hrcc = prd.Hrcc_predict(delta_mu,'$\mu_{x_1}$',b,a,z,sigma_x_1,sigma_x_2,line_labels,prd.gaussian,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
            d = np.round(abs(predicted_hrcc[0][1]-intercept)/ np.sqrt(slope**2 +1),decimals=2)
            d_gxgh_m.append(d)
            delta_slope = np.round(abs(predicted_hrcc[0][0]-slope),decimals=2)
            intercepts = [intercept,predicted_hrcc[0][1]]
        
            vis.graphicPlot_paper(a,b,x_name,y_name,z_name,title,save_name,z_var,fig, ax[nop],cbar,[slope,intercept],hars,[predicted_hrcc[0]],[predicted_hrcc[1]],d,delta_slope,intercepts,linestyle=line_style[nop])
        d_gxgh.append(d_gxgh_m)
        lines_1, labels_1 = ax[0].get_legend_handles_labels()
        lines_2, labels_2 = ax[1].get_legend_handles_labels()
        lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
            return points
        plt.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
        point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        mum_slopes_gxgh.append(num_slopes_gxgh)

if Fig1==1:
    #normal_x_normal_h_plotter_2
    mum_bxgh = [100]
    number_of_opts_gxgh = [2,40]
    cnt = 485#168#
    mum_slopes_gxgh = []
    num_slopes_gxgh = []

    # plt.style.use('ggplot')
    
    fig, ax = plt.subplots(1,2,sharey=True)
    plt.subplots_adjust(wspace = .3)
    z_name = "Average rate of success"
    colors = ['red','darkslategrey','maroon']
    for i in mum_bxgh:
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
        cbar = None
        points = []
        line_style = ['--','-.',':']
        axes_data = []
        for nop in range(len(number_of_opts_gxgh)):
            save_string = str(cnt)+'gxgh_sx=sh_mu_h_vs_mu_x_vs_RCD_nop_'+str(number_of_opts_gxgh[nop]) # str(cnt)+   str(cnt)+'gxgh_mu_h_vs_mu_x_nop_'+str(number_of_opts_gxgh[nop])# 

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot="joined_"+save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=number_of_opts_gxgh[nop],line_labels=number_of_opts_gxgh[nop],z_var_='success_rate',plot_type='paper',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2,mu_m=mu_m_1)
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]

            num_slopes_gxgh.append([slope,hars,intercept])
            
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax[nop].pcolormesh(b,a,z,shading='auto')
            axes_data.append([a,b,z])
            cnt += 7
            ax[len(ax)-nop-1].set_aspect('equal', 'box')
            if isinstance(cbar,type(None))==True:
                cax = fig.add_axes([ax[1].get_position().x1+0.2,ax[1].get_position().y0+0.16,0.02,ax[1].get_position().height*0.58])
                cbar = fig.colorbar(cs,orientation="vertical",cax=cax)
                cbar.set_label(z_name,fontsize=12,fontweight='bold')
                font = {"fontsize":12,"fontweight":'bold'}
                cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":8,"fontweight":'bold'})
                cbar.ax.tick_params(labelsize=8)
                cbar.minorticks_on()
                cs.set_clim(0,1)
                cbar.ax.set_aspect(20)
            ax[nop].set_xlim(min(b),max(b))
            ax[nop].set_ylim(min(a),max(a))
            ax[nop].set_xlabel('Mean option quality $\mu_{q}$',fontsize = 14)
            ax[0].set_ylabel('Mean response threshold $\mu_{h}$',fontsize = 14)
            ax[nop].tick_params(axis='both', which='major',labelsize=14,color='black')
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax[nop].minorticks_on()
            # ax.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
            if nop==0:
                title = '(a) '+ title
            else:
                title = '(b) '+ title
            ax[nop].set_title(title,font,y=1.05,color='black')
            predicted_hrcc = prd.Hrcc_predict(delta_mu,'$\mu_{x_1}$',b,a,z,sigma_x_1,sigma_x_2,line_labels,prd.uniform,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
            d = np.round(abs(predicted_hrcc[0][1]-intercept)/ np.sqrt(slope**2 +1),decimals=2)
            delta_slope = np.round(abs(predicted_hrcc[0][0]-slope),decimals=2)
            intercepts = [intercept,predicted_hrcc[0][1]]
            vis.graphicPlot_paper(a,b,x_name,y_name,z_name,title,save_name,z_var,fig, ax[nop],cbar,[slope,intercept],hars,[predicted_hrcc[0]],[predicted_hrcc[1]],d,delta_slope,intercepts,linestyle=line_style[nop])
        lines_1, labels_1 = ax[0].get_legend_handles_labels()
        lines_2, labels_2 = ax[1].get_legend_handles_labels()
        # lines_3, labels_3 = ax[2].get_legend_handles_labels()
        lines = lines_1 + lines_2 #+ lines_3
        labels = labels_1 + labels_2 #+ labels_3
        
        def onclick(event,points = points):
            color = colors[len(points)]
            lable = ''
            axs_ind = np.where(ax==event.inaxes)[0][0]
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = event.inaxes.plot([axes_data[axs_ind][1][0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 1)
                point_line_2 = event.inaxes.plot([round(event.xdata,1),round(event.xdata,1)],[axes_data[axs_ind][0][0],round(event.ydata,1)],color=color,linewidth = 1)
                point_lable = event.inaxes.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color=color)
                verti=axes_data[axs_ind][2][np.argmin(abs(axes_data[axs_ind][1]-round(event.ydata,1))),np.argmin(abs(axes_data[axs_ind][0]-round(event.xdata,1)))]
                z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=3)
                z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            plt.savefig(save_name,format = "eps",bbox_inches="tight",pad_inches=0)
            return points
        
        fig.savefig(save_name[:-3]+'pdf',format = "pdf",bbox_inches="tight",pad_inches=0)
        fig.savefig(save_name[:-3]+'png',format = "png",bbox_inches="tight",pad_inches=0.0,dpi=300)
        point = fig.canvas.mpl_connect('button_press_event', onclick)

        # fig.legend(lines, labels,loc='upper left',prop=dict(weight='bold',size=12), bbox_to_anchor=(0.15, 0.83),labelcolor=(0.3,0.3,0.3,1),ncol=3,frameon=False)
        plt.show()
        mum_slopes_gxgh.append(num_slopes_gxgh)

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
    col_name,file = save_fitting_data(save_string=gxgh_string)
    continuation = False
    mum_gxgh = [10,50,100,200,500]
    number_of_opts_gxgh = [2,5,8,10,15,20,30,40,80,100]
    file_num = 100#550#250#510#425#250#325#130
    mum_slopes_gxgh = []
    for i in mum_gxgh:
        mu_m_1=i
        sigma_m_1=0
        mu_m_2=i
        sigma_m_2=0
        mu_x_1=5
        mu_x_2=5
        mu_h_1 = mu_x_1
        mu_h_2= mu_x_1
        mu_x = List([mu_x_1,mu_x_2])
        runs = 500
        batch_size = 50
        delta_sigma = 0
        sigma_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [np.round(0.1+i,decimals=1) for i in range(8)]
        sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        # sigma_x = [2]
        # sigma_h = [2]
        num_slopes = []
        for nop in number_of_opts_gxgh:
            # sigma_h = [(0.07*np.log10(nop)+0.57)*sigma_x[0]]
            number_of_options = nop
            save_string = str(file_num)+'gxgh_sigma_h_vs_sigma_x_nop_'+str(nop)#str(file_num)+'D_h*_gxgh_sigma_h_vs_sigma_x_nop_'+str(nop) #str(file_num)+'gxgh_mx=mh_sigma_h_vs_sigma_x_vs_RCD_nop_'+str(nop) # str(file_num)+

            hrcc,predicted_hrcc,a,b,x_name,y_name,title,save_name,z_var,line_labels,min_sig_h = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=0)#,min_sig_h=min_sig_h)
            # hrcc = vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\sigma_{x_1}$',y_var_='$\sigma_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=number_of_options,z_var_='success_rate',plot_type='graphics',gaussian=0,uniform=0,mu_m=mu_m_1,smoothened=1)#,min_sig_h=min_sig_h
            
            # num_slopes.append([hrcc,predicted_hrcc])
            fig, ax = plt.subplots()
            slope = hrcc[0][0]
            intercept = hrcc[0][1]
            hars = hrcc[1]
            z = np.array(z_var).reshape(len(a),len(b))
            cs = ax.pcolormesh(b,a,z,shading='auto')

            ax.set_aspect('equal', 'box')
            cbar = fig.colorbar(cs,orientation='vertical')
            z_name = "Average rate of success"
            cbar.set_label(z_name,fontsize=18,fontweight='bold')
            font = {"fontsize":18,"fontweight":'bold'}
            cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
            cbar.ax.tick_params(labelsize=18)
            cbar.minorticks_on()
            cs.set_clim(0,1)
            ax.set_aspect('equal', 'box')

            ax.set_xlim(min(b),max(b))
            ax.set_ylim(min(a),max(a))
            ax.set_yticks(np.arange(min(a),max(a),1.5))
            # ax[nop].axes.get_xaxis().set_visible(False)
            ax.set_xlabel("Options qualities' std. dev. $\sigma_{q}$",fontsize = 18)
            ax.set_ylabel("Response thresholds' std. dev. $\sigma_{h}$",fontsize = 18)
            ax.tick_params(axis='both', which='major',labelsize=18)
            # ax.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
            ax.minorticks_on()

            z_best_fit = [hrcc[0][0]*bb+hrcc[0][1] for bb in b]
            ax.plot(b,z_best_fit,color = 'red',linewidth=4,linestyle='--')
            
            lines_1, labels_1 = ax.get_legend_handles_labels()

            lines = lines_1 #+ lines_2 + lines_3
            labels = labels_1 #+ labels_2 + labels_3
            
            plt.savefig(save_name[:-3]+'png',format = "png",bbox_inches="tight",pad_inches=0.05)

            file_num += 1
            values = [i,nop,None,hrcc[0][0],hrcc[0][1],hrcc[1],predicted_hrcc,predicted_hrcc,predicted_hrcc]
            # values = [i,nop,None,None,None,hrcc,None,None,None]
            data = {}
            for c in range(len(col_name)):
                data[col_name[c]] = values[c]

            out = pd.DataFrame(data=[data],columns=col_name)
            out.to_csv(file,mode = 'a',header = False, index=False)
            plt.close()
            # print(file_num)
        file_num += 0
        mum_slopes_gxgh.append(num_slopes)


slopes_HARS = 1
def legend_func(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    # labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$D_q=K$',r'$D_q=U$',r'$D_q=N$']
    point_leg1 = []
    labels1 = []
    num_opts = [2,5,8,10,15,20,30,40,80,100]
    for i in range(len(num_opts)):
        # if num_opts[i]==2 or num_opts[i]==10 or num_opts[i]==100:
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(num_opts[i]))
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=2,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Number of options "+r"$n$",title_fontsize=18,markerscale=1.5, bbox_to_anchor=(1, 1),fontsize=18,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)

def legend_func_mum(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    # labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$D_q=K$',r'$D_q=U$',r'$D_q=N$']
    mum = [10,50,100,200,500]
    point_leg1 = []
    labels1 = []
    for i in range(len(mum)):
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(mum[i])+'$n$')
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=2,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Swarm size "+r"$S = m n$",bbox_to_anchor=(1, 1),title_fontsize=18,markerscale=1.5,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)

if slopes_HARS ==1:
    # plt.style.use('ggplot')
    # fig, ax = plt.subplots()
    # bxgh_fit = plot_slopes(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_slopes(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_slopes(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel(r'$\bf \mu_m$',fontproperties)
    # plt.ylabel('Slope of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"slope_mum.eps",bbox_inches="tight",pad_inches=0.2)
    # # plt.show()

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots()  
    # bxgh_fit = plot_inter(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_inter(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_inter(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda linear_fit : ('['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)
    
    # plt.xlabel(r'$\bf \mu_m$',fontproperties)
    # plt.ylabel('Intercept of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"Intercept_mum.eps",bbox_inches="tight",pad_inches=0.2)
    # # plt.show()

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots()
    # bxgh_fit = plot_HARS(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_HARS(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_HARS(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel(r'$\bf \mu_m$',fontproperties)
    # plt.ylabel('HARS of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"HARS_mum.eps",bbox_inches="tight",pad_inches=0.2)
    # # plt.show()


    # # # Fig 4 a
    # path = os.getcwd() + "/results_new/"
    # initials = "mu_q_vs_mu_h"
    # ylab1 = r'$\bf \mu_h\;vs\;\mu_q$'
    # bxgh_string = initials+'_bxgh_full'
    # uxgh_string = initials+'_uxgh_full'
    # gxgh_string = initials+'_gxgh_full'

    # fig, ax = plt.subplots()
    
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,line_style='--',predicted=1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,line_style='--',predicted=1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,line_style='--',predicted=1)

    # function = None#lambda exp_fit : ('['+r'$\bf %.2fn^{-3} %+.2fn^{-2} %+.2fn^{-1} %+.2f$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2),np.round(exp_fit[3],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # ax.set_yticks(np.arange(-2,6,1))
    # lab = [r'$\mu_h^{*}$','Simulation']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [10, 10], c='black',linestyle=styles[i],label=lab[i],linewidth=3) for i in range(len(styles))]
    # plt.ylim(-2.5,5.5)
    # plt.legend(loc='lower left',markerscale=3,prop=dict(size=12,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials+"Intercept_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()
    
    # path = os.getcwd() + "/results_new/"
    # initials = "mu_q_vs_mu_h"
    # ylab1 = r'$\bf \mu_h\;vs\;\mu_q$'
    # bxgh_string = initials+'_bxgh_full'
    # uxgh_string = initials+'_uxgh_full'
    # gxgh_string = initials+'_gxgh_full'
    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='distances',fit_type='a',first_variable='mum',line_style='-',vs_n = 1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='distances',fit_type='a',first_variable='mum',line_style='-',vs_n = 1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='distances',fit_type='a',first_variable='mum',line_style='-',vs_n = 1)

    # ax.set_yticks(np.arange(0,3,1))
    # plt.xlabel(r'$n$',fontproperties)
    # plt.ylabel('Diff.',fontproperties)
    # plt.savefig('Failure_mode'+initials+"distances.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # # # Fig 4 b
    # path = os.getcwd() + "/results_new/"
    # initials = "mu_q_vs_mu_h"
    # ylab1 = r'$\bf \mu_h\;vs\;\mu_q$'
    # bxgh_string = initials+'_bxgh_full'
    # uxgh_string = initials+'_uxgh_full'
    # gxgh_string = initials+'_gxgh_full'
    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)

    # function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # ax.set_yticks(np.arange(0.8,1.2,0.1))
    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials+"Slope_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # # # Fig 4 c
    # path = os.getcwd() + "/results_paper/"
    # initials = "sig_q_vs_mu_q_vs_mu_h"
    # ylab1 = r'$\bf \mu_h\;vs\;\mu_q$'
    # bxgh_string = initials+'_bxgh_1'
    # uxgh_string = initials+'_uxgh_1'
    # gxgh_string = initials+'_gxgh_1'

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=2,plot_against_sig=1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=2,plot_against_sig=1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=2,plot_against_sig=1)

    # function = None #lambda exp_fit : ('['+r'$\bf %+.2fn^{-1} %+.2f$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # # ax.text(13.5,-0.7,s="(c)")
    # plt.xlabel('Standard deviation, '+r'$\sigma_q$',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials+"Intercept_sigma_q.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # # Fig 6 b
    # path = os.getcwd() + "/mu_h=mu_h_pred_s_h_vs_sq/"
    # initials = 'sigma_h_vs_sigma_x_mu_h_pred'
    # fig, ax = plt.subplots()
    # # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',save_string=bxgh_string,function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1,predicted=1)
    # # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',save_string=uxgh_string,function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1,predicted=1)
    # # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',save_string=gxgh_string,function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1,predicted=1)
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='y_intercept_fit',fit_type='n_exp',first_variable='mum',vs_n = 1)
    # # bxgh_fit = plot_inte_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # # uxgh_fit = plot_inte_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # # gxgh_fit = plot_inte_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda exp_fit : ('['+r'$\bf %.2fn^{-3} %+.2fn^{-2} %+.2fn^{-1} %+.2f$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2),np.round(exp_fit[3],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # # ax.text(96,-2.05,s="(a)")
    # # ax.text(39,-2.3,s="(a)")
    # # ax.text(95,0.9,s="(b)")
    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode_full_data'+initials+"Intercept_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # # Fig 6 c
    # path = os.getcwd() + "/mu_h=mu_h_pred_s_h_vs_sq/"
    # initials = 'sigma_h_vs_sigma_x_mu_h_pred'
    # fig, ax = plt.subplots()
    # # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',save_string=bxgh_string,function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)
    # # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',save_string=uxgh_string,function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)
    # # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',save_string=gxgh_string,function='slope_fit',fit_type='a',first_variable='mum',vs_n = 1,predicted=1)
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='slope_fit',fit_type='log',first_variable='mum',vs_n = 1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='slope_fit',fit_type='log',first_variable='mum',vs_n = 1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='slope_fit',fit_type='log',first_variable='mum',vs_n = 1)
    # # bxgh_fit = plot_slopes_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # # uxgh_fit = plot_slopes_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # # gxgh_fit = plot_slopes_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # # ax.text(100,0.9425,s="(b)")
    # # ax.text(39,0.91,s="(b)")
    # # ax.text(96,0.855,s="(b)")
    # # ax.text(95,0.88,s="(c)")
    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials+"Slope_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # Fig S1
    path = os.getcwd() + "/results_new/"
    initials = "mu_q_vs_mu_h"
    ylab1 = r'$\bf \mu_h\;vs\;\mu_q$'
    bxgh_string = initials+'_bxgh_full'
    uxgh_string = initials+'_uxgh_full'
    gxgh_string = initials+'_gxgh_full'

    fig, ax = plt.subplots()

    bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0)
    uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0)
    gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0)

    function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    plt.ylim(0.85,1.2)
    ax.set_yticks(np.arange(0.85,1.25,0.05))
    plt.xlabel('Agents per option, m',fontproperties)
    plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    plt.savefig('Failure_mode'+initials+"Slope_m.pdf",bbox_inches="tight",pad_inches=0.0)
    plt.show()

    fig, ax = plt.subplots()

    bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0)
    uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0)
    gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0)

    function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    plt.ylim(-2,5.5)
    ax.set_yticks(np.arange(-2,5.5,1))
    plt.xlabel('Agents per option, m',fontproperties)
    plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    plt.savefig('Failure_mode'+initials+"intercept_m.pdf",bbox_inches="tight",pad_inches=0.0)
    plt.show()

    # # # Fig S4
    # path1 = os.getcwd() + "/mu_h=mu_x_s_h_vs_s_q/"
    # path = os.getcwd() + "/mu_h=mu_h_pred_s_h_vs_sq/"
    # initials = "sigma_h_vs_sigma_x_mu_h_pred"
    # initials1 = "sigma_h_vs_sigma_x_mu_x"
    # ylab1 = r'$\bf \sigma_h\;vs\;\sigma_q$'
    # bxgh_string = initials+'_bxgh_full'
    # uxgh_string = initials+'_uxgh_full'
    # gxgh_string = initials+'_gxgh_full'
    # bxgh_string1 = initials1+'_bxgh_full'
    # uxgh_string1 = initials1+'_uxgh_full'
    # gxgh_string1 = initials1+'_gxgh_full'

    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path1+bxgh_string1+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path1+uxgh_string1+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path1+gxgh_string1+'.csv',function='slope_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')

    # function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)


    # lab = [r'$\mu_h = \mu_h^{*}$',r'$\mu_h = \mu_q$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0.5,1.5)
    # plt.xlabel('Agents per option, m',fontproperties)
    # plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    # plt.legend(loc='upper right',markerscale=1.5,prop=dict(size=16,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials1+"Slope_m.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path1+bxgh_string1+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path1+uxgh_string1+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path1+gxgh_string1+'.csv',function='y_intercept_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')

    # function = None#lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = None#lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # lab = [r'$\mu_h = \mu_h^{*}$',r'$\mu_h = \mu_q$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.xlabel('Agents per option, m',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.legend(loc='upper right',markerscale=1.5,prop=dict(size=16,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials1+"intercept_m.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots()
    # bxgh_fit = plot_HARS_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_HARS_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_HARS_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options (n)',fontproperties)
    # plt.ylabel('HARS of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"HARS_n.eps",bbox_inches="tight",pad_inches=0.2)
    # # plt.show()


    # # # Fig S2
    # path1 = os.getcwd() + "/mu_h=mu_x_s_h_vs_s_q/"

    # initials1 = "sigma_h_vs_sigma_x_mu_x"
    # ylab1 = r'$\bf \sigma_h\;vs\;\sigma_q$'

    # bxgh_string1 = initials1+'_bxgh_full'
    # uxgh_string1 = initials1+'_uxgh_full'
    # gxgh_string1 = initials1+'_gxgh_full'

    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path1+bxgh_string1+'.csv',function='slope_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path1+uxgh_string1+'.csv',function='slope_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path1+gxgh_string1+'.csv',function='slope_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials1+"Slope_n.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path1+bxgh_string1+'.csv',function='y_intercept_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path1+uxgh_string1+'.csv',function='y_intercept_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path1+gxgh_string1+'.csv',function='y_intercept_fit',fit_type='log',first_variable='m',vs_n = 1,line_style='--')

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.savefig('Failure_mode'+initials1+"intercept_n.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,point_facecolor='black',line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,point_facecolor='black',line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,point_facecolor='black',line_style='-')


    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Agents per option, m',fontproperties)
    # plt.ylabel('HARS',fontproperties)
    
    # lab = [r'$\mu_h = \mu_h^{*}$',r'$\mu_h = \mu_x$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower right',markerscale=6,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('F7c_Failure_mode'+initials+initials1+"HARS_mum.pdf",bbox_inches="tight",pad_inches=0.0)
    
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,point_facecolor='black',line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,point_facecolor='black',line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,point_facecolor='black',line_style='-')


    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('HARS',fontproperties)
    
    # lab = [r'$\mu_h = \mu_h^{*}$',r'$\mu_h = \mu_x$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower right',markerscale=6,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('F7a_Failure_mode'+initials+initials1+"HARS_n.pdf",bbox_inches="tight",pad_inches=0.0)
    
    # plt.show()

    # fig, ax = plt.subplots()
    # print(np.array(d_bxgh))
    # print(np.array(d_uxgh))
    # print(np.array(d_gxgh))
    # d_bxgh_1 = [np.sum(np.array(d_bxgh)[:,i])/len(np.array(d_bxgh)[:,i]) for i in range(len(d_bxgh[0]))]
    # d_uxgh_1 = [np.sum(np.array(d_uxgh)[:,i])/len(np.array(d_uxgh)[:,i]) for i in range(len(d_uxgh[0]))]
    # d_gxgh_1 = [np.sum(np.array(d_gxgh)[:,i])/len(np.array(d_gxgh)[:,i]) for i in range(len(d_gxgh[0]))]


    # bxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_bxgh_1,color='slateblue',linewidth = 3)
    # uxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_uxgh_1,color='lightseagreen',linewidth = 3)
    # gxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_gxgh_1,color='coral',linewidth = 3)

    # # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # function = None
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel("Number of options "+r'$n$',fontproperties)
    # plt.ylabel('Distance between HARS fit \n and prediction in '+ylab1,fontproperties)
    # plt.savefig(initials+"distances.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',save_string=bxgh_string,function='distances',fit_type='a',first_variable='mum',vs_n = 1)
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',save_string=uxgh_string,function='distances',fit_type='a',first_variable='mum',vs_n = 1)
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',save_string=gxgh_string,function='distances',fit_type='a',first_variable='mum',vs_n = 1)

    # # d_bxgh_1 = [np.sum(np.array(d_bxgh)[:,i])/len(np.array(d_bxgh)[:,i]) for i in range(len(d_bxgh[0]))]
    # # d_uxgh_1 = [np.sum(np.array(d_uxgh)[:,i])/len(np.array(d_uxgh)[:,i]) for i in range(len(d_uxgh[0]))]
    # # d_gxgh_1 = [np.sum(np.array(d_gxgh)[:,i])/len(np.array(d_gxgh)[:,i]) for i in range(len(d_gxgh[0]))]


    # # bxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_bxgh_1,color='slateblue',linewidth = 3)
    # # uxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_uxgh_1,color='lightseagreen',linewidth = 3)
    # # gxgh_fit = ax.plot([2,5,8,10,15,20,30,40],d_gxgh_1,color='coral',linewidth = 3)

    # # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # function = None
    # # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel(r'$n$',fontproperties)
    # plt.ylabel('Diff.',fontproperties)
    # plt.savefig('Failure_mode'+initials+"distances.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',f_path1=path+bxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',f_path1=path+uxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',f_path1=path+gxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    

    # # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # # ax.text(460,0.192,s="(b)",fontsize=18,fontweight='bold')
    # plt.xlabel('m',fontproperties)
    # plt.ylabel(r'$\Delta HARS$',fontproperties)
    # plt.savefig('F7d_Failure_mode'+initials1+initials+"HARS_mum_err.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',f_path1=path+bxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',f_path1=path+uxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',f_path1=path+gxgh_string1+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    

    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # ax.text(460,0.192,s="(b)",fontsize=18,fontweight='bold')
    # ax.set_yticks(np.arange(0,0.4,0.1))
    # plt.xlabel('n',fontproperties)
    # plt.ylabel(r'$\Delta HARS$',fontproperties)
    # plt.savefig('F7b_Failure_mode'+initials1+initials+"HARS_mum_err.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path1+bxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path1+uxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path1+gxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':')
    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('HARS',fontproperties)
    
    # # lab = [r'$\mu_h = \mu_h^{*}$',r'$\mu_h = \mu_x$']
    # # styles = ['-','--']
    # # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower right',markerscale=6,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # # plt.savefig('Failure_mode'+initials+"HARS_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style=':',marker_line='o')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style=':',marker_line='o')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style=':',marker_line='o')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-.')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-.')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-.')

    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Agents per option, m',fontproperties)
    # plt.ylabel('HARS',fontproperties)
    
    # lab = [r'$D_h = D_h^{*}$',r'$D_h = D_q$',r'$\mu_h = \mu_h^{*}$',r'$\sigma_h = \sigma_h^{*}$']
    # styles = ['-','--',':','-.']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower right',markerscale=6,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials+initials2+initials3+"HARS_m.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':',marker_line='o')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':',marker_line='o')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string2+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style=':',marker_line='o')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-.')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-.')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string3+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-.')

    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('HARS',fontproperties)
    
    # lab = [r'$D_h = D_h^{*}$',r'$D_h = D_q$',r'$\mu_h = \mu_h^{*}$',r'$\sigma_h = \sigma_h^{*}$']
    # styles = ['-','--',':','-.']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower left',markerscale=6,prop=dict(size=18,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials+initials2+initials3+"HARS_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()


    #Fig 7 (all)
    # path = os.getcwd() + "/F7/"

    # initials = 'Hars_Dh*_'
    # initials1 = 'Hars_Dq_'

    # bxgh_string = initials+'_bxgh_full'
    # uxgh_string = initials+'_uxgh_full'
    # gxgh_string = initials+'_gxgh_full'

    # bxgh_string1 = initials1+'_bxgh_full'
    # uxgh_string1 = initials1+'_uxgh_full'
    # gxgh_string1 = initials1+'_gxgh_full'
    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='--')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='m',vs_n = 1,line_style='-')

    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options, n',fontproperties)
    # plt.ylabel('Average rate of success',fontproperties)
    
    # lab = [r'$D_h = D_h^{*}$',r'$D_h = D_q$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower right',markerscale=4,prop=dict(size=16,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials1+"HARS_n.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()
    
    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='--')

    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string+'.csv',function='mu_hars_fit',fit_type='a',first_variable='n',vs_n = 0,line_style='-')

    # function = None#lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Agents per option, m',fontproperties)
    # plt.ylabel('Average rate of success',fontproperties)
    
    # lab = [r'$D_h = D_h^{*}$',r'$D_h = D_q$']
    # styles = ['-','--']
    # lines = [plt.plot([0, 0], [2, 2], c='black',linestyle=styles[i],label=lab[i],linewidth=5) for i in range(len(styles))]
    # plt.ylim(0,1)
    # plt.legend(loc='lower left',markerscale=4,prop=dict(size=16,weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    # plt.savefig('Failure_mode'+initials+initials1+"HARS_m.pdf",bbox_inches="tight",pad_inches=0.0)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',f_path1=path+bxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',f_path1=path+uxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',f_path1=path+gxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='m',vs_n = 1,line_style='-')
    

    # ax.set_yticks(np.arange(0,0.7,0.2))
    # plt.xlabel('n',fontproperties)
    # plt.ylabel(r'Improvement',fontproperties)
    # plt.savefig('F7b_Failure_mode'+initials1+initials+"HARS_n_err.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = genericPlotter(ax,distribution=1,color='slateblue',f_path=path+bxgh_string1+'.csv',f_path1=path+bxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # uxgh_fit = genericPlotter(ax,distribution=2,color='lightseagreen',f_path=path+uxgh_string1+'.csv',f_path1=path+uxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    # gxgh_fit = genericPlotter(ax,distribution=3,color='coral',f_path=path+gxgh_string1+'.csv',f_path1=path+gxgh_string+'.csv',function='hars_diff',fit_type='a',first_variable='n',vs_n = 0,line_style='-')
    

    # ax.set_yticks(np.arange(0,0.5,0.2))
    # plt.xlabel('n',fontproperties)
    # plt.ylabel(r'Improvement',fontproperties)
    # plt.savefig('F7c_Failure_mode'+initials1+initials+"HARS_m_err.pdf",bbox_inches="tight",pad_inches=0.02)
    # plt.show()



def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))

    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

reflection_plot = 0
prd = yn.Prediction()
font = {"fontsize":18,"fontweight":'bold'}
from matplotlib import rc,rcParams
rc('font', weight='bold',size=18)
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
sigmas = [1/3,1,3]
# sigmas = [1]

if reflection_plot == 1:
    fig,axs = plt.subplots(1,3)#,figsize=(100,30)
    step = 0.0001
    number_of_options = [50]
    dist = prd.gaussian
    # dist = prd.uniform
    mu_x = List([10])
    sigma_x = List([1])
    low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])#,mu_x[1]-np.sqrt(3)*sigma_x[1]])
    high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])#,mu_x[1]+np.sqrt(3)*sigma_x[1]])

    for mus in range(2):
        for sigma_ in range(len(sigmas)):
            if mus == 0:
                # delta_mu = 5
                start1 = np.sum(mu_x)/len(mu_x) - np.sum(sigmas[sigma_])-5
                stop1 = np.sum(mu_x)/len(mu_x) + np.sum(sigmas[sigma_])+5
                dis_x = np.round(np.arange(start1,stop1,step),decimals=4)
                pdf =  dist(dis_x,mu_x,sigma_x)
                area = (np.sum(pdf)*step)
                pdf_x = np.multiply(pdf,1/area)

                for nop in range(len(number_of_options)):
                    mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,stop1,step,dis_x,pdf_x)
                    axs[sigma_].axvline(mean_esmes2m,0,500,color='red',label = r'$\bf \mu_{h}^{*}$',linewidth = 2,linestyle="--")
                    
                    # sigma_h = List([0.17*np.log10(number_of_options[nop]) + 0.46])
                    sig = prd.ICPDF((1-(1/(number_of_options[nop]**2))),mu_x,stop1,step,dis_x,pdf_x) - mean_esmes2m
                    
                    
                    # axs[sigma_].axvline(mean_esmes2m+sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 1,linestyle="--")
                    # axs[sigma_].axvline(mean_esmes2m-sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 1,linestyle="--")
                    # axs[sigma_].axvline(mean_esmes2m+sig,0,500,color='green',label = r'$\bf \sigma_{h-pred}$',linewidth = 1,linestyle="--")

                    axs[sigma_].invert_yaxis()
                    axs[sigma_].plot(dis_x,pdf_x,color="#882255",linewidth=3,label="Quality PDF "+r'$N_q$')
                    
                    slices = []
                    mid_slices=[]
                    for i in range(1,number_of_options[nop],1):
                        ESM = prd.ICPDF(float(i)/number_of_options[nop],mu_x,stop1,step,dis_x,pdf_x)
                        slices.append(np.round(ESM,decimals=3))
                    for i in range(1,2*number_of_options[nop],1):
                        if i%2!=0:
                            mid_slices.append(np.round(prd.ICPDF((i/(2*number_of_options[nop])),mu_x,stop1,step,dis_x,pdf_x),decimals=1))

                    number_of_colors = number_of_options[nop]

                    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                for i in range(number_of_colors)]

                    # for i in range(len(slices)+1):
                    #     if i!=0 and i!=len(slices):
                    #         x1 = np.arange(slices[i-1],slices[i],0.0001)
                    #         pdf1 =  dist(x1,mu_x,sigma_x)
                    #         pdf1 = np.multiply(pdf1,1/area)
                    #         axs[sigma_].fill_between(x1,0,pdf1,facecolor=color[i])
                    #     elif i==0:
                    #         x1 = np.arange(start1,slices[i],0.0001)
                    #         pdf1 =  dist(x1,mu_x,sigma_x)
                    #         pdf1 = np.multiply(pdf1,1/area)
                    #         axs[sigma_].fill_between(x1,0,pdf1,facecolor=color[i])
                    #     elif i==len(slices):
                    #         x1 = np.arange(slices[-1],stop1,0.0001)
                    #         pdf1 =  dist(x1,mu_x,sigma_x)
                    #         pdf1 = np.multiply(pdf1,1/area)
                    #         axs[sigma_].fill_between(x1,0,pdf1,facecolor=color[i])

                    for i in range(50):
                        # ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
                        ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
                        # axs.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
                        
                        axs[sigma_].scatter([max(options_quality)],[np.random.uniform(low = 0.01,high = 0.05)],s=9,edgecolor = 'green',facecolor="green")
                        ind = np.where(options_quality==max(options_quality))
                        options_quality[ind[0]]= -100
                        axs[sigma_].scatter([max(options_quality)],[np.random.uniform(low = 0.01,high = 0.05)],s=9,edgecolor = 'red',facecolor="red",marker="x")
                        ind = np.where(options_quality==max(options_quality))
                        options_quality[ind[0]]= -100
                        # axs[sigma_].scatter(options_quality,(0.02+i*0.01)*np.ones_like(options_quality),s=9,edgecolor = 'black',facecolor="white")
                        # options_quality
                        # axs[sigma_].scatter(max(options_quality),(0.02+i*0.01)],s=9,edgecolor = 'black',facecolor="black")
                        
                        # if sigma_*mus==0:
                        # #     axs[0].text(start1-2,(0.32-i*0.1),'trial '+str(i),font,color='black')
                        #     axs[0].text(start1+2,0.05,r'$\bf \mu_q$',font,color='black') 
            
            if mus == 1:
                mu_h = List([mean_esmes2m])
            else:
                mu_h = mu_x
            
            sigma_h = List([sigmas[sigma_]])
            start = np.sum(mu_h)/len(mu_h) - np.sum(sigma_h)-8
            stop = np.sum(mu_h)/len(mu_h) + np.sum(sigma_h)+8

            dis_h = np.round(np.arange(start,stop,step),decimals=4)
            pdf =  prd.gaussian(dis_h,mu_h,sigma_h)
            area = (np.sum(pdf)*step)
            pdf_h = np.multiply(pdf,1/area)
            axs1 = axs[sigma_].twinx()
            if mus == 1:
                axs1.plot(dis_h,pdf_h,color='indigo',linewidth=3,label="Response threshold PDF "+r'$N_h$')
            else:
                axs1.plot(dis_h,pdf_h,color='indigo',linewidth=3,linestyle='--')
            # for i in range(3):
            #     units = rng.threshold_n(m_units=number_of_options[nop]*100,mu_h=mu_h,sigma_h=sigma_h)
            #     # axs1.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            #     axs1.scatter(units,(0.3/sigmas[sigma_]-i*0.1/sigmas[sigma_])*np.ones_like(units),s=9,edgecolor = 'black')
            #     if sigma_*mus==0:
            # #         axs1.text(start1-2,(0.3/sigmas[sigma_]-i*0.1/sigmas[sigma_]),'trial '+str(i),font,color='black')
            #         axs[0].text(start1+2,-0.05,r'$\bf \mu_h$',font,color='black')
            

            lines_1, labels_1 = axs[sigma_].get_legend_handles_labels()
            lines_2, labels_2 = axs1.get_legend_handles_labels()

            lines = lines_1 + lines_2
            labels = labels_1 + labels_2        
            
            align_yaxis(axs1,0.0,axs[sigma_],0.0)
            axs[sigma_].set_yticks([])
            axs1.set_yticks([])
            axs[sigma_].tick_params(axis='both', which='major', labelsize=18,labelcolor='black')
            axs[sigma_].set_xlabel(r'$\sigma_h =$'+str(np.round(sigmas[sigma_]/sigma_x[0],decimals=1))+'$\sigma_q$', fontsize=18,color='black')
        st = fig.suptitle("Number of samples drawn = "+str(number_of_options[nop]),fontsize=18,fontweight='bold',color='black')
        st.set_y(0.73)
        st.set_x(0.5)
        # plt.title("Number of samples drawn = "+str(number_of_options[nop]),fontsize=18,fontweight='bold',color=(0.3,0.3,0.3,1))
        fig.legend(lines, labels,loc='upper right',prop=dict(weight='bold',size=18), bbox_to_anchor=(0.92, 0.7),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=22,frameon=False)
        # fig.tight_layout()
    plt.show()

# if __name__=="__main__":
#     plt.style.use('ggplot')
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, projection='3d')
#     co = ["slateblue","lightseagreen","coral"]
#     for c in range(len(co)):
#         plot_3d(np.random.uniform(0,1,(2,4,2)),[10,50],[2,5,10,20],ax=ax1,color=co[c],distribution=c+1,save_string=str(c+1)+'000000000',sd=0)
#     plt.show()
#     fig, ax = plt.subplots()
#     plot_slopes(np.random.uniform(0,1,(5,4,2)),[10,50,100,200,500],[2,5,10,20],ax,'bxgh',color='red')
#     plt.legend()
#     plt.xlabel(r'$\mu_m$')
#     plt.ylabel('Slope of best fit')
#     plt.show()

#     fig, ax = plt.subplots()
#     plot_HARS(np.random.uniform(0,1,(5,4,2)),[10,50,100,200,500],[2,5,10,20],ax,'bxgh',color='red')
#     plt.legend()
#     plt.xlabel(r'$\mu_m$')
#     plt.ylabel('HARS of best fit')
#     plt.show()

#     fig, ax = plt.subplots()
#     plot_slopes_1(np.random.uniform(0,1,(5,4,2)),[10,50,100,200,500],[2,5,10,20],ax,'bxgh',color='red')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('Slope of best fit')
#     plt.show()

#     fig, ax = plt.subplots()
#     plot_HARS_1(np.random.uniform(0,1,(5,4,2)),[10,50,100,200,500],[2,5,10,20],ax,'bxgh',color='red')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('HARS of best fit')
#     plt.show()

