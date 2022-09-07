#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

from cProfile import label
import numpy as np
import classexploration1 as yn
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
path = os.getcwd() + "/results1/"

WRC_normal = 0
pval_WRC_normal = 0
pval_WRC_bimodal_x_gaussian_h = 0
pval_WRC_uniform_x_gaussian_h = 0
bimodal_x_normal_h = 1
bimodal_x_normal_h_sigma = 0
uniform_x_uniform_h = 0
uniform_x_uniform_h_sigma = 0
uniform_x_normal_h = 1
uniform_x_normal_h_sigma = 0
normal_x_normal_h = 1
normal_x_normal_h_1 = 0
normal_x_normal_h_sigma = 0

wf = yn.workFlow()
vis = yn.Visualization()
fontproperties = {'weight' : 'bold', 'size' : 18}

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

def save_plot_data(mum_slopes,mum,num_opts,save_string=None):
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
        ax.scatter(x1,y1,s=40,marker=marker[i], facecolors='none', edgecolors=color)
    xf /= len(num_opts)
    yf /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*np.round(linear_fit[0],decimals=2) + np.round(linear_fit[1],decimals=2),linewidth=1,color = color,label=distribution + '['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # ax.text(400,min(dz),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))

    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(20,min(dz),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=2,color = color)
    
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    # ax.set_ylim([0, 1])
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
        ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(num_opts[i])+'_'+distribution)
    xf /= len(num_opts)
    yf /= len(num_opts)
    linear_fit = np.polyfit(xf,yf,1)
    ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=2,color = color)#,label=distribution+'['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # ax.text(400,min(inte),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
    # ax.plot(xf,np.exp(np.array(xf)*exp_fit[0]+exp_fit[1]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    # ax.set_ylim([-0.5, 1])
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
        ax.scatter(x1,y1, facecolors='none', edgecolors=color,s=40,marker=marker[i])#,label=str(mum[i])+'_'+distribution)
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

    # if distribution==2:
    #     exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    #     ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=2,color = color)
    # else:
    #     exp_fit = np.polyfit(np.log10(xf),yf,2)
    #     ax.plot(xf,(np.log10(xf)*np.log10(xf)*exp_fit[0]+np.log10(xf)*exp_fit[1]+exp_fit[2]),linewidth=2,color = color)

    if distribution==2:
        exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
        ax.plot(xf,np.exp(np.array(xf)*exp_fit[0] + exp_fit[1]),linewidth=2,color = color)
    else:
        exp_fit = np.polyfit(np.log10(xf),yf,1)
        ax.plot(xf,(np.log10(xf)*exp_fit[0]+exp_fit[1]),linewidth=2,color = color)

    # ax.set_ylim([0, 1])
    return exp_fit

def plot_inte_n(ax,distribution,color,save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    num_opts = np.unique(y)
    predicted_intercepts = []
    if distribution=="bxgh":
        mu = List([0,5])
        sigma = List([1,1])
    else:
        mu = List([0,0])
        sigma = List([1,1])
    step = 0.0001
    prd = yn.Prediction()
    
    start = np.sum(mu)/len(mu) - 2*np.sum(sigma)-5
    stop = np.sum(mu)/len(mu) + 2*np.sum(sigma)+5
    dis_x = np.arange(start,stop,step)
    if distribution=="uxgh":
        distribution_fn = prd.uniform
    else:
        distribution_fn = prd.gaussian
    pdf =  distribution_fn(dis_x,mu,sigma)
    pdf = np.multiply(pdf,1/(np.sum(pdf)*step))
    for n in num_opts:
        _1 = prd.ICPDF(1-(1/n),mu,stop,step,dis_x,pdf)
        predicted_intercepts.append(_1)

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

        ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(mum[i])+'_'+distribution)
    xf /= len(mum)
    yf /= len(mum)
    # ax.plot(xf,yf,linewidth=1,color = 'black')#,label=distribution+'['+r'$\bf e^{%.2fn^{2} %+.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2),np.round(exp_fit[3],decimals=2))+']')
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'['+str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2))+']')
    # ax.text(20,min(inte),str(np.round(linear_fit[0],decimals=2))+'x + '+str(np.round(linear_fit[1],decimals=2)))
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),3)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*np.array(xf)*exp_fit[0] + np.array(xf)*np.array(xf)*exp_fit[1] + np.array(xf)*exp_fit[2] + exp_fit[3]),linewidth=1,color = color)
    exp_fit = np.polyfit(1/np.array(xf),np.array(yf),3)
    ax.plot(xf,(1/(np.array(xf)*np.array(xf)*np.array(xf)))*exp_fit[0] + (1/(np.array(xf)*np.array(xf)))*exp_fit[1] + (1/(np.array(xf)))*exp_fit[2] + exp_fit[3],linewidth=2,color = color)
    # ax.plot(num_opts,predicted_intercepts,linewidth=3,color = color,linestyle="-.")
    # ax.set_ylim([-0.5, 1])
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

        ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(num_opts[i])+'_'+distribution)
    xf /= len(num_opts)
    yf /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'[ %.2fx %+.2f'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(400,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=2,color = color)#,label=distribution+'['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    
    plt.xticks(fontsize=16,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    ax.set_ylim([0, 1.05])
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
        ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(mum[i])+'_'+distribution)
    xf /= len(mum)
    yf /= len(mum)
    exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=2,color = color)#,label=distribution+'['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(20,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))

    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')
    ax.set_ylim([0, 1.05])
    return exp_fit

def plot_HARS_n_err(ax,distribution,color,save_string,save_string1):
    f_path = path+save_string+'.csv'
    f_path1 = path+save_string1+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    xx,yy,zz,dxx,dyy,dzz,trans1,inte1 = read_slopes_HARS(f_path1,columns_name)
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
        # ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)
    xf /= len(mum)
    yf /= len(mum)

    exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=2,color = color)#,label=distribution+'['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    
    xf1 = np.zeros(len(num_opts))
    yf1 = np.zeros(len(num_opts))
    for i in range(len(mum)):
        x1 = []
        y1 = []
        for j in range(len(num_opts)):
            x1.append(num_opts[j])
            y1.append(trans1[i*len(num_opts)+j])
        xf1 += np.array(x1)
        yf1 += np.array(y1)
        # ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)
    xf1 /= len(mum)
    yf1 /= len(mum)
    exp_fit1 = np.polyfit(xf1,np.log(np.array(yf1)),2)
    # ax.plot(xf1,np.exp(np.array(xf1)*np.array(xf1)*exp_fit1[0]+np.array(xf1)*exp_fit1[1] + exp_fit1[2]),linewidth=2,color = color)#,label=distribution+'['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')

    ax.plot(xf,np.exp(np.array(xf1)*np.array(xf1)*exp_fit1[0]+np.array(xf1)*exp_fit1[1] + exp_fit1[2]) - np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=5,color = color)#,label=distribution+'['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    
    # ax.text(20,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))

    plt.xticks(fontsize=16,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    return exp_fit

def plot_HARS_err(ax,distribution,color,save_string,save_string1):
    f_path = path+save_string+'.csv'
    f_path1 = path+save_string1+'.csv'
    columns_name = ['$\mu_{m}$','n','z','dx','dy','Slope of bestfit','HARS','Intercept']
    x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    xx,yy,zz,dxx,dyy,dzz,trans1,inte1 = read_slopes_HARS(f_path1,columns_name)
    num_opts = np.unique(y)
    mum = np.unique(x)
    
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']

    xf = np.zeros(len(mum))
    yf = np.zeros(len(mum))
    for i in range(len(num_opts)):
        x1 = []
        y1 = []
        for j in range(len(mum)):
            x1.append(mum[j])
            y1.append(trans[i+j*len(num_opts)])
        xf += np.array(x1)
        yf += np.array(y1)

        # ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(num_opts[i])+'_'+distribution)
    xf /= len(num_opts)
    yf /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'[ %.2fx %+.2f'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(400,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))
    log_fit = np.polyfit(np.log10(xf),yf,2)
    # ax.plot(xf,(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=2,color = color)#,label=distribution+'['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    

    xf1 = np.zeros(len(mum))
    yf1 = np.zeros(len(mum))
    for i in range(len(num_opts)):
        x1 = []
        y1 = []
        for j in range(len(mum)):
            x1.append(mum[j])
            y1.append(trans1[i+j*len(num_opts)])
        xf1 += np.array(x1)
        yf1 += np.array(y1)

        # ax.scatter(x1,y1, facecolors='none', edgecolors=color,marker=marker[i],s=40)#,label=str(num_opts[i])+'_'+distribution)
    xf1 /= len(num_opts)
    yf1 /= len(num_opts)
    # linear_fit = np.polyfit(xf,yf,1)
    # ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=1,color = color,label=distribution+'[ %.2fx %+.2f'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    
    # exp_fit = np.polyfit(xf,np.log(np.array(yf)),2)
    # ax.plot(xf,np.exp(np.array(xf)*np.array(xf)*exp_fit[0]+np.array(xf)*exp_fit[1] + exp_fit[2]),linewidth=1,color = color,label=distribution+'['+r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # ax.text(400,min(trans),r'$e^{%.2fx^{2} + %.2fx + %.2f}$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2),np.round(linear_fit[2],decimals=2)))
    log_fit1 = np.polyfit(np.log10(xf1),yf1,2)
    ax.plot(xf,(np.log10(xf1)*np.log10(xf1)*log_fit1[0]+np.log10(xf1)*log_fit1[1]+log_fit1[2])-(np.log10(xf)*np.log10(xf)*log_fit[0]+np.log10(xf)*log_fit[1]+log_fit[2]),linewidth=5,color = color)
    plt.xticks(fontsize=16,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    return log_fit

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
    mum_bxgh = [100]
    number_of_opts_bxgh = [40]
    cnt = 510
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
            num_slopes_bxgh.append([slope,hars,intercept])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
        cnt += 0
        mum_slopes_bxgh.append(num_slopes_bxgh)

if bimodal_x_normal_h_sigma==1:
    continuation = False
    mum_bxgh = [100]
    file_num = 230#0
    mum_slopes_bxgh = []
    number_of_opts_bxgh = [40]
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
            num_slopes.append([slope,hars,intercept])

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            file_num += 1
            print(file_num)
        file_num += 1
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
    continuation = False
    cnt = 429
    mum_uxgh = [10,50,100,200,500]
    number_of_opts_uxgh = [2,5,8,10,15,20,30,40]
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
        runs = 500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes_uxgh = []
        for nop in number_of_opts_uxgh:
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
            num_slopes_uxgh.append([slope,hars,intercept])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
        cnt += 0
        mum_slopes_uxgh.append(num_slopes_uxgh)

if uniform_x_normal_h_sigma==1:
    continuation = False
    mum_uxgh = [10,50,100,200,500]
    number_of_opts_uxgh = [2,3,4,5,8,10,15,20,30,40,80,100]
    cnt = 260#65
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
            num_slopes.append([slope,hars,intercept])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
            print(cnt)
        cnt += 1
        mum_slopes_uxgh.append(num_slopes)

if normal_x_normal_h==1:
    continuation = False
    mum_gxgh = [100]
    cnt = 512
    number_of_opts_gxgh = [2,5,10]
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
        # sigma_x_1 = 1
        # sigma_x_2= 1
        # sigma_h_1=sigma_x_1
        # sigma_h_2=sigma_x_1
        sigma_h_1=1
        sigma_h_2=1
        sigma_x_1 = sigma_h_1/3
        sigma_x_2= sigma_h_1/3
        runs = 500
        batch_size = 50
        delta_mu = 0
        mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]
        num_slopes_gxgh = []
        for nop in number_of_opts_gxgh:
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
            num_slopes_gxgh.append([slope,hars,intercept])
            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'Pval',x_var_='$\mu_{x_1}$',y_var_='Average p-value',num_of_opts=number_of_opts,plot_type='line')
            # vis.data_visualize(file_name=save_string+".csv",save_plot=save_string+'WRC',x_var_='$\mu_{x_1}$',y_var_='Wrong_ranking_cost_without_no_proportion',num_of_opts=number_of_opts,plot_type='line')

            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            cnt += 1
        cnt += 0
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
    continuation = False
    mum_gxgh = [10,50,100,200,500]
    number_of_opts_gxgh = [2,3,4,5,8,10,15,20,30,40,80,100]
    file_num = 325#130
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
            num_slopes.append([slope,hars,intercept])
            message = str(nop)+' number of options simulation finished'
            pushbullet_message('Python Code','Results out! '+message)
            file_num += 1
            print(file_num)
        file_num += 1
        mum_slopes_gxgh.append(num_slopes)

figures3d = 0
# initials = "mu_h=mu_x"
initials1 = "mu_h=mu_h_pred"
initials = "mu_h=mu_h_pred"
# initials = "sigma_h=sigma_x"
ylab2 = r'$\bf \mu_h\;vs.\;\mu_q$'
ylab1 = r'$\bf \sigma_h\;vs.\;\sigma_q$'
bxgh_string = initials+'_bxgh_all_mum2'
uxgh_string = initials+'_uxgh_all_mum2'
gxgh_string = initials+'_gxgh_all_mum2'

bxgh_string1 = initials1+'_bxgh_all_mum2'
uxgh_string1 = initials1+'_uxgh_all_mum2'
gxgh_string1 = initials1+'_gxgh_all_mum2'

saving = 0
if saving:
    save_plot_data(np.array(mum_slopes_bxgh),mum_bxgh,number_of_opts_bxgh,save_string=bxgh_string)
    save_plot_data(np.array(mum_slopes_uxgh),mum_uxgh,number_of_opts_uxgh,save_string=uxgh_string)
    save_plot_data(np.array(mum_slopes_gxgh),mum_gxgh,number_of_opts_gxgh,save_string=gxgh_string)

if figures3d:

    # plt.style.use('ggplot')
    # fig = plt.figure(tight_layout=True)
    # ax1 = fig.add_subplot(111, projection='3d')

    # plot_3d(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # plot_3d(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # plot_3d(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    
    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,['bxgh','uxgh','gxgh']+error+error+error,loc='upper left', bbox_to_anchor=(1.05, 1))
    # plt.show()

    # plt.style.use('ggplot')
    # fig = plt.figure(tight_layout=True)
    # ax1 = fig.add_subplot(111, projection='3d')

    # plot_3d_inter(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # plot_3d_inter(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # plot_3d_inter(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,['bxgh','uxgh','gxgh']+error+error+error,loc='upper left', bbox_to_anchor=(1.05, 1))
    # plt.show()

    # plt.style.use('ggplot')
    # fig = plt.figure(tight_layout=True)
    # ax1 = fig.add_subplot(111, projection='3d')

    # plot_3d(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # plot_3d(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # plot_3d(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")

    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,['bxgh','uxgh','gxgh']+error+error+error,loc='upper left', bbox_to_anchor=(1.05, 1))
    # plt.show()

    # plt.style.use('ggplot')
    # fig = plt.figure(tight_layout=True)
    # ax1 = fig.add_subplot(111, projection='3d')

    # plot_3d_inter(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # plot_3d_inter(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # plot_3d_inter(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")

    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,['bxgh','uxgh','gxgh']+error+error+error,loc='upper left', bbox_to_anchor=(1.05, 1))
    # plt.show()
    ####################################
    # plt.style.use('ggplot')
    # fig, ax1 = plt.subplots()
    # log_fit = plot_2d(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # leg1 = 'bxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    # log_fit = plot_2d(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # leg2 = 'uxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    # log_fit = plot_2d(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    # leg3 = 'gxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'

    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,[leg1,leg2,leg3]+error+error+error,loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # plt.show()

    # plt.style.use('ggplot')
    # fig, ax1 = plt.subplots()
    # log_fit = plot_2d_inter(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1,dist='bxgh')
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # leg1 = 'bxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'
    # log_fit = plot_2d_inter(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1,dist='uxgh')
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # leg2 = 'uxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'
    # log_fit = plot_2d_inter(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1,dist='gxgh')
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    # leg3 = 'gxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'

    # error = [0,.2,.4,.6,.8,1]
    # h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    # h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    # h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    # ax1.legend([legend1,legend2,legend3]+h1+h2+h3,[leg1,leg2,leg3]+error+error+error,loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # plt.show()

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    log_fit = plot_2d(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1)
    legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    leg1 = 'bxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    log_fit = plot_2d(ax=ax1,distribution=3,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1)
    legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    leg2 = 'uxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    log_fit = plot_2d(ax=ax1,distribution=2,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1)
    legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    leg3 = 'gxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'

    error = [0,.2,.4,.6,.8,1]
    h = [(plt.scatter([],[],s=55, c="slateblue",alpha=i),plt.scatter([],[],s=55, c="lightseagreen",alpha=i),plt.scatter([],[],s=55, c="coral",alpha=i)) for i in error]
    
    ax1.legend([legend1,legend2,legend3]+h,[leg1,leg2,leg3]+[str(error[i])+ " HARS" for i in range(len(error))],loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.savefig(initials+"Slope_n_mum10.pdf",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # plt.style.use('ggplot')
    # fig, ax1 = plt.subplots()
    # log_fit = plot_2d_inter(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1,dist='bxgh')
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # leg1 = 'bxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'
    # log_fit = plot_2d_inter(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1,dist='uxgh')
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # leg2 = 'uxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'
    # log_fit = plot_2d_inter(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1,dist='gxgh')
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    # leg3 = 'gxgh ['+r'$\bf %.2flog_{10}^{2}{n} %+.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']'

    # error = [0,.2,.4,.6,.8,1]
    # h = [(plt.scatter([],[],s=55, c="slateblue",alpha=i),plt.scatter([],[],s=55, c="lightseagreen",alpha=i),plt.scatter([],[],s=55, c="coral",alpha=i)) for i in error]
    
    # ax1.legend([legend1,legend2,legend3]+h,[leg1,leg2,leg3]+[str(error[i])+ " HARS" for i in range(len(error))],loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),handler_map={tuple: HandlerTuple(ndivide=None)})
    # plt.savefig("Intercept_n_mum10.pdf",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # plt.style.use('ggplot')
    # fig, ax1 = plt.subplots()
    # # fig1, ax2 = plt.subplots()
    # left, bottom, width, height = [0.35, 0.2, 0.45, 0.4]
    # ax2 = fig.add_axes([left, bottom, width, height])
    # log_fit = plot_2d(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,one = 1,ylab = ylab1)
    # legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    # leg1 = 'bxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    # log_fit = plot_2d(ax=ax2,distribution=3,color="lightseagreen",save_string=uxgh_string,one = 1,ylab = ylab1)
    # legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    # leg2 = 'uxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'
    # log_fit = plot_2d(ax=ax1,distribution=2,color="coral",save_string=gxgh_string,one = 1,ylab = ylab1)
    # legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    # leg3 = 'gxgh ['+r'$\bf %.2flog_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']'

    # error = [0,.2,.4,.6,.8,1]
    # h = [(plt.scatter([],[],s=55, c="slateblue",alpha=i),plt.scatter([],[],s=55, c="lightseagreen",alpha=i),plt.scatter([],[],s=55, c="coral",alpha=i)) for i in error]
    
    # ax1.legend([legend1,legend2,legend3]+h,[leg1,leg2,leg3]+error,loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),handler_map={tuple: HandlerTuple(ndivide=None)})
    # plt.show()
    

err_plt = 0

if err_plt:
    bxgh_string1 = initials+'_bxgh_all_mum2'
    uxgh_string1 = initials+'_uxgh_all_mum2'
    gxgh_string1 = initials+'_gxgh_all_mum2'
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    plot_2d_slope_error(ax=ax1,distribution=1,color="slateblue",save_string=bxgh_string,save_string1=bxgh_string1,one = 1,ylab = ylab1)
    legend1 = plt.Rectangle((0, 0), 1, 1, fc="slateblue")
    leg1 = 'bxgh'
    plot_2d_slope_error(ax=ax1,distribution=2,color="lightseagreen",save_string=uxgh_string,save_string1=uxgh_string1,one = 1,ylab = ylab1)
    legend2 = plt.Rectangle((0, 0), 1, 1, fc="lightseagreen")
    leg2 = 'uxgh'
    plot_2d_slope_error(ax=ax1,distribution=3,color="coral",save_string=gxgh_string,save_string1=gxgh_string1,one = 1,ylab = ylab1)
    legend3 = plt.Rectangle((0, 0), 1, 1, fc="coral")
    leg3 = 'gxgh'

    error = [0,.2,.4,.6,.8,1]
    h1 = [plt.scatter([],[],s=55, c="slateblue",alpha=i) for i in error]
    h2 = [plt.scatter([],[],s=55, c="lightseagreen",alpha=i) for i in error]
    h3 = [plt.scatter([],[],s=55, c="coral",alpha=i) for i in error]
    ax1.legend([legend1,legend2,legend3]+h1+h2+h3,[leg1,leg2,leg3]+error+error+error,loc='upper left', bbox_to_anchor=(1, 1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    plt.show()


slopes_HARS = 0
def legend_func(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$K_q$',r'$U_q$',r'$N_q$']
    point_leg1 = []
    labels1 = []
    num_opts = [2,3,4,5,8,10,15,20,30,40,80,100]
    for i in range(len(marker)):
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(num_opts[i]))
    leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=5,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Number of options "+r"$n$", bbox_to_anchor=(1, 1),fontsize=18,prop=dict(weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)


def legend_func_mum(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$K_q$',r'$U_q$',r'$N_q$']
    mum = [10,50,100,200,500]
    point_leg1 = []
    labels1 = []
    for i in range(len(mum)):
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(mum[i])+'$n$'+' agents')
    leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=5,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Swarm size "+r"$S = m n$",bbox_to_anchor=(1, 1),fontsize=18,prop=dict(weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)


if slopes_HARS ==1:
    # fig, ax = plt.subplots()
    # bxgh_fit = plot_slopes(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_slopes(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_slopes(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel(r'$\bf \mu_m$',fontproperties)
    # plt.ylabel('Slope of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"slope_mum.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()  
    # bxgh_fit = plot_inter(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_inter(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_inter(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda linear_fit : ('['+r'$\bf %.2f\mu_m %+.2f$'%(np.round(linear_fit[0],decimals=2),np.round(linear_fit[1],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)
    
    # plt.xlabel(r'$\bf \mu_m$',fontproperties)
    # plt.ylabel('Intercept of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"Intercept_mum.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    fig, ax = plt.subplots()
    bxgh_fit = plot_HARS(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    uxgh_fit = plot_HARS(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    gxgh_fit = plot_HARS(ax,'gxgh',color='coral',save_string=gxgh_string)

    function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    plt.xlabel("Mean sub-swarm size "+r'$\bf\mu_m$',fontproperties)
    plt.ylabel(r'$\mu_{HARS}$'+' of HARS line in '+ylab1,fontproperties)
    plt.savefig(initials+"HARS_mum.eps",bbox_inches="tight",pad_inches=0.2)
    plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = plot_inte_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_inte_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_inte_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda exp_fit : ('['+r'$\bf %.2fn^{-3} %+.2fn^{-2} %+.2fn^{-1} %+.2f$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2),np.round(exp_fit[3],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # # ax.text(38,0.1,s="(a)")
    # ax.text(95,-0.34,s="(a)")
    # plt.xlabel('Number of options (n)',fontproperties)
    # plt.ylabel('y-intercept of\n HARS line in '+ylab1,fontproperties)
    # plt.tight_layout()
    # plt.savefig(initials+"Intercept_n.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = plot_slopes_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_slopes_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_slopes_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{n} %+.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # # function1 = lambda exp_fit : ('['+r'$e^{%.2fn^{2} %+.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}{n} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2))+']')
    # function1 = lambda exp_fit : ('['+r'$e^{%.2fn %+.2f}$'%(np.round(exp_fit[0],decimals=2),np.round(exp_fit[1],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function1,ax)

    # # ax.text(38,0.9425,s="(b)")
    # ax.text(95,0.51,s="(b)")
    # plt.xlabel('Number of options (n)',fontproperties)
    # plt.ylabel('Slope of\n HARS line in '+ylab1,fontproperties)
    # plt.tight_layout()
    # plt.savefig(initials+"Slope_n.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = plot_HARS_n(ax,'bxgh',color='slateblue',save_string=bxgh_string)
    # uxgh_fit = plot_HARS_n(ax,'uxgh',color='lightseagreen',save_string=uxgh_string)
    # gxgh_fit = plot_HARS_n(ax,'gxgh',color='coral',save_string=gxgh_string)

    # function = lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # plt.xlabel('Number of options (n)',fontproperties)
    # plt.ylabel(r'$\mu_{HARS}$'+' of best fit in '+ylab1,fontproperties)
    # plt.savefig(initials+"HARS_n.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = plot_HARS_n_err(ax,'bxgh',color='slateblue',save_string=bxgh_string,save_string1=bxgh_string1)
    # uxgh_fit = plot_HARS_n_err(ax,'uxgh',color='lightseagreen',save_string=uxgh_string,save_string1=uxgh_string1)
    # gxgh_fit = plot_HARS_n_err(ax,'gxgh',color='coral',save_string=gxgh_string,save_string1=gxgh_string1)

    # function = lambda exp_fit : ('['+r'$\bf e^{%.2fn %+.2f}$'%(np.round(exp_fit[1],decimals=2),np.round(exp_fit[2],decimals=2))+']')
    # legend_func_mum(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # ax.text(2,0.22,s="(a)",fontsize=18,fontweight='bold')
    # plt.xlabel('Number of options (n)',fontproperties)
    # plt.ylabel(r'$\Delta_{\mu_{HARS}}$'+' of \n HARS line in '+ylab1,fontproperties)
    # plt.savefig(initials1+initials+"HARS_n_diff.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

    # fig, ax = plt.subplots()
    # bxgh_fit = plot_HARS_err(ax,'bxgh',color='slateblue',save_string=bxgh_string,save_string1=bxgh_string1)
    # uxgh_fit = plot_HARS_err(ax,'uxgh',color='lightseagreen',save_string=uxgh_string,save_string1=uxgh_string1)
    # gxgh_fit = plot_HARS_err(ax,'gxgh',color='coral',save_string=gxgh_string,save_string1=gxgh_string1)

    # function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}^{2}{\mu_m} %+.2f \log_{10}{\mu_m} %+.2f$'%(np.round(log_fit[0],decimals=2),np.round(log_fit[1],decimals=2),np.round(log_fit[2],decimals=2))+']')
    # legend_func(function,bxgh_fit,uxgh_fit,gxgh_fit,function,ax)

    # ax.text(460,0.192,s="(b)",fontsize=18,fontweight='bold')
    # plt.xlabel("Mean sub-swarm size "+r'$\bf\mu_m$',fontproperties)
    # plt.ylabel(r'$\Delta_{\mu_{HARS}}$'+' of \n HARS line in '+ylab1,fontproperties)
    # plt.savefig(initials1+initials+"HARS_mum_err.eps",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

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
# font = {"fontsize":18,"fontweight":'bold'}
# from matplotlib import rc,rcParams
# rc('font', weight='bold',size=18)
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

if visua == 1:
    step = 0.0001
    
    number_of_options = [30]
    # plt.style.use('ggplot')
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
        axs.axvline(mean_esmes2m,0,500,color='red',label = r'$\bf \mu^*_h$',linewidth = 5)

        mu_h = List([mean_esmes2m])
        # mu_h = mu_x
        # sigma_h = List([1])
        sigma_h = List([0.17*np.log10(number_of_options[nop]) + 0.46])
        sig = prd.ICPDF((1-(1/(number_of_options[nop]**2))),mu_x,stop1,step,dis_x,pdf_x) - mean_esmes2m
        
        
        # axs.axvline(mean_esmes2m+sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 0.5)
        # axs.axvline(mean_esmes2m-sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 0.5)
        # axs.axvline(mean_esmes2m+sig,0,500,color='green',label = r'$\bf \sigma_{h-pred}$',linewidth = 0.5)

        start = np.sum(mu_h)/len(mu_h) - np.sum(sigma_h)-5
        stop = np.sum(mu_h)/len(mu_h) + np.sum(sigma_h)+5

        dis_h = np.round(np.arange(start,stop,step),decimals=4)
        pdf =  prd.gaussian(dis_h,mu_h,sigma_h)
        area = (np.sum(pdf)*step)
        pdf_h = np.multiply(pdf,1/area)

        axs.invert_yaxis()
        axs.plot(dis_x,pdf_x,color='green',label=r'$\bf U_q$',linewidth=5)
        
        

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

        # for i in range(len(slices)+1):
        #     if i!=0 and i!=len(slices):
        #         x1 = np.arange(slices[i-1],slices[i],0.0001)
        #         pdf1 =  dist(x1,mu_x,sigma_x)
        #         pdf1 = np.multiply(pdf1,1/area)
        #         axs.fill_between(x1,0,pdf1,facecolor=color[i])
        #     elif i==0:
        #         x1 = np.arange(start,slices[i],0.0001)
        #         pdf1 =  dist(x1,mu_x,sigma_x)
        #         pdf1 = np.multiply(pdf1,1/area)
        #         axs.fill_between(x1,0,pdf1,facecolor=color[i])
        #     elif i==len(slices):
        #         x1 = np.arange(slices[-1],stop,0.0001)
        #         pdf1 =  dist(x1,mu_x,sigma_x)
        #         pdf1 = np.multiply(pdf1,1/area)
        #         axs.fill_between(x1,0,pdf1,facecolor=color[i])

        for i in range(3):
            ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
            # ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
            # axs.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            # axs.scatter(options_quality,(0.3-i*0.1)*np.ones_like(options_quality),s=3,edgecolor = 'black')
            # axs.text(start1+2,(0.3-i*0.1)-0.01,'trial '+str(i),font,color=(0.3,0.3,0.3,1))
            # axs.text(start1+2,0.05,r'$\bf N_q$',font,color=(0.3,0.3,0.3,1))
            
        axs1 = axs.twinx()
        axs1.plot(dis_h,pdf_h,color='indigo',label=r'$\bf N_h$',linewidth=5)
        for i in range(3):
            units = rng.threshold_n(m_units=number_of_options[nop]*100,mu_h=mu_h,sigma_h=sigma_h)
            # axs1.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            # axs1.scatter(units,(0.3-i*0.1)*np.ones_like(units),s=1,edgecolor = 'black')
            # axs1.text(start1+2,(0.3-i*0.1)+0.01,'trial '+str(i),font,color=(0.3,0.3,0.3,1))
            # axs.text(start1+2,-0.03,r'$\bf N_h$',font,color=(0.3,0.3,0.3,1))
        
        axs.legend(loc='upper left',prop=dict(weight='bold',size=20),labelcolor=(0.3,0.3,0.3,1),frameon = False)
        axs1.legend(loc='center left',bbox_to_anchor=(0, 0.65),prop=dict(weight='bold',size=20),labelcolor=(0.3,0.3,0.3,1),frameon = False)
        # axs.set_aspect('equal')
        # axs1.set_aspect('equal')
        # axs.title.set_text("Number of samples drawn = "+str(number_of_options[nop]))
        # axs.title.set_fontweight('bold')
        # axs.title.set_fontsize(18)
        # axs.title.set_color("dimgray")
        align_yaxis(axs1,0.0,axs,0.0)
        axs.set_yticks([])
        axs1.set_yticks([])
        axs.tick_params(axis='both', which='major', labelsize=18)

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
