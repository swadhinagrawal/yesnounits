# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

from random import uniform
import numpy as np
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import Object as obj
import DecisionMaking as DM
import Params as params
import ParamSet as ps
import os
from functools import partial
import copy
from scipy.special import lambertw
import time
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

def option_assignment(num_opts,num_robo):
    opts = np.arange(1,num_opts+1)
    assig = np.copy(opts)
    for i in range(int(num_robo/num_opts)):
        assig = np.concatenate((assig,np.copy(opts)))
    np.random.shuffle(assig)
    return assig

def deciding_phase(p,robots,time_counter):
    options = parallel(partial(obj.Object,o_type = 'O', p = p),range(p.num_opts))

    opt_assignment =  option_assignment(num_opts = p.num_opts, num_robo = p.num_opts*p.mu_m_1)
    for r in robots[:p.num_opts*p.mu_m_1]:
        r.assigned_opt = opt_assignment[r.id]

    decision = DM.DecisionMaking(robots[:p.num_opts*p.mu_m_1],options,p,time_counter,p.memory_length,p.delta_h)

    matching = decision.compare_with_best(options)
    
    return decision, options


def nullifier(i,robotss,run):
    robotss[i].response = 0	# 0 = "No", 1 = "Yes"
    robotss[i].opt = -1 #self.response*self.assigned_opt
    robotss[i].best_opt = None
    robotss[i].threshold_update = 0

def param_change(i,p,robotss):
    robotss[i].p = p

def looper(p,vP):
    robots = parallel(partial(obj.Object,o_type = 'R', p = p),range(p.num_robots))
    data = {'$\mu_{h_1}$':np.round(p.mu_h_1,decimals=3),'$\sigma_{h_1}$':np.round(p.sigma_h_1,decimals=3),'$x_{max}$ opt No.':None,'$x_{max}$':None,'$CDM$ opt No.':None,'$CDM$':None}
    for i in range(len(robots)):
        data['h_'+str(i)] = np.round(robots[i].threshold,decimals=3)
        data['tr_'+str(i)] = np.round(robots[i].t_r,decimals=3)
        data['Response_'+str(i)] = robots[i].response
        data['rt_step'+str(i)] = robots[i].delta_t_r
    data['Dq'] = p.Dx
    data['$\mu_{q_1}$'] = p.mu_x_1
    data['$\mu_{q_2}$'] = p.mu_x_2
    data['$\sigma_{q_1}$'] = p.sigma_x_1
    data['$\sigma_{q_2}$'] = p.sigma_x_2
    data['n'] = p.num_opts
    out = pd.DataFrame(data=[data],columns=p.data_columns_name)
    out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
    for run in range(20):
        for t in range(10000):
            parallel(partial(nullifier,robotss=robots,run=run),range(p.num_robots))
            decision,options = deciding_phase(p,robots[:p.num_opts*p.mu_m_1],t)
            for r in robots[:p.num_opts*p.mu_m_1]:
                if r.threshold_update:
                    decision.updateThresholds(r,p)
            decision.updateparams(robots[:p.num_opts*p.mu_m_1],p,vP)
            if t%500 == 0:
                data = {'$\mu_{h_1}$':np.round(p.mu_h_1,decimals=3),'$\sigma_{h_1}$':np.round(p.sigma_h_1,decimals=3),'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':np.round(options[decision.ref_best].quality,decimals=3),'$CDM$ opt No.':decision.best_option,'$CDM$':np.round(options[decision.best_option].quality,decimals=3)}
                for i in range(len(robots[:p.num_opts*p.mu_m_1])):
                    data['h_'+str(i)] = np.round(robots[i].threshold,decimals=3)
                    data['tr_'+str(i)] = np.round(robots[i].t_r,decimals=3)
                    data['Response_'+str(i)] = robots[i].response
                    data['rt_step'+str(i)] = robots[i].delta_t_r
                out = pd.DataFrame(data=[data],columns=p.data_columns_name)
                out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
                for repeat in range(100):
                    parallel(partial(nullifier,robotss=robots,run=run),range(p.num_robots))
                    decision,options = deciding_phase(p,robots[:p.num_opts*p.mu_m_1],1)

                    data = {'$\mu_{h_1}$':np.round(p.mu_h_1,decimals=3),'$\sigma_{h_1}$':np.round(p.sigma_h_1,decimals=3),'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':np.round(options[decision.ref_best].quality,decimals=3),'$CDM$ opt No.':decision.best_option,'$CDM$':np.round(options[decision.best_option].quality,decimals=3)}
                    out = pd.DataFrame(data=[data],columns=p.data_columns_name)
                    out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)

        # Update world
        p.Dx = vP.Dx[np.random.randint(3)]
        p.mu_x_1 = (p.mu_x_2 + p.mu_x_1)/2 + 1 #+ np.random.randint(2)*(-2)
        if p.Dx == 'G' or p.Dx == 'U':
            p.mu_x_2 = p.mu_x_1
        else:
            p.mu_x_2 = p.mu_x_1 + 5
        p.sigma_x_1 = 1
        p.sigma_x_2 = 1
        p.num_opts = np.random.randint(101)
        p.mu_m_1 = int(p.num_robots/p.num_opts)
        p.mu_m_2 = int(p.num_robots/p.num_opts)
        
        p.packaging(vP)
        p.data_columns_name = p.data_columns_name[:6]
        p.add_columns(p.num_robots)
        p.pre = p.prefix(vP.results_path)
        open(vP.results_path+str(p.pre),'w+')
        p.save_data(vP.results_path)
        time.sleep(5)
        parallel(partial(param_change,p=p,robotss=robots),range(p.num_robots))

        data['Dq'] = p.Dx
        data['$\mu_{q_1}$'] = p.mu_x_1
        data['$\mu_{q_2}$'] = p.mu_x_2
        data['$\sigma_{q_1}$'] = p.sigma_x_1
        data['$\sigma_{q_2}$'] = p.sigma_x_2
        data['n'] = p.num_opts

        out = pd.DataFrame(data=[data],columns=p.data_columns_name)
        out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)

def paramObjects(fixed_params,vP_,data_columns,path,fixed_params_column):
    p = params.Params(data_columns,path)
    p.initializer(vP_,n = fixed_params[0]['n'], Dm = fixed_params[0]['$D_{m}$'], mum1 = fixed_params[0]['$\mu_{m_{1}}$'],\
        mum2 = fixed_params[0]['$\mu_{m_{2}}$'], sm1 = fixed_params[0]['$\sigma_{m_{1}}$'], sm2 = fixed_params[0]['$\sigma_{m_{2}}$'],\
        dmum = vP_.delta_mu_m, Dx = fixed_params[0]['$D_{x}$'], mux1 = fixed_params[0]['$\mu_{x_{1}}$'],\
        mux2 = fixed_params[0]['$\mu_{x_{2}}$'], sx1 = fixed_params[0]['$\sigma_{x_{1}}$'], sx2 = fixed_params[0]['$\sigma_{x_{2}}$'],\
        dmux = vP_.delta_mu_x, Dh = fixed_params[0]['$D_{h}$'], muh1 = fixed_params[0]['$\mu_{h_{1}}$'],\
        muh2 = fixed_params[0]['$\mu_{h_{2}}$'], sh1 = fixed_params[0]['$\sigma_{h_{1}}$'], sh2 = fixed_params[0]['$\sigma_{h_{2}}$'],\
        dmuh = fixed_params[0]['$\delta_{\mu_{h}}$'], delta_tr = fixed_params[0]['delta_tr'])

    p.packaging(vP_)
    p.add_columns(p.num_robots)
    p.pre = p.prefix(path)
    p.save_params(fixed_params_column,fixed_params,path)
    p.save_data(path)
    return p

if __name__=="__main__":
    path = os.path.realpath(os.path.dirname(__file__)) + "/results/"

    data_columns = ['$\mu_{h_1}$','$\sigma_{h_1}$','$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$']
    
    packed_parameters = []
    vP = ps.variableParams()
    gaussianM = 1
    uniformM = 0
    # for delt_tr in np.arange(0.001,0.21,0.01):
    if gaussianM:
        # for n in vP.num_opts:
        n = 8
        delt_tr = 1/(n*n)
        fixed_params_column = ['n','$D_{m}$','$\mu_{m_{1}}$','$\mu_{m_{2}}$','$\sigma_{m_{1}}$','$\sigma_{m_{2}}$',\
            '$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$',\
            '$D_{h}$','$\mu_{h_{1}}$','$\mu_{h_{2}}$','$\sigma_{h_{1}}$','$\sigma_{h_{2}}$','$\delta_{\mu_{h}}$','delta_tr'] 
        fixed_params = [{'n': n, '$D_{m}$':vP.Dm[0],'$\mu_{m_{1}}$':vP.mu_m_1[1],'$\mu_{m_{2}}$':vP.mu_m_2[1],\
            '$\sigma_{m_{1}}$':vP.sigma_m_1[0],'$\sigma_{m_{2}}$':vP.sigma_m_2[0],'$D_{x}$':vP.Dx[0],\
            '$\mu_{x_{1}}$':17,'$\mu_{x_{2}}$':17,'$\sigma_{x_{1}}$':vP.sigma_x_1[9],\
            '$\sigma_{x_{2}}$':vP.sigma_x_2[9], '$D_{h}$':None, '$\mu_{h_{1}}$': 13, '$\mu_{h_{2}}$': 13,\
            '$\sigma_{h_{1}}$':0, '$\sigma_{h_{2}}$':0,'$\delta_{\mu_{h}}$':0,'delta_tr':delt_tr}]

        packed_parameters.append(paramObjects(fixed_params=fixed_params,vP_=vP,data_columns=copy.copy(data_columns),path=path,fixed_params_column=fixed_params_column))

    elif uniformM:
        for n in vP.num_opts[:5]:
            fixed_params_column = ['n','$D_{m}$','$\mu_{m_{1}}$','$\mu_{m_{2}}$','$\sigma_{m_{1}}$','$\sigma_{m_{2}}$',\
                '$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$',\
                '$D_{h}$','$\mu_{h_{1}}$','$\mu_{h_{2}}$','$\sigma_{h_{1}}$','$\sigma_{h_{2}}$','$\delta_{\mu_{h}}$','delta_tr'] 
            fixed_params = [{'n': n, '$D_{m}$':vP.Dm[0],'$\mu_{m_{1}}$':vP.mu_m_1[1],'$\mu_{m_{2}}$':vP.mu_m_2[1],\
                '$\sigma_{m_{1}}$':vP.sigma_m_1[0],'$\sigma_{m_{2}}$':vP.sigma_m_2[0],'$D_{x}$':vP.Dx[0],\
                '$\mu_{x_{1}}$':vP.mu_x_1[99],'$\mu_{x_{2}}$':vP.mu_x_2[99],'$\sigma_{x_{1}}$':vP.sigma_x_1[9],\
                '$\sigma_{x_{2}}$':vP.sigma_x_2[9], '$D_{h}$':None, '$\mu_{h_{1}}$': 13, '$\mu_{h_{2}}$': 13,\
                '$\sigma_{h_{1}}$':0, '$\sigma_{h_{2}}$':0,'$\delta_{\mu_{h}}$':0,'delta_tr':None}]

            packed_parameters.append(paramObjects(fixed_params=fixed_params,vP_=vP,data_columns=copy.copy(data_columns),path=path,fixed_params_column=fixed_params_column))

    looper(p = packed_parameters[0], vP = vP)

