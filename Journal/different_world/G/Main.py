# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

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

def parallel(func,inputs):
	batch_size = 100
	inps = [(i,) for i in inputs]
	output = []
	for i in range(0,len(inputs),batch_size):
		opt_var = []
		with Pool(20) as processor:#,ray_address="auto") as p:
			opt_var = processor.starmap(func,inps[i:i+batch_size])

		output += list(opt_var)
	return output

def looper(p,vP,path,mem_size,t_step):
    options = parallel(partial(obj.Object,o_type='O',p=p,mem_size=mem_size),range(p.num_opts))
    robots = parallel(partial(obj.Object,o_type='R',p=p,mem_size=mem_size,options=options),range(p.num_robots))
    for j in range(3500):
        # np.random.shuffle(robots)
        options = parallel(partial(obj.Object,o_type='O',p=p,mem_size=mem_size),range(p.num_opts))
        for r in robots:
            r.response = 0
            r.reassignment(options)
        # anim = Animator(robots,options,p)
        decision = DM.DecisionMaking(robots,options,p,j,mem_size,t_step)
        # plt.show()
        matching = decision.compare_with_best(options)
        data = {'$\mu_{h_1}$':p.mu_h_1,'$\sigma_{h_1}$':p.sigma_h_1,'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':options[decision.ref_best].quality,'$CDM$ opt No.':decision.best_option,'$CDM$':options[decision.best_option].quality}
    
        for i in range(len(robots)):
            data[str(i)] = robots[i].threshold
            data['Response_'+str(i)] = robots[i].response
        data['mem_size'] = mem_size
        data['thr_step'] = t_step
        out = pd.DataFrame(data=[data],columns=vP.data_columns_name)
        # print(data)
        out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
        if decision.threshold_update:
            decision.updateThresholds(robots,p,vP)


def paramObjects(fixed_params,vP_):
	p = params.Params()
	p.initializer(vP_,n = fixed_params[0]['n'], Dm = fixed_params[0]['$D_{m}$'], mum1 = fixed_params[0]['$\mu_{m_{1}}$'],\
		mum2 = fixed_params[0]['$\mu_{m_{2}}$'], sm1 = fixed_params[0]['$\sigma_{m_{1}}$'], sm2 = fixed_params[0]['$\sigma_{m_{2}}$'],\
		dmum = vP.delta_mu_m, Dx = fixed_params[0]['$D_{x}$'], mux1 = fixed_params[0]['$\mu_{x_{1}}$'],\
		mux2 = fixed_params[0]['$\mu_{x_{2}}$'], sx1 = fixed_params[0]['$\sigma_{x_{1}}$'], sx2 = fixed_params[0]['$\sigma_{x_{2}}$'],\
		dmux = vP.delta_mu_x, Dh = fixed_params[0]['$D_{h}$'], muh1 = fixed_params[0]['$\mu_{h_{1}}$'], muh2 = fixed_params[0]['$\mu_{h_{2}}$'], sh1 = fixed_params[0]['$\sigma_{h_{1}}$'], sh2 = fixed_params[0]['$\sigma_{h_{2}}$'],\
		dmuh = fixed_params[0]['$\delta_{\mu_{h}}$'])
	p.packaging(vP_)
	return p

if __name__=="__main__":
    path = os.getcwd() + "/results/"
    plt.close()
    plt.ion()
    data_columns = ['$\mu_{h_1}$','$\sigma_{h_1}$','$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$']
    for i in range(250):
        data_columns.append(str(i))
        data_columns.append('Response_'+str(i))
    data_columns.append('mem_size')
    data_columns.append('thr_step')
    vP = ps.variableParams(data_columns,path)
    
    for n in [5]:
        vP.pre = vP.prefix(path)
        fixed_params_column = ['n','$D_{m}$','$\mu_{m_{1}}$','$\mu_{m_{2}}$','$\sigma_{m_{1}}$','$\sigma_{m_{2}}$',\
			'$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$',\
			'$D_{h}$','$\mu_{h_{1}}$','$\mu_{h_{2}}$','$\sigma_{h_{1}}$','$\sigma_{h_{2}}$','$\delta_{\mu_{h}}$'] 
        fixed_params = [{'n': n, '$D_{m}$':vP.Dm[0],'$\mu_{m_{1}}$':vP.mu_m_1[1],'$\mu_{m_{2}}$':vP.mu_m_2[1],\
			'$\sigma_{m_{1}}$':vP.sigma_m_1[0],'$\sigma_{m_{2}}$':vP.sigma_m_2[0],'$D_{x}$':vP.Dx[0],\
			'$\mu_{x_{1}}$':vP.mu_x_1[99],'$\mu_{x_{2}}$':vP.mu_x_2[99],'$\sigma_{x_{1}}$':vP.sigma_x_1[9],\
			'$\sigma_{x_{2}}$':vP.sigma_x_2[9], '$\delta_{\mu_{x}}$':0, '$D_{h}$':None, '$\mu_{h_{1}}$': 20, '$\mu_{h_{2}}$': 20,\
			'$\sigma_{h_{1}}$':0, '$\sigma_{h_{2}}$':0,'$\delta_{\mu_{h}}$':0}]


		# # parallel(partial(looper,copy.copy(P),vP,path))
        thr_for_mem = [0.3,0.5,0.8,1]
        memory_length = range(2,22,2)
        mem_for_thr = [2,10,20,30]
        threshold_step = np.arange(0.2,1.2,0.1)

        for th in thr_for_mem:
            for mem_size in memory_length:
                for times in range(20):
                    P = paramObjects(fixed_params=fixed_params,vP_=vP)
                    vP.save_params(fixed_params_column,fixed_params,path)
                    vP.save_data(P,path)
                    looper(P,vP,path,mem_size,th)
                    
        for m in mem_for_thr:
            for t_step in threshold_step:
                for times in range(20):
                    P = paramObjects(fixed_params=fixed_params,vP_=vP)
                    vP.save_params(fixed_params_column,fixed_params,path)
                    vP.save_data(P,path)
                    looper(P,vP,path,m,t_step)

        
