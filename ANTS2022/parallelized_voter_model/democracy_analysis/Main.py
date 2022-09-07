# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import Animator as anim
import Object as obj
import DecisionMaking as DM
import VoterModel as VM
import Params as params
import ParamSet as ps
import os
from functools import partial
import copy

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

def looper(p,vP,path,no_units):
    for i in range(100):
        options = parallel(partial(obj.Object,o_type='O',p=p),range(p.num_opts))
        qualities = np.array([i.quality for i in options])
        robots_to_assign = np.ones(p.num_opts)*no_units
        p.sub_group_size = np.zeros_like(robots_to_assign)
        for i in range(len(robots_to_assign)):
            maximums = np.where(qualities==max(qualities))[0]
            choosen = np.random.choice(maximums)
            options[choosen].to_be_assigned_count = robots_to_assign[i]
            p.sub_group_size[i] = robots_to_assign[i]
            qualities[choosen] = -100
        no_responses = np.sum(robots_to_assign - np.array([40,39,30,25,21]))
        p.num_robots = int(np.sum(robots_to_assign))
        robots = parallel(partial(obj.Object,o_type='R',p=p,options=options),range(p.num_robots))
		# anim = Animator(robots,options,p)
        decision = DM.DecisionMaking(robots,options,p)
        model = VM.VoterModel()
		# plt.show()
        model.dissemination(robots,options,p)
        matching = decision.compare_with_best(options,model.best_option)
        if isinstance(model.best_option,type(None)) == False and isinstance(decision.ref_best,type(None)) == False:
            out = pd.DataFrame(data=[{'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':options[decision.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations,'No_units':no_responses}],columns=vP.data_columns_name)
            print({'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':options[decision.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations,'No_units':no_responses})
        else:
            out = pd.DataFrame(data=[{'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations,'No_units':no_responses}],columns=vP.data_columns_name)
            print({'$x_{max}$ opt No.':decision.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations,'No_units':no_responses})
        out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)


def paramObjects(fixed_params,vP_):
	p = params.Params()
	p.initializer(vP_,n = fixed_params[0]['n'], Dx = fixed_params[0]['$D_{x}$'], mux1 = fixed_params[0]['$\mu_{x_{1}}$'],\
		mux2 = fixed_params[0]['$\mu_{x_{2}}$'], sx1 = fixed_params[0]['$\sigma_{x_{1}}$'], sx2 = fixed_params[0]['$\sigma_{x_{2}}$'],\
		dmux = vP_.delta_mu_x)
	p.packaging(vP_)
	return p

if __name__=="__main__":
    path = os.getcwd() + "/results/"            # Path where all output results are stored

    vP = ps.variableParams(['$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$','Iterations','No_units'],path)      # Set of all the parameters from where we derive parameters to explore, this also contains some utility functions

    fixed_params_column = ['n','$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$']         # These are the parameters that are kept fixed in these experiments
    fixed_params = [{'n': vP.num_opts[2],'$D_{x}$':vP.Dx[0],'$\mu_{x_{1}}$':vP.mu_x_1[69],'$\mu_{x_{2}}$':vP.mu_x_2[69],\
        '$\sigma_{x_{1}}$':vP.sigma_x_1[9],'$\sigma_{x_{2}}$':vP.sigma_x_2[9]}]

    no_units = [50]
    P = paramObjects(fixed_params=fixed_params,vP_=vP)
    vP.save_params(fixed_params_column,fixed_params,path)
    vP.save_data(P,path)

    for n in no_units:
        looper(P,vP,path,n)
    # parallel(partial(looper,copy.copy(P),vP,path),no_units)