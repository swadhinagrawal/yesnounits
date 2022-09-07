# Hello

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt,exp
import pandas as pd
import os
import time
from functools import partial
import random
from numba.typed import List
from numba import guvectorize, float64, njit,jit
import copy
import random_number_generator as rng
import itertools
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import random_number_generator as rng
from numba.typed import List
import matplotlib.pyplot as plt
import random
from matplotlib.legend_handler import HandlerTuple

path = os.getcwd() + "/results/"
class variableParams:
	def __init__(self):
		#   Option details
		self.num_opts = [2,3,5,8,10,15,20,30,40,80,100]

		self.mu_x_1 = np.arange(0.1,15,0.1)
		self.mu_x_2 = np.arange(0.1,15,0.1)
		self.sigma_x_1 = np.arange(0.1,15,0.1)
		self.sigma_x_2 = np.arange(0.1,15,0.1)
		self.Dx = ['G','U','K']
		self.delta_mu_x = 0

		#   Robot details
		self.mu_m_1 = [10,50,100,200,500]
		self.sigma_m_1 = np.arange(0,15,1)
		self.mu_m_2 = [10,50,100,200,500]
		self.sigma_m_2 = np.arange(0,15,1)
		self.Dm = ['G','U','K']
		self.delta_mu_m = 0

		self.mu_h_1 = np.arange(0.1,15,0.1)
		self.mu_h_2 = np.arange(0.1,15,0.1)
		self.sigma_h_1 = np.arange(0.1,15,0.1)
		self.sigma_h_2 = np.arange(0.1,15,0.1)
		self.Dh = ['G','U','K']
		self.delta_mu_h = 0

		self.data_columns_name = ['$\mu_{h_1}$','$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$','Iterations']
		self.pre = self.prefix()

	@staticmethod
	@guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
	def gaussian(x,mu,sigma,result):
		n = x.shape[0]
		m = mu.shape[0]
		for j in range(x.shape[0]):
			f = 0.0
			for i in range(len(mu)):
				k = 1/(sqrt(2*np.pi)*sigma[i])
				f += k*exp(-((x[j]-mu[i])**2)/(2*sigma[i]**2))
			result[j] = f

	@staticmethod
	@guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
	def uniform(x,mu,sigma,result):
		n = x.shape[0]
		m = mu.shape[0]
		for j in range(n):
			f = 0.0
			for i in range(m):
				a = (mu[i] - np.sqrt(3)*sigma[i])
				b = (mu[i] + np.sqrt(3)*sigma[i])
				if x[j]<=b and x[j]>=a:
					f += 1/abs(b-a)
			result[j] = f

	@staticmethod
	@njit
	def ICPDF(area,mu,stop,step,x,pdf):

		if len(mu)>1:    
			dummy_area = 0.5
			x_ = (mu[0]+mu[1])/2.0
			count = np.argmin(np.abs(x-x_))
		else:
			dummy_area =0.5
			x_ = mu[0]

		count = np.argmin(np.abs(x-x_))
		
		while abs(dummy_area-area)>0.001:
			if dummy_area>area:
				count -= 1
				dummy_area -= pdf[count]*step
				x_ -= step
			elif area>dummy_area:
				count += 1
				dummy_area += pdf[count]*step
				x_ += step
		return x_

	def prefix(self):
		check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))
		count = 0
		for i in check:
			if count==i:
				count+=1
		return count

	def save_data(self,p):		
		save_string = str(self.pre) + '_' + str(time.time())
		f = open(path+save_string+'.csv','a')
		p.data_f_path = path+save_string+'.csv'
		
		columns = pd.DataFrame(data=np.array([self.data_columns_name]))
		columns.to_csv(p.data_f_path,mode='a',header=False,index=False)

	def save_params(self,param_columns_name,fixed_param):
		counter = open(path+str(self.pre),'w+')
		param = open(path+str(self.pre)+'.csv','w+')
		columns_p = pd.DataFrame(data=np.array([param_columns_name]))
		columns_p.to_csv(path+str(self.pre)+'.csv',mode='a',header=False,index=False)
		out_p = pd.DataFrame(data=fixed_param,columns=param_columns_name)	
		out_p.to_csv(path+str(self.pre)+'.csv',mode = 'a',header = False, index=False)	

class Params:
	def __init__(self):
		#   Simulation params
		self.data_f_path = None
		self.save_string = None
		self.columns_name = None
		self.dt = 0.1
		self.boundary_min = 0
		self.boundary_max = 100
		self.robot_size = 10
		self.use_predicted = 0
		
		#   Option details
		self.num_opts = None
		
		self.mu_x_1 = None
		self.mu_x_2 = None
		self.sigma_x_1 = None
		self.sigma_x_2 = None
		self.delta_mu_x = None
		self.Dx = None
		self.x_distribution_fn = None

		self.mu_x = None
		self.sigma_x = None
		self.start_x = None
		self.stop_x = None
		self.step = 0.0001
		self.x = None
		self.colors = None
		self.pdf = None
		self.pdf_distribution_fn = None
		
		#   Robot details
		self.num_robots = None
		self.r_poses = None

		self.mu_m_1 = None
		self.mu_m_2 = None
		self.sigma_m_1 = None
		self.sigma_m_2 = None
		self.Dm = None
		self.delta_mu_m = None
		self.sub_group_size = None
		self.m_distribution_fn = None
		self.mu_m = None
		self.sigma_m = None
		
		#   Response Threshold details
		self.mu_h_1 = None
		self.mu_h_2 = None
		self.sigma_h_1 = None
		self.sigma_h_2 = None
		self.delta_mu_h = None
		self.Dh = None
		self.h_distribution_fn = None
		self.mu_h = None
		self.sigma_h = None

		self.mu_assessment_err = 0
		self.sigma_assessment_err = 0
	
	def initializer(self,**kwargs):
		for key,value in kwargs.items():
			if key == "n":
				self.num_opts = value
			elif key == "mux1":
				self.mu_x_1 = value
			elif key == "mux2":
				self.mu_x_2 = value
			elif key == "sx1":
				self.sigma_x_1 = value
			elif key == "sx2":
				self.sigma_x_2 = value
			elif key == "Dx":
				self.Dx = value
			elif key == "dmux":
				self.delta_mu_x = value
			elif key == "mum1":
				self.mu_m_1 = value
			elif key == "mum2":
				self.mu_m_2 = value
			elif key == "sm1":
				self.sigma_m_1 = value
			elif key == "sm2":
				self.sigma_m_2 = value
			elif key == "Dm":
				self.Dm = value
			elif key == "dmum":
				self.delta_mu_m = value
			elif key == "muh1":
				self.mu_h_1 = value
			elif key == "muh2":
				self.mu_h_2 = value
			elif key == "sh1":
				self.sigma_h_1 = value
			elif key == "sh2":
				self.sigma_h_2 = value
			elif key == "Dh":
				self.Dh = value
			elif key == "dmuh":
				self.delta_mu_h = value
		
		self.packaging()

	def functionChooser(self,variable_array,choices): # variable_array = 1 x n matrix, choices = m x n matrix (m choices for each variable)
		for j in range(len(choices)):
			if variable_array[0] == choices[j][0]:
					for i in range(1,len(variable_array)):
						variable_array[i] = choices[j][i]
		return variable_array

	def packaging(self):
		choices_x = [["G",rng.dx_n,vP.gaussian],["U",rng.dx_u,vP.uniform],["K",rng.dx_n,vP.gaussian]]
		[_,self.x_distribution_fn,self.pdf_distribution_fn] = self.functionChooser(\
			[self.Dx,self.x_distribution_fn,self.pdf_distribution_fn],choices_x)

		self.mu_x = List([self.mu_x_1,self.mu_x_2])
		self.sigma_x = List([self.sigma_x_1,self.sigma_x_2])
		self.start_x = np.sum(self.mu_x)/len(self.mu_x) - 2*np.sum(self.sigma_x)-5
		self.stop_x = np.sum(self.mu_x)/len(self.mu_x) + 2*np.sum(self.sigma_x)+5
		self.x = np.arange(self.start_x,self.stop_x,self.step)
		self.colors = np.random.randint(low = 0, high= 256,size=(self.num_opts,3))
		pdf = self.pdf_distribution_fn(self.x,self.mu_x,self.sigma_x)
		self.pdf = np.multiply(pdf,1/(np.sum(pdf)*self.step))

		choices_m = [["G",rng.units_n],["U",rng.units_u],["K",rng.units_n]]
		[_,self.m_distribution_fn] = self.functionChooser([self.Dm,self.m_distribution_fn],choices_m)

		self.mu_m = List([self.mu_m_1,self.mu_m_2])
		self.sigma_m = List([self.sigma_m_1,self.sigma_m_2])
		self.sub_group_size = self.m_distribution_fn(self.mu_m,self.sigma_m,self.num_opts)
		self.num_robots = int(np.sum(self.sub_group_size))


		r = np.random.choice(np.arange(5,5+(self.boundary_max - self.boundary_min)/(self.num_opts),0.001),int(sqrt(self.num_robots+20)),replace = False)
		theta = np.random.choice(np.arange(0,2*np.pi,0.0001),int(sqrt(self.num_robots+10)*2),replace=False)
		self.r_poses = []
		for i in r:
			for j in theta:
				self.r_poses.append([i,j])

		choices_h= [["G",rng.threshold_n],["U",rng.threshold_u],["K",rng.threshold_n]]
		[_,self.h_distribution_fn] = self.functionChooser([self.Dh,self.h_distribution_fn],choices_h)

		self.mu_h = List([self.mu_h_1,self.mu_h_2])
		self.sigma_h = List([self.sigma_h_1,self.sigma_h_2])

class VoterModel:
	def __init__(self,robots,options,p):
	# def __init__(self,animator,robots,options,p):
		# self.animator = animator
		self.consensus = False
		self.best_option = None
		self.ref_best = None
		self.iterations = 0
		for r in robots:
			assesment_error = np.round(np.random.normal(p.mu_assessment_err,p.sigma_assessment_err),decimals=3)
			if r.threshold<=options[r.assigned_opt-1].quality+assesment_error:
				r.response = 1
				r.opt = r.response*r.assigned_opt

		# 		r.patch.set_color(options[r.assigned_opt-1].color)

		# plt.pause(0.001)

		
	def dissemination(self,robots,options,p):
		t = 0
		best_option = None
		
		yes_respondents = []
		for r in robots:
			if r.response==1:
				yes_respondents.append(r)

		while self.consensus == False:
			# Fails when total number of yes reponding agents is below neighbours number
			# Number of neighbours considered affects the consensus achievment time and success
			consensus_limit = len(yes_respondents)
			if consensus_limit < 1:
				break
			talker = np.random.choice(range(len(yes_respondents)),1,replace = False)[0]

			listener = np.random.choice(range(len(robots)),1,replace = False)[0]

			if yes_respondents[talker].id != robots[listener].id:
				if robots[yes_respondents[talker].id].opt != robots[listener].opt:

					robots[listener].opt = copy.copy(robots[yes_respondents[talker].id].opt)
					
					robots[listener].response = 1

					# robots[listener].patch.set_color(options[robots[listener].opt-1].color)
					if robots[listener] not in yes_respondents:
						yes_respondents.append(robots[listener])

					# plt.pause(0.001)
					# plt.show()


				same = robots[yes_respondents[talker].id].opt
				counter = 1
				opt_counter1 = []
				
				for r in yes_respondents:
					opt_counter1.append(r.opt-1)
					if same == r.opt:
						counter += 1
						
				opt_counter = 100*np.bincount(opt_counter1)/len(yes_respondents)

				# for i in range(len(opt_counter)):
				# 	if isinstance(options[i].progress_bar,type(None)) != True:
				# 		options[i].progress_bar.remove()
				# 	options[i].progress_bar = self.animator.ax.bar(50 + i*10,opt_counter[i],color = options[i].color,alpha = 0.3,width = 5)			
				
				if counter/(p.num_robots) >= 0.99:
					self.consensus = True
					best_option = same
					self.best_option = same -1

				t += 1
		self.iterations = t

	def compare_with_best(self,options):
		opts = []
		for o in range(len(options)):
			opts.append(options[o].quality)
		best_list = np.array(np.where(opts == max(opts)))[0]
		opt_choosen = np.random.randint(0,len(best_list))
		self.ref_best = best_list[opt_choosen]
		if self.ref_best==self.best_option:
			return 1
		else:
			return 0

class Object:
	def __init__(self,id,o_type,p,options=None):
		self.o_type = o_type
		self.id = id

		if self.o_type == "O":
			self.progress_bar = None
			self.pose = np.random.uniform((self.id*p.boundary_max/p.num_opts)+p.boundary_min,((self.id+1)*p.boundary_max/p.num_opts)+p.boundary_min,2)
			self.quality = p.x_distribution_fn(p.mu_x,p.sigma_x,1)[0]
			self.to_be_assigned_count = p.m_distribution_fn(number_of_options=1,mu_m=p.mu_m,sigma_m=p.sigma_m)[0]
			self.assigned_count = 0
			self.color = tuple(p.colors[self.id])
	
		if self.o_type == "R":
			self.patch = None
			self.threshold = p.h_distribution_fn(m_units=1,mu_h=p.mu_h,sigma_h=p.sigma_h)[0]
			self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
			options[self.assigned_opt-1].assigned_count += 1
			while options[self.assigned_opt-1].assigned_count>options[self.assigned_opt-1].to_be_assigned_count:
				options[self.assigned_opt-1].assigned_count -= 1
				self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
				options[self.assigned_opt-1].assigned_count += 1
			
			
			[r,theta] = p.r_poses[self.id]#[np.random.uniform(5,(p.boundary_max - p.boundary_min)/(2*p.num_opts)),np.random.uniform(0,2*np.pi)]
			
			self.pose = options[self.assigned_opt-1].pose + np.array([r*np.cos(theta),r*np.sin(theta)])

			
			self.response = 0	# 0 = "No", 1 = "Yes"
			self.opt = self.response*self.assigned_opt
			self.best_opt = None

class Animator:
	def __init__(self,robots,options,p):
		self.fig,self.ax = plt.subplots()
		self.ax.set_aspect('equal')
		for o in options:
			c = plt.Circle((o.pose[0],o.pose[1]),2*o.quality/p.robot_size,fc = o.color,edgecolor = 'black',linewidth=1)
			self.ax.add_patch(c)
		for r in robots:
			r.patch = plt.Circle((r.pose[0],r.pose[1]), p.robot_size/10,fc = 'maroon',ec = 'maroon')
			self.ax.add_patch(r.patch)

		self.ax.set_xlim(-20,120)
		self.ax.set_ylim(-20,120)
		plt.show()

def looperParallel(p):
	options = parallel(partial(Object,o_type='O',p=p),range(p.num_opts))
	robots = parallel(partial(Object,o_type='R',p=p,options=options),range(p.num_robots))
	anim = Animator()
	model = VoterModel(anim)
	model = VoterModel(robots,options,p)
	plt.show()
	model.dissemination(robots,options,p)
	matching = model.compare_with_best(options)
	if int(model.consensus) and matching:
		result = "achieved on best option!"
	else:
		result = "not achieved on best option!\n best option is "+ options[model.ref_best].color
	anim.ax.text(10,110,s = "Consensus " + result)
	out = pd.DataFrame(data=[{'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations}],columns=vP.data_columns_name)
	out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
	if isinstance(model.best_option,type(None)) == False and isinstance(model.ref_best,type(None)) == False:
		print({'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations})
	#		plt.show()
	#		plt.close()

		return {'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations}
	else:
		print({'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations})
	#		plt.show()
	#		plt.close()

		return {'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations}

def looper(p):
	vP.save_data(p)
	# outputs = parallel(looperParallel,[p for i in range(100)])
	for i in range(100):
		options = parallel_wop(partial(Object,o_type='O',p=p),range(p.num_opts))
		robots = parallel_wop(partial(Object,o_type='R',p=p,options=options),range(p.num_robots))
		# anim = Animator(robots,options,p)
		# model = VoterModel(anim)
		model = VoterModel(robots,options,p)
		plt.show()
		model.dissemination(robots,options,p)
		matching = model.compare_with_best(options)
		if isinstance(model.best_option,type(None)) == False and isinstance(model.ref_best,type(None)) == False:
			out = pd.DataFrame(data=[{'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations}],columns=vP.data_columns_name)

			print({'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations})
		else:
			out = pd.DataFrame(data=[{'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations}],columns=vP.data_columns_name)

			print({'$\mu_{h_1}$':p.mu_h_1,'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':None,'$CDM$ opt No.':model.best_option,'$CDM$':None,'Iterations':model.iterations})
		out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
		
	# out = pd.DataFrame(data=outputs,columns=vP.data_columns_name)
	# out.to_csv(p.data_f_path,mode = 'a',header = False, index=False)
	# print(outputs)
		
def parallel(func,inputs):
	batch_size = 20
	inps = [(i,) for i in inputs]
	output = []
	for i in range(0,len(inputs),batch_size):
		opt_var = []
		with Pool(20) as processor:#,ray_address="auto") as p:
			opt_var = processor.starmap(func,inps[i:i+batch_size])
		if i + batch_size<= len(inputs):
			print("\r Percent of input processed : {}%".format(np.round(100*(i+batch_size)/len(inputs)),decimals=1), end="")
		else:
			print("\r Percent of input processed : {}%".format(100, end=""))
		output += list(opt_var)
	return output

def parallel_wop(func,inputs):
	batch_size = 20
	inps = [(i,) for i in inputs]
	output = []
	for i in range(0,len(inputs),batch_size):
		opt_var = []
		with Pool(20) as processor:#,ray_address="auto") as p:
			opt_var = processor.starmap(func,inps[i:i+batch_size])

		output += list(opt_var)
	return output

# Need to change for variable parameters for each experiment
def paramObjects(varP,fixed_params):
	p = Params()
	predicted = 0
	p.initializer(n = fixed_params[0]['n'], Dm = fixed_params[0]['$D_{m}$'], mum1 = fixed_params[0]['$\mu_{m_{1}}$'],\
		mum2 = fixed_params[0]['$\mu_{m_{2}}$'], sm1 = fixed_params[0]['$\sigma_{m_{1}}$'], sm2 = fixed_params[0]['$\sigma_{m_{2}}$'],\
		dmum = vP.delta_mu_m, Dx = fixed_params[0]['$D_{x}$'], mux1 = fixed_params[0]['$\mu_{x_{1}}$'],\
		mux2 = fixed_params[0]['$\mu_{x_{2}}$'], sx1 = fixed_params[0]['$\sigma_{x_{1}}$'], sx2 = fixed_params[0]['$\sigma_{x_{2}}$'],\
		dmux = vP.delta_mu_x, Dh = fixed_params[0]['$D_{h}$'], muh1 = varP[0], muh2 = varP[0], sh1 = fixed_params[0]['$\sigma_{h_{1}}$'], sh2 = fixed_params[0]['$\sigma_{h_{2}}$'],\
		dmuh = fixed_params[0]['$\delta_{\mu_{h}}$'])
	
	if predicted == 0 and isinstance(fixed_params[0]['$\mu_{h_{1}}$'],type(None)) == True and isinstance(fixed_params[0]['$\mu_{h_{2}}$'],type(None)) == True and isinstance(fixed_params[0]['$\sigma_{h_{1}}$'],type(None)) == True and isinstance(fixed_params[0]['$\sigma_{h_{2}}$'],type(None)) == True:
		fixed_params[0]['$\mu_{h_{1}}$'] = vP.ICPDF(1-(1/p.num_opts),p.mu_x,p.stop_x,p.step,p.x,p.pdf)
		fixed_params[0]['$\mu_{h_{2}}$'] = p.mu_h_1 + p.delta_mu_h
		choices_h = [['G',(0.12*np.log10(p.num_opts) + 0.56)*p.sigma_x_1],['U',p.sigma_x_1*np.exp(-0.07*p.num_opts+0.7)],['K',(0.12*np.log10(p.num_opts) + 0.54)*p.sigma_x_1]]
		[_,fixed_params[0]['$\sigma_{h_{1}}$']] = p.functionChooser([p.Dh,fixed_params[0]['$\sigma_{h_{1}}$']],choices_h)
		fixed_params[0]['$\sigma_{h_{2}}$'] = fixed_params[0]['$\sigma_{h_{1}}$']
		p.mu_h_1 = fixed_params[0]['$\mu_{h_{1}}$']
		p.mu_h_2 = fixed_params[0]['$\mu_{h_{2}}$']
		p.sigma_h_1 = fixed_params[0]['$\sigma_{h_{1}}$']
		p.sigma_h_2 = fixed_params[0]['$\sigma_{h_{2}}$']
		predicted = 1
	return p

if __name__=="__main__":
	path = os.getcwd() + "/results/"
	# plt.close()
	# plt.ion()
	vP = variableParams()
	variabling_params = list(itertools.product(vP.mu_h_1))

	fixed_params_column = ['n','$D_{m}$','$\mu_{m_{1}}$','$\mu_{m_{2}}$','$\sigma_{m_{1}}$','$\sigma_{m_{2}}$',\
		'$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$',\
		'$D_{h}$','$\mu_{h_{1}}$','$\mu_{h_{2}}$','$\sigma_{h_{1}}$','$\sigma_{h_{2}}$','$\delta_{\mu_{h}}$'] 
	fixed_params = [{'n': vP.num_opts[2], '$D_{m}$':vP.Dm[0],'$\mu_{m_{1}}$':vP.mu_m_1[1],'$\mu_{m_{2}}$':vP.mu_m_2[1],\
		'$\sigma_{m_{1}}$':vP.sigma_m_1[0],'$\sigma_{m_{2}}$':vP.sigma_m_2[0],'$D_{x}$':vP.Dx[0],\
		'$\mu_{x_{1}}$':vP.mu_x_1[69],'$\mu_{x_{2}}$':vP.mu_x_2[69],'$\sigma_{x_{1}}$':vP.sigma_x_1[9],\
		'$\sigma_{x_{2}}$':vP.sigma_x_2[9], '$D_{h}$':vP.Dh[0], '$\mu_{h_{1}}$': None, '$\mu_{h_{2}}$': None,\
		'$\sigma_{h_{1}}$':vP.sigma_h_1[9], '$\sigma_{h_{2}}$':vP.sigma_h_2[9],'$\delta_{\mu_{h}}$':vP.delta_mu_h}]

	inputs = parallel(partial(paramObjects,fixed_params=fixed_params),variabling_params)

	vP.save_params(fixed_params_column,fixed_params)

	parallel(looper,inputs)
	
	
	
	
		

    

