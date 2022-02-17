import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import cos, sqrt,exp
from statsmodels.stats.proportion import proportions_ztest as pt
import pandas as pd
import os
import time
import random
from numba.typed import List
from numba import guvectorize, float64, njit,jit
import copy
import datetime as date
import random_number_generator as rng


class VoterModel:
	def __init__(self,animator):
		self.animator = animator
		self.consensus = False
		self.best_option = None
		self.ref_best = None
		self.iterations = 0
		for r in robots:
			assesment_error = np.round(np.random.normal(p.mu_assessment_err,p.sigma_assessment_err),decimals=3)
			if r.threshold<=options[r.assigned_opt-1].quality+assesment_error:
				r.response = 1
				r.opt = r.response*r.assigned_opt

				r.patch.set_color(options[r.assigned_opt-1].color)

		plt.pause(0.001)
		
	def dissemination(self,robots,options):
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
			talker = np.random.choice(range(len(yes_respondents)),1,replace = False)[0]

			listener = np.random.choice(range(len(robots)),1,replace = False)[0]

			if yes_respondents[talker].id != robots[listener].id:
				if robots[yes_respondents[talker].id].opt != robots[listener].opt:

					robots[listener].opt = copy.copy(robots[yes_respondents[talker].id].opt)
					
					robots[listener].response = 1

					robots[listener].patch.set_color(options[robots[listener].opt-1].color)
					if robots[listener] not in yes_respondents:
						yes_respondents.append(robots[listener])

					plt.pause(0.001)
					plt.show()


				same = robots[yes_respondents[talker].id].opt
				counter = 1
				opt_counter1 = []
				
				for r in yes_respondents:
					opt_counter1.append(r.opt-1)
					if same == r.opt:
						counter += 1
						
				opt_counter = 100*np.bincount(opt_counter1)/len(yes_respondents)

				for i in range(len(opt_counter)):
					if isinstance(options[i].progress_bar,type(None)) != True:
						options[i].progress_bar.remove()
					options[i].progress_bar = self.animator.ax.bar(50 + i*10,opt_counter[i],color = options[i].color,alpha = 0.3,width = 5)			
				
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


class Params:
	def __init__(self):

		#   Simulation params
		self.dt = 0.1

		#   Environment params
		self.boundary_min = 0
		self.boundary_max = 100
		
		#   Option details
		self.num_opts = 3
		self.mu_x_1 = 10
		self.mu_x_2 = 10
		self.sigma_x_1 = 1
		self.sigma_x_2 = 1
		self.delta_mu_x = 0
		self.x_distribution_type = "G"
		if self.x_distribution_type == "G":
			self.x_distribution_fn = rng.dx_n
			self.pdf_distribution_fn = self.gaussian
		elif self.x_distribution_type == "U":
			self.x_distribution_fn = rng.dx_u
			self.pdf_distribution_fn = self.uniform
		elif self.x_distribution_type == "K":
			self.x_distribution_fn = rng.dx_n
			self.pdf_distribution_fn = self.gaussian
			
		self.mu_x = List([self.mu_x_1,self.mu_x_2])
		self.sigma_x = List([self.sigma_x_1,self.sigma_x_2])
		self.start_x = np.sum(self.mu_x)/len(self.mu_x) - 2*np.sum(self.sigma_x)-5
		self.stop_x = np.sum(self.mu_x)/len(self.mu_x) + 2*np.sum(self.sigma_x)+5
		self.step = 0.0001
		self.x = np.arange(self.start_x,self.stop_x,self.step)
		
		
		#   Robot details
		self.num_robots = 150
		self.robot_size = 10
		self.mu_m_1 = 50
		self.sigma_m_1 = 0
		self.mu_m_2 = 50
		self.sigma_m_2 = 0
		self.mu_m = List([self.mu_m_1,self.mu_m_2])
		self.sigma_m = List([self.sigma_m_1,self.sigma_m_2])
		self.m_distribution_type = "G"
		if self.m_distribution_type == "G":
			self.m_distribution_fn = rng.units_n
		elif self.m_distribution_type == "U":
			self.m_distribution_fn = rng.units_u
		elif self.m_distribution_type == "K":
			self.m_distribution_fn = rng.units_n
		
		r = np.random.choice(np.arange(5,(self.boundary_max - self.boundary_min)/(2*self.num_opts),0.1),int(sqrt(self.num_robots+10)),replace = False)
		theta = np.random.choice(np.arange(0,2*np.pi,0.0001),int(sqrt(self.num_robots+10)*2),replace=False)
		self.r_poses = []
		for i in r:
			for j in theta:
				self.r_poses.append([i,j])
		
		#   Response Threshold details
		self.h_distribution_type = "G"
		if self.h_distribution_type == "G":
			self.h_distribution_fn = rng.threshold_n
		elif self.h_distribution_type == "U":
			self.h_distribution_fn = rng.threshold_u
		elif self.h_distribution_type == "K":
			self.h_distribution_fn = rng.threshold_n
			
		self.delta_mu_h = 0
		
		
		pdf = self.pdf_distribution_fn(self.x,self.mu_x,self.sigma_x)
		pdf = np.multiply(pdf,1/(np.sum(pdf)*self.step))
#		self.mu_h_1 = self.ICPDF(1-(1/self.num_opts),self.mu_x,self.stop_x,self.step,self.x,pdf)
#		self.mu_h_2 = self.ICPDF(1-(1/self.num_opts),self.mu_x,self.stop_x,self.step,self.x,pdf)
#		self.sigma_h_1 = (0.12*np.log10(self.num_opts) + 0.56)*self.sigma_x_1
#		self.sigma_h_2 = (0.12*np.log10(self.num_opts) + 0.56)*self.sigma_x_1
		self.mu_h_1 = self.mu_x_1
		self.mu_h_2 = self.mu_x_2
		self.sigma_h_1 = self.sigma_x_1
		self.sigma_h_2 = self.sigma_x_2
		
		self.mu_h = List([self.mu_h_1,self.mu_h_2])
		self.sigma_h = List([self.sigma_h_1,self.sigma_h_2])

		self.mu_assessment_err = 0
		self.sigma_assessment_err = 0
		self.save_string = None
		self.columns_name = None
		
	def save_runs(self):
		columns_name = ['$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$','Iterations']
		self.columns_name = columns_name
		check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))
		count = 0
		for i in check:
			if count==i:
				count+=1
		   
		save_string = str(count)+'_'+str(time.time())
		self.save_string = save_string
		f = open(path+save_string+'.csv','a')
		f_path = path+save_string+'.csv'
		
		columns = pd.DataFrame(data=np.array([columns_name]))
		columns.to_csv(f_path,mode='a',header=False,index=False)
#		out = pd.DataFrame(data=data_array,columns=columns_name)
#		out.to_csv(f_path,mode = 'a',header = False, index=False)tmux

		param = open(path+str(count)+'.csv','w+')
		counter = open(path+str(count),'w+')
		param_columns_name = ['n','$m_{i}$','$D_{m}$','$\mu_{m_{1}}$','$\mu_{m_{2}}$','$\sigma_{m_{1}}$','$\sigma_{m_{2}}$','$D_{x}$','$\mu_{x_{1}}$','$\mu_{x_{2}}$','$\sigma_{x_{1}}$','$\sigma_{x_{2}}$','$\delta_{\mu_{x}}$','$D_{h}$','$\mu_{h_{1}}$','$\mu_{h_{2}}$','$\sigma_{h_{1}}$','$\sigma_{h_{2}}$','$\delta_{\mu_{h}}$'] #[nop,number of robots per option,Dm,mu_m_1,mu_m_2,sigma_m_1,sigma_m_2,Dx,mu_x_1,mu_x_2,sigma_x_1,sigma_x_2,delta_mu_x,Dh,mu_h_1,mu_h_2,sigma_h_1,sigma_h_2,delta_mu_h]
		columns_p = pd.DataFrame(data=np.array([param_columns_name]))
		columns_p.to_csv(path+str(count)+'.csv',mode='a',header=False,index=False)
		out_p = pd.DataFrame(data=[{'n':self.num_opts,'$m_{i}$':np.round(self.num_robots/self.num_opts),'$D_{m}$':self.m_distribution_type,'$\mu_{m_{1}}$':self.mu_m_1,'$\mu_{m_{2}}$':self.mu_m_2,'$\sigma_{m_{1}}$':self.sigma_m_1,'$\sigma_{m_{2}}$':self.sigma_m_2,'$D_{x}$':self.x_distribution_type,'$\mu_{x_{1}}$':self.mu_x_1,'$\mu_{x_{2}}$':self.mu_x_2,'$\sigma_{x_{1}}$':self.sigma_x_1,'$\sigma_{x_{2}}$':self.sigma_x_2,'$\delta_{\mu_{x}}$':self.delta_mu_x,'$D_{h}$':self.h_distribution_type,'$\mu_{h_{1}}$':self.mu_h_1,'$\mu_{h_{2}}$':self.mu_h_2,'$\sigma_{h_{1}}$':self.sigma_h_1,'$\sigma_{h_{2}}$':self.sigma_h_2,'$\delta_{\mu_{h}}$':self.delta_mu_h}],columns=param_columns_name)
		out_p.to_csv(path+str(count)+'.csv',mode = 'a',header = False, index=False)	

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

class Object:
	def __init__(self,id,o_type,array):
		self.o_type = o_type
		self.id = id
		self.array = array

		if self.o_type == "O":
			self.progress_bar = None
			self.pose = np.random.uniform((self.id*p.boundary_max/p.num_opts)+p.boundary_min,((self.id+1)*p.boundary_max/p.num_opts)+p.boundary_min,2)
			self.quality = p.x_distribution_fn(p.mu_x,p.sigma_x,1)[0]
			self.to_be_assigned_count = p.m_distribution_fn(number_of_options=1,mu_m=p.mu_m,sigma_m=p.sigma_m)[0]
			self.assigned_count = 0
			self.color = color[self.id]
	
		if self.o_type == "R":
			self.patch = None
			self.threshold = p.h_distribution_fn(m_units=1,mu_h=p.mu_h,sigma_h=p.sigma_h)[0]
			self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
	        
			while options[self.assigned_opt-1].assigned_count>options[self.assigned_opt-1].to_be_assigned_count:
				self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
			
			
			[r,theta] = p.r_poses[self.id]#[np.random.uniform(5,(p.boundary_max - p.boundary_min)/(2*p.num_opts)),np.random.uniform(0,2*np.pi)]
			
			self.pose = options[self.assigned_opt-1].pose + np.array([r*np.cos(theta),r*np.sin(theta)])

			
			self.response = 0	# 0 = "No", 1 = "Yes"
			self.opt = self.response*self.assigned_opt
			self.best_opt = None

class Animator:
	def __init__(self):
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

def generator(count,o_type):
	array = []
	for i in range(count):
            	array.append(Object(i,o_type,array))
	return array

if __name__=="__main__":
	path = os.getcwd() + "/results/"
	plt.close()
	plt.ion()
	
	color = ["darkviolet","seagreen","orange","navy","hotpink"]
	data_a = []
	p = Params()
	p.save_runs()
	for i in range(100):
		options = generator(p.num_opts,'O')
		robots = generator(p.num_robots,'R')
		anim = Animator()
		model = VoterModel(anim)
		plt.show()
		model.dissemination(robots,options)
		matching = model.compare_with_best(options)
		if int(model.consensus) and matching:
			result = "achieved on best option!"
		else:
			result = "not achieved on best option!\n best option is "+ options[model.ref_best].color
		anim.ax.text(10,110,s = "Consensus " + result)
#		data_a.append({'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations})
		out = pd.DataFrame(data=[{'$x_{max}$ opt No.':model.ref_best,'$x_{max}$':options[model.ref_best].quality,'$CDM$ opt No.':model.best_option,'$CDM$':options[model.best_option].quality,'Iterations':model.iterations}],columns=p.columns_name)
		out.to_csv(path+p.save_string+'.csv',mode = 'a',header = False, index=False)
		plt.show()
		plt.close()
	
		

    

