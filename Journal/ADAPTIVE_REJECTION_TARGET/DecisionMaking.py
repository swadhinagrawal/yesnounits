# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import copy

def saturate(a):
	if a>=1.0:
		a = 0.9999
	if a<=0.0:
		a = 0.0001
	return a

class DecisionMaking:
	def __init__(self,robots,options,p,counter,mem_size,t_step):
		self.ref_best = None
		self.best_option = None
		self.votes = np.zeros(p.num_opts)
		self.counter = counter
		self.mem_size = mem_size
		self.t_step = t_step
		self.threshold_update = 0
		for r in robots:
			assesment_error = np.round(np.random.normal(p.mu_assessment_err,p.sigma_assessment_err),decimals=3)
			
			if r.threshold<=options[r.assigned_opt-1].quality+assesment_error:
				r.response = 1
				r.opt = r.response*r.assigned_opt
				self.votes[r.assigned_opt-1] += 1
			else:
				r.response = 0
				r.opt = 0

			if self.counter%self.mem_size != 0 or self.counter<self.mem_size:
				r.memory[self.counter%(self.mem_size)] = r.response
				self.threshold_update = 0
			else:
				self.threshold_update = 1
		
	def updateThresholds(self,robots,p,vP_):
		mu = 0
		sigma = 0

		for r in robots:

			peak_at = 1/p.num_opts#(p.alpha-1)/(p.alpha+p.beta-2)
			val = 0.1/p.num_opts# 0.2/p.num_opts
			if r.personal_accepting_barrier - peak_at > val - (val**3)/3:#np.arctan(0.175/p.num_opts):#0.2
				r.personal_accepting_barrier -= r.mario_update_step
				r.personal_accepting_barrier = saturate(r.personal_accepting_barrier)
			elif r.personal_accepting_barrier - peak_at < -(val - (val**3)/3):#-np.arctan(0.175/p.num_opts):
				r.personal_accepting_barrier += r.mario_update_step
				r.personal_accepting_barrier = saturate(r.personal_accepting_barrier)

			condition = np.round(np.sum(r.memory)/len(r.memory),decimals=3)-np.round(r.personal_accepting_barrier,decimals=3)
			# condition = np.round(np.sum(r.memory)/len(r.memory),decimals=3)-(np.round(r.personal_accepting_barrier,decimals=3)**2)/(np.round(r.personal_accepting_barrier,decimals=3) + p.num_opts - 2)
			if condition < -0.01:
				r.threshold -= self.t_step # np.random.random()#0.1 #abs(np.sum(r.memory)/len(r.memory)-r.personal_accepting_barrier)
				r.memory = np.zeros(self.mem_size)
				r.memory[0] = r.response

			elif condition > 0.01:
				r.threshold += self.t_step#np.random.random()#0.1 #abs(np.sum(r.memory)/len(r.memory)-r.personal_accepting_barrier)
				r.memory = np.zeros(self.mem_size)
				r.memory[0] = r.response

			mu += r.threshold/len(robots)
			

		if self.counter%(self.mem_size+1)!=0:
			for r in robots:
				sigma += (mu - r.threshold)**2
			sigma = (sigma/(len(robots)-1))**0.5
			p.mu_h_1 = mu
			p.sigma_h_1 = sigma
			p.mu_h_2 = mu
			p.sigma_h_2 = sigma
			p.packaging(vP_)

	def compare_with_best(self,options):
		opts = []
		for o in range(len(options)):
			opts.append(options[o].quality)
		best_list = np.array(np.where(opts == max(opts)))[0]
		opt_choosen = np.random.randint(0,len(best_list))
		self.ref_best = best_list[opt_choosen]

		best_decided = np.array(np.where(self.votes == max(self.votes)))[0]
		best_option_index = np.random.randint(0,len(best_decided))
		self.best_option = best_decided[best_option_index]
		if self.ref_best==best_decided[best_option_index]:
			return 1
		else:
			return 0