# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import copy

class DecisionMaking:
	def __init__(self,robots,options,p,counter):
	# def __init__(self,animator,robots,options,p):
		# self.animator = animator
		self.ref_best = None
		self.best_option = None
		self.votes = np.zeros(p.num_opts)
		self.counter = counter
		for r in robots:
			assesment_error = np.round(np.random.normal(p.mu_assessment_err,p.sigma_assessment_err),decimals=3)
			if r.threshold<=options[r.assigned_opt-1].quality+assesment_error:
				r.response = 1
				r.opt = r.response*r.assigned_opt
				self.votes[r.assigned_opt-1] += 1
			r.memory.append(r.response)
		# 		r.patch.set_color(options[r.assigned_opt-1].color)
		# plt.pause(0.001)

	def updateThresholds(self,robots,p,vP_):
		mu = 0
		sigma = 0

		for r in robots:
			if self.counter%11==0:
				r.memory = []
			else:
				if np.round(np.sum(r.memory)/len(r.memory),decimals=3)-np.round(r.personal_accepting_barrier,decimals=3)<-0.1:
					r.threshold -= np.random.random()#0.1 #abs(np.sum(r.memory)/len(r.memory)-r.personal_accepting_barrier)
					# for i in range(len(r.memory)):
					# 	keeper = np.random.randint(low=0,high=2)
					# 	r.memory[i] += keeper
						
					# r.memory = list((np.array(r.memory)>=1).astype(int))

				elif np.round(np.sum(r.memory)/len(r.memory),decimals=3)-np.round(r.personal_accepting_barrier,decimals=3)>0.1:
					r.threshold += np.random.random()#0.1 #abs(np.sum(r.memory)/len(r.memory)-r.personal_accepting_barrier)
					# for i in range(len(r.memory)):
					# 	keeper = np.random.randint(low=0,high=2)
					# 	r.memory[i] *= keeper
				mu += r.threshold/len(robots)
			

		if self.counter%11!=0:
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