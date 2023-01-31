# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np

class Object:
	def __init__(self,id,o_type,p,option=None):
		self.o_type = o_type
		self.id = id
		self.p = p
		if self.o_type == "O":
			self.quality = p.x_distribution_fn(p.mu_x,p.sigma_x,1)[0]
			self.to_be_assigned_count = p.m_distribution_fn(number_of_options=1,mu_m=p.mu_m,sigma_m=p.sigma_m)[0]
			self.assigned_count = 0

		if self.o_type == "R":
			self.threshold = p.h_distribution_fn(m_units=1,mu_h=p.mu_h,sigma_h=p.sigma_h)[0]#p.mu_h_1#
			self.assigned_opt = option
			
			self.delta_t_r = np.random.uniform(0.0001,0.999)
			
			self.t_r = np.random.uniform(0.001,0.999)#np.random.choice(np.arange(0.0001,0.9999,0.0001),p=p.pdf_mario)

			self.response = 0	# 0 = "No", 1 = "Yes"
			self.opt = -1 #self.response*self.assigned_opt
			self.best_opt = None
			self.memory = np.zeros(p.memory_length)
			self.threshold_update = 0
