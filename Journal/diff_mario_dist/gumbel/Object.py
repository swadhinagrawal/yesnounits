# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
from scipy.special import lambertw

class Object:
	def __init__(self,id,o_type,p,mem_size,options=None,pdf_g = None):
		self.o_type = o_type
		self.id = id
		self.p = p
		if self.o_type == "O":
			self.progress_bar = None
			self.pose = np.random.uniform((self.id*p.boundary_max/p.num_opts)+p.boundary_min,((self.id+1)*p.boundary_max/p.num_opts)+p.boundary_min,2)
			self.quality = p.x_distribution_fn(p.mu_x,p.sigma_x,1)[0]
			self.to_be_assigned_count = p.m_distribution_fn(number_of_options=1,mu_m=p.mu_m,sigma_m=p.sigma_m)[0]
			self.assigned_count = 0
			self.color = tuple(p.colors[self.id])
	
		if self.o_type == "R":
			self.patch = None
			self.threshold = p.mu_h_1#p.h_distribution_fn(m_units=1,mu_h=p.mu_h,sigma_h=p.sigma_h)[0]
			self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
			options[self.assigned_opt-1].assigned_count += 1
			while options[self.assigned_opt-1].assigned_count>options[self.assigned_opt-1].to_be_assigned_count:
				options[self.assigned_opt-1].assigned_count -= 1
				self.assigned_opt = np.random.randint(low = 1,high = p.num_opts+1)
				options[self.assigned_opt-1].assigned_count += 1
			
			
			# [r,theta] = p.r_poses[self.id]#[np.random.uniform(5,(p.boundary_max - p.boundary_min)/(2*p.num_opts)),np.random.uniform(0,2*np.pi)]
			
			# self.pose = options[self.assigned_opt-1].pose + np.array([r*np.cos(theta),r*np.sin(theta)])

			# #   Gumbel distribution
			# x_n = np.sqrt(1*lambertw((5**2)/(2*np.pi)))# 5 = num of opts, 1 = var_sigma, mean = 0.5
			# n_rho_xn = x_n/1
			# step = 0.0001
			# start_g = 0
			# stop_g = 1.0001

			# dis_x_g = np.round(np.arange(start_g,stop_g,step),decimals=4)
			# pdf_g =  [n_rho_xn*np.exp(-n_rho_xn*(i-0.5-x_n)-np.exp(-n_rho_xn*(i-0.5-x_n))) for i in dis_x_g]
			# variance_g = np.pi/(np.sqrt(6)*n_rho_xn)
			# pdf_g = pdf_g/np.sum(pdf_g)

			self.personal_accepting_barrier = np.random.choice(np.arange(0,1.0001,0.0001),p=pdf_g)
			self.response = 0	# 0 = "No", 1 = "Yes"
			self.opt = self.response*self.assigned_opt
			self.best_opt = None
			self.memory = np.zeros(mem_size)
	
	def reassignment(self,options):
		self.assigned_opt = np.random.randint(low = 1,high = self.p.num_opts+1)
		options[self.assigned_opt-1].assigned_count += 1
		while options[self.assigned_opt-1].assigned_count>options[self.assigned_opt-1].to_be_assigned_count:
			options[self.assigned_opt-1].assigned_count -= 1
			self.assigned_opt = np.random.randint(low = 1,high = self.p.num_opts+1)
			options[self.assigned_opt-1].assigned_count += 1