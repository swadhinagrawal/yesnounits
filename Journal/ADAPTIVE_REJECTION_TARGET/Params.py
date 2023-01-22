# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import random_number_generator as rng
from numba.typed import List
from math import sqrt,exp
import pandas as pd
import time, os
from scipy.special import lambertw

class Params:
	def __init__(self,column_names,path):
		#   Simulation params
		self.data_f_path = None                 #   Data file name for this set of params   (string)
		self.save_string = None                 #   String for data file naming (string)
		self.columns_name = None                #   Columns name for data file  (list of strings)
		self.dt = 0.1                           #   Simulation time step size   (Float)
		self.boundary_min = 0                   #   Minimum of boundary of simulation area  (Float)
		self.boundary_max = 100                 #   Maximum of boundary of simulation area  (Float)
		self.robot_size = 10                    #   Body size of each robot or units    (Float)
		self.use_predicted = 0                  #   If to use the predicted response threshold distribution (Boolean)
		
		#   Option details
		self.num_opts = None                    #   Number of options in the env    (Int)
		
		self.mu_x_1 = None                      #   1st Mean of Quality-PDF (Float)
		self.mu_x_2 = None                      #   2nd Mean of Quality-PDF (Float)
		self.sigma_x_1 = None                   #   1st Variance of Quality-PDF (Float)
		self.sigma_x_2 = None                   #   2nd Variance of Quality-PDF (Float)
		self.delta_mu_x = None                  #   Difference between two means of Quality-PDF (Float)
		self.Dx = None                          #   PDF type of option qualities (Character)
		self.x_distribution_fn = None           #   Quality-PDF function (Object)

		self.mu_x = None                        #   Mean of Quality-PDF (list of Floats)
		self.sigma_x = None                     #   Variance of Quality-PDF (list of Floats)
		self.start_x = None                     #   Starting value of PDF   (Float)
		self.stop_x = None                      #   Stoping value of PDF   (Float)
		self.step = 0.0001                      #   Resolution step size of PDF   (Float)
		self.x = None                           #   x-values of PDF (list of Floats)
		self.colors = None                      #   Colours associated to each option   (list of Tuples)
		self.pdf = None                         #   Values of PDF (list of Floats)
		self.pdf_distribution_fn = None         #   PDF function (Object)
		
		#   Robot details
		self.num_robots = None                  #   Total number of robots in the env    (Int)
		self.r_poses = None                     #   Position of the robots around the option it is assigned to (2D array nx2)

		self.mu_m_1 = None                      #   1st Mean of subgroup size-PDF (Float)
		self.mu_m_2 = None                      #   2nd Mean of subgroup size-PDF (Float)
		self.sigma_m_1 = None                   #   1st Variance of subgroup size-PDF (Float)
		self.sigma_m_2 = None                   #   2nd Variance of subgroup size-PDF (Float)
		self.Dm = None                          #   PDF type for number of robots to be assigned to each option (Character)
		self.delta_mu_m = None                  #   Difference between two means of subgroup size-PDF (Float)
		self.sub_group_size = None              #   Subgroup size for each option (list of Int)
		self.m_distribution_fn = None			#   Subgroup size-PDF function (Object)
		self.mu_m = None						#   Mean of Subgroup size-PDF (list of Floats)
		self.sigma_m = None						#   Variance of Subgroup size-PDF (list of Floats)
		
		#   Response Threshold details
		self.mu_h_1 = None						#   1st Mean of response threshold-PDF (Float)
		self.mu_h_2 = None						#   2nd Mean of response threshold-PDF (Float)
		self.sigma_h_1 = None					#   1st Variance of response threshold-PDF (Float)
		self.sigma_h_2 = None					#   2nd Variance of response threshold-PDF (Float)
		self.delta_mu_h = None					#   Difference between two means of response threshold-PDF (Float)
		self.Dh = None							#   PDF type for response thresholds (Character)
		self.h_distribution_fn = None			#   Response thresholds-PDF function (Object)
		self.mu_h = None						#   Mean of response thresholds-PDF (list of Floats)
		self.sigma_h = None						#   Variance of response thresholds-PDF (list of Floats)

		self.mu_assessment_err = 0				#   Mean of error in measurement-PDF (list of Floats)
		self.sigma_assessment_err = 0			#   Variance of error in measurement-PDF (list of Floats)

		#	Mario distribution

		self.threshold_step = 0.3
		self.memory_length = 10
		self.step = 0.0001
		self.start = 0.0001
		self.stop = 0.9999
		self.mario = 'beta'
		# self.mario = 'gumbel'
		# self.mario = 'uniform'
		self.pdf_mario = None

		#	Data logging
		self.data_columns_name = column_names   #['$\mu_{h_1}$','$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$','Iterations']
		
		self.pre = self.prefix(path)
		
	def initializer(self,vP,**kwargs):
		'''
		This assigns values to parameters
		'''
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
			elif key == "beta":
				self.beta = value
		
		self.packaging(vP)

	def functionChooser(self,variable_array,choices): # variable_array = 1 x n matrix, choices = m x n matrix (m choices for each variable)
		'''
		This helps selectng value for a variable with multiple choices
		'''
		for j in range(len(choices)):
			if variable_array[0] == choices[j][0]:
					for i in range(1,len(variable_array)):
						variable_array[i] = choices[j][i]
		return variable_array

	def packaging(self,vP):
		'''
		This changes parameter values for a new run for repeatative experiments
		'''
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

		self.mu_h = List([self.mu_h_1,self.mu_h_2])
		self.sigma_h = List([self.sigma_h_1,self.sigma_h_2])
		
		if self.mario == 'beta':
			self.distri_x = np.round(np.arange(self.start,self.stop,self.step),decimals=4)
			self.alpha = 20.0
			self.B_alpha_beta = np.math.factorial(self.alpha-1)*np.math.factorial(self.beta-1)/np.math.factorial(self.alpha+self.beta-1)
			self.pdf_beta = np.array([(x**(self.alpha-1))*((1-x)**(self.beta-1))/self.B_alpha_beta for x in self.distri_x])
			self.pdf_mario = self.pdf_beta/np.sum(self.pdf_beta)

		elif self.mario == 'gumbel':
			#   Gumbel distribution
			x_n = np.sqrt(0.5*lambertw((self.num_opts**2)/(2*np.pi)))# 5 = num of opts, 0.5 = var_sigma, mean = 0.5
			n_rho_xn = x_n/0.5

			dis_x_g = np.round(np.arange(self.start,self.stop,self.step),decimals=4)
			pdf_g =  [n_rho_xn*np.exp(-n_rho_xn*(i-0.5-x_n)-np.exp(-n_rho_xn*(i-0.5-x_n))) for i in dis_x_g]
			variance_g = np.pi/(np.sqrt(6)*n_rho_xn)
			self.pdf_mario = np.flip(np.array(pdf_g).real/np.sum(np.array(pdf_g).real))

		elif self.mario == 'uniform':
			dis_x_u = np.round(np.arange(self.start,self.stop,self.step),decimals=4)
			self.pdf_mario = np.ones(len(dis_x_u))*1/len(dis_x_u)

		
	def prefix(self,path):
		'''
		Input: Data storage path
        Output: Counter for creating new files at the input path
		'''
		check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))
		count = 0
		for i in check:
			if count==i:
				count+=1
		return count

	def add_columns(self,num_robots):
		for i in range(num_robots):
			self.data_columns_name.append(str(i))
			self.data_columns_name.append('Response_'+str(i))
		self.data_columns_name.append('mem_size')
		self.data_columns_name.append('thr_step')
		self.data_columns_name.append('t')

	def save_data(self,path):
		'''
		Input: p (parameter object for current running set of experiments)
        Outputs: Creates a new file to store data, and stores the file name at the parameters object
		'''
		save_string = str(self.pre) + '_' + str(time.time())
		f = open(path+save_string+'.csv','a')
		self.data_f_path = path+save_string+'.csv'
		
		columns = pd.DataFrame(data=np.array([self.data_columns_name]))
		columns.to_csv(self.data_f_path,mode='a',header=False,index=False)

	def save_params(self,param_columns_name,fixed_param,path):
		'''
		Input: param_columns_name (parameter names of current running set of experiments), fixed_param (parameter values that are fixed through out all runs in this experiment)
        Outputs: Creates a new file to store parameters and stores the fixed parameters
		'''
		counter = open(path+str(self.pre),'w+')
		param = open(path+str(self.pre)+'.csv','w+')
		columns_p = pd.DataFrame(data=np.array([param_columns_name]))
		columns_p.to_csv(path+str(self.pre)+'.csv',mode='a',header=False,index=False)
		out_p = pd.DataFrame(data=fixed_param,columns=param_columns_name)	
		out_p.to_csv(path+str(self.pre)+'.csv',mode = 'a',header = False, index=False)	