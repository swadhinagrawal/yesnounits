# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


import numpy as np
from numba.typed import List
from math import sqrt,exp
from numba import guvectorize, float64, njit
import os,time
import pandas as pd

class variableParams:
	def __init__(self,column_names,path):
		#   Option details
		self.num_opts = [2,3,5,8,10,15,20,30,40,80,100]

		self.mu_x_1 = np.arange(0.1,15.1,0.1)
		self.mu_x_2 = np.arange(0.1,15.1,0.1)
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

		#	Data logging
		self.data_columns_name = column_names   #['$\mu_{h_1}$','$x_{max}$ opt No.','$x_{max}$','$CDM$ opt No.','$CDM$','Iterations']
		self.pre = self.prefix(path)

	@staticmethod
	@guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
	def gaussian(x,mu,sigma,result):
		'''
		Inputs: mu (list), sigma (list)
        Outputs: list of probability density values using a gaussian PDF
		'''
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
		'''
		Inputs: mu (list), sigma (list)
        Outputs: list of probability density values using a gaussian PDF
		'''
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
	def ICPDF(area,mu,step,x,pdf):
		'''
		Inputs: area, mu (list), step (resolution), x (x-values), pdf (probability dsitribution function)
        Outputs: Calculates the x-value for which area under PDF is = area (input) Inverse Cummulative Distribution Function value.
		'''
		if len(mu)>1:    
			dummy_area = 0.5
			x_ = (mu[0]+mu[-1])/2.0
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

	def save_data(self,p,path):
		'''
		Input: p (parameter object for current running set of experiments)
        Outputs: Creates a new file to store data, and stores the file name at the parameters object
		'''
		save_string = str(self.pre) + '_' + str(time.time())
		f = open(path+save_string+'.csv','a')
		p.data_f_path = path+save_string+'.csv'
		
		columns = pd.DataFrame(data=np.array([self.data_columns_name]))
		columns.to_csv(p.data_f_path,mode='a',header=False,index=False)

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