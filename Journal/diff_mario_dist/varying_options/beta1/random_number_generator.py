#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
from numba import cuda,njit
import matplotlib.pyplot as plt
import random
import time
import random_number_generator as rng


@njit
def units_n(mu_m,sigma_m,number_of_options):
      '''
      Inputs: mu_m (list of peaks), sigma_m (list of variances), number_of_options (number of options in env)
      Outputs: array consisting number of units to be assigned to each option
      '''
      a = np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_m),number_of_options)
      for i in range(len(peak_choice)):
            k = int(np.random.normal(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            while k<=0:
                  k = int(np.random.normal(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            a[i] = k      
      return a

@njit
def units_u(mu_m,sigma_m,number_of_options):
      '''
      Inputs: mu_m (list of peaks), sigma_m (list of variances), number_of_options (number of options in env)
      Outputs: array consisting number of units to be assigned to each option
      '''
      a = np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_m),number_of_options)
      for i in range(len(peak_choice)):
            k = int(np.random.uniform(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            while k<=0:
                  k = int(np.random.uniform(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            a[i] = k      
      return a

@njit
def threshold_n(mu_h,sigma_h,m_units,h_type=3):
      '''
      Inputs: mu_h (list of peaks), sigma_h (list of variances), m_units (number of units assigned to a given option)
      Outputs: array consisting response thresholds of each unit assigned to a particular option
      '''
      a = np.zeros(m_units)
      peak_choice = np.random.randint(0,len(mu_h),m_units)
      for i in range(len(peak_choice)):
            a[i] = round(np.random.normal(mu_h[peak_choice[i]],sigma_h[peak_choice[i]]),h_type)
      return a

@njit
def threshold_u(mu_h,sigma_h,m_units,h_type=3):
      '''
      Inputs: mu_h (list of peaks), sigma_h (list of variances), m_units (number of units assigned to a given option)
      Outputs: array consisting response thresholds of each unit assigned to a particular option
      '''
      a = np.zeros(m_units)
      peak_choice = np.random.randint(0,len(mu_h),m_units)
      for i in range(len(peak_choice)):
            a[i] = round(np.random.uniform(mu_h[peak_choice[i]],sigma_h[peak_choice[i]]),h_type)
      return a

@njit
def dx_n(mu_x,sigma_x,number_of_options,x_type=3):
      '''
      Inputs: mu_x (list of peaks), sigma_x (list of variances), number_of_options (number of options in env)
      Outputs: array consisting quality values for each option
      '''
      Dx =  np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_x),number_of_options)
      # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
      # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
      for i in range(len(peak_choice)):
            Dx[i] = round(np.random.normal(mu_x[peak_choice[i]],sigma_x[peak_choice[i]]),x_type)
      return Dx

@njit
def dx_u(mu_x,sigma_x,number_of_options,x_type=3):
      '''
      Inputs: mu_x (list of peaks), sigma_x (list of variances), number_of_options (number of options in env)
      Outputs: array consisting quality values for each option
      '''
      Dx =  np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_x),number_of_options)
      # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
      # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
      for i in range(len(peak_choice)):
            Dx[i] = round(np.random.uniform(mu_x[peak_choice[i]],sigma_x[peak_choice[i]]),x_type)
      return Dx

def ref_highest_qual(Dx):
      '''
      Inputs: Dx (list of quality values of each option)
      Outputs: index of option with highest quality value, in case of multiple same quality option, a randomly choosen from the list of best options is returned
      '''
      best_list = np.array(np.where(Dx == max(Dx)))[0]
      opt_choosen = np.random.randint(0,len(best_list))
      return best_list[opt_choosen]

def quality(distribution,mu_x,sigma_x,number_of_options,x_type=3):
      '''
      Inputs: distribution (function to generate quality values), mu_x (list of peaks), sigma_x (list of variances), number_of_options (number of options in env)
      Outputs: Index of best option and the list of qualities for the set of options
      '''
      dis_x = distribution(mu_x,sigma_x,number_of_options)
      ref = ref_highest_qual(dis_x)
      return ref,dis_x


if __name__ == '__main__':
      
      for k in range(1000):
            # t1 = time.time()
            # x,samples = RandomNumberGenerator(mu= [10,15],sigma = [1,1],alpha=[0.45,0.55])
            # t2 = time.time()
            # print(t2-t1)
            x = np.arange(0,30,0.0001)
            t1 = time.time()
            ref,samples = quality([10,20],[1,1],10)
            t2 = time.time()
            print(t2-t1)
            y = np.zeros_like(x)
            for i in range(len(samples)):
                  j = np.where(np.round(x,decimals=2)==np.round(samples[i],decimals=2))
                  y[j] += 1
            plt.plot(x,y)
            plt.show()