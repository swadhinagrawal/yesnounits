#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
from numba import cuda,njit,vectorize,jit
import matplotlib.pyplot as plt
import random
import time
import random_number_generator as rng

@njit(parallel=True)
def gaussian(x,mu,sigma,alpha):
      f = 0
      for i in range(len(mu)):
            k = 1/(2.5066282746310002*sigma[i])
            f += alpha[i]*k*np.exp(-((x-mu[i])**2)/(2*sigma[i]**2))
      return f

@njit(parallel=True)
def RandomNumberGenerator(mu,sigma,alpha):
      start = mu[0] - 10
      stop = mu[-1] + 10
      x = np.arange(start,stop,0.0001)
      pdf =  [int(100*gaussian(i,mu,sigma,alpha)) for i in x]
      data = [0]
      for i in range(len(pdf)):
            data += [x[i] for j in range(pdf[i])]
      del(data[0])
      samples = np.random.choice(np.array(data),size=10)
      return x,samples
      
@njit
def units_n(mu_m,sigma_m,number_of_options):
      a = np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_m),number_of_options)
      for i in range(len(peak_choice)):
            k = int(np.random.normal(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            while k<=0:
                  k = int(np.random.normal(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            a[i] = k      
      return a

@njit
def threshold_n(mu_h,sigma_h,m_units,h_type=3):
      a = np.zeros(m_units)
      peak_choice = np.random.randint(0,len(mu_h),m_units)
      for i in range(len(peak_choice)):
            a[i] = round(np.random.normal(mu_h[peak_choice[i]],sigma_h[peak_choice[i]]),h_type)
      return a

@njit
def units_u(mu_m,sigma_m,number_of_options):
      a = np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_m),number_of_options)
      for i in range(len(peak_choice)):
            k = int(np.random.uniform(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            while k<=0:
                  k = int(np.random.uniform(mu_m[peak_choice[i]],sigma_m[peak_choice[i]]))
            a[i] = k      
      return a

@njit
def threshold_u(mu_h,sigma_h,m_units,h_type=3):
      a = np.zeros(m_units)
      peak_choice = np.random.randint(0,len(mu_h),m_units)
      for i in range(len(peak_choice)):
            a[i] = round(np.random.uniform(mu_h[peak_choice[i]],sigma_h[peak_choice[i]]),h_type)
      return a

def ref_highest_qual(Dx):
      """
      Provides known highest quality option
      """
      best_list = np.array(np.where(Dx == max(Dx)))[0]
      opt_choosen = np.random.randint(0,len(best_list))
      return best_list[opt_choosen]

@njit
def dx_n(mu_x,sigma_x,number_of_options,x_type=3):
      """
      Provides distribution of quality stimulus for each option upto specified decimal places
      """
      Dx =  np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_x),number_of_options)
      # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
      # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
      for i in range(len(peak_choice)):
            Dx[i] = round(np.random.normal(mu_x[peak_choice[i]],sigma_x[peak_choice[i]]),x_type)
      return Dx

@njit
def dx_u(mu_x,sigma_x,number_of_options,x_type=3):
      """
      Provides distribution of quality stimulus for each option upto specified decimal places
      """
      Dx =  np.zeros(number_of_options)
      peak_choice = np.random.randint(0,len(mu_x),number_of_options)
      # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
      # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
      for i in range(len(peak_choice)):
            Dx[i] = round(np.random.uniform(mu_x[peak_choice[i]],sigma_x[peak_choice[i]]),x_type)
      return Dx
          
def quality(distribution,mu_x,sigma_x,number_of_options,x_type=3):
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