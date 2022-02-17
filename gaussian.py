
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
import statistics
  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)/2
  
plt.plot(x_axis, norm.pdf(x_axis, mean, sd),color='indigo',linewidth=3)
plt.show()


  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)
  
plt.plot(x_axis, uniform.pdf(x_axis, mean, sd),color='indigo',linewidth=3)
plt.show()

x_axis = np.arange(-10, 10, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)/4
  
plt.plot(x_axis, norm.pdf(x_axis, mean-4, sd)+norm.pdf(x_axis, mean+4, sd),color='indigo',linewidth=3)
plt.show()