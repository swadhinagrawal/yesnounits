# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__=="__main__":
	path = os.getcwd() + "/results/"
	run_no = 16
	# file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
	params = pd.read_csv(path+str(run_no)+'.csv')
	data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(str(run_no))] == str(run_no) and len(f)>10]))[0]

	param_columns = list(params.columns)

	data = pd.read_csv(path+data_files)
	data_columns = list(data.columns)
	
	
	plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\mu_{h_1}$'])
	plt.xlabel("Time",fontsize = 18)
	plt.ylabel("$\mu_{h_1}$",fontsize = 18)
	plt.show()

	plt.plot(range(len(data["$\mu_{h_1}$"])),data['$\sigma_{h_1}$'])
	plt.xlabel("Time",fontsize = 18)
	plt.ylabel("$\sigma_{h_1}$",fontsize = 18)
	plt.show()

	thresholds = [i for i in data.iloc[-1][6:]]
	bins = np.linspace(min(thresholds),max(thresholds),50)
	distribution = np.bincount(np.digitize(thresholds, bins))
	mean = np.sum(thresholds)/len(thresholds)

	plt.plot(bins,distribution[1:])
	plt.plot([mean,mean],[0,max(distribution)],c='red')

	plt.xlabel("$h_{i}$",fontsize = 18)
	plt.ylabel("Frequency",fontsize = 18)
	plt.show()
