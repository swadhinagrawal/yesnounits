# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def file_reader(f):
	op = pd.read_csv(path+f)
	pass

if __name__=="__main__":
	path = os.getcwd() + "/results/"
	run_no = 4
	file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
	params = pd.read_csv(path+file_identifier[run_no]+'.csv')
	data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(file_identifier[run_no])] == file_identifier[run_no] and len(f)>10]))[0]

	param_columns = list(params.columns)

	data = pd.read_csv(path+data_files)
	data_columns = list(data.columns)
	sucess_rate = []
	no_units = data['No_units'].unique()

	total = len(data['No_units'])
	for sets in range(len(no_units)-1):
		success = 0
		repeats = 100
		for i in range(repeats):
			if data["$x_{max}$ opt No."][sets*repeats+i] == data["$CDM$ opt No."][sets*repeats+i] and np.isnan(data["$x_{max}$ opt No."][sets*repeats+i]) == False and np.isnan(data["$CDM$ opt No."][sets*repeats+i]) == False:
				success += 1
			else:
				pass
		sucess_rate.append(np.round(success/repeats,decimals=2))

	
	plt.plot(np.array(no_units[:-1]),np.array(sucess_rate))
	plt.xlabel("No-Units",fontsize = 18)
	plt.ylabel("Average rate of success",fontsize = 18)
	plt.show()
