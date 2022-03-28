# Hello

from unicodedata import decimal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from matplotlib.legend_handler import HandlerTuple
import voter_model as vm
from numba.typed import List
import random

def file_reader(f):
	op = pd.read_csv(path+f)
	pass

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))

    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


if __name__=="__main__":
	path = os.getcwd() + "/results_evolving_fixed_within_range/"
	run_no = 2
	file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
	params = pd.read_csv(path+file_identifier[run_no]+'.csv')
	data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(file_identifier[run_no])] == file_identifier[run_no] and len(f)>10]))[0]

	param_columns = list(params.columns)
	mux = np.array(params["$\mu_{x_{1}}$"])
	op_f = pd.read_csv(path+data_files)
	data_columns = list(op_f.columns)
	sucess_rate = {}
	n = [op_f['n'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	mu_h_pred = [op_f['$\mu_{h_{pred}}$'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	sigma_h_pred = [op_f['$\sigma_{h_{pred}}$'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	mu_h_evolved = [op_f['$\mu_{h_1}$'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	sigma_h_evolved = [op_f['$\sigma_{h_1}$'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	no_yes_ratio = op_f['no:yes']
	ratio_after_replication = [op_f['after_replication_no:yes'][i] for i in range(0,len(op_f['$\mu_{h_1}$']),20)]
	
	# print(no_yes_ratio)

	total = len(op_f["$\mu_{h_1}$"])
	for sets in range(0,total,20):
		success = 0
		for i in range(20):
			if op_f["$x_{max}$ opt No."][sets+i] == op_f["$CDM$ opt No."][sets+i] and np.isnan(op_f["$x_{max}$ opt No."][sets+i]) == False and np.isnan(op_f["$CDM$ opt No."][sets+i]) == False:
				success += 1
			else:
				pass
		sucess_rate[str(mu_h_evolved[int(sets/20)])] = np.round(success/20,decimals=2)

	plt.scatter(np.array(n)-1-np.array(ratio_after_replication),np.array(mu_h_pred)-np.array(mu_h_evolved))
	plt.xlabel("(n-1) - ratio_after_replication",fontsize = 18)
	plt.ylabel("$\mu_{h_{pred}} - \mu_{h_{evolved}}$",fontsize = 18)
	plt.show()
	plt.scatter(np.array(n)-1-np.array(ratio_after_replication),np.array(sigma_h_pred)-np.array(sigma_h_evolved))
	plt.xlabel("(n-1) - ratio_after_replication",fontsize = 18)
	plt.ylabel("$\sigma_{h_{pred}} - \sigma_{h_{evolved}}$",fontsize = 18)
	plt.show()
	succ_rate = np.zeros_like(mu_h_evolved)
	for j in sucess_rate:
		index = np.where(np.array(mu_h_evolved) == float(j))[0]
		succ_rate[index] += sucess_rate[j]
	plt.scatter(np.array(n)-1-np.array(ratio_after_replication),succ_rate)
	plt.xlabel("(n-1) - ratio_after_replication",fontsize = 18)
	plt.ylabel("Average rate of success",fontsize = 18)
	plt.show()
	plt.plot(np.array(n),np.array(sigma_h_evolved))
	plt.xlabel("n",fontsize = 18)
	plt.ylabel("$\sigma_{h_{evolved}}$",fontsize = 18)
	plt.show()
	plt.plot(np.array(n),np.array(mu_h_evolved))
	plt.xlabel("n",fontsize = 18)
	plt.ylabel("$\mu_{h_{evolved}}$",fontsize = 18)
	plt.show()
	plt.plot(np.array(n),np.array(mu_h_pred))
	plt.xlabel("n",fontsize = 18)
	plt.ylabel("$\mu_{h_{pred}}$",fontsize = 18)
	plt.show()
	plt.plot(np.array(n),np.array(sigma_h_pred))
	plt.xlabel("n",fontsize = 18)
	plt.ylabel("$\sigma_{h_{pred}}$",fontsize = 18)
	plt.show()
	# plt.xlabel("n - ratio_after_replication",fontsize = 18)
	# plt.ylabel("$\mu_{h_{pred}} - \mu_{h_{evolved}}$",fontsize = 18)

	# vp = vm.variableParams()

	# mean_x = List([mux[0],mux[0]])
	# sigma_x = List([np.array(params["$\sigma_{x_{1}}$"])[0],np.array(params["$\sigma_{x_{1}}$"])[0]])
	# start = stop = np.sum(mean_x)/len(mean_x) - 2*np.sum(sigma_x)-5
	# stop = np.sum(mean_x)/len(mean_x) + 2*np.sum(sigma_x)+5
	# step = 0.001
	# x = np.arange(start,stop,step)

	# pdf = vp.gaussian(x,mean_x,sigma_x)
	# area = (np.sum(pdf)*step)
	# pdf = np.multiply(pdf,1/area)
	# mu_h_pred = vp.ICPDF(1-(1/np.array(params['n'])[0]),List([mux[0],mux[0]]),stop,step,x,pdf)
	
	# fig, ax = plt.subplots()
	# ax.invert_yaxis()
	# ax.plot(x,pdf,c='orange')
	# slices = []
	# mid_slices=[]
	# for i in range(1,np.array(params['n'])[0],1):
	# 	ESM = vp.ICPDF(float(i)/np.array(params['n'])[0],mean_x,stop,step,x,pdf)
	# 	slices.append(np.round(ESM,decimals=3))


	# number_of_colors = np.array(params['n'])[0]

	# color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
	# 		for i in range(number_of_colors)]

	# for i in range(len(slices)+1):
	# 	if i!=0 and i!=len(slices):
	# 		x1 = np.arange(slices[i-1],slices[i],0.0001)
	# 		pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
	# 		pdf1 = np.multiply(pdf1,1/area)
	# 		ax.fill_between(x1,0,pdf1,facecolor=color[i])
	# 	elif i==0:
	# 		x1 = np.arange(start,slices[i],0.0001)
	# 		pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
	# 		pdf1 = np.multiply(pdf1,1/area)
	# 		ax.fill_between(x1,0,pdf1,facecolor=color[i])
	# 	elif i==len(slices):
	# 		x1 = np.arange(slices[-1],stop,0.0001)
	# 		pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
	# 		pdf1 = np.multiply(pdf1,1/area)
	# 		ax.fill_between(x1,0,pdf1,facecolor=color[i])
	
	# axs1 = ax.twinx()
	# axs1.axvline(mu_h_pred,0,500,color='red',linewidth = 0.5)
	# axs1.plot(muh,succ_rate)
	# align_yaxis(axs1,0.00,ax,0.00)
	
	# plt.tight_layout()
	# # # plt.title(title,font,y=-0.28,color=(0.3,0.3,0.3,1))
	# # plt.legend(loc='upper center', bbox_to_anchor=(0.56, 1.17),prop=dict(weight='bold',size=12),labelcolor=(0.3,0.3,0.3,1),frameon=False)  #bbox earlier(0.0,1.1)
	# # plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
	# # plt.minorticks_on()
	# # plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)

	# plt.savefig(path+"third_plot.png",format = "png",bbox_inches="tight",pad_inches=0.2)
	# plt.show()
	
		

    

