# Hello

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import voter_model as vm
from numba.typed import List
import random
import ants_analysis


icpdf = ants_analysis.Prediction()
def file_reader(f):
	op = pd.read_csv(path+f)
	pass

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))

    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


if __name__=="__main__":
	path = os.getcwd() + "/results/"
	run_no = 5
	file_identifier = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
	data_files = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(file_identifier[run_no])] == file_identifier[run_no]]))

	data = pd.read_csv(path+data_files[0])
	data_columns = list(data.columns)

	run_no_1 = 2
	file_identifier_1 = np.sort(np.array([f for f in os.listdir(path) if '.' not in f]))
	data_files_1 = np.sort(np.array([f for f in os.listdir(path) if '.csv' in f and f[:len(file_identifier_1[run_no_1])] == file_identifier_1[run_no_1]]))

	data_1 = pd.read_csv(path+data_files_1[0])

	muh = np.unique(data["muH1"])

	majority_ars = data["ARS"]
	voter_ars = data['voter_ARS']
	voter_ars_1 = data_1['voter_ARS']

	vp = vm.variableParams()

	mean_x = List([data["muQ1"][0],data["muQ1"][0]])
	sigma_x = List([1,1])
	pdf,area,x,start,stop = ants_analysis.PDF(vp.gaussian,mean_x,sigma_x)
	step = 0.0001
	mu_h_pred = icpdf.ICPDF(1-(1/5),mean_x,step,x,pdf)

	fig, ax = plt.subplots()
	ax.invert_yaxis()
	ax.plot(x,pdf,c='orange')
	slices = []
	mid_slices=[]
	for i in range(1,5,1):
		ESM = icpdf.ICPDF(float(i)/5,mean_x,step,x,pdf)
		slices.append(np.round(ESM,decimals=3))
	# for i in range(1,2,1):
	# 	ESM = vp.ICPDF(float(i)/2,mean_x,stop,step,x,pdf)
	# 	slices.append(np.round(ESM,decimals=3))

	number_of_colors = 5
	random.seed(16812)
	color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			for i in range(number_of_colors)]

	for i in range(len(slices)+1):
		if i!=0 and i!=len(slices):
			x1 = np.arange(slices[i-1],slices[i],0.0001)
			pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
			pdf1 = np.multiply(pdf1,1/area)
			ax.fill_between(x1,0,pdf1,facecolor=color[i])
		elif i==0:
			x1 = np.arange(start,slices[i],0.0001)
			pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
			pdf1 = np.multiply(pdf1,1/area)
			ax.fill_between(x1,0,pdf1,facecolor=color[i])
		elif i==len(slices):
			x1 = np.arange(slices[-1],stop,0.0001)
			pdf1 =  vp.gaussian(x1,mean_x,sigma_x)
			pdf1 = np.multiply(pdf1,1/area)
			ax.fill_between(x1,0,pdf1,facecolor=color[i])
	
	axs1 = ax.twinx()
	axs1.axvline(mu_h_pred,0,500,color='red',linewidth = 0.5)
	# axs1.plot(muh,voter_ars,linewidth = 5)
	axs1.plot(muh,voter_ars_1,c='brown')
	# axs1.plot(list(muh),majority_ars,c="green")
	# smoothened = smooth(majority_ars[:-1],30)

	vote_diff = []
	vote_diff_13 = []
	vote_diff_14 = []
	vote_diff_15 = []
	no_proportion = []
	yes_votes_best = []
	sum_yes_without_best = []
	for ind,row in data.iterrows():
		sum_ = 0
		sum_13 = 0
		sum_14 = 0
		sum_15 = 0
		sum_best = 0
		sum_rest = 0
		avg_no_proportion = 0
		for i in range(200):
			search_array = [str((i,max([0,int(j%5)]))) for j in range(5*i,5*(i+1))]
			votes_array = [row[k] for k in search_array]
			# avg_no_proportion += np.sum(votes_array)/(50*5)
			avg_no_proportion += np.sum(votes_array)/(5)
			best_opt = max(votes_array)
			sum_best += best_opt
			sum_rest += np.sum(votes_array) - best_opt
			index_best = np.where(votes_array == best_opt)[0][0]
			votes_array[index_best] = -10
			second_best = max(votes_array)
			index_best_2 = np.where(votes_array == second_best)[0][0]
			votes_array[index_best_2] = -10
			best_3 = max(votes_array)
			index_best_3 = np.where(votes_array == best_3)[0][0]
			votes_array[index_best_3] = -10
			best_4 = max(votes_array)
			index_best_4 = np.where(votes_array == best_4)[0][0]
			votes_array[index_best_4] = -10
			best_5 = max(votes_array)
			index_best_5 = np.where(votes_array == best_5)[0][0]
			votes_array[index_best_5] = -10
			sum_ += best_opt - second_best
			sum_13 += best_opt - best_3
			sum_14 += best_opt - best_4
			sum_15 += best_opt - best_5

		vote_diff.append(sum_/200)
		vote_diff_13.append(sum_13/200)
		vote_diff_14.append(sum_14/200)
		vote_diff_15.append(sum_15/200)
		yes_votes_best.append(sum_best/200)
		sum_yes_without_best.append(sum_rest/200)
		# no_proportion.append(1 - avg_no_proportion/200)
		no_proportion.append(50 - avg_no_proportion/200)
	axs1.plot(muh,vote_diff,c=color[-2])#/max(vote_diff)

	axs1.plot(muh,yes_votes_best,c=color[-1])
	axs1.plot(muh,sum_yes_without_best,c='red',linewidth = 3)
	# axs1.plot(muh,vote_diff_13/max(vote_diff),c=color[-3])
	# axs1.plot(muh,vote_diff_14/max(vote_diff),c=color[-4])
	# axs1.plot(muh,vote_diff_15/max(vote_diff),c=color[-5])
	axs1.plot(muh,no_proportion,c='black')

	

	avg_exploitation = None
	avg_yes_votes = []
	for opt in range(number_of_colors):
		a = data[str(0)+'_exploited_agents_'+str(opt+1)]
		for run in range(1,200):
			a += data[str(run)+'_exploited_agents_'+str(opt+1)]
		a = a/200
		if opt==0:
			avg_exploitation = a
		# a = a/max(avg_exploitation)
		a = smooth(a,30)
		plt.plot(muh,a,color[opt])


		a = list(data[str((0,opt))])
		for run in range(1,200):
			a += data[str((run,opt))]
		a = a/200
		avg_yes_votes.append(a)
	avg_yes_votes = np.array(avg_yes_votes)
	votes = [avg_yes_votes[_][0] for _ in range(number_of_colors)]
	normalizer = 1#max(votes)
	for opt in range(number_of_colors):
		for mu in range(0,len(muh)):
			avg_yes_votes[:,mu] = np.sort(avg_yes_votes[:,mu])
		a = smooth(avg_yes_votes[opt],5)
		plt.plot(muh,a/normalizer,c = color[opt])


	align_yaxis(axs1,0.00,ax,0.00)
	plt.xlabel("$\mu_{h}$",fontsize = 18)
	# plt.ylabel("Average rate of success",fontsize = 18)
	plt.tight_layout()

	# plt.savefig(path+"6_overlapped_plot.png",format = "png",bbox_inches="tight",pad_inches=0.2)
	plt.show()
