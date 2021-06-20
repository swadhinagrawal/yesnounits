# !/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

from operator import le
from numba.core.types import scalars
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import count_nonzero
from statsmodels.stats.proportion import proportions_ztest as pt
import pandas as pd
import os
import time
import random_number_generator as rng
from numba import  njit
from sklearn import linear_model
import random
import requests
import json
from numba.typed import List
from math import sqrt,exp
from numba import guvectorize, float64
from scipy import stats

path = os.getcwd() + "/results/"

def pushbullet_message(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = 'o.YlTBKuQWnkOUsCP9ZxzWC9pvFNz1G0mi'
    # resp = requests.post('https://api.pushbullet.com/v2/pushes', 
    #                      data=json.dumps(msg),
    #                      headers={'Authorization': 'Bearer ' + TOKEN,
    #                               'Content-Type': 'application/json'})
    # if resp.status_code != 200:
    #     raise Exception('Error',resp.status_code)
    # else:
    #     print ('Message sent')

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err,sigma_assessment_err):
        self.err_type = err_type
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_assessment_err = mu_assessment_err
        self.sigma_assessment_err = sigma_assessment_err
        self.votes = None
        self.no_votes = None
        self.no_proportion = None
        self.yes_stats = []
        self.max_ratio_pvalue = None
        self.y_ratios = []

    def vote_counter(self,assigned_units,Dx,pc= None):
        self.votes,self.no_votes = self.counter(self.number_of_options,self.mu_assessment_err,self.sigma_assessment_err,assigned_units,self.err_type,Dx)
        self.no_proportion = [self.no_votes[i]/pc[i] for i in range(self.number_of_options)]
        self.y_ratios= [self.votes[j]/pc[j] for j in range(self.number_of_options)]
        self.hypothesis_testing(pc)
        self.max_ratio_pvalue = self.hypothesis_testing_top_two(pc)
    
    @staticmethod
    @njit
    def counter(nop,e_mu,e_sigma,ass_u,e_type,Dx):
        votes = [0 for i in range(nop)]
        no_votes = [0 for i in range(nop)]
        for i in range(nop):
            for j in range(len(ass_u[i])):
                assesment_error = round(np.random.normal(e_mu,e_sigma),e_type)
                if (ass_u[i][j] < (Dx[i] + assesment_error)):
                    votes[i] += 1
                else:
                    no_votes[i] += 1
        return votes,no_votes

    def hypothesis_testing(self,pc):
        for i in range(self.number_of_options-1):
            self.yes_stats.append([])
            for j in range(i+1,self.number_of_options):
                self.yes_stats[i].append(pt([self.votes[i],self.votes[j]],[pc[i],pc[j]],verbose=False))

    def hypothesis_testing_top_two(self,pc):
        max_1 = [max(self.y_ratios),self.y_ratios.index(max(self.y_ratios))]
        self.y_ratios[max_1[1]] = 0
        max_2 = [max(self.y_ratios),self.y_ratios.index(max(self.y_ratios))]
        self.y_ratios[max_1[1]] = max_1[0]
        pvalue = pt([self.votes[max_1[1]],self.votes[max_2[1]]],[pc[max_1[1]],pc[max_2[1]]],verbose=False)
        return pvalue

    def best_among_bests_no(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        available_opt = np.array(np.where(np.array(self.no_votes) == min(self.no_votes)))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1
        else:
            return 0

    def best_among_bests(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1
        else:
            return 0
    
    def quorum_voting(self,assigned_units,Dx,ref_highest_quality):
        """
        success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
        """
        units_used = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            count = 0
            while count<self.quorum:
                if units_used[i] == len(assigned_units[i]):
                    break
                if (assigned_units[i][units_used[i]] < (Dx[i] + assesment_error[units_used[i]])):
                    count += 1
                units_used[i] += 1
        units_used = np.array(units_used)
        loc = np.array(np.where(units_used == min(units_used)))[0]
        opt_choosen = np.random.randint(0,len(loc))

        flag = 0
        for i in range(self.number_of_options):
            if units_used[i] == len(assigned_units[i]):
                flag += 1
        if flag==self.number_of_options:
            quorum_reached = 0
            result = 0
            return result,quorum_reached

        if loc[opt_choosen] ==  ref_highest_quality:
            result = 1
            quorum_reached = 1
            return result,quorum_reached
        else:
            quorum_reached = 1
            result = 0
            return result,quorum_reached


class Qranking:
    def __init__(self,number_of_options):
        self.n = number_of_options
        self.ref_rank = np.zeros((self.n,self.n))
        self.exp_rank = np.zeros((self.n,self.n))
        self.exp_rank_w_n = np.zeros((self.n,self.n))

    def ref_ranking(self,oq,y_ratios,no_votes):
        for i in range(len(oq)):
            for j in range(i+1,self.n):
                if oq[i]>oq[j]:
                    self.ref_rank[i,j] = 1
                if y_ratios[i]>y_ratios[j] and no_votes[i]<no_votes[j]:
                    self.exp_rank[i,j] = 1
                elif y_ratios[i]<y_ratios[j] and no_votes[i]>no_votes[j]:
                    self.exp_rank[i,j] = 0
                elif y_ratios[i]<y_ratios[j] and no_votes[i]<no_votes[j]:
                    self.exp_rank[i,j] = 0.5
                elif y_ratios[i]>y_ratios[j] and no_votes[i]>no_votes[j]:
                    self.exp_rank[i,j] = 0.5

    def ref_ranking_w_n(self,oq,y_ratios,no_votes):
        for i in range(len(oq)):
            for j in range(i+1,self.n):
                if y_ratios[i]>y_ratios[j]:
                    self.exp_rank_w_n[i,j] = 1
                elif y_ratios[i]<y_ratios[j]:
                    self.exp_rank_w_n[i,j] = 0
                else:
                    self.exp_rank_w_n[i,j] = 0.5

    def incorrectness_cost(self,exp_rank):
        measure_of_incorrectness = 0
        for i in range(self.n):
            for j in range(i+1,self.n):
                measure_of_incorrectness += abs(exp_rank[i,j]-self.ref_rank[i,j])
        measure_of_incorrectness = 2*measure_of_incorrectness/(self.n*(self.n - 1))
        return measure_of_incorrectness           #   Higher measure of incorrectness more bad is the ranking by units votes


class workFlow:
    def __init__(self):
        pass

    def majority_decision(self,number_of_options,Dx,assigned_units,ref_highest_quality,pc,mu_assessment_err=0,sigma_assessment_err=0,\
        err_type=0,quorum = None):

        DM = Decision_making(number_of_options=number_of_options,err_type=err_type,\
        mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
        DM.quorum = quorum
        DM.vote_counter(assigned_units,Dx,pc)
        majority_dec = DM.best_among_bests_no(ref_highest_quality)
        qrincorrectness = Qranking(number_of_options)
        qrincorrectness.ref_ranking(Dx,DM.y_ratios,DM.no_votes)
        incorrectness = qrincorrectness.incorrectness_cost(qrincorrectness.exp_rank)
        qrincorrectness.ref_ranking_w_n(Dx,DM.y_ratios,DM.no_votes)
        incorrectness_w_n = qrincorrectness.incorrectness_cost(qrincorrectness.exp_rank_w_n)
        if quorum == None:
            # plt.scatter(Dx,DM.votes)
            # plt.show()
            return majority_dec,incorrectness,incorrectness_w_n,DM.yes_stats,DM.max_ratio_pvalue
        else:
            result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
            return result,quorum_reached,majority_dec
    
    def one_run(self,distribution_m,distribution_h,distribution_x,mu_m,sigma_m,mu_h,sigma_h,mu_x,sigma_x,number_of_options=None,h_type=3,x_type=3,err_type=0,mu_assessment_err= 0,\
        sigma_assessment_err=0,quorum= None):

        pc = rng.units(distribution_m,number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)

        units_distribution = []
        
        for i in pc:
            units_distribution.append(rng.threshold(distribution_h,m_units=int(i),mu_h=mu_h,sigma_h=sigma_h))
        
        ref,qc = rng.quality(distribution_x,number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

        dec,_,_,_,_ = self.majority_decision(number_of_options=number_of_options,Dx = qc,\
            assigned_units=units_distribution,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=ref,quorum=quorum,pc=pc)
        
        if dec == 1:
            print("success")

        else:
            print("failed")

    def multi_run(self,distribution_m,distribution_h,distribution_x,mu_m,sigma_m,mu_h,sigma_h,mu_x,sigma_x,number_of_options=None,h_type=3,x_type=3,err_type=0,mu_assessment_err= 0,\
        sigma_assessment_err=0,quorum= None):

        pc = List(distribution_m(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m))

        units_distribution = []
        
        for i in pc:
            units_distribution.append(List(distribution_h(m_units=int(i),mu_h=mu_h,sigma_h=sigma_h)))
        
        ref,qc = rng.quality(distribution = distribution_x,number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc,\
            assigned_units=List(units_distribution),err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=ref,quorum=quorum,pc=pc)

        return dec


class Prediction:
    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
    def gaussian(x,mu,sigma,result):
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
    def ICPDF(area,mu,stop,step,x,pdf):
        # import faulthandler; faulthandler.enable()
        if len(mu)>1:    
            dummy_area = 0.5
            x_ = (mu[0]+mu[1])/2.0
            count = np.argmin(np.abs(x-x_))
        else:
            dummy_area =0.5
            x_ = mu[0]

        count = np.argmin(np.abs(x-x_))
        
        while abs(dummy_area-area)>0.0001:
            if dummy_area>area:
                count -= 1
                dummy_area -= pdf[count]*step
                x_ -= step
            elif area>dummy_area:
                count += 1
                dummy_area += pdf[count]*step
                x_ += step
        return x_

    @staticmethod
    def optimization(x,y,z,max_iter=2000,d = 0.2):
        iterations = 0
        goodness_of_fit = -np.Inf
        iter = []
        fit_cost = []
        while iterations<max_iter:
            selected = np.random.randint(0,len(x),int(2*len(x)/3))
            maybeInliers_x = []
            maybeInliers_y = []
            maybeInliers_z = []
            for i in selected:
                maybeInliers_z.append(z[i])
                maybeInliers_x.append(x[i])
                maybeInliers_y.append(y[i])
            
            [slope,intercept] = np.polyfit(maybeInliers_x,maybeInliers_y,deg=1)
            bestFit = [slope,intercept]
            alsoInliers = []
            set = range(len(x))
            for point1 in np.setdiff1d(set,selected):
                dis = abs(slope*x[point1]-y[point1]+intercept)/((slope*slope + 1)**0.5)
                if dis<=d:
                    alsoInliers.append(point1)

            inliers_x = maybeInliers_x + [x[i] for i in alsoInliers]
            inliers_y = maybeInliers_y + [y[i] for i in alsoInliers]
            
            for i in alsoInliers:
                maybeInliers_z.append(z[i])
            params = np.polyfit(inliers_x,inliers_y,deg=1)
            thisgoodness = sum(maybeInliers_z)/len(maybeInliers_z)
            if thisgoodness > goodness_of_fit:
                bestFit = params
                goodness_of_fit = thisgoodness

            iterations += 1
            iter.append(iterations)
            fit_cost.append(goodness_of_fit)
        
        return [bestFit,np.round(goodness_of_fit,decimals=3),iter,fit_cost]

    @staticmethod
    def z_extractor(x,y,x_line,z):
        z = np.array(z).reshape(len(y),len(x))
        z_extracted = []
        for i in range(len(x)):
            z_loc = np.argmin(np.abs(np.array(y)-x_line[i]))
            z_extracted.append(z[z_loc,i])
        return z_extracted

    @staticmethod
    def Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,distribution_fn,ICPDF_fn,hrcc_area,extractor,optimizer,nop):
        x_1 = []
        step = 0.01
        for i in x:
            if x_var_ == '$\mu_{x_1}$':
                mu = List([i,i+delta_mu])
            else:
                mu = List([i-delta_mu,i])
            
            sigma = List([sigma_x_1,sigma_x_2])
            start = np.sum(mu)/len(mu) - np.sum(sigma)-5
            stop = np.sum(mu)/len(mu) + np.sum(sigma)+5
            dis_x = np.arange(start,stop,step)
            pdf =  distribution_fn(dis_x,mu,sigma)
            pdf = np.multiply(pdf,1/(np.sum(pdf)*step))
            print(np.sum(pdf)*step)
            _1 = ICPDF_fn(hrcc_area,mu,stop,step,dis_x,pdf)
            x_1.append(_1)
            # print(np.round(len(x_1)/len(x),decimals=2),end="\r")

        z_extracted = extractor(x,y,x_1,z)
        hrcc = optimizer(x,x_1,z_extracted)
        return hrcc


class Visualization:
    def data_visualize(self,file_name,save_plot,x_var_,y_var_,z_var_,plot_type,gaussian=1,uniform=0,cbar_orien=None,line_labels=None,sigma_x_1=None,\
        data =None,num_of_opts=None,delta_mu=None,sigma_x_2 = None,z1_var_=None):
        # gives data as array

        op = pd.read_csv(path+file_name)
        opt_var = []

        for j in range(len(op[x_var_])):
            a = {}
            for i in op:
                a[str(i)] = op[str(i)][j]
            opt_var.append(a)

        x = []
        y = []
        z = []  # Make sure that it is in ordered form as y variable (i.e. 1st column of data file)
        z1 = []
        z_max = max(op[z_var_])
        z_best = []
        z_only_best = []
        xa = []
        ya = []

        for i in opt_var:
            # x,y,z variables
            if i[x_var_] not in x:
                x.append(i[x_var_])
            if i[y_var_] not in y:
                y.append(i[y_var_])
            z.append(i[z_var_])

            # x,y,z for only HRCC
            if i[z_var_] >= z_max-0.05:
                z_best.append(i[z_var_])
                z_only_best.append(i[z_var_])
                xa.append(i[x_var_])
                ya.append(i[y_var_])
            else:
                z_best.append(min(z))
            
            if z1_var_ != None:
                z1.append(i[z1_var_])

            print(np.round(100*len(z)/len(opt_var),decimals=2),end="\r")
        print(np.round(100*len(z)/len(opt_var),decimals=2))
        prd = Prediction()
        if 'mu' in x_var_:
            HRCC = prd.optimization(xa,ya,z_only_best)
        else:
            HRCC=[None,None,None]
        
        if plot_type == 'graphics':
            self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],line_labels=line_labels)
            if gaussian ==1:
                self.linePlot(HRCC[2],HRCC[3],x_name='Number of iterations',y_name='Average HRCC',z_name=[str(HRCC[1])],title='Maximizing HRCC for best fit',save_name=path+save_plot+'HRCC.pdf')
                # Mean of ESM and ES2M
                predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.gaussian,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
                d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_meanESMES2M.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

                # # ESM non integral

                # predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.gaussian,prd.ICPDF,1.0-(1.0/(2*line_labels)),prd.z_extractor,prd.optimization,line_labels)
                # d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                # delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                # self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_ESM.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

                # # ES2M non integral

                # predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.gaussian,prd.ICPDF,1.0-(3.0/(2*line_labels)),prd.z_extractor,prd.optimization,line_labels)
                # d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                # delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                # self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_ES2M.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

            if uniform ==1:
                self.linePlot(HRCC[2],HRCC[3],x_name='Number of iterations',y_name='Average HRCC',z_name=[str(HRCC[1])],title='Maximizing HRCC for best fit',save_name=path+save_plot+'HRCC.pdf')
                # mean ESM and ES2M
                predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.uniform,prd.ICPDF,1.0-(1.0/(line_labels)),prd.z_extractor,prd.optimization,line_labels)
                d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_meanESMES2M.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

                # # ESM non integral
                # predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.uniform,prd.ICPDF,1.0-(1.0/(2*line_labels)),prd.z_extractor,prd.optimization,line_labels)
                # d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                # delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                # self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_ESM.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

                # # ES2M non integral
                # predicted_hrcc = prd.Hrcc_predict(delta_mu,x_var_,x,y,z,sigma_x_1,sigma_x_2,line_labels,prd.uniform,prd.ICPDF,1.0-(3.0/(2*line_labels)),prd.z_extractor,prd.optimization,line_labels)
                # d = np.round(abs(predicted_hrcc[0][1]-HRCC[0][1])/ np.sqrt(HRCC[0][0]**2 +1),decimals=2)
                # delta_slope = np.round(abs(predicted_hrcc[0][0]-HRCC[0][0]),decimals=2)
                # self.graphicPlot(a= y,b=x,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD_ES2M.pdf',cbar_loc=cbar_orien,z_var=z,z_max_fit = HRCC[0],z_max_fit_lab=HRCC[1],options_line=[predicted_hrcc[0]],line_labels=[line_labels,predicted_hrcc[1]],d=d,delta_slope=delta_slope)

        elif plot_type == 'line':
            z = z + z1
            self.linePlot(x = x,y = z,z = y,x_name=x_var_,y_name=z_var_,z_name = num_of_opts,title="Number of options",save_name=path + save_plot+".pdf")

        elif plot_type == 'bar':
            self.barPlot(quorum,opt_v[str(sig_m[i])],save_name[i],"maj")

    def linePlot(self,x,y,x_name,y_name,z_name,title,save_name):
        c = ["blue","green","red","purple","brown","yellow","black","orange","pink"]
        line_style = ["-","--",":","-."]
        fig = plt.figure(figsize=(15, 8), dpi= 90, facecolor='w', edgecolor='k')
        plt.style.use("ggplot")
        for i in range(len(z_name)):
            plt.plot(x,[y[s] for s in range(i*len(x),(i+1)*len(x),1)],c = c[i],linewidth = 1,linestyle=line_style[i%len(line_style)])

        plt.ylim(top = max(y)+(max(y)-min(y))/10,bottom = min(y)-(max(y)-min(y))/10)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend(z_name,markerscale = 3, title = title)
        plt.savefig(save_name,format = "pdf")
        plt.show()

    def graphicPlot(self,a,b,x_name,y_name,z_name,title,save_name,cbar_loc,z_var,z_max_fit=None,z_max_fit_lab = None,options_line=None,line_labels=None,d=None,delta_slope=None):
        points = []
        fig, ax = plt.subplots()
        z = np.array(z_var).reshape(len(a),len(b))
        cs = ax.pcolormesh(b,a,z,shading='auto')
        colors = ["black","brown"]
        if isinstance(options_line, type(None)) == False:
            for j in range(len(options_line)):
                ESM = [options_line[j][0]*bb+options_line[j][1] for bb in b]
                plt.plot(b,ESM,color = colors[j],linestyle='-',label = str(line_labels[0]) + ' options, HRCC = '+str(line_labels[1])+', d = '+str(d)+', $\Delta m$ = '+str(delta_slope),linewidth=0.8)
        if isinstance(z_max_fit, type(None)) == False:
            z_best_fit = [z_max_fit[0]*bb+z_max_fit[1] for bb in b]
            plt.plot(b,z_best_fit,color = 'darkgreen',label = 'HRCC = '+str(z_max_fit_lab),linewidth=0.8)
        cbar = fig.colorbar(cs,orientation=cbar_loc)
        cbar.set_label(z_name,fontsize=14)
        cbar.set_ticks(np.arange(0,1,0.1))
        cbar.minorticks_on()
        cs.set_clim(0,1)
        ax.set_aspect('equal', 'box')
        plt.xlim(min(b),max(b))
        plt.ylim(min(a),max(a))
        plt.xlabel(x_name,fontsize = 14)
        plt.ylabel(y_name,fontsize = 14)
        if 'mu' in x_name:
            plt.legend(title = 'HRCC Prediction and Best fit HRCC',loc='upper left')
        plt.title(title)
        plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
        
        def onclick(event,points = points):
            color = 'red'
            lable = str(len(points)+1)
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = ax.plot([b[0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 0.5)
                point_line_2 = ax.plot([round(event.xdata,1),round(event.xdata,1)],[a[0],round(event.ydata,1)],color=color,linewidth = 0.5)
                point_lable = ax.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14)
                verti=z[np.argmin(abs(b-round(event.ydata,1))),np.argmin(abs(a-round(event.xdata,1)))]
                z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=0.5)
                z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=8)
                points.append([point_line_1,point_line_2,point_lable,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4:
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            
            plt.savefig(save_name,format = "pdf")
            return points

        point = fig.canvas.mpl_connect('button_press_event', onclick)
        pushbullet_message('Python Code','Pick the point! ')
        plt.show()
        
        
    def barPlot(self,quor,opt_v,save_name,correct):
        fig, ax = plt.subplots()
        ax.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
        ax.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
        ax.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),bottom=list(map(itemgetter("success_rate"), opt_v)),width=1,color = "orange",edgecolor='black')
        plt.plot(quor,list(map(itemgetter(correct), opt_v)),color ="red")
        plt.xlabel('Quorum size')
        plt.ylabel('Rate of choice')
        plt.savefig(save_name,format = "pdf")
        # plt.show()


####################### Do not remove   #################
prd = Prediction()


crosscheck = 0
if crosscheck == 1:
    fig = plt.figure()
    ax = fig.add_subplot(121)
    step = 0.0001
    mu = List([5,8])
    sigma = List([1,1])
    start = np.sum(mu)/len(mu) - np.sum(sigma)-5
    stop = np.sum(mu)/len(mu) + np.sum(sigma)+5
    x = np.arange(start,stop,step)
    pdfg = prd.gaussian(x,mu,sigma)
    area = np.sum(pdfg)*step
    pdfg = pdfg/area
    print(np.sum(pdfg)*step)
    plt.plot(x,pdfg)
    pdfu = prd.uniform(x,mu,sigma)
    area = np.sum(pdfu)*step
    pdfu = pdfu/area
    print(np.sum(pdfu)*step)
    plt.plot(x,pdfu)
    x_ = prd.ICPDF(0.8,mu,stop,step,x,pdfg)
    print(x_)
    print(prd.gaussian(List([x_]),mu,sigma))
    ax.fill_between(x[:np.argmin(np.abs(x-x_))],0,pdfg[:np.argmin(np.abs(x-x_))],facecolor='blue')

    x_ = prd.ICPDF(0.8,mu,stop,step,x,pdfu)
    print(x_)
    print(prd.uniform(List([x_]),mu,sigma)/area)
    ax.fill_between(x[:np.argmin(np.abs(x-x_))],0,pdfu[:np.argmin(np.abs(x-x_))],facecolor='orange')
    plt.show()

check_qualityrange = 0
if check_qualityrange == 1:
    fig = plt.figure()
    ax = fig.add_subplot(121)
    step = 0.00001
    mu_x=List([5,10])
    sigma_x=List([5,5])
    start = np.sum(mu_x)/len(mu_x) - np.sum(sigma_x)-max(sigma_x)*45
    stop = np.sum(mu_x)/len(mu_x) + np.sum(sigma_x)+max(sigma_x)*45
    dis_x = np.round(np.arange(start,stop,step),decimals=4)
    pdf =  prd.gaussian(dis_x,mu_x,sigma_x)
    area = (np.sum(pdf)*step)
    pdf = np.multiply(pdf,1/area)
    ax.plot(dis_x,pdf)
    ax1 = fig.add_subplot(122)
    number_of_options = [10]
    for nop in number_of_options:
        slices = []
        mid_slices=[]
        for i in range(1,nop,1):
            ESM = prd.ICPDF(float(i)/nop,mu_x,stop,step,dis_x,pdf)
            slices.append(np.round(ESM,decimals=3))
        print(slices)
        for i in range(2*nop-2,0,-1):
            if i%2!=0:
                mid_slices.append(np.round(prd.ICPDF(1.0-(i/(2*nop)),mu_x,stop,step,dis_x,pdf),decimals=3))

        number_of_colors = nop

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
        for i in range(len(slices)+1):
            if i!=0 and i!=len(slices):
                x1 = np.arange(slices[i-1],slices[i],0.0001)
                pdf1 =  prd.gaussian(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                ax.fill_between(x1,0,pdf1,facecolor=color[i])
            elif i==0:
                x1 = np.arange(start,slices[i],0.0001)
                pdf1 =  prd.gaussian(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                ax.fill_between(x1,0,pdf1,facecolor=color[i])
            elif i==len(slices):
                x1 = np.arange(slices[-1],stop,0.0001)
                pdf1 =  prd.gaussian(x1,mu_x,sigma_x)
                pdf1 = np.multiply(pdf1,1/area)
                ax.fill_between(x1,0,pdf1,facecolor=color[i])
        bests = []
        for i in range(100):
            ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=nop)
            best = max(options_quality)
            bests.append(best)
        slices.append(stop)
        slices.append(stop+1)

        hist, bin_edges = np.histogram(bests,slices) # make the histogram
        # # Plot the histogram heights against integers on the x axis
        ax1.bar(range(1,len(hist)+1,1),hist,width=1) 

        # # Set the ticks to the middle of the bars
        ax1.set_xticks([0.5+i for i,j in enumerate(hist)])

        # # Set the xticklabels to a string that tells us what the bin edges were
        ax1.set_xticklabels(['{}'.format(np.round(slices[i],decimals=2)) for i,j in enumerate(hist)])

        ESM = prd.ICPDF(1.0-(1/(2*nop)),mu_x,stop,step,dis_x,pdf)

        ESMi = 0
        areas = np.round(np.arange(0,1,step),decimals=4)
        for area in areas:
            inverse_P =  prd.ICPDF(area**(1/nop),mu_x,stop,step,dis_x,pdf)
            ESMi += inverse_P*step

        print(ESM)
        print(ESMi)
        ax2 = fig.add_subplot(133)
        plt.axvline(ESM,0,500,color='orange',label = 'Non-integral')
        plt.axvline(ESMi,0,500,color='red',label = 'Integral')
        plt.legend()
    plt.show()

distributions = 0
if distributions == 1:
    step = 0.0001
    var_mu = List([i for i in range(10)])
    var_sigma = List([1+i for i in range(10)])
    fig,axs = plt.subplots(len(var_mu),2*len(var_sigma))
    for j in range(len(var_mu)):
        delta_mu = 5 +j
        for k in range(len(var_sigma)):
            delta_sig = 0+k
            mu = List([var_mu[j],var_mu[j]+delta_mu])
            sigma = List([var_sigma[k],var_sigma[k]+delta_sig])
            start = np.sum(mu)/len(mu) - np.sum(sigma)-5
            stop = np.sum(mu)/len(mu) + np.sum(sigma)+5
            x = np.arange(start,stop,step)


            pdfg = prd.uniform(x,mu,sigma)
            area = np.sum(pdfg)*step
            pdfg = pdfg/area
            axs[j,k].plot(x,pdfg)


    plt.show()

#########################################################

distributions1 = 0
if distributions1 == 1:
    step = 0.0001
    var_mu = [10]
    number_of_options = [2,5,10]
    var_sigma = List([1+i*2 for i in range(0,2)])
    fig,axs = plt.subplots(len(number_of_options),2*len(var_sigma))
    dist = prd.uniform
    for j in range(len(var_mu)):
        delta_mu = 5
        for k in range(len(var_sigma)):
            mu_x = List([var_mu[j],var_mu[j]+delta_mu])
            sigma_x = List([var_sigma[k],var_sigma[k]])
            low = List([mu_x[0]-np.sqrt(3)*sigma_x[0],mu_x[1]-np.sqrt(3)*sigma_x[1]])
            high = List([mu_x[0]+np.sqrt(3)*sigma_x[0],mu_x[1]+np.sqrt(3)*sigma_x[1]])
            start = np.sum(mu_x)/len(mu_x) - np.sum(sigma_x)-5
            stop = np.sum(mu_x)/len(mu_x) + np.sum(sigma_x)+5

            dis_x = np.round(np.arange(start,stop,step),decimals=4)
            pdf =  dist(dis_x,mu_x,sigma_x)
            area = (np.sum(pdf)*step)
            pdf = np.multiply(pdf,1/area)

            for nop in range(len(number_of_options)):
                axs[nop,2*k].plot(dis_x,pdf)
                slices = []
                mid_slices=[]
                for i in range(1,number_of_options[nop],1):
                    ESM = prd.ICPDF(float(i)/number_of_options[nop],mu_x,stop,step,dis_x,pdf)
                    slices.append(np.round(ESM,decimals=3))
                for i in range(1,2*number_of_options[nop],1):
                    if i%2!=0:
                        mid_slices.append(np.round(prd.ICPDF((i/(2*number_of_options[nop])),mu_x,stop,step,dis_x,pdf),decimals=1))

                number_of_colors = number_of_options[nop]

                color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]
                for i in range(len(slices)+1):
                    if i!=0 and i!=len(slices):
                        x1 = np.arange(slices[i-1],slices[i],0.0001)
                        pdf1 =  dist(x1,mu_x,sigma_x)
                        pdf1 = np.multiply(pdf1,1/area)
                        axs[nop,2*k].fill_between(x1,0,pdf1,facecolor=color[i])
                    elif i==0:
                        x1 = np.arange(start,slices[i],0.0001)
                        pdf1 =  dist(x1,mu_x,sigma_x)
                        pdf1 = np.multiply(pdf1,1/area)
                        axs[nop,2*k].fill_between(x1,0,pdf1,facecolor=color[i])
                    elif i==len(slices):
                        x1 = np.arange(slices[-1],stop,0.0001)
                        pdf1 =  dist(x1,mu_x,sigma_x)
                        pdf1 = np.multiply(pdf1,1/area)
                        axs[nop,2*k].fill_between(x1,0,pdf1,facecolor=color[i])
                bests = []
                for i in range(1000):
                    ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
                    available_opt = np.array(np.where(np.array(options_quality) == max(options_quality)))[0]
                    opt_choosen = np.random.randint(0,len(available_opt))
                    best = options_quality[available_opt[opt_choosen]]
                    bests.append(best)
                slices = [start]+slices+[stop]
                bins = np.zeros(number_of_options[nop])
                for b in bests:
                    for s in range(len(slices)-1):
                        if b>slices[s] and b< slices[s+1]:
                            bins[s] += 1
                            break
                # # Plot the histogram heights against integers on the x axis
                print([slices,mid_slices])
                axs[nop,2*k+1].bar(mid_slices,bins,width=0.2)


    plt.show()



    