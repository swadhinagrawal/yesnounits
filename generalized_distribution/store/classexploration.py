# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest as pt
import pandas as pd
import os
import time
import random_number_generator as rng
from numba import  njit


path = os.getcwd() + "/results/"


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
    @njit(parallel = True)
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


class workFlow:
    def __init__(self):
        self.check_count = 0

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

        pc = distribution_m(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)

        units_distribution = []
        
        for i in pc:
            units_distribution.append(distribution_h(m_units=int(i),mu_h=mu_h,sigma_h=sigma_h))
        
        ref,qc = rng.quality(distribution = distribution_x,number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc,\
            assigned_units=units_distribution,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=ref,quorum=quorum,pc=pc)

        return dec
    
    @staticmethod
    @njit
    def gaussian(x,mu,sigma):
        f = 0
        for i in range(len(mu)):
            k = 1/np.sqrt(2*np.pi*sigma[i])
            f += k*np.exp(-((x-mu[i])**2)/(2*sigma[i]**2))
        return f

    @staticmethod
    @njit
    def finding_gaussian_base(area,mu,sigma,start,stop,step,x,pdf):
        if area<=0.25:
            dummy_area =0.25
            x_ = mu[0]
        elif area>0.25 and area<=0.5:
            dummy_area =0.5
            x_ = (mu[0]+mu[1])/2
        elif area>0.5 and area<=0.75:
            dummy_area =0.75
            x_ = mu[1]
        elif area>0.75 and area<=1:
            dummy_area =1
            x_ = stop
        count = np.where(x==round(x_,6))[0][0]
        while abs(dummy_area-area)>0.000001:
            dummy_area -= pdf[count]*step
            x_ -= step
            count -= 1
        return x_

class Visualization:
    def data_visualize(self,file_name,save_plot,x_var_,y_var_,z_var_,plot_type,cbar_orien=None,line_labels=None,sigma_x_1=None,\
        data =None,num_of_opts=None,delta_mu=None,sigma_x_2 = None,z1_var_=None):
        if data == None:
            op = pd.read_csv(path+file_name)
            opt_var = []

            for j in range(len(op[x_var_])):
                a = {}
                for i in op:
                    a[str(i)] = op[str(i)][j]
                opt_var.append(a)
        else:
            opt_var = data
        
        x = []
        y = []
        z = []  # Make sure that it is in ordered form as y variable (i.e. 1st column of data file)
        z1 = []
        for i in opt_var:
            if i[x_var_] not in x:
                x.append(i[x_var_])
            if i[y_var_] not in y:
                y.append(i[y_var_])
            z.append(i[z_var_])
            if z1_var_ != None:
                z1.append(i[z1_var_])

            print(np.round(len(z)/len(opt_var),decimals=2),end="\r")
        print(np.round(len(z)/len(opt_var),decimals=2))

        
        if plot_type == 'graphics':
            wf = workFlow()
            x_1 = []
            x_2 = []
            for i in x:
                # t1 = time.time()
                if x_var_ == '$\mu_{x_1}$':
                    mu = [i,i+delta_mu]
                else:
                    mu = [i-delta_mu,i]
                sigma = [sigma_x_1,sigma_x_2]
                start = mu[0] - sigma[0]-5
                stop = mu[-1] + sigma[1]+5
                step = 0.0001
                dis_x = np.round(np.arange(start,stop,step),decimals=6)
                pdf =  [wf.gaussian(i,mu,sigma) for i in dis_x]
                pdf = np.multiply(pdf,1/(np.sum(pdf)*step))
                
                _1 = wf.finding_gaussian_base(1.0-(1.0/(line_labels[0])),mu,sigma,start,stop,step,dis_x,pdf)
                _2 = wf.finding_gaussian_base(1.0-(1.0/(line_labels[1])),mu,sigma,start,stop,step,dis_x,pdf)
                x_1.append(_1)
                x_2.append(_2)
                print(np.round(len(x_1)/len(x),decimals=2),end="\r")
                # t2 = time.time()
                # print(t2-t1)

            self.graphicPlot(a= y,b=x,array= opt_var,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z,options_line=[x_1,x_2],line_labels=line_labels)
            # delta_area = 0.01
            # areas = np.round(np.arange(0,1,delta_area),decimals=2)
            # expected_highest_quality = []
            # for i in x:
            #     integral = 0
            #     mu = [i-delta_mu,i]
            #     sigma = [sigma_x_1,sigma_x_2]
            #     start = mu[0] - sigma[0]-5
            #     stop = mu[-1] + sigma[1]+5
            #     step = 0.0001
            #     dis_x = np.round(np.arange(start,stop,step),decimals=6)
            #     pdf =  [wf.gaussian(i,mu,sigma) for i in dis_x]
            #     pdf = np.multiply(pdf,1/(np.sum(pdf)*step))
            #     for area in areas:
            #         # print(area)
            #         inverse_P =  wf.finding_gaussian_base(area,mu,sigma,start,stop,step,dis_x,pdf)
            #         # print(np.round(inverse_P,decimals=2))
            #         integral += np.round(inverse_P,decimals=2)*delta_area
            #         integral = np.round(integral,decimals=2)
            #         print(integral)
            #     expected_highest_quality.append(integral)
                
            # self.graphicPlot(a= y,b=x,array= opt_var,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name=z_var_,title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z,options_line=[expected_highest_quality],line_labels=['expected highest quality'])

        elif plot_type == 'line':
            z = z + z1
            self.linePlot(x = x,y = z,z = y,x_name=x_var_,y_name=z_var_,z_name = num_of_opts,title="Number of options",save_name=path + save_plot+".pdf")

        elif plot_type == 'bar':
            self.barPlot(quorum,opt_v[str(sig_m[i])],save_name[i],"maj")

    def linePlot(self,x,y,z,x_name,y_name,z_name,title,save_name):
        c = ["blue","green","red","purple","brown","yellow","black","orange","pink"]
        line_style = ["-","--",":","-."]
        fig = plt.figure(figsize=(15, 8), dpi= 90, facecolor='w', edgecolor='k')
        plt.style.use("ggplot")
        for i in range(len(z_name)):
            plt.plot(x,[y[s] for s in range(i*len(x),(i+1)*len(x),1)],c = c[i],linewidth = 1,linestyle=line_style[i%len(line_style)])

        plt.ylim(top = 0.3,bottom = -0.1)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend(z_name,markerscale = 3, title = title)
        plt.savefig(save_name,format = "pdf")
        plt.show()

    # def graphicPlot(self,a,b,array,x_name,y_name,z_name,title,save_name,cbar_loc,options_line,line_labels,z_var = None):
    #     plt.legend()

    def graphicPlot(self,a,b,array,x_name,y_name,z_name,title,save_name,cbar_loc,z_var,line_labels=None,options_line=None):
        fig, ax = plt.subplots()
        z = np.array(z_var).reshape(len(a),len(b))
        cs = ax.pcolormesh(b,a,z)
        colors = ["black","blue"]
        if options_line != None:
            for j in range(len(options_line)):
                plt.plot(b,options_line[j],color = colors[j],linestyle='-.',label = str(line_labels[j]))
        cbar = fig.colorbar(cs,orientation=cbar_loc)
        cbar.set_label(z_name,fontsize=14)
        cbar. set_ticks(np.arange(min(z_var),max(z_var),(max(z_var)-min(z_var))/10))
        ax.set_aspect('equal', 'box')
        # ax.xaxis.set_ticks(np.arange(min(b),max(b),int(max(b)-min(b))/10))
        # ax.yaxis.set_ticks(np.arange(min(a),max(a),int(max(a)-min(a))/10))
        plt.xlim(min(b),max(b))
        plt.ylim(min(a),max(a))
        plt.xlabel(x_name,fontsize = 14)
        plt.ylabel(y_name,fontsize = 14)
        plt.title(title)
        plt.legend()
        plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            plt.plot([b[0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color= 'red',linewidth = 0.5)
            plt.plot([round(event.xdata,1),round(event.xdata,1)],[a[0],round(event.ydata,1)],color= 'red',linewidth = 0.5)
            ax.text(int(event.xdata+1), int(event.ydata+1), "(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14)
            plt.savefig(save_name,format = "pdf")
        # plt.savefig(save_name,format = "pdf")
        point = fig.canvas.mpl_connect('button_press_event', onclick)
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
