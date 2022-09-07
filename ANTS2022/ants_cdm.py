#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Response threshold distributions to improve best-of-n decisions in minimalistic robot swarms

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import pandas as pd
import os
from numba.typed import List
import itertools
from numba import  njit
from numba.typed import List
# Daemonic parallel processing
from multiprocessing import Pool
# Ray cluster (Comment Daemonic parallel processing Pool function and uncomment ray multi processing)
# import ray
# from ray.util.multiprocessing import Pool
# Importing custom libraries
import random_number_generator as rng       #   Custum Random number generator
import ants_analysis as analizer            #   Plotters and post processing data

# Initializing the cluster
# ray.init(address='auto', _redis_password='5241590000000000')

# Global initializers and objects

path = os.getcwd() + "/results/"        # Path to save results

NqNh_mu = 1                                # Set this 1 for running the case of NqNh_mu_h_vs_mu_q
UqNh_mu = 0                                # Set this 1 for running the case of UqNh_mu_h_vs_mu_q
KqNh_mu = 0                                # Set this 1 for running the case of KqNh_mu_h_vs_mu_q
NqNh_sigma = 0                             # Set this 1 for running the case of NqNh_sigma_h_vs_sigma_q
UqNh_sigma = 0                             # Set this 1 for running the case of UqNh_sigma_h_vs_sigma_q
KqNh_sigma = 0                             # Set this 1 for running the case of KqNh_sigma_h_vs_sigma_q

#####################   START READING THIS CODE FROM LINE 247   ############################

# Classes

#   Class for counting "yes" votes received by each option
class Decision_making:
    def __init__(self,number_of_options):
        self.number_of_options = number_of_options
        self.votes = None                               #   Stores count of yes votes for each options

    def vote_counter(self,assigned_units,list_quality):
        self.votes = self.counter(self.number_of_options,assigned_units,list_quality)
    
    @staticmethod
    @njit
    def counter(nop,ass_u,list_quality):
        '''
        Inputs: number of options, assigned number of agents to each options(array of size 'n'), set of qualities of all the options
        Output: List of counts of 'yes' votes for each option as an array 
        '''
        votes = np.zeros(nop)
        for i in range(nop):
            for j in range(len(ass_u[i])):  #   For each agent, this checks if option quality to which this agent is assigned if higher than the agents response threshold
                if (ass_u[i][j] < list_quality[i]):
                    votes[i] += 1           #   If option quality is higher than the response threshold, count is incremented by +1
        return votes

    def verification(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple options having the same number of votes
        """
        if np.sum(self.votes)==0:       #   If all the robots reject all the options, the run is considered unsuccessful
            return 0
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]  #   Finds the index of option with largest acceptance
        opt_choosen = np.random.randint(0,len(available_opt))   #   In case of multiple option with same number of acceptance, one is choosen at random
        if available_opt[opt_choosen] ==  ref_highest_quality:  #   If collectively found option matches the actual highest quality option, the run is considered as successful
            return 1
        else:
            return 0

class workFlow:
    def voting_process(self,number_of_options,list_quality,assigned_units,ref_highest_quality):
        '''
        Inputs: number of options, list of option qualities, list of assigned agents ,reference best option
        '''
        DM = Decision_making(number_of_options=number_of_options)       # Instance of decision making class is initialized
        DM.vote_counter(assigned_units,list_quality)                              # Individual reponses are taken and votes are counted 
        majority_dec = DM.verification(ref_highest_quality)             # Success of decision making is decided in Boolean
        return majority_dec, DM.votes

    def collective_decision_making(self,distribution_m,distribution_h,distribution_x,mu_m,sigma_m,mu_h,sigma_h,mu_x,sigma_x,number_of_options=None,h_type=3,x_type=3):
        '''
        Inputs: Dm, Dh, Dx (all three functions), mean of Dm, variance of Dm, mean of Dh, variance of Dh, mean of Dx, variance of Dx, number of options, decimal places for thresholds and qualities
        Output: Success of the decision making as Boolean
        '''
        list_m = List(distribution_m(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m))    #   List of number of agents assigned to each option

        units_distribution = []
        
        for i in list_m:
            units_distribution.append(List(distribution_h(m_units=int(i),mu_h=mu_h,sigma_h=sigma_h)))   #   Array of size nxm_i where each row consists of response thresholds of m_i robots assigned to that i'th option 
        
        ref,list_quality = rng.quality(distribution = distribution_x,number_of_options=number_of_options,\
            x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)   #   Generates list of qualities for n options and the reference best quality

        dec,votes = self.voting_process(number_of_options=number_of_options,list_quality = list_quality,\
            assigned_units=List(units_distribution),ref_highest_quality=ref)    #   Decision making is called

        return dec, votes

# Utility Functions

def save_data(save_string,continuation=0):
    check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))   #   Get a list of counters in the folder where data file needs to be created
    count = 0
    for i in check:
        if count==i:
            count+=1        #   Set count to next integer of what is the last integer count present in results folder
    
    if continuation==0: #   If its a new experiment
        save_string = str(count)+save_string    #   Generate the new save string
        f1 = open(path+str(count),'w+') #   Create a file with counter as name
        return save_string,f1   #   Return the data file name
    else:
        check = os.listdir(path)   #   Get a list of counters in the folder where data file needs to be created
        save_string = check[-1][:-4]  #   Otherwise return the name of last file created in the folder
        f1 = None
        return save_string,f1

def looper(**kwargs):
    '''
    Inputs: One complete parameters set
    Outputs: Average rate of success
    '''
    for key,value in kwargs.items():
        if key == "n":
            n = value
        elif key == "muQ1":
            muQ1 = value
        elif key == "muQ2":
            muQ2 = value
        elif key == "sigmaQ1":
            sigmaQ1 = value
        elif key == "sigmaQ2":
            sigmaQ2 = value
        elif key == "Dq":
            Dq = value
        elif key == "muM1":
            muM1 = value
        elif key == "muM2":
            muM2 = value
        elif key == "sigmaM1":
            sigmaM1 = value
        elif key == "sigmaM2":
            sigmaM2 = value
        elif key == "Dm":
            Dm = value
        elif key == "muH1":
            muH1 = value
        elif key == "muH2":
            muH2 = value
        elif key == "sigmaH1":
            sigmaH1 = value
        elif key == "sigmaH2":
            sigmaH2 = value
        elif key == "Dh":
            Dh = value
        elif key == "runs":
            runs = value
        elif key == "vars":
            variables = value
    
    if Dq == 'U':
        low_q_1 = -np.sqrt(3)*sigmaQ1                         #   Lower bound of distribution from which quality stimulus are sampled randomly
        high_q_1 = np.sqrt(3)*sigmaQ1                         #   Upper bound of distribution from which quality stimulus are sampled randomly
        Dq = rng.dx_u
        sigmaQ1 = muQ1 + high_q_1
        sigmaQ2 = muQ1 + high_q_1
        muQ1 += low_q_1
        muQ2 += low_q_1
        
    else:
        Dq = rng.dx_n     
    if Dm == 'U':
        Dm = rng.units_u
    else:
        Dm = rng.units_n
    if Dh == 'U':
        Dh = rng.threshold_u
    else:
        Dh = rng.threshold_n

    count = 0
    yes_votes = []
    for k in range(runs):
        success, votes = wf.collective_decision_making(distribution_m=Dm,distribution_x=Dq,distribution_h=Dh,\
            mu_h=List([muH1,muH2]),sigma_h=List([sigmaH1,sigmaH2]),mu_x=List([muQ1,muQ2]),sigma_x=List([sigmaQ1,sigmaQ2]),number_of_options=n,\
            mu_m=List([muM1,muM2]),sigma_m=List([sigmaM1,sigmaM2]))
        if success == 1:
            count += 1
        yes_votes.append(votes)

    #   Inserting data into the data file
    output = {}
    for key,value in kwargs.items():
        if key in variables:
            output[key] = value
    output["ARS"] = count/runs

    for i in range(runs):
        for j in range(n):
            output[str((i,j))] = yes_votes[i][j]

    return output

def wrapper_kwargs(fn,kwargs):
    #   Wraper function for keyword based function input in parallel processing
    return fn(**kwargs)

def parallelFeeder(columns_name,kwargs_iter,f_path,batch_size):
    #   Feeder for parallel processing the parameters set
    outputSet = []
    progress = 0
    for i in range(0,len(kwargs_iter),batch_size):  
        with Pool(20) as p:#,ray_address="auto") as p:
            args_for_starmap = zip(itertools.repeat(looper), kwargs_iter[i:i+batch_size])
            outputSet = p.starmap(wrapper_kwargs, args_for_starmap)

        cols = list(outputSet[0].keys())[5:]
        f = open(f_path[:-4]+'_'+str(outputSet[0]['muM1'])+'_'+str(outputSet[0]['n'])+'.csv','a')    #   Otherwise creat a new file with specified name
        columns = pd.DataFrame(data=np.array([columns_name+cols]))   #   Add column names
        columns.to_csv(f_path[:-4]+'_'+str(outputSet[0]['muM1'])+'_'+str(outputSet[0]['n'])+'.csv',mode='a',header=False,index=False)
        out = pd.DataFrame(data=outputSet,columns=columns_name+cols)
        out.to_csv(f_path[:-4]+'_'+str(outputSet[0]['muM1'])+'_'+str(outputSet[0]['n'])+'.csv',mode = 'a',header = False, index=False)
        progress +=1
        print("\r Percent of input processed : {}%".format(np.round(100*progress*batch_size/len(kwargs_iter)),decimals=1), end="")

def usemuHPredicted(muQ1,muQ2,sigmaQ1,sigmaQ2,n,DQ):
    '''
    Inputs: mean quality peak 1, mean quality peak 2, standard deviation of quality distribution (list), number of options, PDF function
    Outputs: Predicted mean for response threshold distribution (list)
    '''
    if DQ == 'U':
        dist = analizer.Prediction.uniform
    else:
        dist = analizer.Prediction.gaussian

    step = 0.0001
    muH1 = None
    sigma_q = List([sigmaQ1,sigmaQ2])
    mu_q = List([muQ1,muQ2])
    pdf_q,area,dis_q,start1,stop1 = analizer.PDF(dist,mu_q,sigma_q) #   Generating the PDF for quality distribution using mean and standard deviation
    mean_esmes2m = analizer.Prediction.ICPDF(1-(1/int(n)),mu_q,step,dis_q,pdf_q)   #   Calculating the predicted mean from ICPDF
    muH1 = mean_esmes2m
    muH2 = muH1
    return muH1,muH2

def paramFeeder(continuation,f_path,var1,var2,num_opts,muM):   # var1 = quality, var2 = threshold
    '''
    Inputs: continuation boolean, data file name to be created, columns name of parameters that needs to be stored, var1, var2, number of options, mean swarm size
    Outputs: Data file saving name, parameter file created
    '''
    if continuation:                    #   If its not a new experiment, and is a continuation of stopped experiment
        f1 = pd.read_csv(f_path)        #   Read the last created file
        lastData = f1.iloc[-1]       #   Get the data from last row
        mi = np.where(muM == lastData['muM1'])[0][0] #   Last sub-swarm size that was under process
        ni = np.where(num_opts == lastData['n'])[0][0]    #   Last number of options that was under process

        inputs = []

        for n in num_opts[ni+1:]:
            for i in var1:
                for j in var2:
                    inputs.append((muM[mi],n,i,j))    #   Create tuples of left-out cases 
        inputs += list(itertools.product(muM[mi+1:],num_opts,var1,var2))    #   Create tuples of leftout cases
    else:
        inputs = list(itertools.product(muM,num_opts,var1,var2))    #   Create tuples using set product of all the varying set of variables
    return inputs

def mu_h_vs_mu_q(var1,var2,sigmaQ1,sigmaQ2,delta_muQ,file_name_string,DQ,predicted_sigmaH=0,continuation=0):
    columns_name = ["muM1","n","muH1","muQ1","ARS"] #   Name of columns where data needs to be stored
    #   List of column names for variables that needs to be stored, check line 295 for variable names format. "ARS" stands for
    #  average rate of success
    # mu_m = [10,50,100,200,500]  #   List of mean sub-swarm size cases that needs to be explored
    mu_m = [50]  #   List of mean sub-swarm size cases that needs to be explored
    # num_opts = [2,5,8,10,15,20,30,40,80,100]    #   List of number of options cases that needs to be explored
    num_opts = [5]    #   List of number of options cases that needs to be explored
    sigma_m_1 = 0         #   Fixed variance of sub-swarm sizes
    sigma_m_2 = 0         #   Fixed variance of 2nd peak of sub-swarm sizes, generalized for bimodal case
    sigma_q_1 = sigmaQ1       #   Variance of response threshold distribution (Fixed)
    sigma_q_2 = sigmaQ2       #   Variance of response threshold distribution (Fixed), generalized for bimodal case
    sigma_h_1 = sigma_q_1 #   Variance of option quality distribution (Fixed)
    sigma_h_2 = sigma_q_1 #   Variance of option quality distribution (Fixed), generalized for bimodal case

    runs = 200          #   Number of runs for each set of parameters
    batch_size = len(var2)*len(var1)    #   Batch size for multi-processing
    save_string = file_name_string        #   Data file name string
    save_string,param = save_data(save_string,continuation) #   Adding counter to data file name for multiple runs and in case of continuing previous runs
    f_path = path+save_string+'.csv'        #   Final data file name to be processed
    if isinstance(param,type(None))==False: #   Writting fixed parameter values to count of data files name prefix
        param.write("sigma_q_1 : "+str(sigma_q_1)+"\n")
        param.write("sigma_q_2 : "+str(sigma_q_2)+"\n")
        if not predicted_sigmaH:
            param.write("sigma_h_1 : "+str(sigma_h_1)+"\n")
            param.write("sigma_h_2 : "+str(sigma_h_2)+"\n")
        param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
        param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
        param.write("delta_muQ : "+str(delta_muQ)+"\n")
    inputs = paramFeeder(continuation,f_path,var1,var2,num_opts,mu_m)  #   Creating list of tuples of varying parameters using set product method
    kwargs_iter = []
    for i in inputs:
        if predicted_sigmaH:
            if DQ == 'K':
                sigma_h_1 = [(0.05*np.log10(i[1])+0.58)*sigma_q_1]
                sigma_h_2 = sigma_h_1
            elif DQ == 'U':
                sigma_h_1 = [(-0.07*np.log10(i[1])+0.67)*sigma_q_1]
                sigma_h_2 = sigma_h_1
            elif DQ == 'N':
                sigma_h_1 = [(0.07*np.log10(i[1])+0.57)*sigma_q_1]
                sigma_h_2 = sigma_h_1
        #   Creating list of dictionaries to pass keyword based parameters into the parallel processing function through another function wrapper
        kwargs_iter.append(dict(n = i[1], muQ1 = i[2], muQ2 = i[2] + delta_muQ, sigmaQ1 = sigma_q_1, sigmaQ2 = sigma_q_2,\
            Dq = DQ, muM1 = i[0], muM2 = i[0], sigmaM1 = sigma_m_1, sigmaM2 = sigma_m_2, Dm = 'N', muH1 = i[3], muH2 = i[3],\
            sigmaH1 = sigma_h_1, sigmaH2 = sigma_h_2, Dh = 'N', runs = runs, vars = columns_name[:-1]))
    new_string = ''
    counter = 0
    for i in save_string:
        if i == '_':
            counter += 1
        if counter != 2:
            new_string += i
    f_path = path+new_string+'.csv'
    parallelFeeder(columns_name,kwargs_iter,f_path,batch_size)  #   Parallelly processing the parameter sets 

def sigma_h_vs_sigma_q(var1,var2,muQ1,muQ2,delta_muQ,file_name_string,DQ,predicted_muH=0,predicted_sigmaH=0,continuation=0):
    columns_name = ["muM1","n","sigmaH1","sigmaQ1","ARS"] #   Name of columns where data needs to be stored
    #   List of column names for variables that needs to be stored, check line 295 for variable names format. "ARS" stands for
    #  average rate of success
    mu_m = [10,50,100,200,500]  #   List of mean sub-swarm size cases that needs to be explored
    num_opts = [2,5,8,10,15,20,30,40,80,100]    #   List of number of options cases that needs to be explored
    sigma_m_1 = 0         #   Fixed variance of sub-swarm sizes
    sigma_m_2 = 0         #   Fixed variance of 2nd peak of sub-swarm sizes, generalized for bimodal case
    mu_q_1 = muQ1       #   Variance of response threshold distribution (Fixed)
    mu_q_2 = muQ2 + delta_muQ      #   Variance of response threshold distribution (Fixed), generalized for bimodal case
    mu_h_1 = (mu_q_1 + mu_q_2)/2    #   Variance of option quality distribution (Fixed)
    mu_h_2 = (mu_q_1 + mu_q_2)/2 #   Variance of option quality distribution (Fixed), generalized for bimodal case
    runs = 200          #   Number of runs for each set of parameters
    batch_size = len(var2)*len(var1)    #   Batch size for multi-processing
    save_string = file_name_string        #   Data file name string
    save_string,param = save_data(save_string,continuation) #   Adding counter to data file name for multiple runs and in case of continuing previous runs
    f_path = path+save_string+'.csv'        #   Final data file name to be processed
    if isinstance(param,type(None))==False: #   Writting fixed parameter values to count of data files name prefix
        param.write("mu_q_1 : "+str(mu_q_1)+"\n")
        param.write("mu_q_2 : "+str(mu_q_2)+"\n")
        if not predicted_muH:
            param.write("mu_h_1 : "+str(mu_h_1)+"\n")
            param.write("mu_h_2 : "+str(mu_h_2)+"\n")
        param.write("sigma_m_1 : "+str(sigma_m_1)+"\n")
        param.write("sigma_m_2 : "+str(sigma_m_2)+"\n")
        param.write("delta_muQ : "+str(delta_muQ)+"\n")
    
    
    inputs = paramFeeder(continuation,f_path,var1,var2,num_opts,mu_m)  #   Creating list of tuples of varying parameters using set product method
    kwargs_iter = []
    for i in inputs:
        sig_h = i[3]
        if predicted_sigmaH:
            if DQ == 'K':
                sig_h = (0.05*np.log10(i[1])+0.58)*i[2]
            elif DQ == 'U':
                sig_h = (-0.07*np.log10(i[1])+0.67)*i[2]
            elif DQ == 'N':
                sig_h = (0.07*np.log10(i[1])+0.57)*i[2]
        if predicted_muH:
            mu_h_1,mu_h_2 = usemuHPredicted(mu_q_1,mu_q_2,i[2],i[2],i[1],DQ)
        #   Creating list of dictionaries to pass keyword based parameters into the parallel processing function through another function wrapper
        kwargs_iter.append(dict(n = i[1], muQ1 = mu_q_1, muQ2 = mu_q_2, sigmaQ1 = i[2], sigmaQ2 = i[2],\
            Dq = DQ, muM1 = i[0], muM2 = i[0], sigmaM1 = sigma_m_1, sigmaM2 = sigma_m_2, Dm = 'N', muH1 = mu_h_1, muH2 = mu_h_2,\
            sigmaH1 = sig_h, sigmaH2 = sig_h, Dh = 'N', runs = runs, vars = columns_name[:-1]))
    if continuation:
        new_string = ''
        counter = 0
        for i in save_string:
            if i == '_':
                counter += 1
            if counter != 2:
                new_string += i
        f_path = path+new_string+'.csv'
    parallelFeeder(columns_name,kwargs_iter,f_path,batch_size)  #   Parallelly processing the parameter sets 

if __name__=='__main__':
    wf = workFlow()
    
    if NqNh_mu:
        continuation = 0            #   If running this code after stopping it in between, set this to 1
        mu_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
        mu_h_vs_mu_q(mu_q,mu_h,sigmaQ1=1,sigmaQ2=1,delta_muQ=0,file_name_string="NQNh_mu",DQ='N',continuation=continuation)
        
    if UqNh_mu:
        continuation = 0            #   If running this code after stopping it in between, set this to 1
        mu_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
        mu_h_vs_mu_q(mu_q,mu_h,sigmaQ1=1,sigmaQ2=1,delta_muQ=0,file_name_string="UQNh_mu",DQ='U',continuation=continuation)
        
    if KqNh_mu:
        continuation = 0            #   If running this code after stopping it in between, set this to 1
        mu_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
        mu_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
        mu_h_vs_mu_q(mu_q,mu_h,sigmaQ1=1,sigmaQ2=1,delta_muQ=5,file_name_string="KQNh_mu",DQ='K',continuation=continuation)

    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="NQNh_sigma",DQ='N',continuation=continuation)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="UQNh_sigma",DQ='U',continuation=continuation)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [np.round(i*0.1,decimals=1) for i in range(151)] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=10,delta_muQ=5,file_name_string="KQNh_sigma",DQ='K',continuation=continuation)

    # # CASE 1
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="NQNh_naive",DQ='N',continuation=continuation)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="UQNh_naive",DQ='U',continuation=continuation)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=10,delta_muQ=5,file_name_string="KQNh_naive",DQ='K',continuation=continuation)

    # # CASE 2
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="NQNh_pred",DQ='N',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="UQNh_pred",DQ='U',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=10,delta_muQ=5,file_name_string="KQNh_pred_",DQ='K',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)

    # # CASE 3
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="NQNh_mu_pred",DQ='N',continuation=continuation,predicted_muH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="UQNh_mu_pred",DQ='U',continuation=continuation,predicted_muH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=10,delta_muQ=5,file_name_string="KQNh_mu_pred",DQ='K',continuation=continuation,predicted_muH=1)

    # # CASE 4
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="NQNh_sigma_pred",DQ='N',continuation=continuation,predicted_sigmaH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=7.5,muQ2=7.5,delta_muQ=0,file_name_string="UQNh_sigma_pred",DQ='U',continuation=continuation,predicted_sigmaH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [10] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [10] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=10,delta_muQ=5,file_name_string="KQNh_sigma_pred",DQ='K',continuation=continuation,predicted_sigmaH=1)

    # # CASE 1
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="NQNh_naive",DQ='N',continuation=continuation)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="UQNh_naive",DQ='U',continuation=continuation)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=2.5,muQ2=7.5,delta_muQ=5,file_name_string="KQNh_naive",DQ='K',continuation=continuation)

    # # CASE 2
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="NQNh_pred",DQ='N',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="UQNh_pred",DQ='U',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=2.5,muQ2=7.5,delta_muQ=5,file_name_string="KQNh_pred_",DQ='K',continuation=continuation,predicted_sigmaH=1,predicted_muH=1)

    # # CASE 3
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="NQNh_mu_pred",DQ='N',continuation=continuation,predicted_muH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="UQNh_mu_pred",DQ='U',continuation=continuation,predicted_muH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=2.5,muQ2=7.5,delta_muQ=5,file_name_string="KQNh_mu_pred",DQ='K',continuation=continuation,predicted_muH=1)

    # # CASE 4
    # if NqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="NQNh_sigma_pred",DQ='N',continuation=continuation,predicted_sigmaH=1)
        
    # if UqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=5,muQ2=5,delta_muQ=0,file_name_string="UQNh_sigma_pred",DQ='U',continuation=continuation,predicted_sigmaH=1)
        
    # if KqNh_sigma:
    #     continuation = 0            #   If running this code after stopping it in between, set this to 1
    #     sigma_q = [2] #   List of mean option quality cases that needs to be explored
    #     sigma_h = [2] #   List of mean response thresholds cases that needs to be explored
    #     sigma_h_vs_sigma_q(sigma_q,sigma_h,muQ1=2.5,muQ2=7.5,delta_muQ=5,file_name_string="KQNh_sigma_pred",DQ='K',continuation=continuation,predicted_sigmaH=1)

