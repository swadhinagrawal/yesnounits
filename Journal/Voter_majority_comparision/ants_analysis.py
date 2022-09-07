#!/Users/swadhinagrawal/opt/anaconda3/envs/cdm/bin/python
# Response threshold distributions to improve best-of-n decisions in minimalistic robot swarms

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

# Importing open source libraries
import numpy as np
import pandas as pd
import os
from numba.typed import List
import matplotlib.pyplot as plt
import random
from matplotlib import rc,rcParams
import matplotlib
# matplotlib.use('Agg')
from functools import partial
import pandas as pd
from numba import  njit,jit
from math import cos, sqrt,exp
from numba import guvectorize, float64
from scipy.optimize import differential_evolution
from scipy.special import lambertw

# Importing custom libraries
import random_number_generator as rng

path = os.getcwd() + "/results/"        # Path to save results

font = {"fontsize":18,"fontweight":'bold'}
fontproperties = {'weight' : 'bold', 'size' : 18}
rc('font', weight='bold',size=18)
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def align_yaxis(ax1, v1, ax2, v2):
    '''adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1'''
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))

    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

class FittingHARS:
    @staticmethod
    def objective_fn(coeff,vars):    # coeff = [m,c]
        x = np.array(vars[0])
        z = np.array(vars[1])
        y = coeff[0]*x+coeff[1]
        z = np.array(z).reshape(len(x),len(x))
        z_extracted = []
        for i in range(len(x)):
            z_loc = np.argmin(np.abs(np.array(x)-y[i]))
            z_extracted.append(z[z_loc,i])
        
        hars = np.sum(z_extracted)/len(z_extracted)
        return -hars

    def optimization(self,x,y,z):
        bounds = bounds = [(0, 2), (0, 1)]#[(0, 2), (0, 8)]#[(0, 2), (0, 1)]#
        bestFit = differential_evolution(partial(self.objective_fn,vars=[x,z]), bounds,seed=1000)
        print([bestFit.x,np.round(-bestFit.fun,decimals=3)])
        return [bestFit.x,np.round(-bestFit.fun,decimals=3)]

class Prediction:
    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
    def gaussian(x,mu,sigma,result):
        '''
        Inputs: base, list of means, list of variance (in case of bimodal distributions) 
        Output: list of Normal PDF values
        '''
        n = x.shape[0]
        m = mu.shape[0]
        for j in range(x.shape[0]):         #   For all base values we need the PDF values
            f = 0.0
            for i in range(len(mu)):        #   We calculate the PDF value due to each modal and superimpose them to 
                #   get PDF of bimodal distribution (We can later also make it weighted by adding weights list for each peak)
                k = 1/(sqrt(2*np.pi)*sigma[i])
                f += k*exp(-((x[j]-mu[i])**2)/(2*sigma[i]**2))
            result[j] = f

    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:],float64[:])], '(n),(m),(m)->(n)')
    def uniform(x,mu,sigma,result):
        '''
        Inputs: base, list of means, list of variance (in case of bimodal distributions) 
        Output: list of Uniform PDF values
        '''
        n = x.shape[0]
        m = mu.shape[0]
        for j in range(n):      #   For all base values we need the PDF values
            f = 0.0
            for i in range(m):  #   We calculate the PDF value due to each modal and superimpose them to 
                #   get PDF of bimodal distribution (We can later also make it weighted by adding weights list for each peak)
                a = (mu[i] - np.sqrt(3)*sigma[i])   #   Lower bound of PDF
                b = (mu[i] + np.sqrt(3)*sigma[i])   #   Upper bound of PDF
                if x[j]<=b and x[j]>=a:
                    f += 1/abs(b-a)
            result[j] = f

    @staticmethod
    @njit
    def ICPDF(area,mu,step,x,pdf):
        '''
        Inputs: Area where we need base of PDF, list of mean of PDF, resolution of PDF, base of PDF, PDF function values
        Output: The base where cummulative area under the PDF is as desired
        '''
        if len(mu)>1:                           #   If the PDF is bimodal
            dummy_area = 0.5                    #   We make a dummy area variable to start at a known point closest to the area where we need the base
            x_ = (mu[0]+mu[1])/2.0              #   The base where dummy area under the PDF is true is the mean of means
        else:
            dummy_area =0.5
            x_ = mu[0]                          #   If the PDF is unimodal, the 50% area is at the mean of PDF

        count = np.argmin(np.abs(x-x_))         #   We find the index of element in our base list closest to the known base point for dummy area
        
        while abs(dummy_area-area)>0.001:       #   While the distance between dummy area and the desired area is not small enough, we keep searching the point 
            if dummy_area>area:                 #   If desired area is smaller than the dummy
                count -= 1                      #   We decrease the index count in base list
                dummy_area -= pdf[count]*step   #   Hence, decrease the area under the PDF accordingly
                x_ -= step                      #   Also, move the search point location towards left
            elif area>dummy_area:               #   Otherwise, if desired area > dummy
                count += 1                      #   We increment the index of base list
                dummy_area += pdf[count]*step   #   Hence, add area to existing dummy
                x_ += step                      #   Also, move the search point rightward
        return x_

    @staticmethod
    def HARS_prediction(delta_mu,x,ars,sigma_q,distribution_fn,ICPDF_fn,hrcc_area):
        x_1 = []
        step = 0.0001
        for i in x:
            mu = List([i,i+delta_mu])

            pdf,area,dis_x,start,stop = PDF(distribution_fn,mu,sigma_q)
            _1 = ICPDF_fn(hrcc_area,mu,step,dis_x,pdf)
            x_1.append(_1)
            print(np.round(100*len(x_1)/len(x),decimals=2),end="\r")

        hars = FittingHARS.objective_fn(coeff=None,z=ars,x=x,y=x_1)
        [slope,intercept] = np.polyfit(x,x_1,deg=1)
        bestFit = [slope,intercept]
        return [bestFit,hars]

class Visualization:
    def data_reading(self,file_name,save_plot,varXlabel,varYlabel,plot_type,cbar_orien,Dq):
        data = pd.read_csv(path+file_name)

        m = np.unique(data['muM1'])
        n = np.unique(data['n'])
        var_x = np.unique(data[varXlabel])
        var_y = np.unique(data[varYlabel])
        ars = np.array(data['ARS']).reshape((len(m),len(n),len(var_x),len(var_y)))

        fitting = FittingHARS()

        HRCC_fit = fitting.optimization(x=var_x,y=var_y,z=ars.T)
        if plot_type == 'colourmapIndividual':
            if 'mu' not in varXlabel:
                # self.HeatMap(a=var_y,b=var_x,intensity=ars.T,x_name=r'$\sigma_{q}$',y_name=r'$\sigma_{h}$',z_name='Average rate of success',title="Number_of_options = "+str(n[0]),save_name=path+save_plot+'.pdf',cbar_loc=cbar_orien,fitted_HARS = HRCC_fit)
                analysis = [HRCC_fit,var_x[0],ars[0,0,0,0]]
            else:
                if Dq == 'N' or Dq == 'K':
                    predicted_hrcc = prd.HARS_prediction(delta_mu,var_x,ars.T,List([var_x[0],var_x[0]]),prd.gaussian,prd.ICPDF,1.0-(1.0/n[0]))
                    # self.HeatMap(a=var_y,b=var_x,intensity=ars.T,x_name=r'$\mu_{q}$',y_name=r'$\mu_{h}$',z_name='Average rate of success',title="Number_of_options = "+str(n[0]),save_name=path+save_plot+'.pdf',cbar_loc=cbar_orien,fitted_HARS = HRCC_fit,predicted_HARS = predicted_hrcc)
                if Dq == 'U':
                    predicted_hrcc = prd.HARS_prediction(delta_mu,var_x,ars.T,List([var_x[0],var_x[0]]),prd.uniform,prd.ICPDF,1.0-(1.0/n[0]))
                    # self.HeatMap(a=var_y,b=var_x,intensity=ars.T,x_name=r'$\mu_{q}$',y_name=r'$\mu_{h}$',z_name='Average rate of success',title="Number_of_options = "+str(n[0]),save_name=path+save_plot+'.pdf',cbar_loc=cbar_orien,fitted_HARS = HRCC_fit,predicted_HARS = predicted_hrcc)
                analysis = [HRCC_fit,predicted_hrcc,var_x[0],ars[0,0,0,0]]

        elif plot_type == 'colourmapCombined':
            analysis = [HRCC_fit,var_x[0],ars[0,0,0,0]]
        return analysis

    def HeatMap(self,a,b,intensity,x_name,y_name,z_name,title,save_name,cbar_loc,fitted_HARS,predicted_HARS=None):
        points = []
        fig, ax = plt.subplots()
        intensity = np.array(intensity).reshape(len(a),len(b))
        cs = ax.pcolormesh(b,a,intensity,shading='auto')

        if isinstance(predicted_HARS, type(None)) == False:
            muh_star = [predicted_HARS[0][0]*bb+predicted_HARS[0][1] for bb in b]
            plt.plot(b,muh_star,color = "black",linestyle=':',label = 'HARS = '+str(predicted_HARS[1]),linewidth=4)

        fitted_HARS_line = [fitted_HARS[0][0]*bb+fitted_HARS[0][1] for bb in b]
        plt.plot(b,fitted_HARS_line,linestyle='--',color = 'red',label = 'fitted HARS = '+str(fitted_HARS[1]),linewidth=4)

        cbar = fig.colorbar(cs,orientation=cbar_loc)

        cbar.set_label(z_name,fontsize=18,fontweight='bold')

        cbar.set_ticks(np.arange(0,1.1,0.1),{"fontsize":24,"fontweight":'bold'})
        cbar.ax.tick_params(labelsize=18)
        cbar.minorticks_on()
        cs.set_clim(0,1)
        ax.set_aspect('equal', 'box')
        plt.xlim(min(b),max(b))
        plt.ylim(min(a),max(a))
        plt.xlabel(x_name,fontsize = 18)
        plt.ylabel(y_name,fontsize = 18)
        plt.xticks(fontsize=18,fontweight='bold')
        plt.yticks(fontsize=18,fontweight='bold')

        plt.title(title,font,y=-0.28,color=(0.3,0.3,0.3,1))
        plt.legend(loc='upper center', bbox_to_anchor=(0.6, 1.2),prop=dict(weight='bold',size=12),labelcolor=(0,0,0,1))
        # plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
        plt.minorticks_on()
        # plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
        
        def onclick(event,points = points):
            color = 'red'
            lable = str(len(points)+1)
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                point_line_1 = ax.plot([b[0],round(event.xdata,1)],[round(event.ydata,1),round(event.ydata,1)],color=color,linewidth = 0.5)
                point_line_2 = ax.plot([round(event.xdata,1),round(event.xdata,1)],[a[0],round(event.ydata,1)],color=color,linewidth = 0.5)
                point_lable = ax.text(int(event.xdata+1), int(event.ydata+1), lable+"(%2.1f,%2.1f)"%(event.xdata,event.ydata),fontsize=14,color='red')
                verti=intensity[np.argmin(abs(b-round(event.ydata,1))),np.argmin(abs(a-round(event.xdata,1)))]
                z_point = cbar.ax.plot([0,1],[verti,verti],color=color,linewidth=0.5)
                z_p_l = cbar.ax.text(0.4,verti+0.005,lable,fontsize=14,color='red')
                points.append([point_line_1,point_line_2,point_lable,z_point,z_p_l])
            else:
                for p in range(len(points[-1])):
                    if p!=2 and p!=4 :
                        points[-1][p][0].remove()
                    else:
                        points[-1][p].remove()
                del points[-1]
            
            plt.savefig(save_name,format = "eps",bbox_inches="tight",pad_inches=0.2)
            return points
        plt.savefig(save_name,format = "eps",bbox_inches="tight",pad_inches=0)
        point = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def HeatMap_combined(self,a,b,intensity,x_name,y_name,z_name,title,save_name,cbar_loc,fitted_HARS,ax,predicted_HARS=None):
        intensity = np.array(intensity).reshape(len(a),len(b))

        if isinstance(predicted_HARS, type(None)) == False:
            muh_star = [predicted_HARS[0][0]*bb+predicted_HARS[0][1] for bb in b]
            ax.plot(b,muh_star,color = "black",linestyle=':',label = 'HARS = '+str(predicted_HARS[1]),linewidth=4)

        fitted_HARS_line = [fitted_HARS[0][0]*bb+fitted_HARS[0][1] for bb in b]
        ax.plot(b,fitted_HARS_line,linestyle='--',color = 'red',label = 'fitted HARS = '+str(fitted_HARS[1]),linewidth=4)

def genericPlotter(ax,distribution,color,f_path,function,fit_type,first_variable,vs_n = 0,predicted = 0,plot_against_sig=0,point_facecolor='none',line_style='-.',f_path1=None,marker_line=None):
    file = pd.read_csv(f_path)
    mum = np.unique(file['$\mu_{m}$'])
    n = np.unique(file['n'])
    
    if plot_against_sig:
        sig_q = np.unique(file['$\sigma_{q}$'])
        n = sig_q
        slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
        slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    else:
        slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
        y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
        mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
        slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
        y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
        mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if not isinstance(f_path1,type(None)):
        file = pd.read_csv(f_path1)

        if plot_against_sig:
            sig_q = np.unique(file['$\sigma_{q}$'])
            n = sig_q
            slope_fit = np.array(file['slope_fit']).reshape((len(mum),len(n)))
            y_intercept_fit = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
            mu_hars_fit = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
            slope_predicted = np.array(file['slope*']).reshape((len(mum),len(n)))
            y_intercept_predicted = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
            mu_hars_predicted = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
        else:
            slope_fit1 = np.array(file['slope_fit']).reshape((len(mum),len(n)))
            y_intercept_fit1 = np.array(file['y_intercept_fit']).reshape((len(mum),len(n)))
            mu_hars_fit1 = np.array(file['fit '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n)))
            slope_predicted1 = np.array(file['slope*']).reshape((len(mum),len(n)))
            y_intercept_predicted1 = np.array(file['y_intercept*']).reshape((len(mum),len(n)))
            mu_hars_predicted1 = np.array(file['* '+ r'$\mu_{HARS}$']).reshape((len(mum),len(n))) #x,y,z,dx,dy,dz,trans,inte = read_slopes_HARS(f_path,columns_name)
    
    if function=="slope_fit":
        variable = slope_fit
    elif function == "y_intercept_fit":
        variable = y_intercept_fit
    elif function == "mu_hars_fit":
        variable = mu_hars_fit
    elif function == "slope_predicted":
        variable = slope_predicted
    elif function == "y_intercept_predicted":
        variable = y_intercept_predicted
    elif function == "mu_hars_predicted":
        variable = mu_hars_predicted
    elif function == 'distances':
        variable = abs(y_intercept_predicted - y_intercept_fit)
    elif function == 'hars_diff':
        variable = mu_hars_fit1-mu_hars_fit

    
    if predicted ==1:
        predicted_intercepts = y_intercept_predicted
        predicted_slope = slope_predicted

        if distribution==1:
            predicted_intercepts -= 2.5

        if 'intercept' in function:
            ax.plot(n,predicted_intercepts[0,:],linewidth=3,color = color)
        elif 'slope' in function:
            ax.plot(n,predicted_slope[0,:],linewidth=3,color = color)
            
    if predicted ==2:
        predicted_intercepts = y_intercept_predicted[0,:]
        if distribution==1:
            predicted_intercepts -= 2.5

        ax.plot(n[1:],predicted_intercepts[1:],linewidth=3,color = color)

    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.weight'] = 'bold'

    if first_variable == "n":
        first_var = n
        second_var = mum
    else:
        first_var = mum
        second_var = n
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    xf = np.zeros(len(second_var))
    yf = np.zeros(len(second_var))

    for i in range(len(first_var)):
        # if first_var[i]==2 or first_var[i]==10 or first_var[i]==100: 
        x1 = []
        y1 = []
        for j in range(len(second_var)):
            x1.append(second_var[j])
            # index = i+j*len(n)
            if vs_n:
                if distribution==1 and function == "y_intercept_fit" and not plot_against_sig:
                    # y1.append(variable[i,j])
                    y1.append(variable[i,j]-2.5)
                elif distribution==1 and plot_against_sig:
                    y1.append(variable[i,j]-2.5)
                else:
                    if not plot_against_sig:
                        y1.append(variable[i,j])
                    else:
                        y1.append(variable[i,j])
                # index = i*len(n)+j
            # y1.append(variable[i,j])
            else:
                if distribution==1 and function == "y_intercept_fit" and not plot_against_sig:
                    y1.append(variable[j,i]-2.5)
                    # y1.append(variable[j,i])
                else:
                    y1.append(variable[j,i])
                    # y1.append(variable[i,j])

        xf += np.array(x1)
        yf += np.array(y1)
        # ax.scatter(x1,y1,s=50,marker=marker[i], facecolors=point_facecolor, edgecolors=color)
            
    xf /= len(first_var)
    yf /= len(first_var)
    # ax.plot(xf,yf, color=color,alpha = 0.5,linewidth=6)
    # ax.plot(xf,yf, color=color,linewidth=8)
    # ax.plot(xf,yf, color=color,linewidth=8,linestyle=line_style)
    ax.plot(xf,yf, color=color,linewidth=3,linestyle=line_style,marker=marker_line)

    if fit_type == "log":
        log_fit = np.polyfit(np.log10(xf),yf,1)
        ax.plot(xf,(np.log10(xf)*log_fit[0]+log_fit[1]),linewidth=3,color = color,linestyle=line_style)
        return_var = log_fit
    elif fit_type == "linear":
        linear_fit = np.polyfit(xf,yf,1)
        ax.plot(xf,np.array(xf)*linear_fit[0] + linear_fit[1],linewidth=3,color = color)
        return_var = linear_fit
    elif fit_type == "exp":
        if distribution==2:
            exp_fit = np.polyfit(xf,np.log(np.array(yf)),1)
            ax.plot(xf,np.exp(np.array(xf)*exp_fit[0] + exp_fit[1]),linewidth=3,color = color,linestyle=line_style)
        else:
            exp_fit = np.polyfit(np.log10(xf),yf,1)
            ax.plot(xf,(np.log10(xf)*exp_fit[0]+exp_fit[1]),linewidth=3,color = color,linestyle=line_style)
        return_var = exp_fit
    elif fit_type == "n_exp":
        exp_fit = np.polyfit(1/np.array(xf),np.array(yf),3)
        ax.plot(xf,(1/(np.array(xf)*np.array(xf)*np.array(xf)))*exp_fit[0] + (1/(np.array(xf)*np.array(xf)))*exp_fit[1] + (1/(np.array(xf)))*exp_fit[2] + exp_fit[3],linewidth=3,color = color,linestyle=line_style)
        return_var = exp_fit
    else:
        return_var = None
    
    # ax.set_yticks(np.arange(0,2,0.5))
    # plt.xticks(fontsize=32,fontweight='bold')
    # plt.yticks(fontsize=32,fontweight='bold')
    # plt.xticks(fontsize=18,fontweight='bold')
    # plt.yticks(fontsize=18,fontweight='bold')
    plt.xticks(fontsize=22,fontweight='bold')
    plt.yticks(fontsize=22,fontweight='bold')
    # ax.set_ylim([0, 2])
    return return_var

def PDF(pdf_func,mu,sigma):
    step = 0.0001                                           #   Resolution of the PDF
    start = np.sum(mu)/len(mu) - np.sum(sigma)-5            #   Start point of PDF (left tail end)
    end = np.sum(mu)/len(mu) + np.sum(sigma)+5              #   End point of PDF (right tail end)
    base = np.round(np.arange(start,end,step),decimals=4)   #   Base axis of the PDF
    pdf =  pdf_func(base,mu,sigma)                          #   PDF function values at each base values (array)
    area = (np.sum(pdf)*step)                               #   Area under the PDF
    normalized_pdf = np.multiply(pdf,1/area)                #   Normalized PDF
    return normalized_pdf,area,base,start,end

def PDFportion(pdf_func,mu,sigma,area,slice_i_1,slice_i,ax,color_i):
    x1 = np.arange(slice_i_1,slice_i,0.0001)            #   Get the base of the slice 
    pdf1 =  pdf_func(x1,mu,sigma)                       #   Calculate the PDF values at the slice region only
    pdf1 = np.multiply(pdf1,1/area)                     #   Normalizing PDF
    ax.fill_between(x1,0,pdf1,facecolor=color_i)        #   Filling the region with the colour

def slicing(n,mu,sigma,area,end,base,pdf,pdf_func,ax):
    step = 0.0001                                                                               #   Resolution of the PDF
    slices = []                                                                                 #   List of slice boundaries
    mid_slices=[]                                                                               #   List of slice centers
    for i in range(1,n,1):                                                                      #   Looping over number of options to get slice boundaries
        ESM = prd.ICPDF(float(i)/n,mu,step,base,pdf)                                        #   Calculating slice boundary using ICPDF function
        slices.append(np.round(ESM,decimals=3))
    for i in range(1,2*n,1):                                                                    #   Looping over number of options to get slice centers
        if i%2!=0:
            mid_slices.append(np.round(prd.ICPDF((i/(2*n)),mu,step,base,pdf),decimals=1))   #   Calculating slice centers using ICPDF function
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(n)]                                                              #   Generating colours for each slice

    for i in range(len(slices)+1):
        if i!=0 and i!=len(slices):                                                             #   If slice boundary is not the either end of distribution
            PDFportion(pdf_func,mu,sigma,area,slices[i-1],slices[i],ax,color[i])
        elif i==0:                                                                              #   If slice boundary is left tail
            PDFportion(pdf_func,mu,sigma,area,start1,slices[i],ax,color[i])
        elif i==len(slices):                                                                    #   If slice boundary is right tail
            PDFportion(pdf_func,mu,sigma,area,slices[-1],stop1,ax,color[i])

def QualityWiseOptions(Dx,mu_x,sigma_x,n,ax,once):
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]                                                              #   Generating colours for each option
    color = ['#004D40','#1E88E5']
    lable = [r"$q^{\bf '}$",r"$q^{\bf ''}$"]
    legendss = []
    for i in range(50):
        ref_qual,options_quality = rng.quality(distribution=Dx,mu_x=mu_x,sigma_x=sigma_x,number_of_options=n)    #   Sampling qualities for each option
        if once == 1:
            for j in range(2):    
                if i==0:
                    ax.scatter([max(options_quality)],[np.random.uniform(low = 0.1*j ,high = j/10 +0.07)],s=9,edgecolor = color[j],facecolor=color[j],label=lable[j])  # Plotting the maximum quality option in sequence
                    legendss.append(lable[j])
                else:
                    ax.scatter([max(options_quality)],[np.random.uniform(low = 0.1*j ,high = j/10 +0.07)],s=9,edgecolor = color[j],facecolor=color[j])  # Plotting the maximum quality option in sequence
                ind = np.where(options_quality==max(options_quality))   #   Finding the maximum quality value
                options_quality[ind[0]]= -100                           #   Removing the maximum quality to get nex maximum in the list
                once = 0
        else:
            for j in range(2):    
                ax.scatter([max(options_quality)],[np.random.uniform(low = 0.1*j ,high = j/10 +0.07)],s=9,edgecolor = color[j],facecolor=color[j])  # Plotting the maximum quality option in sequence
                ind = np.where(options_quality==max(options_quality))   #   Finding the maximum quality value
                options_quality[ind[0]]= -100                 
    return legendss

def save_fitting_data(save_string):
    f_path = path+save_string+'.csv'
    columns_name = ['$\mu_{m}$','n','$\sigma_{q}$','slope_fit','y_intercept_fit','fit '+ r'$\mu_{HARS}$','slope*','y_intercept*','* '+ r'$\mu_{HARS}$']
    f = open(f_path,'a')
    columns = pd.DataFrame(data=np.array([columns_name]))
    columns.to_csv(f_path,mode='a',header=False,index=False)
    return columns_name,f_path

prd = Prediction()
vis = Visualization()

Fig2 = 0
if Fig2 == 1:
    step = 0.0001
    var_mu = [10]
    number_of_options = [2,5,10,100]
    sd_g = []
    var_sigma = List([1])
    fig,axs = plt.subplots(len(number_of_options))
    plt.subplots_adjust(left=0.3,bottom=0.05, right=0.7, top=0.95, wspace=0.2,hspace=0.5)
    # fig,axs = plt.subplots(len(number_of_options),len(var_sigma))
    dist = prd.gaussian
    for j in range(len(var_mu)):
        delta_mu = 5

        mu_x = List([var_mu[j]])
        sigma_x = List([var_sigma[0]])
        low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])#,mu_x[1]-np.sqrt(3)*sigma_x[1]])
        high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])#,mu_x[1]+np.sqrt(3)*sigma_x[1]])
        start = np.sum(mu_x)/len(mu_x) - np.sum(sigma_x)-5
        stop = np.sum(mu_x)/len(mu_x) + np.sum(sigma_x)+5

        dis_x = np.round(np.arange(start,stop,step),decimals=4)
        pdf =  dist(dis_x,mu_x,sigma_x)
        area = (np.sum(pdf)*step)
        pdf = np.multiply(pdf,1/area)

        for nop in range(len(number_of_options)):
            # if nop == 0:
            #     axs[nop].plot(dis_x,pdf,c='#882255',linewidth = 4,label = r'$D_q$')
            # else:
                
            axs[nop].plot(dis_x,pdf,c='#882255',linewidth = 2)
            
            axs[nop].invert_yaxis()
            axs[nop].axes.get_yaxis().set_visible(False)

            slices = []
            mid_slices=[]
            for i in range(1,number_of_options[nop],1):
                ESM = prd.ICPDF(float(i)/number_of_options[nop],mu_x,step,dis_x,pdf)
                slices.append(np.round(ESM,decimals=3))
            for i in range(1,2*number_of_options[nop],1):
                if i%2!=0:
                    mid_slices.append(np.round(prd.ICPDF((i/(2*number_of_options[nop])),mu_x,step,dis_x,pdf),decimals=1))

            number_of_colors = number_of_options[nop]

            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
            for i in range(len(slices)+1):
                if i!=0 and i!=len(slices):
                    x1 = np.arange(slices[i-1],slices[i],0.0001)
                    pdf1 =  dist(x1,mu_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    axs[nop].fill_between(x1,0,pdf1,facecolor=color[i])
                elif i==0:
                    x1 = np.arange(start,slices[i],0.0001)
                    pdf1 =  dist(x1,mu_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    axs[nop].fill_between(x1,0,pdf1,facecolor=color[i])
                elif i==len(slices):
                    x1 = np.arange(slices[-1],stop,0.0001)
                    pdf1 =  dist(x1,mu_x,sigma_x)
                    pdf1 = np.multiply(pdf1,1/area)
                    axs[nop].fill_between(x1,0,pdf1,facecolor=color[i])
            bests = []
            for i in range(10000):
                # ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
                ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
                # # print(max(options_quality))
                # options_quality[list(options_quality).index(max(options_quality))] = -np.Inf
                # options_quality[list(options_quality).index(max(options_quality))] = -np.Inf
                # options_quality[list(options_quality).index(max(options_quality))] = -np.Inf
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
            # print([slices,mid_slices])
            # axs[nop,2*k+1].bar(mid_slices,bins,width=0.2)
            axs1 = axs[nop].twinx()
            ticks = [tick for tick in axs1.get_yticks() if tick >=0]
            axs1.set_yticks(np.arange(0, 800))
            axs1.bar(mid_slices,bins/10000,width=0.2,color='indigo')

            #   Gumbel distribution
            x_n = np.sqrt(var_sigma[0]*lambertw((number_of_options[nop]**2)/(2*np.pi)))
            n_rho_xn = x_n/var_sigma[0]

            start_g = mu_x[0] - var_sigma[0] - 5
            stop_g = mu_x[0] + var_sigma[0] + 5

            dis_x_g = np.round(np.arange(start_g,stop_g,step),decimals=4)
            pdf_g =  [n_rho_xn*np.exp(-n_rho_xn*(i-mu_x[0]-x_n)-np.exp(-n_rho_xn*(i-mu_x[0]-x_n))) for i in dis_x_g]
            variance_g = np.pi/(np.sqrt(6)*n_rho_xn)
            sd_g.append(variance_g)
            # area_g = (np.sum(pdf_g)*step)
            # pdf_g = np.multiply(pdf_g,1/area_g)
            axs1.plot(dis_x_g,pdf_g,c='#004D40',label='Gumbel PDF',linewidth = 2)


            pdf_cg = []
            dis_x_cg = []
            for i in range(len(slices[:-1])):
                sum__ = 0
                for j in np.arange(slices[i],slices[i+1],step):
                    sum__ += n_rho_xn*np.exp(-n_rho_xn*(j-mu_x[0]-x_n)-np.exp(-n_rho_xn*(j-mu_x[0]-x_n)))*step
                pdf_cg.append(sum__)
                # dis_x_cg.append(mid_slices[2*i+1])
                
            # pdf_cg =  [np.exp(-np.exp(-n_rho_xn*(i-mu_x[0]-x_n))) for i in dis_x_g]
            # axs1.plot(dis_x_g,pdf_cg,c='#882255',linestyle='-.',label = 'Gumbel Cummulative DF')
            # if nop == 0:
            #     axs1.plot(mid_slices,pdf_cg,c='#1E88E5',linestyle='-.',label = 'Area under Gumbel slice',linewidth = 4)
            # else:
            #     axs1.plot(mid_slices,pdf_cg,c='#1E88E5',linestyle='-.',linewidth = 4)

            mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,step,dis_x,pdf)
            axs1.axvline(mean_esmes2m,0,500,color='red',label = r'$\bf \mu_{h}^{*}$',linewidth = 2,linestyle='--')

            axs1.legend(loc='upper left',frameon=False,fontsize=14)
            # axs1.axvline(mean_esmes2m,0,500,color='red',label = r'$\bf \mu_{h}^{*}$',linewidth = 1)
            axs[nop].title.set_text("Number of samples drawn,"+r"$\bf n$"+" = "+str(number_of_options[nop]))
            axs[nop].title.set_fontweight('bold')
            align_yaxis(axs1,0,axs[nop],0)
        plt.yticks(fontweight='bold',fontsize = 14)
        plt.xticks(fontweight='bold',fontsize = 14)
        
        plt.savefig('optimal_mu_h_gumbel.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
        def onclick(event):
            plt.savefig('optimal_mu_h_gumbel.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.05)
        point = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # fig,axs = plt.subplots()
    # axs.plot(number_of_options,sd_g)
    # plt.show()

Fig3_sch = 0
if Fig3_sch == 1:
    step = 0.0001
    
    number_of_options = [30]
    # plt.style.use('ggplot')
    fig,axs = plt.subplots()

    # fig.tight_layout(pad=0.5)
    dist = prd.gaussian
    # dist = prd.uniform
    # delta_mu = 4
    mu_x = List([10,15])
    sigma_x = List([1,1])
    low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])#,mu_x[1]-np.sqrt(3)*sigma_x[1]])
    high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])#,mu_x[1]+np.sqrt(3)*sigma_x[1]])
    
    pdf_x,area,dis_x,start1,stop1 = PDF(dist,mu_x,sigma_x)

    
    for nop in range(len(number_of_options)):
        mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,step,dis_x,pdf_x)
        axs.axvline(mean_esmes2m,0,500,color='red',label = r'$\bf \mu^*_h$',linewidth = 5,linestyle='--')

        mu_h = List([mean_esmes2m])
        # mu_h = mu_x
        # sigma_h = List([1])
        sigma_h = List([0.17*np.log10(number_of_options[nop]) + 0.46])
        sig = prd.ICPDF((1-(1/(number_of_options[nop]**2))),mu_x,step,dis_x,pdf_x) - mean_esmes2m
        
        
        # axs.axvline(mean_esmes2m+sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 0.5)
        # axs.axvline(mean_esmes2m-sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 0.5)
        # axs.axvline(mean_esmes2m+sig,0,500,color='green',label = r'$\bf \sigma_{h-pred}$',linewidth = 0.5)

        pdf_h,area,dis_h,start,stop = PDF(prd.gaussian,mu_h,sigma_h)

        axs.invert_yaxis()
        axs.plot(dis_x,pdf_x,color='green',label=r'$\bf D_q$',linewidth=5)

        for i in range(3):
            # ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
            ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
            # axs.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            # axs.scatter(options_quality,(0.3-i*0.1)*np.ones_like(options_quality),s=3,edgecolor = 'black')
            # axs.text(start1+2,(0.3-i*0.1)-0.01,'trial '+str(i),font,color=(0.3,0.3,0.3,1))
            # axs.text(start1+2,0.05,r'$\bf N_q$',font,color=(0.3,0.3,0.3,1))
            
        axs1 = axs.twinx()
        axs1.plot(dis_h,pdf_h,color='indigo',label=r'$\bf D_h$',linewidth=5)
        for i in range(3):
            units = rng.threshold_n(m_units=number_of_options[nop]*100,mu_h=mu_h,sigma_h=sigma_h)

        
        axs.legend(loc='upper left',prop=dict(weight='bold',size=20),frameon = False)
        axs1.legend(loc='center left',bbox_to_anchor=(0, 0.65),prop=dict(weight='bold',size=20),frameon = False)

        align_yaxis(axs1,0.0,axs,0.0)
        axs.set_yticks([])
        axs1.set_yticks([])
        axs.set_xticks([])
        axs1.set_xticks([])
        axs.tick_params(axis='both', which='major', labelsize=18)
        def onclick(event):
            plt.savefig('Fig3c.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.01)
        point = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

sigmas = [1/3,1,3]

Fig5 = 0
if Fig5 == 1:
    legends = []
    fig,axs = plt.subplots(1,3)
    plt.subplots_adjust(left=0.01,bottom=0.3, right=0.99, top=0.63, wspace=0.01)
    step = 0.0001
    number_of_options = [50]
    dist = prd.gaussian
    # dist = prd.uniform
    mu_x = List([10])
    sigma_x = List([1])
    low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])
    high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])
    for mus in range(2):
        for sigma_ in range(len(sigmas)):
            if mus == 0:
                pdf_x,area,dis_x,start1,stop1 = PDF(dist,mu_x,sigma_x,)

                for nop in range(len(number_of_options)):
                    mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,step,dis_x,pdf_x)
                    if sigma_+nop == 0:    
                        axs[sigma_].axvline(mean_esmes2m,0,500,color='red',linewidth = 3,linestyle="--",label=r'$\bf \mu_{h}^{*}$')
                        legends.append(r'$\bf \mu_{h}^{*}$')
                    else:
                        axs[sigma_].axvline(mean_esmes2m,0,500,color='red',linewidth = 3,linestyle="--")
                    
                    # sigma_h = List([0.17*np.log10(number_of_options[nop]) + 0.46])
                    # sig = prd.ICPDF((1-(1/(number_of_options[nop]**2))),mu_x,stop1,step,dis_x,pdf_x) - mean_esmes2m
                    
                    
                    # axs[sigma_].axvline(mean_esmes2m+sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 1,linestyle="--")
                    # axs[sigma_].axvline(mean_esmes2m-sigma_h[0],0,500,color='brown',label = r'$\bf \sigma_{h-fit}$',linewidth = 1,linestyle="--")
                    # axs[sigma_].axvline(mean_esmes2m+sig,0,500,color='green',label = r'$\bf \sigma_{h-pred}$',linewidth = 1,linestyle="--")

                    axs[sigma_].invert_yaxis()
                    axs[sigma_].plot(dis_x,pdf_x,color="#882255",linewidth=5)
                    
                    # slicing(number_of_options[nop],mu_x,sigma_x,area,stop1,dis_x,pdf_x,dist,axs[sigma_])
                    
                    # QualityWiseOptions(rng.dx_u,low,high,number_of_options[nop],axs[sigma_])
                    if sigma_+nop == 0:
                        leg = QualityWiseOptions(rng.dx_n,mu_x,sigma_x,number_of_options[nop],axs[sigma_],once=1)
                    else:
                        leg = QualityWiseOptions(rng.dx_n,mu_x,sigma_x,number_of_options[nop],axs[sigma_],once = 0)
                    legends += leg

            
            if mus == 1:
                mu_h = List([mean_esmes2m])
            else:
                mu_h = mu_x
            
            sigma_h = List([sigmas[sigma_]])

            pdf_h,area,dis_h,start,stop = PDF(prd.gaussian,mu_h,sigma_h)

            axs1 = axs[sigma_].twinx()
            if mus == 1:
                if sigma_== 0:
                    axs1.plot(dis_h,pdf_h,color='indigo',linewidth=5,label=r'$N(\mu_h=\mu_h^*)$')
                    legends.append(r'$N(\mu_h=\mu_h^*)$')
                else:
                    axs1.plot(dis_h,pdf_h,color='indigo',linewidth=5)

            else:
                if sigma_ == 0:
                    axs1.plot(dis_h,pdf_h,color='indigo',linewidth=5,linestyle='--',label=r'$N(\mu_h=\mu_x)$')
                    legends.append(r'$N(\mu_h=\mu_x)$')
                else:
                    axs1.plot(dis_h,pdf_h,color='indigo',linewidth=5,linestyle='--')
            
            
            align_yaxis(axs1,0.0,axs[sigma_],0.0)
            if sigma_ ==0:
                num = "(a) "
            elif sigma_ == 1:
                num = "(b) "
            elif sigma_==2:
                num = "(c) "
            # axs[sigma_].text(7,0.4,s=num)
            axs[sigma_].set_yticks([])
            axs1.set_yticks([])
            axs[sigma_].tick_params(axis='both', which='major', labelsize=18,labelcolor='black')
            axs[sigma_].set_xlabel(num+r'$\sigma_h =$'+str(np.round(sigmas[sigma_]/sigma_x[0],decimals=1))+'$\sigma_q$', fontsize=18,color='black')
        
        # last.set_lable("Response threshold PDF "+r'$N_h$')
        # st = fig.suptitle("Number of samples drawn = "+str(number_of_options[nop]),fontsize=18,fontweight='bold',color='black')
        # st.set_y(0.73)
        # st.set_x(0.5)
        # colors = ['indigo','red','#882255','#004D40','#1E88E5']
        # linestyle = ['--','-']
        # point_leg = [plt.plot([0, 0], [0,0.5], c=colors[0],linestyle=linestyle[i]) for i in range(len(linestyle))]
        # point_leg += [plt.plot([0, 0], [0,0.5], c=colors[1+i],linestyle=linestyle[i]) for i in range(len(linestyle))]
        # point_leg += [plt.scatter([0],[0],s=10,c=colors[3+i]) for i in range(len(linestyle))]
        # point_leg = [plt.Rectangle((0, 0), 0.5, 0.5, fc=colors[i]) for i in range(len(colors))]
        labels = [r'$N(\mu_h=\mu_x)$',r'$N(\mu_h=\mu_h^*)$', r'$\bf \mu_{h}^{*}$',r"$q^{\bf '}$",r"$q^{\bf ''}$"]
        # fig.legend(point_leg,labels,loc='upper right',prop=dict(weight='bold',size=20), bbox_to_anchor=(0.95, 0.7),ncol=5,columnspacing=3,frameon=False)
    fig.legend(loc='upper right',markerscale=4,prop=dict(weight='bold',size=20), bbox_to_anchor=(0.95, 0.7),ncol=5,columnspacing=6,frameon=False)
    
    def onclick(event):
        plt.savefig('Fig5.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.0)
    point = fig.canvas.mpl_connect('button_press_event', onclick)
    print(legends)
    plt.show()

sliced_reflection_figure_with_points = 0
if sliced_reflection_figure_with_points == 1:
    step = 0.0001
    
    number_of_options = [10]

    fig,axs = plt.subplots()

    # fig.tight_layout(pad=0.5)
    dist = prd.gaussian
    # delta_mu = 4
    mu_x = List([10])
    sigma_x = List([1])
    low = List([mu_x[0]-np.sqrt(3)*sigma_x[0]])#,mu_x[1]-np.sqrt(3)*sigma_x[1]])
    high = List([mu_x[0]+np.sqrt(3)*sigma_x[0]])#,mu_x[1]+np.sqrt(3)*sigma_x[1]])
    
    pdf_x,area,dis_x,start1,stop1 = PDF(dist,mu_x,sigma_x)
    
    for nop in range(len(number_of_options)):
        mean_esmes2m = prd.ICPDF(1-(1/number_of_options[nop]),mu_x,step,dis_x,pdf_x)
        axs.axvline(mean_esmes2m,0,500,color='red',label = r'$\mu_{ESM,ES2M}$',linewidth = 0.5)

        mu_h = List([mean_esmes2m])
        # mu_h = mu_x
        sigma_h = List([0.15*np.log10(number_of_options[nop]) + 0.5])
        pred_var = prd.ICPDF(1-(1/(number_of_options[nop])**2),mu_x,step,dis_x,pdf_x)
        axs.axvline(pred_var,0,500,color='orange',label = r'$\mu_{ESM,ES2M}$',linewidth = 0.5)

        axs.axvline(mean_esmes2m+sigma_h[0],0,500,color='darkgreen',label = r'$\mu_{ESM,ES2M}$',linewidth = 0.5)
        axs.axvline(mean_esmes2m-sigma_h[0],0,500,color='purple',label = r'$\mu_{ESM,ES2M}$',linewidth = 0.5)
        
        pdf_h,area,dis_h,start,stop = PDF(prd.gaussian,mu_h,sigma_h)

        axs.plot(dis_x,pdf_x,color='black')
        
        axs.invert_yaxis()

        slicing(number_of_options[nop],mu_x,sigma_x,area,stop1,dis_x,pdf_x,dist,axs)

        for i in range(3):
            # ref_qual,options_quality = rng.quality(distribution=rng.dx_u,mu_x=low,sigma_x=high,number_of_options=number_of_options[nop])
            ref_qual,options_quality = rng.quality(distribution=rng.dx_n,mu_x=mu_x,sigma_x=sigma_x,number_of_options=number_of_options[nop])
            axs.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            axs.scatter(options_quality,(0.3-i*0.1)*np.ones_like(options_quality),s=3,edgecolor = 'black')
            axs.text(start1+2,(0.3-i*0.1)-0.01,'trial '+str(i))
            axs.text(start1+2,0.03,r'$\mu_x$')
            
        axs1 = axs.twinx()
        axs1.plot(dis_h,pdf_h,color='maroon')
        for i in range(3):
            units = rng.threshold_n(m_units=number_of_options[nop]*100,mu_h=mu_h,sigma_h=sigma_h)
            axs1.axhline((0.3-i*0.1),0,500,color='black',linewidth = 0.5,alpha=0.25)
            axs1.scatter(units,(0.3-i*0.1)*np.ones_like(units),s=1,edgecolor = 'black')
            axs1.text(start1+2,(0.3-i*0.1)+0.01,'trial '+str(i))
            axs.text(start1+2,-0.01,r'$\mu_h$')
        
        axs.legend(loc='upper right')

        axs.title.set_text("Number of samples drawn = "+str(number_of_options[nop]))
        align_yaxis(axs1,0.0,axs,0.0)
        axs.set_yticks([])
        axs1.set_yticks([])

    plt.show()

overlapping_distributions = 0
if overlapping_distributions == 1:
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
    x_ = prd.ICPDF(0.8,mu,step,x,pdfg)
    print(x_)
    print(prd.gaussian(List([x_]),mu,sigma))
    ax.fill_between(x[:np.argmin(np.abs(x-x_))],0,pdfg[:np.argmin(np.abs(x-x_))],facecolor='blue')

    x_ = prd.ICPDF(0.8,mu,step,x,pdfu)
    print(x_)
    print(prd.uniform(List([x_]),mu,sigma)/area)
    ax.fill_between(x[:np.argmin(np.abs(x-x_))],0,pdfu[:np.argmin(np.abs(x-x_))],facecolor='orange')
    plt.show()

distributions_checker_board = 0
if distributions_checker_board == 1:
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


ylab2 = r'$\bf \mu_h\;vs.\;\mu_q$'
ylab1 = r'$\bf \sigma_h\;vs.\;\sigma_q$'


def legend_func(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$K_q$',r'$U_q$',r'$N_q$']
    point_leg1 = []
    labels1 = []
    num_opts = [2,3,4,5,8,10,15,20,30,40,80,100]
    for i in range(len(marker)):
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(num_opts[i]))
    leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=5,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Number of options "+r"$n$", bbox_to_anchor=(1, 1),fontsize=18,prop=dict(weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)

def legend_func_mum(func,bxgh_fit,uxgh_fit,gxgh_fit,func1,axes):
    marker=['o','s','*','D','X','p','d','v','^','P','H','8']
    colors = ['slateblue','lightseagreen','coral']
    point_leg = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    labels = ['bxgh'+func(bxgh_fit),'uxgh'+func1(uxgh_fit),'gxgh'+func(gxgh_fit)]
    dist_lab = [r'$K_q$',r'$U_q$',r'$N_q$']
    mum = [10,50,100,200,500]
    point_leg1 = []
    labels1 = []
    for i in range(len(mum)):
        point_leg1.append(plt.scatter([],[], edgecolor='black',s=40,marker=marker[i], facecolor='white'))#,label=str(num_opts[i])+'_'+distribution))
        labels1.append(str(mum[i])+'$n$'+' agents')
    leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1))
    leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.14),prop=dict(weight='bold',size=16),labelcolor=(0,0,0,1),ncol=3,columnspacing=5,frameon=False)
    # leg1 = plt.legend(point_leg,labels,loc='upper center', bbox_to_anchor=(0.5, 1.7),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1))
    # leg3 = plt.legend(point_leg,dist_lab,loc='upper center', bbox_to_anchor=(0.5, 1.1),fontsize=12,prop=dict(weight='bold'),labelcolor=(0.3,0.3,0.3,1),ncol=3,columnspacing=3,frameon=False)
    leg2 = plt.legend(point_leg1,labels1,loc='upper left', title="Swarm size "+r"$S = m n$",bbox_to_anchor=(1, 1),fontsize=18,prop=dict(weight='bold'),labelcolor=(0,0,0,1),frameon=False)
    axes.add_artist(leg1)
    axes.add_artist(leg2)
    axes.add_artist(leg3)

def CurveFitting(bxgh_string,uxgh_string,gxgh_string,fit_types,legendFunc,xlab,ylab,save_string,fit_var,text=None,t_x=0,t_y=0,function = None,fit_type=None,first_variable=None,vs_n=0):
    fig, ax = plt.subplots()
    line_styles = ['--','-.',':','-']
    for i in range(len(bxgh_string)):
        bxgh_fit = genericPlotter(ax,3,color='slateblue',f_path=path+bxgh_string[i]+'.csv',function=function,fit_type=fit_type,first_variable=first_variable,vs_n = vs_n,line_style=line_styles[i])
        uxgh_fit = genericPlotter(ax,2,color='lightseagreen',f_path=path+uxgh_string[i]+'.csv',function=function,fit_type=fit_type,first_variable=first_variable,vs_n = vs_n,line_style=line_styles[i])
        gxgh_fit = genericPlotter(ax,1,color='coral',f_path=path+gxgh_string[i]+'.csv',function=function,fit_type=fit_type,first_variable=first_variable,vs_n = vs_n,line_style=line_styles[i])

    # functions = []
    # if not isinstance(fit_types,type(None)):
    #     for fit_type in fit_types:
    #         if fit_type=="log":
    #             function = lambda log_fit : ('['+r'$\bf %.2f \log_{10}{%s} %+.2f$'%(np.round(log_fit[0],decimals=2),fit_var,np.round(log_fit[1],decimals=2))+']')
            
    #         elif fit_type=="linear":
    #             function = lambda linear_fit : ('['+r'$\bf %.2f%s %+.2f$'%(np.round(linear_fit[0],decimals=2),fit_var,np.round(linear_fit[1],decimals=2))+']')
            
    #         elif fit_type=="n_exp":
    #             function = lambda exp_fit : ('['+r'$\bf %.2f%s^{-3} %+.2f%s^{-2} %+.2f%s^{-1} %+.2f$'%(np.round(exp_fit[0],decimals=2),fit_var,np.round(exp_fit[1],decimals=2),fit_var,np.round(exp_fit[2],decimals=2),fit_var,np.round(exp_fit[3],decimals=2))+']')
    #         elif fit_type=='exp':
    #             function = lambda exp_fit : ('['+r'$\bf e^{%.2f%s %+.2f}$'%(np.round(exp_fit[1],decimals=2),fit_var,np.round(exp_fit[2],decimals=2))+']')

    #         functions.append(function)
    
    # if len(functions)==1:
    #     functions.append(function)

    # legendFunc(functions[0],bxgh_fit,uxgh_fit,gxgh_fit,functions[1],ax)

    # ax.text(38,0.1,s=text)
    # ax.text(95,-0.34,s=text)
    ax.text(t_x,t_y,s=text)
    plt.xlabel(xlab,fontproperties)
    plt.ylabel(ylab,fontproperties)
    # plt.savefig(save_string,bbox_inches="tight",pad_inches=0.2)
    plt.tight_layout()
    plt.show()

# slopes_HARS = 0
# if slopes_HARS ==1:

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['log'],legend_func,"Mean sub-swarm size "+r'$\bf\mu_m$','Slope of \n HARS line in '+ylab1,initials+"slope_mum.eps","\mu_m")

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['linear'],legend_func,"Mean sub-swarm size "+r'$\bf\mu_m$','y-intercept of \n HARS line in '+ylab1,initials+"Intercept_mum.eps","\mu_m")

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['log'],legend_func,"Mean sub-swarm size "+r'$\bf\mu_m$',r'$\mu_{HARS}$'+' of HARS line in '+ylab1,initials+"HARS_mum.eps","\mu_m")

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['n_exp'],legend_func,'Number of options, n','y-intercept of\n HARS line in '+ylab1,initials+"Intercept_n.eps","n","(a)",95,-0.34)

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['log','exp'],legend_func_mum,'Number of options, n','Slope of\n HARS line in '+ylab1,initials+"Slope_n.eps","n","(b)",38,0.9425)

#     CurveFitting(genericPlotter,bxgh_string,uxgh_string,gxgh_string,['exp'],legend_func_mum,'Number of options, n',r'$\mu_{HARS}$'+' of best fit in '+ylab1,initials+"HARS_n.eps","n")

if __name__=='__main__':
    KqNh_mu = 1
    UqNh_mu = 1
    NqNh_mu = 1
    # if KqNh_mu==1:
    #     cnt = 0
    #     save_string = str(cnt)+'KQNh'
    #     vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='muQ1',varYlabel='muH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='K')

    initials = "sigma_predicted"
    bxgh_string = initials+'_bxgh_2_sigma'
    uxgh_string = initials+'_uxgh_2_sigma'
    gxgh_string = initials+'_gxgh_2_sigma'
    initials = "mu_predicted"
    bxgh_string1 = initials+'_bxgh_2_sigma'
    uxgh_string1 = initials+'_uxgh_2_sigma'
    gxgh_string1 = initials+'_gxgh_2_sigma'
    initials = "naive"
    bxgh_string2 = initials+'_bxgh_2_sigma'
    uxgh_string2 = initials+'_uxgh_2_sigma'
    gxgh_string2 = initials+'_gxgh_2_sigma'
    initials = "predicted"
    bxgh_string3 = initials+'_bxgh_2_sigma'
    uxgh_string3 = initials+'_uxgh_2_sigma'
    gxgh_string3 = initials+'_gxgh_2_sigma'
    
    if NqNh_mu==1:
        col_name,file = save_fitting_data(save_string=gxgh_string2)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 12
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'NQNh_naive_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='N')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    if UqNh_mu==1:
        col_name,file = save_fitting_data(save_string=uxgh_string2)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 13
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'UQNh_naive_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='U')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)

    if KqNh_mu==1:
        col_name,file = save_fitting_data(save_string=bxgh_string2)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 14
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'KQNh_naive_'+str(m)+'_'+str(n)
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='K')
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
        
    if NqNh_mu==1:
        col_name,file = save_fitting_data(save_string=gxgh_string3)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 15
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'NQNh_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='N')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    if UqNh_mu==1:
        col_name,file = save_fitting_data(save_string=uxgh_string3)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 16
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'UQNh_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='U')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)

    if KqNh_mu==1:
        col_name,file = save_fitting_data(save_string=bxgh_string3)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 17
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'KQNh_pred__'+str(m)+'_'+str(n)
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='K')
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
        
    if NqNh_mu==1:
        col_name,file = save_fitting_data(save_string=gxgh_string1)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 18
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'NQNh_mu_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='N')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    if UqNh_mu==1:
        col_name,file = save_fitting_data(save_string=uxgh_string1)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 19
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'UQNh_mu_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='U')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    if KqNh_mu==1:
        col_name,file = save_fitting_data(save_string=bxgh_string1)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 20
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'KQNh_mu_pred_'+str(m)+'_'+str(n)
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='K')
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
        
    if NqNh_mu==1:
        col_name,file = save_fitting_data(save_string=gxgh_string)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 21
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'NQNh_sigma_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='N')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)

    if UqNh_mu==1:
        col_name,file = save_fitting_data(save_string=uxgh_string)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 22
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'UQNh_sigma_pred_'+str(m)+'_'+str(n)
                # data = pd.read_csv(path+save_string+'.csv')
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='U')
                # hars.append(data['ARS'])
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    if KqNh_mu==1:
        col_name,file = save_fitting_data(save_string=bxgh_string)
        mum = [10,50,100,200,500]
        n_options = [2,5,8,10,15,20,30,40,80,100]
        cnt = 23
        # hars = []
        for m in mum:
            for n in n_options:
                save_string = str(cnt)+'KQNh_sigma_pred_'+str(m)+'_'+str(n)
                [hrcc,sigma_q,ars] = vis.data_reading(file_name=save_string+".csv",save_plot=save_string,varXlabel='sigmaQ1',varYlabel='sigmaH1',plot_type='colourmapIndividual',cbar_orien='vertical',Dq='K')
                values = [m,n,sigma_q,None,None,ars,None,None,None]
                data = {}
                for c in range(len(col_name)):
                    data[col_name[c]] = values[c]

                out = pd.DataFrame(data=[data],columns=col_name)
                out.to_csv(file,mode = 'a',header = False, index=False)
    
    CurveFitting([bxgh_string,bxgh_string1,bxgh_string2,bxgh_string3],[uxgh_string,uxgh_string1,uxgh_string2,uxgh_string3],[gxgh_string,gxgh_string1,gxgh_string2,gxgh_string3],legendFunc=legend_func,xlab="Mean sub-swarm size "+r'$\bf\mu_m$',ylab='ARS',save_string=initials+"ars_mum.pdf",fit_var=None,fit_types=None,function = 'mu_hars_fit',fit_type='a',first_variable='n',vs_n=0)
    CurveFitting([bxgh_string,bxgh_string1,bxgh_string2,bxgh_string3],[uxgh_string,uxgh_string1,uxgh_string2,uxgh_string3],[gxgh_string,gxgh_string1,gxgh_string2,gxgh_string3],legendFunc=legend_func,xlab="Number of options "+r'$\bf n$',ylab='ARS',save_string=initials+"ars_n.pdf",fit_var=None,fit_types=None,function = 'mu_hars_fit',fit_type='a',first_variable='m',vs_n=1)
                
