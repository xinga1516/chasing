# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:08:12 2022
to linear fit the drive frequency and the rate of increase, 
get the intercept and coefficient of the fitting line.

@author: xinyue Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import matplotlib.gridspec as gridspec
import scipy
import os
import math
import pandas as pd
import re

plt.rcParams.update({'font.size': 30,
                     'font.family': "Times New Roman",
                     "axes.titlesize": 2,
                     "axes.labelsize": 30,
                     
                     "legend.fontsize": 2,
                     "axes.labelpad": 1.0,
                     "axes.linewidth":1.2,
                     "axes.labelcolor": "black",
                     "xtick.color": "black",
                     "ytick.color": "black",
                     "xtick.major.size": 6.0,
                     "xtick.major.width":0.9,
                     "xtick.major.pad":1.8,
                     "xtick.direction":'inout',
                     "ytick.direction":'inout',
                     "ytick.major.size": 6.0,
                     "ytick.major.width":0.9,
                     "ytick.major.pad":1.5,
                     "legend.framealpha": 0,
                     "patch.facecolor": "white"})

def huber_loss(theta, x, y, delta=0.005):
    '''

    Parameters
    ----------
    theta : TYPE
        initial parameter, the first one is the intercept, second coeffcient
    x : TYPE
        fitting 
    y : TYPE
        DESCRIPTION.
    delta : TYPE, optional
        DESCRIPTION. The default is 0.005.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    diff = abs(y-(theta[0]+theta[1]*x))
    return ((diff < delta)*diff**2/2+(diff >= delta)*delta*(diff-delta/2)).sum()


def sca(x1,y1,colors1,x,y,coefficient,intercept,coefficient1,intercept1,x_label,y_label,title,colors=None,label=None):

    figure, ax = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(bottom=0.15)
    
    ax.set_title(title,fontsize=30)
    ax.set_xlabel(x_label,fontsize=30)
    ax.set_ylabel(y_label,fontsize=30)
    plt.scatter(x,y,c=colors,cmap='afmhot_r')
    plt.scatter(x1,y1,c=colors1,cmap='afmhot_r')
    plt.plot(x,[coefficient*i+intercept for i in x],c='green')
    plt.plot(x,[coefficient1*i+intercept1 for i in x],c='yellow')
    if colors is None:
        plt.show()
        return
    
    bar = plt.colorbar()
    bar.set_label(label)
    plt.show()
    return



def file_data(filename):
    '''
    read the raw data of the .csv files and compute the relative rate of increase
    the last return value is for linear regression 
    for the whole data
    start: seperate the range into 8 parts, the start part no
    end: the end part no
    '''
    rate_dr_plot = []
    increase = []
    dataXY = []
    rate_dr=[]
    low_percentage = 0
    dataXY_sample = np.array([[1,2],])
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("GEN:"):
                contents = line.split()
                rate_dr = rate_dr+[float(contents[3])]
        for i in range(len(rate_dr)-1):
            if(float(rate_dr[i+1])!=0 and rate_dr[i]!=0):
                relative_increase = (float(rate_dr[i+1])-float(rate_dr[i]))/rate_dr[i]
                if (relative_increase>-0.8 and relative_increase<1):
                    rate_dr_plot.append(rate_dr[i])
                    increase.append(relative_increase)
                    # exclude the tail
                    if(rate_dr[i]>0 and rate_dr[i]<0.9): # min .05
                        # 0.225*ger-0.0825
                        dataXY.extend([rate_dr[i],relative_increase])
                        if rate_dr[i]<0.10:
                            low_percentage = low_percentage + 1
        
        dataXY = np.array(dataXY).reshape(int(len(dataXY)/2),2)
        low_percentage = low_percentage/len(dataXY) if len(dataXY)!=0 else 1
        
        seperate_no = 20
        seperator = np.linspace(min(rate_dr_plot), 0.5,seperate_no+1) # min 0.05max(rate_dr_plot)
        if low_percentage > 0.7:
            start = 2
            end = seperate_no
        else:
            start = 1
            end = seperate_no -8
        # seperate the whole dataXY into 8 part, but don't use the last two parts
        number_list = []
        for i in range(start,end):
            con1 = dataXY[:,0]>seperator[i]
            con2 = dataXY[:,0]<seperator[i+1]
            con = [(con1[j] and con2[j]) for j in range(len(dataXY))]
            locals()['dataXY_phase'+str(i)] = dataXY[con]
            number_list.append(len(locals().get('dataXY_phase'+str(i))))
        
        while 0 in number_list:
            number_list.remove(0)
        #print(number_list)
        if len(number_list)!=0 :
            sample_number = int(0.2*min(number_list))
        else:
            return rate_dr_plot,increase,dataXY,low_percentage
        
        # if len(locals().get('dataXY_phase0'))!=0:
        #     dataXY_sample_no = np.random.choice(range(len(locals().get('dataXY_phase0'))),sample_number,replace=False)
        #     dataXY_sample = locals().get('dataXY_phase0')[dataXY_sample_no]
        #     print(len(dataXY_sample))
        for i in range(start,end):
            if len(locals().get('dataXY_phase'+str(i)))!=0:
                dataXY_sample_no = np.random.choice(range(len(locals().get('dataXY_phase'+str(i)))),sample_number,replace=False)
                dataXY_sample = np.concatenate((dataXY_sample,locals().get('dataXY_phase'+str(i))[dataXY_sample_no]),axis=0)
                # print(len(dataXY_sample))
            
            
        nan_no = [0]
        for i in range(len(dataXY_sample)):
            x = dataXY_sample[i]
            if math.isnan(x[0]) or math.isnan(x[1]):
                nan_no.append(i)
        dataXY_sample = np.delete(dataXY_sample,nan_no,axis=0)    
        # print(len(dataXY_sample))
        
    # print(low_percentage)
        
    return rate_dr_plot,increase,dataXY_sample,low_percentage

def file_data1(filename):
    '''
    read the raw data of the .out files and compute the relative rate of increase for r2 allele
    the last return value is for linear regression 
    for the whole data
    start: seperate the range into 8 parts, the start part no
    end: the end part no
    '''
    rate_dr_plot = []
    increase = []
    dataXY = []
    rate_dr=[]
    low_percentage = 0
    dataXY_sample = np.array([[1,2],])
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("GEN:"):
                contents = line.split()
                rate_dr = rate_dr+[float(contents[5])]
        for i in range(len(rate_dr)-1):
            if(float(rate_dr[i+1])!=0 and rate_dr[i]!=0):
                relative_increase = (float(rate_dr[i+1])-float(rate_dr[i]))/rate_dr[i]
                if (relative_increase>-0.8 and relative_increase<1):
                    rate_dr_plot.append(rate_dr[i])
                    increase.append(relative_increase)
                    # exclude the tail
                    if(rate_dr[i]>0 and rate_dr[i]<0.9): # min .05
                        # 0.225*ger-0.0825
                        dataXY.extend([rate_dr[i],relative_increase])
        
        dataXY = np.array(dataXY).reshape(int(len(dataXY)/2),2)
        
        seperate_no = 8
        seperator = np.linspace(min(rate_dr_plot), max(rate_dr_plot),seperate_no+1) # min 0.05
        
        start = 1
        end = seperate_no -1
        # seperate the whole dataXY into 8 part, but don't use the last two parts
        number_list = []
        for i in range(start,end):
            con1 = dataXY[:,0]>seperator[i]
            con2 = dataXY[:,0]<seperator[i+1]
            con = [(con1[j] and con2[j]) for j in range(len(dataXY))]
            locals()['dataXY_phase'+str(i)] = dataXY[con]
            number_list.append(len(locals().get('dataXY_phase'+str(i))))
        
        while 0 in number_list:
            number_list.remove(0)
        #print(number_list)
        if len(number_list)!=0 :
            sample_number = min(number_list)
        else:
            return rate_dr_plot,increase,dataXY,low_percentage
        
        # if len(locals().get('dataXY_phase0'))!=0:
        #     dataXY_sample_no = np.random.choice(range(len(locals().get('dataXY_phase0'))),sample_number,replace=False)
        #     dataXY_sample = locals().get('dataXY_phase0')[dataXY_sample_no]
        #     print(len(dataXY_sample))
        for i in range(start,end):
            if len(locals().get('dataXY_phase'+str(i)))!=0:
                dataXY_sample_no = np.random.choice(range(len(locals().get('dataXY_phase'+str(i)))),sample_number,replace=False)
                dataXY_sample = np.concatenate((dataXY_sample,locals().get('dataXY_phase'+str(i))[dataXY_sample_no]),axis=0)
                # print(len(dataXY_sample))
            
            
        nan_no = [0]
        for i in range(len(dataXY_sample)):
            x = dataXY_sample[i]
            if math.isnan(x[0]) or math.isnan(x[1]):
                nan_no.append(i)
        dataXY_sample = np.delete(dataXY_sample,nan_no,axis=0)    
        # print(len(dataXY_sample))
        
    # print(low_percentage)
        
    return rate_dr_plot,increase,dataXY_sample


def transform_range(val, prev_min, prev_max, new_min, new_max):
    # Transform a value from one range to another.
    return (((val - prev_min) * (new_max - new_min)) / (prev_max - prev_min)) + new_min


def heatmap(xcol, ycol, zcol, x_label,y_label,title,output):
    """
    Plots a heatmap.
    """
    x_dim = len(set(xcol))
    x_min = min(xcol)
    x_max = max(xcol)
    y_dim = len(set(ycol))
    y_min = min(ycol)
    y_max = max(ycol)
    count_data = len(xcol)
    # print(x_max)
    # print(x_min)
    plot_data = [[0.0 for i in range(x_dim)] for i in range(y_dim)]

    with open(output,'w') as r:
        r.write(x_label + ","+y_label+","+title+",\n")
        for i in range(count_data):
            x = xcol[i]
            y = ycol[i]
            z = zcol[i]
            x_coord = int(transform_range(x, x_min, x_max, 0, x_dim - 1) + 0.5)
            y_coord = int(transform_range(y, y_min, y_max, 0, y_dim - 1) + 0.5)
            plot_data[y_coord][x_coord] = z
            r.write(str(x)+","+str(y)+","+str(z)+",\n")

    fig = plt.figure(figsize=(8,8.5))  # confure the desired figure size.
    #fig = plt.figure()
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig) #place
    ax = fig.add_subplot(spec[0, 0])

    # Adjust the next line if the figure or legends are not well aligned with the edge of the image.
    #fig.set_tight_layout({"pad":0.0, "w_pad":0.0, "h_pad":0.0})
    ax.set_title(f"{title}",fontsize=30)  
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6]])
    #ax.set_yticks(np.arrange(data.shape[]))
    ax.set_xticks(np.linspace(0, x_dim-1, 6))  
    ax.set_yticks(np.linspace(0, y_dim-1, 6))  
    ax.set_xticklabels([round(i,2) for i in np.linspace(x_min,x_max,6)])
    ax.set_yticklabels([round(i,2) for i in np.linspace(y_min,y_max,6)])
    
    im = ax.imshow(plot_data, cmap=plt.cm.hot_r, rasterized=False,origin='lower',aspect='auto')
    plt.colorbar(im)
    
    plt.show()
    
def r2(x,y,theta):
    y_mean = np.mean(y) 
    up = sum((y - (theta[0]+theta[1]*x))**2)
    down = sum((y - y_mean)**2)
    return 1-up/down

def main():
    with open('embryo resistance/drive_speed0.csv','r') as f1:
        #f1.write("embryo resistance,coefficient,zero-frequency relative increase rate,maximum drive frequency,\n")
        for root,dirs,files in os.walk('embryo resistance/0-6results/'):
            for file in files:
                print(file)
                rate_dr_plot,increase,dataXY= file_data1(root+file)
                # print(low_percentage)
                # print(dataXY)
                if (len(dataXY) >0 ):
                    theta = scipy.optimize.fmin(huber_loss, x0=(1, -1), args=(dataXY[:,0], dataXY[:,1]), disp=False)
                    print(theta)
                    print(-theta[0]/theta[1])
                    loss = huber_loss(theta, dataXY[:,0], dataXY[:,1])
                    r = r2(dataXY[:,0], dataXY[:,1],theta)
                    print("loss: "+str(loss))
                    #f1.write(file.replace(".out",'')+","+str(abs(theta[1]))+","+str(theta[0])+","+str(-theta[0]/theta[1])+",\n")
                    
                    # print(len(dataXY))
                    # print(low_percentage)
                    figure, ax = plt.subplots(dpi=400,figsize=(8,8))
                    plt.subplots_adjust(bottom=0.15)
                    plt.gca().patch.set_facecolor('white')
                    figure.set_facecolor('white')
                    
                    ax.set_xlabel("r2 frequency",fontsize=30)
                    ax.set_ylabel("Relative increase of r2 frequency",fontsize=30)
                    color = ['red' if x in dataXY else 'blue' for x in rate_dr_plot]
                    plt.scatter(rate_dr_plot,increase,c=color)
                    plt.plot(rate_dr_plot,[theta[1]*i+theta[0] for i in rate_dr_plot],c='green')
                    # 显示坐标轴线 
                    ax.spines['left'].set_visible(True) 
                    ax.spines['bottom'].set_visible(True)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False) 
                    # 自定义坐标轴线的样式和颜色
                    ax.spines['left'].set_linestyle('-')
                    ax.spines['bottom'].set_linestyle('-')
                    ax.spines['left'].set_color('black')
                    ax.spines['bottom'].set_color('black')
                    
                    #ax.set_title("germline resistance="+file.replace(".out",''),fontsize=30)
                    #plt.text(0.1, 0.6,'y='+format(theta[1],'.2f')+'*x+'+format(theta[0],'.2f'))#+'\nr2='+str(r)
                    
                    plt.show()
    
    
if __name__ == "__main__":
    main()