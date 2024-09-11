# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:44:53 2024

@author: Xinyue Zhang
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import style
import math
from sklearn.cluster import DBSCAN
from itertools import combinations


def find_cluster(df):
    '''
    find the classification of all chasing cells, and the number of local chasing.
    and visualize the classification.
    the adjacent chasing cells are regarded as the same cluster, maximum 

    Parameters
    ----------
    df : TYPE dataframe
        the coordinates of the chasing cells

    Returns the number of local chasing.
    -------
    None.

    '''
    # fig,ax = plt.subplots(dpi=300)
    # plt.scatter(df['x'],df['y'])
    # plt.show()  
    df.insert(2,'traversal',[0 for i in range(len(df))])
    cluster_no = 1
    for index,row in df.iterrows():
        if row['traversal']==0:
            df.loc[index,'traversal'] = cluster_no
            cluster_no = cluster_no + 1
        base_no = row['traversal']
        group_set = [base_no]
        for index1,row1 in df.iterrows():
            distance = (row1['x']-row['x'])**2 + (row1['y']-row['y'])**2
            if distance <= 2:
                if row1['traversal'] == 0:
                    df.loc[index1,'traversal'] = base_no
                else:
                    group_set = group_set+[row1['traversal']]
        if len(set(group_set))!=1: # merge the group set
            group_no = min(group_set)
            for index1,row1 in df.iterrows():
                if row1['traversal'] in group_set and row1['traversal']!=group_no:
                    df.loc[index1,'traversal'] = group_no
    groups = df.groupby(by=['traversal'])
    df = df.set_index('traversal',drop=False)
    if (len(groups)!=1):
        for name,group in groups:
            if (len(group)==1):
                df = df.drop(name)
    group_set = set(df['traversal'])
    #print(df['traversal'])
    
    
    return df,len(group_set)

def merge_sets_with_intersection(sets_list):
    merged_sets = []
    merged = True

    while merged:
        merged = False
        new_sets_list = []
        merged_list = [0 for i in range(len(sets_list))]

        for i in range(len(sets_list)):
            for j in range(i + 1, len(sets_list)):
                if sets_list[i].intersection(sets_list[j]):
                    merged_set = sets_list[i].union(sets_list[j])
                    new_sets_list.append(merged_set)
                    merged_list[i]=1
                    merged_list[j]=1
                    merged = True
                    break
            if merged_list[i]==0:
                new_sets_list.append(sets_list[i])

        merged_sets = new_sets_list
        sets_list = new_sets_list

    return merged_sets

def merge_clusters(df,df0):
    if (len(df0)>1 and len(set(df0['traversal']))>1):
        group_set = []
        combines = list(combinations(set(df0['traversal']),2))
        for i in combines:
            c1 = min([(a['x']-b['x'])**2+(a['y']-b['y'])**2 for index,b in (df.loc[df['traversal']==i[1]]).iterrows() for index,a in (df.loc[df['traversal']==i[0]]).iterrows()])
            if c1<=16: #*2
                group_set = group_set+[set(i)]
                
        group_set = merge_sets_with_intersection(group_set)
        df = df.set_index('traversal', drop=False)
        for s in group_set:
            c_no = s.pop()
            for j in s:
                df.loc[j, 'traversal'] = c_no
        df = df.reset_index(drop=True)
    return df


def DBSCAN_find(df,area,cell_width):
    '''
    find clusters of chasing cells using DBSCAN algorithm

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = df.reset_index(drop=True)
    X = [[row['x'], row['y']] for index, row in df.iterrows()]
    clusters = DBSCAN(eps=np.sqrt(5)*0.05/cell_width, min_samples=4).fit_predict(X) #np.sqrt(5)*0.05/cell_width
    #print(clusters)
    df.insert(2, 'traversal', clusters)
    df = df.set_index('traversal', drop=False)
    if (-1 in df['traversal']):
        df = df.drop(-1)
    df = df.reset_index(drop=True)
    # merge the cells at the edges of area
    df0 = df.loc[df['x'] == 0]
    df=merge_clusters(df, df0)
    
    df0 = df.loc[df['y'] == 0]
    df=merge_clusters(df, df0)
    
    df0 = df.loc[df['x'] == round(area/cell_width)-1]
    df=merge_clusters(df, df0)
    
    df0 = df.loc[df['y']==round(area/cell_width)-1]
    df=merge_clusters(df, df0)
    
    groups = df.groupby(by='traversal')
    df = df.set_index('traversal', drop=False)
    for name,group in groups:
        if(len(group)<=round(3*0.05/cell_width)):
            df = df.drop(name)

    return df,len(set(df['traversal']))

def visualize(filename):
    #plot the actual individuals' positions 
    with open(filename, 'r') as f:
        data = f.read()
    data = data[1:].split('G\n')
    for gen in range(len(data)):
        data[gen] = data[gen][1:-1].split('\n')
    dft = pd.DataFrame()
    xs = [[] for _ in range(len(data))]
    ys = [[] for _ in range(len(data))]
    cs = [[] for _ in range(len(data))]
    for gen in range(len(data)):
        for ind in range(len(data[gen])):
            split = data[gen][ind].split()
            if len(split) > 0:  # Skip empty generations.
                xs[gen].append(int(split[0], 16) / 4095)
                ys[gen].append(int(split[1], 16) / 4095)
                cs[gen].append([int(split[2], 16) / 255, int(split[3], 16) / 255, int(split[4], 16) / 255])
        dft = pd.concat([dft,pd.DataFrame([[gen+1,xs[gen],ys[gen],cs[gen]]])])
    dft.columns=['gen','xs','ys','cs']
    dft = dft.set_index('gen')
    dft['chasing cell xs'] = -1
    dft['chasing cell ys'] = -1
    dft['chasing cell cluster'] = -1
    dft['chasing cell xs'] = dft['chasing cell xs'].astype('object')
    dft['chasing cell ys'] = dft['chasing cell ys'].astype('object')
    dft['chasing cell cluster'] = dft['chasing cell cluster'].astype('object')
    print(len(dft))
    
    cell_width=0.05#/math.sqrt(2)#0.01/math.sqrt(2/math.pi)
    # plot the cluster of chasing cells
    with open('area/test.out','r') as f1:
        lines = f1.readlines()
        for line in lines:
            if line.startswith("SIM BOUND:"):
                area = float(line.replace("SIM BOUND: ", ''))
            if line.startswith("GEN:"):
                last_gen = int((line.split())[1])
            if line.startswith("chasing cells: "):
                cells = [int(i) for i in (line.replace("chasing cells: ",'')).split()]
                if len(cells)>1:
                    df = pd.DataFrame()
                    for no in cells:
                        x = math.floor(no/(1/cell_width*area))
                        y = no%(round(1/cell_width*area))
                        df = pd.concat([df,pd.DataFrame([[x,y]])])
                    df.columns = ['x','y']
                    df.index = range(len(df))
                    df,cluster = DBSCAN_find(df,area,cell_width)
                    #print(np.array(df['x']))
                    dft.at[last_gen+10,'chasing cell xs']=np.array(df['x'])#.tobytes()
                    dft.at[last_gen+10,'chasing cell ys'] = np.array(df['y'])#.tobytes()
                    dft.at[last_gen+10,'chasing cell cluster'] = np.array(df['traversal'])#.tobytes()
        print(last_gen)
    
    # plot the comparison graph
    #print([isinstance(i,np.ndarray) for i in dft['chasing cell xs']])
    dft_plot = dft.loc[[isinstance(i,np.ndarray) for i in dft['chasing cell xs']]]
    print(dft_plot)
    for index,row in dft.iterrows():
        fig,(ax1,ax2) = plt.subplots(ncols=2, dpi=400,figsize=(17,8.5))
        # ax1.xaxis.set_visible(False)
        # ax1.yaxis.set_visible(False)
        ax1.set_title("generation {}".format(index-10),fontsize=60)
        ax1.set_ylim(0, area)
        ax1.set_xlim(0, area)
        ax1.scatter(row['xs'],row['ys'], color=row['cs'], s=10)
        ax1.set_xticks(np.linspace(0,area,int(1/cell_width*area+1)))
        ax1.set_yticks(np.linspace(0,area,int(1/cell_width*area+1)))
        ax1.set_xticklabels(['0']+['']*9+['0.5']+['']*9+['1.0']+['']*7+['1.4'],fontsize=50)
        ax1.set_yticklabels(['0']+['']*9+['0.5']+['']*9+['1.0']+['']*7+['1.4'],fontsize=50)
        
        if (isinstance(dft.loc[index,'chasing cell xs'],np.ndarray)):
            ax2.scatter(cell_width*np.array(row['chasing cell xs']+0.5),cell_width*np.array(row['chasing cell ys']+0.5),c=row['chasing cell cluster'])
            ax2.set_xlim([0,area])
            ax2.set_ylim([0,area])
            ax2.set_xticks(np.linspace(0,area,int(1/cell_width*area+1)))
            ax2.set_yticks(np.linspace(0,area,int(1/cell_width*area+1)))
            ax2.set_xticklabels(['0']+['']*9+['0.5']+['']*9+['1.0']+['']*7+['1.4'],fontsize=50)
            ax2.set_yticklabels(['0']+['']*9+['0.5']+['']*9+['1.0']+['']*7+['1.4'],fontsize=50)
            ax2.grid(alpha=0.5) # alpha controls the width of grid lines

        plt.show()
    
    
def main():
    re = []
    for root,dirs,files in os.walk('area/0-0results/'):#low density growth rate/0-3results/
        for file in files:
            print(file)
            #cell_width = float(file.replace('.out',''))
            cell_width = 0.05#0.05/np.sqrt(2)
            with open(root+file) as f:
                clusters = []
                pop_sizes = []
                pop_sizes_per_cluster = []
                lines = f.readlines()
                for line in lines:
                    if line.startswith("SIM BOUND:"):
                        area = float(line.replace("SIM BOUND: ", ''))
                    if ' ' not in line and line.strip()!='':
                        line = line.strip()
                        if len(line) <= 6 and line!='1':
                            pop_size = int(line)
                    if line.startswith('GEN:'):
                        line = line.strip()
                        con = line.split()
                        this_gen = int(con[1])
                    if line.startswith("chasing cells: ") and this_gen>=80:
                        cells = [int(i) for i in (line.replace("chasing cells: ",'')).split()]
                        if len(cells)>3:
                            df = pd.DataFrame()
                            for no in cells:
                                x = math.floor(no/(1/cell_width*area))
                                y = no%(round(1/cell_width*area))
                                df = pd.concat([df,pd.DataFrame([[x,y]])])
                            df.columns = ['x','y']
                            df = df.reset_index()
                            df,cluster = DBSCAN_find(df,area,cell_width)
                            if (cluster!=0):
                                clusters = clusters + [cluster]
                                pop_sizes = pop_sizes + [pop_size]
                                pop_sizes_per_cluster = pop_sizes_per_cluster + [pop_size/cluster]*cluster
                re.append(clusters)
                re.append(pop_sizes)
                re.append(pop_sizes_per_cluster)
                print(np.mean(clusters))
                print(np.mean(pop_sizes))
    re = (pd.DataFrame(re)).transpose()     
    re.to_csv('chasing_number_cha.csv')
if __name__ == "__main__":
    # df = pd.DataFrame([[1,2],[1,3],[1,4],[14,5],[15,5],[15,6],[8,1],[20,1]],columns=['x','y'])
    # df,n=DBSCAN_find(df)
    # print(df)
    
    main()
   
    #visualize('area/test_movie')#'area/test_movie' 'area/overlapping_fecundity_movie'
    
    
    
    