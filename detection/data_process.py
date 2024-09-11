# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:49:29 2024

@author: Xinyue Zhang
"""
import pandas as pd
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from pandas.api.types import CategoricalDtype
import scipy.stats as st


plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20,
                     'font.family': "Times New Roman",
                     "axes.titlesize": 20,     
                     "axes.labelsize": 20,
                     "savefig.pad_inches": 0,
                     "legend.fontsize": 10,
                     "axes.labelpad": 0.1,
                     "axes.linewidth":1.2,
                     "text.color": "black",
                     "axes.labelcolor": "black",
                     "xtick.color": "black",
                     "ytick.color": "black",
                     "xtick.major.size": 5.0,
                     "xtick.major.width":1.5,
                     "xtick.major.pad":0.1,
                     "ytick.major.size": 5.0,
                     "ytick.major.width":1.5,
                     "ytick.major.pad":0.1,
                     "legend.framealpha": 0.2,
                     "patch.facecolor": "white"})


def summary_raw(filename,feature_list):
    '''
    read the raw data of spatial results with 20 repeats, and write the summary
    to the sheet "summary"

    Returns
    -------
    None.

    '''
    data = pd.read_excel(filename,index_col=False)
    #data = data.loc[data['somatic_fitness']==0.9]
    df = pd.DataFrame()
    pop_size=0
    weighted_I=0
    var=0
    var_I=0
    I_SD=0
    groups = data.groupby(by=feature_list)#'drive_conversion','growth curve','embryo_resistance'
    for name,group in groups:
        no = float(len(group))
        suppressed_without_chasing = sum(group['suppressed_without_chasing'])
        suppressed_with_chasing = sum(group['suppressed_with_chasing'])
        suppressed_with_chasing_rate = suppressed_with_chasing/no
        suppression = suppressed_without_chasing+suppressed_with_chasing
        drive_loss_with_chasing = len(group.loc[(group['drive_loss']==1) & (group['pop_persist']-group['gen_chasing_start']>5)])
        long_term_chasing = len(group.loc[(group['chasing_or_equilibrium']==1)])
        #avg_fertile_females = sum(group['avg_fertile_female']*group['duration_of_chasing'])/sum(group['duration_of_chasing'])
        avg_fertile_females = np.mean(group['avg_fertile_female'])
        #pop_size = np.mean(group['pop size'])
        # process the chasing gen
        suppressed_with_chasing = 0
        chasing_gen = 0
        pop_size = 0
        pop_size_gen = 0
        I_w = 0
        I_w_list = []
        chasing_gen_threshold = 80 # from this generation, regrad it as chasing generation
        for index,row in group.iterrows():
            if (row['gen_suppressed']>=chasing_gen_threshold*row['generation time']) and (row['gen_suppressed']!=10000):
                suppressed_with_chasing = suppressed_with_chasing + 1
                chasing_gen = chasing_gen + row['gen_suppressed']/row['generation time']-chasing_gen_threshold
            if (row['gen_suppressed']>=80*row['generation time']) and (row['gen_suppressed']!=10000):
                pop_size = pop_size+row['pop_size']*(row['gen_suppressed']-chasing_gen_threshold)
                pop_size_gen = pop_size_gen + row['gen_suppressed']-chasing_gen_threshold
                I_w = I_w + row['weighted I']*row['pop_size']*(row['gen_suppressed']-chasing_gen_threshold)
                I_w_list = I_w_list + [row['weighted I']]
            if (row['chasing_or_equilibrium']==1):
                chasing_gen = chasing_gen + 375-chasing_gen_threshold
            if (row['drive_loss']==1) and (row['pop_persist']>=chasing_gen_threshold*row['generation time']):
                chasing_gen = chasing_gen + row['pop_persist']/row['generation time']-chasing_gen_threshold
        #chasing_gen = sum(group['duration_of_chasing'])
        print(name)
        print(chasing_gen)
        print(suppressed_with_chasing)
        if (chasing_gen!=0):
            suppression_rate_when_chasing = suppressed_with_chasing/chasing_gen
            suppression_confidence_up = suppressed_with_chasing/chasing_gen + 1.96*np.sqrt(suppression_rate_when_chasing*(1-suppression_rate_when_chasing)/chasing_gen)
            suppression_confidence_down = suppressed_with_chasing/chasing_gen - 1.96*np.sqrt(suppression_rate_when_chasing*(1-suppression_rate_when_chasing)/chasing_gen)
            driveloss_rate_when_chasing = drive_loss_with_chasing/chasing_gen
            driveloss_confidence_up = drive_loss_with_chasing/chasing_gen + 1.96*np.sqrt(driveloss_rate_when_chasing*(1-driveloss_rate_when_chasing)/chasing_gen)
            driveloss_confidence_down = drive_loss_with_chasing/chasing_gen - 1.96*np.sqrt(driveloss_rate_when_chasing*(1-driveloss_rate_when_chasing)/chasing_gen)
        else:
            suppression_rate_when_chasing = -1
            suppression_confidence_up = -1
            suppression_confidence_down = -1
            driveloss_rate_when_chasing = -1
            driveloss_confidence_up = -1
            driveloss_confidence_down = -1
        driveloss_without_chasing = sum(group['drive_loss'])-drive_loss_with_chasing
        coe = group.loc[group['chasing_or_equilibrium']==1]#group.loc[group['gen_chasing_start']!=0]#group.loc[group['chasing_or_equilibrium']==1]
        if (len(coe)!=0):
            I = np.mean(coe['I'])
            I_SD = np.std(coe['I'])
            I_CI = 1.96*np.std(coe['I'])/np.sqrt(len(coe))
            #weighted_I = np.mean(coe['weighted I'])
            weighted_I_SD = np.std(coe['weighted I'])
            #weighted_I_CI = 1.96*np.std(coe['weighted I'])/np.sqrt(len(coe))
            I_w = I_w + sum(coe['weighted I']*coe['pop_size']*(375-chasing_gen_threshold))*np.mean(group['generation time'])
            I_w_list = I_w_list + list(coe['weighted I'])
            
            var = np.mean(coe['var_nni_across_time'])
            var_I = np.mean(coe['var_I_across_time'])
            pop_size = pop_size + sum(coe['pop_size'])*(375-chasing_gen_threshold)*np.mean(group['generation time'])
            pop_size_gen = pop_size_gen + len(coe)*(375-chasing_gen_threshold)
        else:
            I = 0
            I_SD = 0
            I_CI=0
            #weighted_I = 0
            var = 0
            var_I = 0
            weighted_I_SD = 0
            #weighted_I_CI = 0
        if (pop_size!=0):
            weighted_I = I_w/pop_size
            weighted_I_CI = 1.96*np.std(I_w_list)/np.sqrt(len(I_w_list))
            pop_size = pop_size/pop_size_gen
        if isinstance(name, tuple):
            df = pd.concat([df,pd.DataFrame([list(name)+[suppressed_without_chasing/no,suppressed_with_chasing_rate,suppression/no,long_term_chasing/no,
                                         suppression_rate_when_chasing,suppression_confidence_up,suppression_confidence_down,
                                         I,I_SD,weighted_I,var,var_I,drive_loss_with_chasing/no,driveloss_rate_when_chasing,driveloss_confidence_up,driveloss_confidence_down,
                                         driveloss_without_chasing/no,1-suppressed_without_chasing/no-driveloss_without_chasing/no,chasing_gen,avg_fertile_females,I_CI,pop_size,weighted_I_SD,weighted_I_CI]])])
        else:
            df = pd.concat([df,pd.DataFrame([[name,suppressed_without_chasing/no,suppressed_with_chasing_rate,suppression/no,long_term_chasing/no,
                                         suppression_rate_when_chasing,suppression_confidence_up,suppression_confidence_down,
                                         I,I_SD,weighted_I,var,var_I,drive_loss_with_chasing/no,driveloss_rate_when_chasing,driveloss_confidence_up,driveloss_confidence_down,
                                         driveloss_without_chasing/no,1-suppressed_without_chasing/no-driveloss_without_chasing/no,chasing_gen,avg_fertile_females,I_CI,pop_size,weighted_I_SD,weighted_I_CI]])])
    df.columns=feature_list+['suppressed without chasing rate','suppressed with chasing rate','suppression rate','long term chasing or equilibrium rate',
                             'suppression rate when chasing','suppression rate up confidence','suppression rate down confidence',
                'I','I_SD','weighted I','var nni across time','var I across time','drive loss with chasing rate','drive loss rate when chasing','driveloss rate up confidence','driveloss rate down confidence',
                'drive loss rate without chasing','chasing rate','p sample size','average fertile females','I_CI','pop size','weighted_I_SD','weighted_I_CI']
    # write the summary data to summary sheet
    book = load_workbook(filename)
    with pd.ExcelWriter(filename) as writer:
        writer.book = book
        df.to_excel(writer,sheet_name='summary',index=False)
        
    return df
        
def heatmap(df,feature1,feature2,y_feature):
    x_dim = len(set(df[feature1]))
    y_dim = len(set(df[feature2]))
    df = df.sort_values(by=[feature1,feature2])
    y = np.array(df[y_feature]).reshape(x_dim,y_dim)
    index=list(set(df[feature1]))
    index.sort()
    columns=list(set(df[feature2]))
    columns.sort()
    heatdf = pd.DataFrame(y,index=index,columns=columns)
    
    fig,ax = plt.subplots(dpi=300)
    ax = sns.heatmap(heatdf,cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(y_feature)
    return

def line_plot(df,x_feature,label_feature,y_feature,title):
    fig,ax = plt.subplots(dpi=300)
    if(y_feature=="I"):
        df = df.loc[df['I']!=0]
        #ax.set_ylim([0.65,1.0])
    if(y_feature=="suppression rate when chasing"):
        df = df.loc[df['suppression rate when chasing']!=-1]
    if(label_feature==''):
        df = df.sort_values(by=x_feature)
        plt.plot(df[x_feature],df[y_feature])
        if(y_feature=="suppression rate when chasing"):
            plt.fill_between(df[x_feature], df['suppression rate up confidence'], df['suppression rate down confidence'],alpha=0.4)
        if(y_feature=='drive loss rate when chasing'):
            plt.fill_between(df[x_feature], df['driveloss rate up confidence'], df['driveloss rate down confidence'],alpha=0.4)
    else:
        groups = df.groupby(by=label_feature)
        for name,group in groups:
            group = group.sort_values(by=x_feature)
            #print(group[x_feature])
            plt.plot(group[x_feature],group[y_feature],label=label_feature+" "+str(name))
            if(y_feature=="suppression rate when chasing"):
                plt.fill_between(group[x_feature], group['suppression rate up confidence'], group['suppression rate down confidence'],alpha=0.4)#,facecolor='salmon'
            if(y_feature=='drive loss rate when chasing'):
                plt.fill_between(group[x_feature], group['driveloss rate up confidence'], group['driveloss rate down confidence'],alpha=0.4)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(title)
    plt.legend()
    plt.show()
    
def outcome_plot(df,x_feature):
    df = df.sort_values(by=x_feature)
    fig,ax = plt.subplots(figsize=(6,6), dpi=300)
    fig.set_facecolor('white')
    plt.gca().patch.set_facecolor('white')
    plt.plot(df[x_feature],df['suppressed without chasing rate'],label='suppressed without chasing rate')
    #plt.plot(df[x_feature],df['suppressed with chasing rate'],label='suppressed with chasing rate')
    plt.plot(df[x_feature],df['chasing rate'],label='chasing rate')
    #plt.plot(df[x_feature],df['long term chasing or equilibrium rate'],label='long term chasing or equilibrium rate')
    #plt.plot(df[x_feature],df['drive loss with chasing rate'],label='drive loss with chasing rate')
    plt.plot(df[x_feature],df['drive loss rate without chasing'],label='drive loss rate without chasing')
    #plt.plot(df[x_feature],df['suppression rate'],label='suppression rate')
    ax.set_xlabel(x_feature)
    ax.set_ylabel('rate of outcome')
    plt.legend(fontsize=15)
    plt.show()
    
    df1 = df.loc[df["suppression rate when chasing"]!=-1]
    fig,ax = plt.subplots(figsize=(6,6),dpi=300)
    fig.set_facecolor('white')
    plt.gca().patch.set_facecolor('white')
    plt.plot(df1[x_feature],df1['suppression rate when chasing'],label='suppression rate when chasing')
    plt.fill_between(df1[x_feature], df1['suppression rate up confidence'], df1['suppression rate down confidence'],alpha=0.4)
    for i in range(len(df1)):
        plt.annotate(int((df1.iloc[i])['p sample size']), xy=((df1.iloc[i])[x_feature],(df1.iloc[i])['suppression rate when chasing']),fontsize=14)
    df1 = df.loc[df['drive loss rate when chasing']!=-1]
    plt.plot(df1[x_feature],df1['drive loss rate when chasing'],label='drive loss rate when chasing')
    plt.fill_between(df1[x_feature], df1['driveloss rate up confidence'], df1['driveloss rate down confidence'],alpha=0.4)
    ax.set_xlabel(x_feature)
    ax.set_ylabel('rate of different chasing outcome')
    plt.legend(fontsize=15)
    plt.show()

    fig,ax = plt.subplots(dpi=300)
    df = df.loc[(df['I']!=0) & (df['I']!=-1)]
    plt.plot(df[x_feature],df['I'],label='I')
    #plt.plot(df[x_feature],df['weighted I'],label='weighted I')
    ax.set_xlabel(x_feature)
    ax.set_ylabel('I')
    #plt.ylim(0.6,1.1)
    plt.legend(fontsize=15)
    plt.show()
    
    fig,ax = plt.subplots(dpi=300)
    plt.plot(df[x_feature],df['var nni across time'],label='var nni across time')
    ax.set_xlabel(x_feature)
    ax.set_ylabel('var nni across time')
    #plt.ylim(0.6,1.1)
    plt.show()
    
    fig,ax = plt.subplots(dpi=300)
    plt.plot(df[x_feature],df['var I across time'],label='var I across time')
    ax.set_xlabel(x_feature)
    ax.set_ylabel('var I across time')
    #plt.ylim(0.6,1.1)
    plt.show()
    
def errorbar_plot(df,x_feature):
    fig,ax = plt.subplots(figsize=(6,6),dpi=300)
    df = df.sort_values(by=x_feature)
    plt.errorbar(df[x_feature], df['suppression rate when chasing'], yerr=df['suppression rate up confidence']-df['suppression rate when chasing'],
                 color='black',capsize=3,linestyle="None",marker='s',markersize=7)
    ax.set_ylabel('suppression rate when chasing')
    
    
def fit_weighted_I():
    '''
    to fit all the simulation sample points with variant germline resistance

    Returns
    -------
    None.

    '''
    file = 'germline resistance/0-14_result.xlsx'
    df = pd.read_excel(file,sheet_name='0-14_result',index_col=False)
    x = []
    y = []
    chasing_gen_threshold = 150
    for index,row in df.iterrows():
        if (row['gen_suppressed']>=chasing_gen_threshold*row['generation time']):
            if row['drive_loss']==1:
                if (row['pop_persist']>=chasing_gen_threshold*row['generation time']) and (row['pop_persist']!=10000):
                    x = x+[row['germline_resistance']]
                    y = y+[row['weighted I']]
            else:
                x = x+[row['germline_resistance']]
                y = y+[row['weighted I']]
        if row['chasing_or_equilibrium']==1:
            x = x+[row['germline_resistance']]
            y = y+[row['weighted I']]
    theta = st.linregress(x,y)
    plt.scatter(x,y)
    plt.plot(x,theta[0]*np.array(x)+theta[1])
    print(theta)
    
        
    
def main():
    filename = 'conversion/0-12_result.xlsx'#'germline resistance/0-14_result.xlsx'#'curves/0-1_result_testI.xlsx'#'lifespan/0-2_result_all.xlsx'#'embryo resistance/0-6_result.xlsx'#'area/0-0_result_I.xlsx'#'conversion/0-12_result.xlsx'#'population density/0-4_result_ov.xlsx'#'resource/0-9_result.xlsx'#'migration distance/0-8_result.xlsx'#'population density/0-4_result_ov.xlsx'#'low density growth rate/0-3_result.xlsx'#'competition distance/0-5_result3_convex.xlsx'
    feature_list = ['drive_conversion','somatic_fitness']#['germline_resistance']#['growth curve']#['model']#['sim bound','somatic_fitness']#['model','population density']#['migration rate']#["growth curve","drive_conversion",'embryo_resistance']#["population density","model"]#['model','remate chance']#['low density growth rate']#["sim bound","drive_conversion"]#["competition distance","growth curve"]
    df = summary_raw(filename,feature_list)
    df = pd.read_excel(filename,sheet_name='summary',index_col=False)
    # curve_order = CategoricalDtype(['concave', 'linear', 'convex'], ordered=True)
    # df['growth curve'] = df['growth curve'].astype(curve_order)
    y_feature = 'average fertile females'#'drive loss rate when chasing'#'var nni across time'#'suppression rate when chasing' 'I' 'suppressed without chasing rate'
    
    #errorbar_plot(df, 'growth curve')
    #df = df.loc[(df['drive_conversion']==0.94)]
    #df = df.loc[(df['model']=='overlapping_fecundity') | (df['model']=='discrete_fecundity')]
    #line_plot(df, feature_list[0], feature_list[1],y_feature,'')
    
    #heatmap(df,'drive conversion','sim bound',y_feature)
    
    #df = df.loc[(df['model']=='overlapping_viability')]# & (df['sim bound']!=1.0)
    outcome_plot(df, feature_list[0])
    
    # # student's t test
    # df = df.loc[(df['embryo_resistance']==0.4) & (df['I']!=0)]
    # g1 = (df.loc[df['growth curve']=="concave"]).sort_values(by=['drive_conversion','embryo_resistance'])
    # g2 = (df.loc[df['growth curve']=="convex"]).sort_values(by=['drive_conversion','embryo_resistance'])
    # print(ttest_rel(g1[y_feature],g2[y_feature]))

if __name__ == "__main__":
    #main()
    fit_weighted_I()