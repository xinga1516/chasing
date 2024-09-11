# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:17:55 2023
call for run slim script with specified parameters and output the video and 
gc, wt frequency changing graph.

@author: xinyue Zhang
"""
from argparse import ArgumentParser
import itertools
import subprocess
import sys
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import style
from argparse import ArgumentParser
import ffmpeg
from scipy.stats import f as f_test
import numpy as np
from scipy.special import comb
import re


SLIM = "slim"
# plt.style.use('classic')
plt.rcParams.update({'font.size': 25,
                      'font.family': 'Times New Roman',
                      "axes.titlesize": 25,
                      "axes.labelsize": 25,
                      "legend.fontsize": 15,
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

def animate(i, subplot, xs, ys, cs, individual_dot_size):
    """
    Update the animation each frame with the data for that generation.
    """
    print("   Drawing generation {}/{} \r".format(i, len(xs)), end="", flush=True)
    subplot.clear()
    subplot.set_title("Month {}".format(i),fontsize=30)
    subplot.set_ylim(0, 1)
    subplot.set_xlim(0, 1)
    subplot.scatter(xs[i], ys[i], color=cs[i], s=individual_dot_size)


def video(source):
    """
    Program flow:
    1. Open file with position and color data output from slim.
    2. Parse data into lists for x coords, y coords, and colors.
    3. Make matplotlib plot with an animation function that uses the data.
    4. Export the plot as an mp4.
    """
    # set video parameters
    fps = 10 # Frames (generations) per sec for the animation.
    dimensions = 1080 # Dimensions of the animation. Default 1080 (1080*1080 animation)
    individual_dot_size = 10 # Dot size of individuals. Lower this for high pops, or things will be messy.
    dark = True # the light of background
    
    # Read and parse data from the file into a list of
    # x coords, y coords, and colors for each generation:
    print("Reading {}...".format(source))
    with open(source, 'r') as f:
        data = f.read()
    data = data[1:].split('G\n')
    for gen in range(len(data)):
        data[gen] = data[gen][1:-1].split('\n')
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

    print("Generating animation...")
    # Create and configure a matplotlib figure, then call the animation function using the data:
    if dark:
        style.use('dark_background')
    else:
        style.use('default')
    fig = plt.figure(figsize=(dimensions/100, dimensions/100))
    subplot = fig.add_subplot(1,1,1)
    subplot.xaxis.set_visible(False)
    subplot.yaxis.set_visible(False)
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=1000/fps, fargs=(subplot, xs, ys, cs,individual_dot_size))
    fig.tight_layout(pad=2.5)
    #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    # Save the animation, and then display it in a window.
    anim.save('{}.mp4'.format(source), writer=animation.writers['ffmpeg'](fps=fps)) # , writer=animation.writers['ffmpeg'](fps=fps)
    #anim.save('{}.gif'.format(source))
    # plt.show()  # Not particularly useful to show the plot after outputting the movie, but might have a purpose?
    print("Animation written to {}.mp4.".format(source))
    os.remove(source)
    return 


        
        
def relation_plot(df):
    '''
    plot the changing allele frequency, gc, and empty cell number

    Parameters
    ----------
    df : TYPE dataframe
        information containing generation and allele frequency and so on.

    Returns
    -------
    fig : TYPE figure
        DESCRIPTION.

    '''
    fig,ax = plt.subplots(dpi=400,figsize=(11,9.5))
    fig.set_facecolor('white')
    plt.gca().patch.set_facecolor('white')
    # fig.subplots_adjust(right=1.5)
# ax.plot(df['gen'],df['drive frequency'],label='drive frequency',color='red')
    ax.plot(df['gen'],df['r2 frequency'],label='r2 frequency',color='blue')

    ax2 = ax.twinx()
    ax2.plot(df['gen'],df['nondrive gc'],label='nondrive gc',color='green')
#ax2.plot(df['gen'],df['overall gc'],label='overall gc',color='grey')

    ax3 = ax.twinx()
    ax3.plot(df['gen'],df['empty space number'],label='empty space number',color='red')
    ax3.spines['right'].set_position(('data',500))

    ax.set_xlabel("generation")
    ax.set_ylabel("allele frequency")
    ax2.set_ylabel("gc coefficient")
    ax3.set_ylabel("empty space number")
# ax2.set_ylim(0,0.8)
# ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    fig.legend(fontsize=15)
    plt.tight_layout()
    fig = ax.get_figure()
    return fig

def parse_slim(slim_out):
    """
    read the slim output and got the wt, dr, and gc data.
    the spatial record file contains the data of 20 simulations.

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.

    Returns dataframe contains wt, dr, gc data.
    -------
    None.

    """
    suppressed_with_chasing = 0
    suppressed_without_chasing = 0
    chasing_or_equilibrium = 0
    long_term_chasing = 0
    equilibrium = 0
    gen_suppressed = 10000
    gen_chasing_start = 0
    duration_of_chasing = 0
    drive_loss = 0
    pop_persist = 10000
    avg_fertile_female = 1000000
    is_persist = 0
    number_of_fertile_females = []
    generation_time = 1
    area = 1
    
    # chasing dynamic test variables
    I = -1
    I_w = -1
    var_nni_across_time = -1
    var_I_across_time = -1
    
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    slim_out = slim_out.split('\n')
    for line in slim_out:
        # collect the population dynamics
        if ' ' not in line and line!='':
            line = line.strip()
            if len(line) <= 6 and line!='1':
                pop_size = int(line)
        if line.startswith('GEN:'):
            line1 = line.strip()
            line1 = line1.split(' ')
            df1 = pd.concat([df1,pd.DataFrame([[int(line1[1]),float(line1[3]),pop_size]])])#,float(line1[5])
        if line.startswith('WT_AL'):
            line2 = line.strip()
            line2 = line2.split(' ')
            df2 = pd.concat([df2,pd.DataFrame([[int(line2[1]),int(line2[2]),int(line2[3]),float(line2[4]),float(line2[6]),float(line2[8])]])])
        
        # direct outcome collect
        if line.startswith('POP_PERSISTS'):
            drive_loss = 1
            pop_persist = int(line.replace('POP_PERSISTS:: ',''))
        if line.startswith('CHASING START'):
            gen_chasing_start = int(line.replace('CHASING START:: ',''))
        if line.startswith('SUPPRESSED WITHOUT CHASING'):
            suppressed_without_chasing = 1
            gen_suppressed = int(line.replace('SUPPRESSED WITHOUT CHASING:: ',''))
        if line.startswith('SUPPRESSED WITH CHASING'):
            suppressed_with_chasing = 1
            gen_suppressed = int(line.replace('SUPPRESSED WITH CHASING:: ',''))
        if line.startswith('ENDING_AFTER_375'):
            is_persist = 1
        if line.startswith("FERTILE_FEMALES:: "):
            spaced_line = line.split()
            number_of_fertile_females.append(int(spaced_line[1]))
        if line.startswith("GENERATION TIME:"):
            generation_time = float(line.replace('GENERATION TIME: ',''))
        if line.startswith("SIM BOUND:"):
            area = float(line.replace('SIM BOUND: ',''))
        
    if (len(df1)!=0):
        df1.columns = ['gen','drive frequency','pop size'] #,'r2 frequency'
    if (len(df2)!=0):
        df2.columns = ['wt allele number','gen','pop size','rate drive carriers','average nni','var nni']
        df = pd.merge(df1,df2,how='right')# how='right',on=['gen','pop size']
        '''parse the outcome of drive'''
        record = df
        #standard_var = (0.26136/np.sqrt(record['pop size']))**2
        # weighted chasing aggregation
        record = record.loc[record['average nni']!= 0]
        #record['average nni'] = [float(i) for i in record['average nni']]
        # the random distribution var and mean 
        standard_mean = 0.5/np.sqrt(record['pop size'])*area
        # what we need is the I during chasing
        if (record.iloc[-1])['gen']>80*generation_time:
            chasing_record = record.loc[record['gen']>80*generation_time]  
            chasing_standard_mean = 0.5/np.sqrt(chasing_record['pop size'])*area
            I = np.mean(chasing_record['average nni']/chasing_standard_mean)
            I_w = sum(chasing_record['pop size']*chasing_record['average nni']/chasing_standard_mean)/sum(chasing_record['pop size'])
            var_nni_across_time = np.var(chasing_record['average nni'])
            var_I_across_time = np.var(chasing_record['average nni']/chasing_standard_mean)
        else:
            I = np.mean(record['average nni']/standard_mean)
            I_w = sum(record['pop size']*record['average nni']/standard_mean)/sum(record['pop size'])
            var_nni_across_time = np.var(record['average nni'])
            var_I_across_time = np.var(record['average nni']/standard_mean)
            
        # # plot the change of I with generations
        # plt.plot(record['gen'],record['average nni']/standard_mean)
        # plt.show()
    else:
        # non-wt frequency don't reach 20% 
        df = df1
         
    if(len(df)>80*generation_time):
        pop_size = np.mean(df['pop size'].iloc[int(80*generation_time):])
    else:
        pop_size = np.mean(df['pop size'].iloc[10:])
     
        
    if is_persist == 1:
        chasing_or_equilibrium = 1
        if gen_chasing_start == 0:
            equilibrium = 1
        else:
            long_term_chasing = 1
    
    if (gen_chasing_start!=0):
        avg_fertile_female = np.mean(number_of_fertile_females[gen_chasing_start:])
        duration_of_chasing = df.iloc[-1]['gen']+1-gen_chasing_start
        # if chasing gen is too short, then see it as no chasing
        if duration_of_chasing<5:
            if suppressed_with_chasing==1:
                suppressed_without_chasing=1
                suppressed_with_chasing=0
                gen_chasing_start = 0
                duration_of_chasing=0
                avg_fertile_female = np.mean(number_of_fertile_females)
            if drive_loss==1:
                gen_chasing_start = 0
                duration_of_chasing = 0
                avg_fertile_female = np.mean(number_of_fertile_females)
    else:
        avg_fertile_female = np.mean(number_of_fertile_females)
    
    
    return [generation_time,suppressed_without_chasing,suppressed_with_chasing,gen_suppressed,gen_chasing_start,
            duration_of_chasing,chasing_or_equilibrium,I,I_w,var_nni_across_time,var_I_across_time,
            long_term_chasing,equilibrium,avg_fertile_female,drive_loss,pop_persist,pop_size]




def run_slim(command_line_args,filename):
    """
    Runs SLiM using subprocess.
    Args:
        command_line_args: list; a list of command line arguments.
    return: The entire SLiM output as a string.
    """
    slim = subprocess.Popen(command_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
    out, err = slim.communicate()
    # with open(filename,'w') as f:
    #     f.write(out)
    print(out)
    return out


def configure_slim_command_line(args_dict):
    """
    Sets up a list of command line arguments for running SLiM.
    Args:
        args_dict: a dictionary of arg parser arguments.
    Return
        clargs: A formated list of the arguments.
    """
# We're running SLiM, so the first arg is simple:
    clargs = SLIM + " "
# The filename of the source file must be the last argument:
    source = args_dict.pop("source")
# Add each argument from arg parser to the command line arguemnts for SLiM:
    for arg in args_dict:
        if isinstance(args_dict[arg], bool):
            clargs += f"-d {arg}={'T' if args_dict[arg] else 'F'} " #字符串前加f可以解析大括号中的表达式
        elif isinstance(args_dict[arg],str):
            clargs += f"-d {arg}='{args_dict[arg]}' "
        else:
            clargs += f"-d {arg}={args_dict[arg]} "
    #print(clargs)
# Add the source file, and return the string split into a list.
    clargs += source
    return clargs.split()

def main():
    parser = ArgumentParser()
    parser.add_argument('-src', '--source', default="discrete_fecundity.slim", type=str,
    help=r"SLiM file to be run. Default 'generic_spatial.slim'") #字符串前加r可以去掉有特殊意义的字符（忽略转义）
    parser.add_argument('-o', '--outputfile', default="0-0results.csv", type=str,
    help=r"output file.")
    parser.add_argument('-header', '--print_header', action='store_true', default=False,
    help='If this is set, python prints a header for a csv file.')
    #parser.add_argument('-migrationvalue','--SPEED', default=0.05*math.sqrt(2/math.pi), type=float,help='value')
    parser.add_argument('-side','--SIM_BOUND',default=1.0,type=float)
    parser.add_argument('-lowdensity','--GROWTH_AT_ZERO_DENSITY', default=6.0, type=float)
    parser.add_argument('-popdensity','--EFFECTIVE_POPULATION_DENSITY', default=20000, type=int)
    parser.add_argument('-video','--OUTPUT_SPATIAL_DATA',default=False,type=bool)
    parser.add_argument('-embryo','--EMBRYO_RESISTANCE_CUT_RATE_F_INPUT', default=0.1, type=float)
    parser.add_argument('-germline','--WHOLE_GERMLINE_RESISTANCE_CUT_RATE', default=0.1, type=float)
    parser.add_argument('-fitness','--SOMATIC_FITNESS_MULTIPLIER_F', default=0.9, type=float)
    parser.add_argument('-conversion','--DRIVE_CONVERSION', default=0.94, type=float)
    parser.add_argument('-growthcurve','--DENSITY_GROWTH_CURVE',default='concave',type=str)
    parser.add_argument("-migdis",'--AVERAGE_DISTANCE',default=0.05,type=float)
    parser.add_argument("-dendis",'--DENSITY_INTERACTION_DISTANCE',default=0.01,type=float)
    parser.add_argument('-r','--runs', default=1, type=int) # without changing normally
    parser.add_argument('-gl','--test_genetic_load',default=0.0,type=float)
    parser.add_argument('-remate','--REMATE_CHANCE',default=1.0,type=float)
    parser.add_argument('-lifespan','--WEEKS_OF_GENERATION',default=4,type=int)
    args_dict = vars(parser.parse_args()) #vars()转换成字典
    # print(args_dict)
    model = (args_dict['source']).replace(".slim",'')
    output_file = args_dict.pop("outputfile")
    germline_resistance = args_dict['WHOLE_GERMLINE_RESISTANCE_CUT_RATE']
    embryo_resistance = args_dict['EMBRYO_RESISTANCE_CUT_RATE_F_INPUT']
    drive_conversion = args_dict['DRIVE_CONVERSION']
    somatic_fitness = args_dict['SOMATIC_FITNESS_MULTIPLIER_F']
    low_density_growth_rate = args_dict['GROWTH_AT_ZERO_DENSITY']
    migdis = args_dict['AVERAGE_DISTANCE']
    args_dict['AVERAGE_DISTANCE'] = args_dict['AVERAGE_DISTANCE']/math.sqrt(2/math.pi)
    genetic_load = args_dict['test_genetic_load']
    competition_dis = args_dict['DENSITY_INTERACTION_DISTANCE']
    side = args_dict['SIM_BOUND']
    pop_density = args_dict['EFFECTIVE_POPULATION_DENSITY']
    remate = args_dict['REMATE_CHANCE']
    lifespan = args_dict['WEEKS_OF_GENERATION']
    runs = int(args_dict.pop("runs"))
# The '-header' argument prints a header for the output. This can
# help generate a nice CSV by adding this argument to the first SLiM run:
    f = open(output_file, 'a')
    if args_dict.pop("print_header",None):
        f.write("model,migration rate,low density growth rate,remate chance,lifespan,drive_conversion,somatic_fitness,embryo_resistance,growth curve,sim bound,population density,competition distance,generation time,suppressed_without_chasing,suppressed_with_chasing,gen_suppressed,gen_chasing_start,\
                duration_of_chasing,chasing_or_equilibrium,I,weighted I,var_nni_across_time,var_I_across_time,long_term_chasing,equilibrium,avg_fertile_female,drive_loss,pop_persist,pop_size")
        f.write('\n')
    f.close()
# Next, assemble the command line arguments in the way we want to for SLiM:
# Run the file with the desired arguments.
    clargs = configure_slim_command_line(args_dict)
    for i in range(runs):
        file_name='0-4results/slim_'+str(i)+'c'+str(drive_conversion)+"a"+str(side)
        #file_name="3-0results/gl"+str(genetic_load)#"2-2results/g"+str(germline_resistance)+"e"+str(embryo_resistance)+"c"+str(drive_conversion)+"f"+str(somatic_fitness)+"l"+str(low_density_growth_rate)+"md"+str(migdis)
        #args_dict["OUTPUT_FILE"] = file_name+"_movie"
        #print(clargs)
        #print(args_dict)
        
        out = run_slim(clargs,file_name+".out")
        results = parse_slim(out)
        
        with open(output_file,'a') as fo:
            fo.write(model+","+str(migdis)+","+str(low_density_growth_rate)+","+str(remate)+","+str(lifespan)+","+str(drive_conversion)+","+str(somatic_fitness)+","+str(embryo_resistance)+","+str(germline_resistance)+","+args_dict['DENSITY_GROWTH_CURVE']+','+str(side)+","+str(pop_density)+","+str(competition_dis)+",")
            for i in results:
                fo.write(str(i)+",")
            fo.write("\n")
        # fig = relation_plot(df)
        # fig.savefig("0-0results"+'/'+file_name+'.png')
        # make video
        #video(file_name+'_movie')#file_name+'_movie')
        

if __name__ =="__main__":
    main()
    
    # # 2-2-1 inn vs nnn
    # output_file = "3-0results1.csv"
    # results = pd.DataFrame()
    # folder = '3-0results/'
    # files = os.listdir(folder)
    # for file in files:
    #     if file.endswith(".out"):
    #         with open(folder+file,'r') as f:
    #             content = f.read()
    #             result = parse_slim(content)
    #             parameters = re.split(r'[efc]',(file.replace("slim_0g","")).replace(".out",''))
    #             results = pd.concat([results,pd.DataFrame([[parameters[1],parameters[2],parameters[0],parameters[3],file.replace(".out","_movie.mp4")]+result])])
    # results.columns = ["drive_conversion","somatic_fitness","germline_resistance","embryo_resistance","video","suppressed_without_chasing",
    #                     "suppressed_with_chasing","gen_suppressed","gen_chasing_start","duration_of_chasing","long_term_chasing","equilibrium",
    #                     "avg_fertile_female","drive_loss","pop_persist","avg_normalized_nni","avg_normalized_variance_nni","log_average_vnnn","log_average_annn",
    #                     "log_average_vnni","log_average_anni","z_two_diff_rate","avg_nni_significant","avg_nnn_significant","avg_nnn_toEquilibrium_sig",
    #                     "var_nnn_significant","var_nni_significant"]
    # results.to_csv(output_file,index=False)
    
    # df = pd.read_csv('3-0results1.csv',index_col=False)
    # fig,ax = plt.subplots(figsize=(8,8))
    # feature1 = 'avg_nni_significant'#'nni_significant'
    # feature2 = 'avg_nnn_toEquilibrium_sig'#'nnn_significant'
    # df1 = df.loc[(df[feature1]==1) & (df[feature2]==0)]
    # df2 = df.loc[(df[feature1]==0) & (df[feature2]==1)]
    # df3 = df.loc[(df[feature1]==1) & (df[feature2]==1)]
    # df4 = df.loc[(df[feature1]==0) & (df[feature2]==0)]
    # #print(df)
    # feature3 = 'log_average_vnnn'
    # feature4 = 'log_average_vnni'
    # plt.scatter(df1[feature3],df1[feature4],label='only avg nni significantly different')
    # plt.scatter(df2[feature3],df2[feature4],label='only avg nnn significantly different')
    # plt.scatter(df3[feature3],df3[feature4],label='both avg nni and nnn significantly different')
    # plt.scatter(df4[feature3],df4[feature4],label='no significantly different')
    # ax.set_xlabel(feature3)
    # ax.set_ylabel(feature4)
    # ax.set_title('nni to random, nnn to equilibrium distribution, p=0.01, confidence=0.05')#equilibrium distribution
    # lgd = plt.legend(fontsize=18,scatterpoints=1)
    # # for handle in lgd.legend_handles:
    # #     handle.set_sizes([10])
    # plt.plot()
    # plt.show()
    
    # # 0-2
    # files = os.listdir('0-2results/0-2results')
    # for file in files:
    #     with open('0-2results/0-2results/'+file,'r') as f:
    #         out = f.read()
    #         parse_slim(out, 1)
    
    # # 0-0
    # files = os.listdir('area/0-0results/')
    # results = pd.DataFrame()
    # for file in files:
    #     with open('area/0-0results/'+file,'r') as f:
    #         outs = f.read()
    #         outs = outs.split('// Initial random seed:')
    #         for out in outs[1:]:
    #             result = parse_slim(out)
    #             results = pd.concat([results,pd.DataFrame([[file.replace('.out','')]+result])])
    # results.to_csv("area/chasing_number.csv",index=False)
                
    # # 0-11
    # files = os.listdir('conversion/0-11results/')
    # results = pd.DataFrame()
    # for file in files:
    #     with open('conversion/0-11results/'+file,'r') as f:
    #         outs = f.read()
    #         outs = outs.split('// Initial random seed:')
    #         for out in outs[1:]:
    #             result = parse_slim(out)
    #             content = re.split(r'[cg]',file.replace('.out',''))
    #             print(content)
    #             results = pd.concat([results,pd.DataFrame([[float(content[1]),int(content[2])*0.05]+result])])
    # results.to_csv("conversion/0-11result.csv",index=False)
                