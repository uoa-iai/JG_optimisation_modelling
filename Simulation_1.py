#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import collections

import pickle

import matplotlib.pyplot as plt

from SA import obj_funct

#timing
import time

class TimerError(Exception):
    #Report timer errors
    print("TIMER ERROR")

class Timer:
    """A class to implement simple code timing"""
    def __init__(self):
        self._start_time = None
        
    def start(self):
        #Check if timer is not running by checking state of start time
        if self._start_time is not None:
            raise TimerError(f"Timer is running, use .stop() to stop it")
        
        self._start_time = time.perf_counter() # time.perf_counter() is the current value of an arbitrary timer
    
    def stop(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running, use .start() to start it")
        
        #calculate runtime
        runtime = time.perf_counter() - self._start_time
        self._start_time = None
        return runtime

if __name__ == "__main__":
    
    #Load PDF for latency
    kdefile = open('./Pickles/kdePickle','rb')
    lat_kde = pickle.load(kdefile)
    kdefile.close()

    lanfile = open('./Pickles/LAN_Pickle','rb')
    lat_lan = pickle.load(lanfile)
    lanfile.close()

    pbadFile = open('./Pickles/pbadPickle','rb')
    p_bad = pickle.load(pbadFile)
    pbadFile.close()



    #CONSTRAINTS
    wp = 1000
    bcrit = 5
    buf_min = bcrit+1
    buf_max = wp
    fact_min = 0
    fact_max = 1
    
    #MODE SELECT
    sim_num = 10000
    mode = 'omx' #omx or tb3
    
    if mode == 'omx':
        #Optimised using NSGA2 with aggregation - OM-X        
        buf = 126
        a_lat = 0.779001524842345
        a_buf = 0.12525509424920095
        a_acc = 0.8955133250442148
    
    if mode == 'tb3':
        #TB3 - Quadratic Smoothing
        buf = 80
        a_lat = 0.8342855939600385
        a_buf = 0.011781929313469802
        a_acc = 0.2810546088039789

    params = (lat_kde,lat_lan,p_bad,mode, wp)
    variables = np.array([buf, a_lat, a_buf, a_acc])
    
    speed_list = []
    smooth_list = []
    wait_list = []
    count_list = []
    vel_ideal = []
    vel_comp = []
    acc_ideal = []
    acc_comp = []
    time_lists = []
    time_vels = []
    
    #id for indexing multiple latencies
    lat_id = [0,1,2,3]
    lat_val = [10,50,100,300]
    
    tm = Timer()
    tm.start()
    time_hist = collections.deque(maxlen=50)

    for i in range(0,sim_num):
        runtime = tm.stop()
        time_hist.append(runtime)
        print(f"{i}/{sim_num}    Iteration: {runtime:0.4f} seconds   Time Remaining: {((sum(time_hist)/len(time_hist))*(sim_num-i))/60:0.4f} minutes")
        tm.start()
        cost_list = obj_funct(variables,*params,sim=True)
        speed_list.append(cost_list[0])
        smooth_list.append(cost_list[1])
        wait_list.append(cost_list[2])
        count_list.append(cost_list[3])
        vel_ideal = cost_list[4]
        vel_comp = cost_list[5]
        acc_ideal = cost_list[6]
        acc_comp = cost_list[7]
        time_lists = cost_list[8]
        time_vels = cost_list[9]
        time_vow = cost_list[10]
        rt_list = cost_list[11]
    
    speed_costs     = [[],[],[],[]]
    smooth_costs    = [[],[],[],[]]
    wait_costs      = [[],[],[],[]]
    count_costs     = [[],[],[],[]]
    vel_list        = [[],[],[],[],[]]
    acc_list        = [[],[],[],[],[]]
    
    for data in speed_list:
        speed_costs[0].append(data[0])
        speed_costs[1].append(data[1])
        speed_costs[2].append(data[2])
        speed_costs[3].append(data[3])
        
    for data in smooth_list:
        smooth_costs[0].append(data[0])
        smooth_costs[1].append(data[1])
        smooth_costs[2].append(data[2])
        smooth_costs[3].append(data[3])
    
    for data in wait_list:
        wait_costs[0].append(data[0])
        wait_costs[1].append(data[1])
        wait_costs[2].append(data[2])
        wait_costs[3].append(data[3])
        
    for data in count_list:
        count_costs[0].append(data[0])
        count_costs[1].append(data[1])
        count_costs[2].append(data[2])
        count_costs[3].append(data[3])
        
    vel_list[0] = vel_ideal[0]
    for i in range(0,4):
        vel_list[i+1] = vel_comp[i]
    
    acc_list[0] = acc_ideal[0]
    for i in range(0,4):
        acc_list[i+1] = acc_comp[i]
                 
    
    fig, ax = plt.subplots()
    ax.set_title('Speed Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Speed Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in speed_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Smooth Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Smooth Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in smooth_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Wait Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Wait Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in wait_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Count Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Count Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in count_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Velocity Per Waypoint')
    ax.set(xlabel='Waypoints', ylabel='Velocity (m/s)')
    iter = 0
    for item in vel_list:
        ind = range(len(item))
        if iter == 0:
            ax.plot(ind,item,c='m',label='ideal')
        else:
            ax.plot(ind,item,label=str(lat_val[iter - 1])+" ms")
        iter += 1
            
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Acceleration Per Waypoint')
    ax.set(xlabel='Waypoints', ylabel='Acceleration (m/s^2)')
    iter = 0
    for item in acc_list:
        ind = range(len(item))
        if iter == 0:
            ax.plot(ind,item,c='m',label='ideal')
        else:
            ax.plot(ind,item,label=str(lat_val[iter - 1])+" ms")
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Velocity over time')
    ax.set(xlabel='Time (ms)', ylabel='Velocity (m/s)')
    iter = 0
    
    ax.step(time_lists[3], time_vow[3], c='m',label = 'ideal')
    
    for item in time_vels:
        ax.step(time_lists[iter],item,label=str(lat_val[iter])+" ms")
        iter += 1
    ax.legend(loc="upper right")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.set_title('Required Retransmissions Per Waypoint')
    ax.set(xlabel='Waypoints', ylabel='Retransmission Count')
    ind = range(len(rt_list))
    plt.step(ind,rt_list)
    plt.show()

    # fix, [(ax1, ax2),(ax3, ax4),(ax5,ax6),(ax7,ax8)] = plt.subplots(4, 2)
    # ax1.hist(speed_list, bins=100)
    # ax1.set_title('Speed Cost Distribution Over 500 Simulations')
    # ax1.set(xlabel='Speed Cost Value', ylabel='Frequency (samples)')
    # ax2.hist(smooth_list, bins=100)
    # ax2.set_title('Smooth Cost Distribution Over 500 Simulations')
    # ax2.set(xlabel='Smooth Cost Value', ylabel='Frequency (samples)')
    # ax3.hist(wait_list, bins=100)
    # ax3.set_title('Wait Cost Distribution Over 500 Simulations')
    # ax3.set(xlabel='Wait Cost Value', ylabel='Frequency (samples)')
    # ax4.hist(count_list, bins=100)
    # ax4.set_title('Wait Count Cost Distribution Over 500 Simulations')
    # ax4.set(xlabel='Wait Count Cost Value', ylabel='Frequency (samples)')
    # ax5.plot(range(len(vel_ideal)),vel_ideal)
    # ax6.plot(range(len(vel_comp)),vel_comp)
    # ax7.plot(range(len(acc_ideal)),acc_ideal)
    # ax8.plot(range(len(acc_comp)),acc_comp)
    # fix.tight_layout()
    # plt.show()

###
#TODO:
#Put 4 latency charts on the same plot
###