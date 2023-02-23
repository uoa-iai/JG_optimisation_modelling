#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import collections
import pickle
import matplotlib.pyplot as plt

import sklearn

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
    print(sklearn.__version__)
    
    #Load PDF for latency
    kdefile = open('./SYD_LDN_LAT_KDE','rb')
    lat_kde = pickle.load(kdefile)
    kdefile.close()

    lanfile = open('./LAN_LAT_KDE','rb')
    lat_lan = pickle.load(lanfile)
    lanfile.close()

    pbadFile = open('./SYD_LDN_ConLoss_pbad','rb')
    p_bad = pickle.load(pbadFile)
    pbadFile.close()

    pbadFile = open('./LAN_ConLoss_pbad','rb')
    p_bad_lan = pickle.load(pbadFile)
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
        buf = 156
        a_lat = 0.779001524842345
        a_buf = 0.12525509424920095
        a_acc = 0.8955133250442148
    
    if mode == 'tb3':
        #TB3 - Quadratic Smoothing
        buf = 120
        a_lat = 0.8342855939600385
        a_buf = 0.011781929313469802
        a_acc = 0.2810546088039789

    search = True

    params = (lat_kde,lat_lan,p_bad,p_bad_lan,mode, wp)
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
        
    #Report stoppage percentages
    print(f"stoppage of 10 ms {1-(count_costs[0].count(0)/len(count_costs[0]))} -  1 CNT {count_costs[0].count(1)}-- {count_costs[0].count(1)/len(count_costs[0])} -  2 --CNT {count_costs[0].count(2)} {count_costs[0].count(2)/len(count_costs[0])} - {(len(count_costs[0]) - count_costs[0].count(0))} / {len(count_costs[0])}")
    print(f"stoppage of 50 ms {1-(count_costs[1].count(0)/len(count_costs[1]))} -  1 CNT {count_costs[1].count(1)}-- {count_costs[1].count(1)/len(count_costs[1])} -  2 --CNT {count_costs[1].count(2)} {count_costs[1].count(2)/len(count_costs[1])} - {(len(count_costs[1]) - count_costs[1].count(0))} / {len(count_costs[1])}")
    print(f"stoppage of 100 ms {1-(count_costs[2].count(0)/len(count_costs[2]))} -  1 CNT {count_costs[2].count(1)}-- {count_costs[2].count(1)/len(count_costs[2])} -  2 --CNT {count_costs[2].count(2)} {count_costs[2].count(2)/len(count_costs[2])} - {(len(count_costs[2]) - count_costs[2].count(0))} / {len(count_costs[2])}")
    print(f"stoppage of 300 ms {1-(count_costs[3].count(0)/len(count_costs[3]))} -  1 CNT {count_costs[3].count(1)}-- {count_costs[3].count(1)/len(count_costs[3])} -  2 --CNT {count_costs[3].count(2)} {count_costs[3].count(2)/len(count_costs[3])} - {(len(count_costs[3]) - count_costs[3].count(0))} / {len(count_costs[3])}")
    
        
    vel_list[0] = vel_ideal[0]
    for i in range(0,4):
        vel_list[i+1] = vel_comp[i]
    
    acc_list[0] = acc_ideal[0]
    for i in range(0,4):
        acc_list[i+1] = acc_comp[i]
                 
    #SPD_COST
    fig, ax = plt.subplots()
    ax.set_title('Speed Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Speed Cost Value (ms)', ylabel='Frequency (samples)')
    iter = 0
    for item in speed_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.savefig(f'{mode}_spd_cost_{sim_num}.png', bbox_inches='tight')
    
    #SMTH_COST
    fig, ax = plt.subplots()
    ax.set_title('Smooth Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Smooth Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in smooth_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.savefig(f'{mode}_smth_cost_{sim_num}.png', bbox_inches='tight')
    
    #WAIT_COST
    fig, ax = plt.subplots()
    ax.set_title('Wait Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Wait Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in wait_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.savefig(f'{mode}_wait_cost_{sim_num}.png', bbox_inches='tight')
    
    #CNT_COST
    fig, ax = plt.subplots()
    ax.set_title('Count Cost Distribution Over '+str(sim_num)+' Simulations')
    ax.set(xlabel='Count Cost Value', ylabel='Frequency (samples)')
    iter = 0
    for item in count_costs:
        ax.hist(item, bins=100, label=str(lat_val[iter])+" ms", histtype=u'step')
        iter += 1
    ax.legend(loc="upper right")
    plt.savefig(f'{mode}_count_cost_{sim_num}.png', bbox_inches='tight')
    
    #VT
    fig, ax = plt.subplots()
    ax.set_title('Velocity over time')
    ax.set(xlabel='Time (ms)', ylabel='Velocity (m/s)')
    iter = 0
    
    ax.step(time_lists[3], time_vow[3], c='m',label = 'ideal')
    
    for item in time_vels:
        ax.step(time_lists[iter],item,label=str(lat_val[iter])+" ms")
        iter += 1
    ax.legend(loc="lower right")
    plt.savefig(f'{mode}_VT_{sim_num}.png', bbox_inches='tight')
    
    #RT
    fig, ax = plt.subplots()
    ax.set_title('Required Retransmissions Per Waypoint')
    ax.set(xlabel='Waypoints', ylabel='Retransmission Count')
    ind = range(len(rt_list))
    plt.step(ind,rt_list)
    plt.savefig(f'{mode}_RT_{sim_num}.png', bbox_inches='tight')