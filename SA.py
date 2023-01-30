#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import collections
from matplotlib import pyplot as plt
from functools import reduce
   
import pickle

#Globals
alg_iter = 0

#Callback function to show progress
def callback_prog(x, f, context):
    global alg_iter
    print(str(alg_iter)+"/"+"1000       "+"buf: "+str(x[0])+" LAT_FACT: "+str(x[1])+" BUF_FACT: "+str(x[2]))
    alg_iter += 1


def obj_wrapper(variables, *params):
    lat_kde,lat_lan,p_bad = params
    buf, a_lat, a_buf, a_acc = variables
    buf = round(buf)
    
    var = np.array([buf, a_lat, a_buf, a_acc])
    par = (lat_kde,lat_lan,p_bad)
    
    i_count = 5
    i_sum = [0,0,0,0]
    for i in range(0,i_count): 
        res = obj_funct(var, *par) #run the objective function
        for j in range(0,len(i_sum)):
            i_sum[j]+=res[j][0]
    return i_sum
    
    

"""
A function that runs one iteration of the robot simulation
"""
def obj_funct(variables, *params, sim = False):
    lat_kde,lat_lan,p_bad = params
    buf, a_lat, a_buf, a_acc = variables
    
#CONSTANTS
    wp = 1000

    #OM-X
    I_recv = 10
    d = 0.05
    V_min = 0.05
    V_max = 0.5
    V_ow = V_max
    bcrit = 5
    
    #TB3
    # wp = 200
    # I_recv = 150
    # d = 0.1
    # V_min = 0.1
    # V_max = 0.26
    # V_ow = V_max
    # bcrit = 3
    
    #Network Conditions
    L_list = [10,50,100,300]
    L_default = 100
    LANavg = 8
    Lmax = 500

    #Loss rate
    pRate = 0.05
    p_bad[0] = pRate
    
    #Initialise lists
    #aggregate costs
    Z_Speed = [] #convert to ms scale
    Z_Smooth = []  # convert to ms scale
    Z_Wait = []
    Z_Count = []
    vel_ilist = []
    vel_clist = []
    acc_ilist = []
    acc_clist = []
    time_list = []
    time_vel = []
    time_vow = []
    rt_store = []
    
    first = True
    
    #Constants
    #wait cooldown
    rt_cd = buf
    
    rt_list = [0]*wp
    #Ploss status array
    pLoss = [0,1]
    #Initialise V_cw
    V_cw = V_ow
    
    #Loop for multiple latencies
    for Lavg in L_list:
        tgrace = 0
        
        #Initialise circular buffer for latency smoothing
        lat_hist = collections.deque(maxlen=30)
        I_hist = collections.deque(maxlen = int(buf-bcrit))
    
        #band-aid for training
        if not sim:
            Lavg = L_default

        #I_snd history
        I_hist = []

        #List of costs within tests
        zspeed = [0]*wp
        zsmooth = [0]*wp
        zwait = [0]*wp
        zcount = [0]*wp
        
        speed_cost = 0
        smooth_cost = 0

        #Simulation results
        t_oper = [0]*wp
        t_ideal = [0]*wp
        t_stop = [0]*wp
        
        #Initialising Velocity and Acceleration Charts vs distance
        if sim:
            vel_ideal = [0]*wp
            vel_comp = [0]*wp
            acc_ideal = [0]*wp
            acc_comp = [0]*wp
        
        #Markov Setup
        mCount = 0
        mList = [0]
        success = False
        
        #Time simulation
        vel_toriginal = [V_ow]
        vel = [0]
        time = [0]
        time_run = []
        
        wait_cost = 0
        count_cost = 0
        
        
        

        #Fill rt_list, keeping the same network pattern for all latencies
        if first:
            # for n in range(0,wp):
            #     #Sample initial packet loss 
            #     rt_count = 0
                
            #     while True: #Retransmit until successful
            #         if(rt_count < len(p_bad)):
            #             pRate = p_bad[rt_count]
            #         else:
            #             pRate = 0
            #         pL = np.random.choice(pLoss,p=[1-pRate,pRate])
            #         if pL > 0:
            #             rt_count += 1
            #         else:
            #             break
            
            #     rt_list[n] = rt_count
            
            ############### batch
            jmp_ind = int(I_recv/5)
            rt_prev = 0
            rt_count = 0
            while len(mList) > 0:
                
                try:
                    if rt_prev >= len(p_bad) or rt_count >= len(p_bad):
                        pRate = 0
                    else:
                        pRate = reduce((lambda x, y: x*y),p_bad[rt_prev:rt_count+1])
                except:
                    pRate = 0
                if np.random.choice(pLoss,p=[1-pRate,pRate]) > 0:
                    for packet in mList:
                        #Test for packet failure
                        rt_list[packet] += 1
                else:
                    success = True
                            
                mList = [value for value in mList if value != -1]
                if success:
                    rt_count = 0
                    rt_prev = 0
                    mList = []
                else:
                    rt_prev = rt_count+1
                    rt_count += jmp_ind #Markov transition interval
                
                mCount += 1
                
                if mCount < wp:
                    mList.append(mCount)
                    success = False
            ####### end batch
            #backup for other latencies
            rt_store = rt_list.copy()
            first = False
        else:
            #restore list for other latencies
            rt_list = rt_store.copy()


        #For each waypoint
        for n in range(0,wp):
            #Sample latency offset
            Loff = lat_kde.sample(1)[0][0]
            LANoff = lat_lan.sample(1)[0][0]
            Lsum = (Loff + Lavg + LANoff + LANavg)
            lat_hist.append(Lsum)
            Lsmoothed = sum(lat_hist)/float(len(lat_hist))

            b_loss = 0
            b_iter = 1
            #Find position of next lost packet
            b_stop = min(n+int(buf),wp)
            for m in range(n+1,b_stop):
                if rt_list[m] - (buf - b_iter) > 0:
                    b_loss = buf - b_iter
                    break
                b_iter += 1


            #Calcualte Scaling Factor
            klat = a_lat*(1-((Lmax-Lsmoothed)/Lmax))
            kbuf = a_buf*(1-(buf-b_loss)/buf)

            #store previous
            V_prev = V_cw

            #Scale Velocity - QUADRATIC
            V_cw = ((V_ow-(V_ow*klat))/(-1*(buf)**2))*((a_buf*b_loss)**2-buf**2)
            #Linear limiter on upwards acceleration
            V_cw = min(V_prev + V_max*a_acc,V_cw)
            V_cw = max(V_cw, V_min)
            
            #Calculate the new consumption period - originally in seconds -> convert to ms
            I_snd = I_recv*(V_ow/V_cw)
            I_hist.append(I_snd)
            
                      
            #TIME SIMULATION
            #obtain execution timestamp            
            if n == 0:
                #Express wait time while filling initial buffer to b_oper
                tgrace = max(rt_list[0:int(buf)])*I_recv
                init_time = buf*I_recv + tgrace
                for i in range(0,int(buf)):
                    rt_list[i] = 0
                time_run.append(init_time)
                time.append(time_run[n]) 
                vel.append(0)
                vel_toriginal.append(V_ow)
                wait_cost += time_run[n]
            else:
                time_run.append(time_run[n-1]+I_snd)
                
                #Calculate the number of retransmissions remaining when attempted execution
                rt_max = rt_list[n]-int(buf*(I_recv*len(I_hist)/(sum(I_hist))))
                             
                #Calculate initial wait time
                time_rt = time_run[n-1]+ 2*Lsmoothed + rt_max*I_recv - tgrace if rt_max > 0 else 0
                #If additional waiting time is incurred, travel through the buffer to find the total waiting time
                if(time_run[n] < time_rt):
                    #increment wait count cost
                    count_cost += 1
                    for j in range(n+1, int(n+buf)):
                        try:
                            cnt = j-n
                            #Find number of retransmissions
                            rt_try = rt_list[j]-int(buf*(I_recv*(len(I_hist)-cnt)/(sum(list(I_hist)[cnt:]))))
                            #store if larger 
                            if rt_try > rt_max:
                                rt_max = rt_try
                                #recalculate waiting time to reflect max wait time for buffer
                                time_rt = time_run[n-1]+ 2*Lsmoothed + rt_max*I_recv - tgrace
                            #set rt count to 0
                            rt_list[j] = 0
                        except IndexError:
                            break
                    #log the time waited
                    wait_cost += time_rt - time_run[n]
                    #grace time is given accounting for simultaneous retransmission
                    tgrace = time_rt - time_run[n]
                    #Log the new execution time
                    time_run[n] = time_rt + I_snd
                    #log the velocities and timestamp and increment the smooth costs FOR EACH WAITING PERIOD
                    vel.append(0)
                    vel_toriginal.append(V_ow)
                    time.append(time_rt)
                    #increment cost for waiting
                    if(len(time) > 1):
                        speed_cost += ( V_ow - list(reversed(vel))[0] ) * ( list(reversed(time))[0] - list(reversed(time))[1] )
                        smooth_cost += abs(( V_ow - list(reversed(vel))[1] ) - ( V_ow - list(reversed(vel))[0] ))
                    else:
                        speed_cost += ( V_ow - list(reversed(vel))[0] ) * list(reversed(time))[0]
                        smooth_cost += 0
                
                #decay the grace time
                tgrace = max(0, tgrace - I_snd)
                vel.append(V_cw)
                vel_toriginal.append(V_ow)
                time.append(time_run[n]) 
                #increment costs for running
                if(len(time) > 1):
                    speed_cost += ( V_ow - list(reversed(vel))[0] ) * ( list(reversed(time))[0] - list(reversed(time))[1] )
                    smooth_cost += abs(( V_ow - list(reversed(vel))[1] ) - ( V_ow - list(reversed(vel))[0] ))
                else:
                    speed_cost += ( V_ow - list(reversed(vel))[0] ) * list(reversed(time))[0]
                    smooth_cost += ( V_ow - list(reversed(vel))[0] )

            #Calculate error
            zspeed[n] = abs(V_ow - V_cw)
            # print(count_cost)
            #OLD
            # if n > 0:
            #     zsmooth[n] = abs(zspeed[n]-zspeed[n-1])

            #Use retransmission to check for wait time - Lsum intentional
            # if n>buf:
            #     zwait[n] = 2*Lsum+rt_list[n]*(I_recv)-sum(I_hist)-tgrace
            #     #calculate tgrace for next iteration
            #     zwait[n] = max(zwait[n],0)
            #     tgrace += zwait[n]
            # else:
            #     zwait[n] = I_snd*rt_count
            #     zwait[n] = max(zwait[n],0)

            # #calculate zcount
            # if zwait[n] > 0 and rt_cd <= 0:
            #     rt_cd = buf
            #     zcount[n] = 1
            # else:
            #     rt_cd -= 1
            #     zcount[n] - 0
            
            #Calculate Simulation Results
            t_ideal[n] = d/V_ow*1000
            t_oper[n] = d/V_cw*1000
            t_stop[n] = zwait[n]
            
            #Store value for charts
            if sim:
                vel_ideal[n] = V_ow
                vel_comp[n] = V_cw
                if n > 0:
                    acc_ideal[n] = V_ow - vel_ideal[n-1]
                    acc_comp[n] = V_cw - vel_comp[n-1]
                else:
                    acc_ideal[n] = V_ow
                    acc_comp[n] = V_cw

        #aggregate costs
        Z_Speed.append(speed_cost) #convert to ms scale
        Z_Smooth.append(smooth_cost) # convert to ms scale
        Z_Wait.append(wait_cost)
        Z_Count.append(2**count_cost)
        if sim:
            vel_ilist.append(vel_ideal)
            vel_clist.append(vel_comp)
            acc_ilist.append(acc_ideal)
            acc_clist.append(acc_comp)
            time_list.append(time)
            time_vel.append(vel)
            time_vow.append(vel_toriginal)
        else:
            break
        
    if sim:
        return([Z_Speed,Z_Smooth,Z_Wait,Z_Count,vel_ilist, vel_clist, acc_ilist, acc_clist, time_list, time_vel, time_vow, rt_store])
    else:
        return([Z_Speed,Z_Smooth,Z_Wait,Z_Count])

### MAIN FUNCTION ###

if __name__ == "__main__":
    pass

"""
Results:

██████████████████████████████████████████████████████████████████████████████████

"""