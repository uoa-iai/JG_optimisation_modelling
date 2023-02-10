#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import collections
from functools import reduce
import sys
   

def obj_wrapper(variables, *params):
    """This function is for use with the NSGA-II algorithm it loads parameters and runs the simulation 5 times, aggregating the results"""
    
    #Load parameters
    lat_kde,lat_lan,p_bad = params
    buf, a_lat, a_buf, a_acc = variables
    buf = round(buf)
    
    var = np.array([buf, a_lat, a_buf, a_acc])
    par = (lat_kde,lat_lan,p_bad)
    
    #Run simulation 5 times, aggregating the results
    i_count = 5
    i_sum = [0,0,0,0]
    for i in range(0,i_count): 
        res = obj_funct(var, *par) #run the objective function
        for j in range(0,len(i_sum)):
            i_sum[j]+=res[j][0]
    return i_sum
    
    


def obj_funct(variables, *params, sim = False):
    """ A function that runs one iteration of the robot simulation """
    
    ### INITIALISATION ###
    
    #Load parameters
    lat_kde,lat_lan,p_bad, mode = params
    buf, a_lat, a_buf, a_acc = variables
    
    #CONSTANTS
    wp = 1000

    #Which robot to simulate
    if mode == 'omx':
        I_recv = 10 #ms
        d = 0.05 #m
        V_min = 0.05 #m/s
        V_max = 0.5 #m/s
        bcrit = 5

    elif mode == 'tb3':
        I_recv = 150 #ms
        d = 0.1 #m
        V_min = 0.1 #m/s
        V_max = 0.26 #m/s
        bcrit = 5
    
    else:
        print("INVALID MODE")
        sys.exit()
        
    #Set trajectory - A constant maximum Speed
    V_ow = V_max
    #Initialise V_cw
    V_cw = V_ow
    
    #Network Conditions
    L_list = [10,50,100,300]
    L_default = 100
    LANavg = 8
    Lmax = 500

    #Loss rate
    pRate = 0.05 #initial loss rate as observed from testing
    p_bad[0] = pRate
    
    #Initialise lists
    
    #Cost Aggregates
    Z_Speed = [] #convert to ms scale
    Z_Smooth = []  # convert to ms scale
    Z_Wait = []
    Z_Count = []
    #Charting
    vel_ilist = []
    vel_clist = []
    acc_ilist = []
    acc_clist = []
    time_list = []
    time_vel = []
    time_vow = []
    #Retransmission Lists
    rt_store = [] #for retaining
    rt_list = [0]*wp
    #oneshot
    first = True
    #Ploss status array
    pLoss = [0,1]
    
    
    ### MAIN OBJECTIVE FUNCTION ###
    
    #Loop for multiple latencies
    for Lavg in L_list:
        #grace time for retransmission
        tgrace = 0
        
        #Initialise circular buffer for latency smoothing
        lat_hist = collections.deque(maxlen=30)
        I_hist = collections.deque(maxlen = int(buf-bcrit))
    
        #default latency for training
        if not sim:
            Lavg = L_default

        #I_snd history
        I_hist = []
        
        speed_cost = 0
        smooth_cost = 0
        wait_cost = 0
        count_cost = 0

        #Initialising Speed and Acceleration Charts vs distance
        if sim:
            vel_ideal = [0]*wp
            vel_comp = [0]*wp
            acc_ideal = [0]*wp
            acc_comp = [0]*wp
        
        #Markov Setup
        mCount = 0
        mList = [0]
        success = False
        
        #Time simulation initialisation
        vel_toriginal = [V_ow]
        vel = [0]
        time = [0]
        time_run = []
        
        ### RETRANSMISSION LIST GENERATION ###

        #Fill rt_list, keeping the same network pattern for all latencies
        if first:
            #temporal step size
            jmp_ind = int(I_recv/5)
            rt_count = 0
            #loop through all waypoints by feeding into mList
            while len(mList) > 0:
                try:
                    #Out of range, default to initial state
                    if rt_count >= len(p_bad):
                        pRate = p_bad[0]
                    else:
                        #assign pRate
                        pRate = p_bad[rt_count]
                except:
                    pRate = p_bad[0]
                if np.random.choice(pLoss,p=[1-pRate,pRate]) > 0:
                    for packet in mList:
                        #Test for packet failure
                        rt_list[packet] += 1
                else:
                    success = True
                            
                mList = [value for value in mList if value != -1]
                if success:
                    rt_count = 0
                    mList = []
                else:
                    #traverse markov chain
                    rt_count += 1 #Markov transition interval
                    for i in range(0,jmp_ind-1):
                        rt_count = rt_count+1 if np.random.choice(pLoss,p=[1-p_bad[rt_count],p_bad[rt_count]]) > 0 else 0
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


        ### LATENCY SAMPLING AND SPEED SCALING PER WAYPOINT ###

        #For each waypoint
        for n in range(0,wp):
            #Sample latency offset
            Loff = lat_kde.sample(1)[0][0]
            LANoff = lat_lan.sample(1)[0][0]
            Lsum = (Loff + Lavg + LANoff + LANavg)
            lat_hist.append(Lsum)
            #Smoothing using moving average filter
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

            #store previous speed
            V_prev = V_cw

            #Scale Speed - QUADRATIC
            V_cw = ((V_ow-(V_ow*klat))/(-1*(buf)**2))*((a_buf*b_loss)**2-buf**2)
            #Linear limiter on upwards acceleration
            V_cw = min(V_prev + V_max*a_acc,V_cw)
            V_cw = max(V_cw, V_min)
            
            #Calculate the new consumption period - ms
            I_snd = I_recv*(V_ow/V_cw)
            I_hist.append(I_snd)
            
            ### RUNNING TIME AND WAITING TIME SIMULATION ###
            
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
                rt_max = rt_list[n]-int( sum(I_hist)/I_recv )
                             
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
                            rt_try = rt_list[j]-int(sum(list(I_hist)[cnt:])/I_recv )
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
        Z_Count.append(count_cost)
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
        
    if sim: #for visualisation
        return([Z_Speed,Z_Smooth,Z_Wait,Z_Count,vel_ilist, vel_clist, acc_ilist, acc_clist, time_list, time_vel, time_vow, rt_store])
    else: #for training
        return([Z_Speed,Z_Smooth,Z_Wait,Z_Count])

### MAIN FUNCTION ###

if __name__ == "__main__":
    pass

"""
Results:

██████████████████████████████████████████████████████████████████████████████████

"""