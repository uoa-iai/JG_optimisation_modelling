import numpy as np
from matplotlib import pyplot as plt
import collections

# d = collections.deque(maxlen=10)
# for i in range(0,20):
#     d.append(i)
#     print(d)
#     print(sum(d)/float(len(d)))

V_ow = 0.5
V_prev = V_ow
Lsmoothed = 100
buf = 5

I_recv = 15
I_hist = collections.deque(maxlen=buf)

#grace time provided by simultaneous retransmission
tgrace = 0

vel = []
time = []
time_run = [0.0]
wait = [0.0]

rt_list = [0,0,0,0,0,0,0,0,0,400,350,325,300,275,250,225,200,0,0,0,0,0,0,0,0,0,0,0]

for n in range(1,len(rt_list)):
    #obtain loss in terms of buffer
    for j in range(n, n+buf+1):
        if j < len(rt_list):
            if(rt_list[j] > 0):
                b_loss = j-n-buf
                break
            else:
                b_loss = 0
        
    #obtain velocity, capping acceleration for smoothing - quadratic deceleration
    V_cw = max(((V_ow-(V_ow*0.75))/(-1*(buf)**2))*((b_loss)**2-buf**2),0.05)
    V_cw = min(V_cw, V_prev+V_ow*0.01)
    V_prev = V_cw
    
    I_snd = I_recv*(V_ow/V_cw)
    #obtain execution timestamp
    time_run.append(time_run[n-1]+I_snd)
    
    #Calculate initial wait time
    time_rt = time_run[n-1]+ 2*Lsmoothed + rt_list[n]*I_recv - tgrace if rt_list[n]>0 else 0
    
    rt_credit = 2*Lsmoothed + rt_list[n]*I_recv - tgrace 
    #If additional waiting time is incurred, fill up to b_oper buffer again
    if(time_run[n] < time_rt): #Time taken for packet to arrive means that we have to wait
        #Check other packetime_run in b_oper 
        for j in range(n+1, n+buf):
            try:
                #take an interval
                rt_credit -= I_recv
                #Check if additional waiting time is required
                if rt_credit < 2*Lsmoothed + rt_list[j]*I_recv:
                    #add the additional wait time to time_rt
                    time_rt += 2*Lsmoothed + rt_list[j]*I_recv - rt_credit 
                    #set rt_list to 0 
                    rt_list[j] = 0
                else:
                    #consider it done
                    rt_list[j] = 0
            except IndexError:
                break
        #grace time is given accounting for simultaneous retransmission
        tgrace = time_rt - time_run[n]
        #Log the new execution time
        time_run[n] = time_rt + I_snd
        vel.append(0)
        time.append(time_rt)
        
    tgrace = max(0, tgrace - I_snd)
    I_hist.append(I_snd)
    vel.append(V_cw)
    time.append(time_run[n])

plt.step(time,vel)
plt.show()