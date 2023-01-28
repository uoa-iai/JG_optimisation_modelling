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
lat = 100
buf = 5

I_recv = 15
I_hist = collections.deque(maxlen=buf)

#grace time provided by simultaneous retransmission
tgrace = 0

vel = []
time = []

rt = [0,0,0,0,0,0,0,0,0,400,350,325,300,275,250,225,200,0,0,0,0,0,0,0,0,0,0,0]
ts = [0.0]
wait = [0.0]
for i in range(1,len(rt)):
    for j in range(i, i+buf+1):
        if j < len(rt):
            if(rt[j] > 0):
                b_loss = j-i-buf
                break
            else:
                b_loss = 0
        
    
    print(b_loss)
    #obtain velocity, capping acceleration for smoothing - quadratic deceleration
    V_cw = max(((V_ow-(V_ow*0.75))/(-1*(buf)**2))*((b_loss)**2-buf**2),0.05)
    V_cw = min(V_cw, V_prev+V_ow*0.01)
    V_prev = V_cw
    
    I_snd = I_recv*(V_ow/V_cw)
    #obtain execution timestamp
    ts.append(ts[i-1]+I_snd)
    
    t_rt = ts[i-1]+ 2*lat + rt[i]*I_recv - tgrace if rt[i]>0 else 0
    
    rt_credit = 2*lat + rt[i]*I_recv
    #If additional waiting time is incurred, fill up to b_oper buffer again
    if(ts[i] < t_rt): #Time taken for packet to arrive means that we have to wait
        #Check other packets in b_oper 
        for j in range(i+1, i+buf):
            try:
                #take an interval
                rt_credit -= I_recv
                #Check if additional waiting time is required
                if rt_credit < 2*lat + rt[j]*I_recv:
                    #add the additional wait time to t_rt
                    t_rt += 2*lat + rt[j]*I_recv - rt_credit 
                    #set rt to 0 
                    rt[j] = 0
                else:
                    #consider it done
                    rt[j] = 0
            except IndexError:
                break
                
            # if t_rt < 2*lat+rt[j]*I_recv:
            #     t_rt += 
            pass     
        #Process the buffer waiting time - t_rt is the time at +0
        wait.append(t_rt)   
        #the grace time for the following packets is the additional wait time incurred
        #i.e. the difference between the expected execution and the actual
        tgrace = t_rt - ts[i]
        ts[i] = t_rt + I_snd
        vel.append(0)
        time.append(t_rt)
    else:
        wait.append(0)
        
    t_grace = max(0, tgrace - I_snd)
    I_hist.append(I_snd)
    vel.append(V_cw)
    time.append(ts[i])
    
plt

print("TS")
print(ts)
print("WAIT")
print(wait)

plt.step(time,vel)
plt.show()