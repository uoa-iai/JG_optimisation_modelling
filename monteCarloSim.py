#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os

import pickle

#Load PDF for latency
kdefile = open('kdePickle','rb')
lat_kde = pickle.load(kdefile)
# print(kde.sample(1)[0][0])
kdefile.close()

lanfile = open('LAN_Pickle','rb')
lat_lan = pickle.load(lanfile)
# print(kde.sample(1)[0][0])
lanfile.close()

#Load PDF for retransmission attempts
# rtfile = open('rtPickle','rb')
# rt_kde = pickle.load(rtfile)
# rtfile.close()

#define parameters
wp = 1000 #number of waypoints
sim_num = 1000
buf = 12 #buffer size 
tSnd = 15 #time between transmissions [ms]

Lavg = 100 #average latency [ms]
LANavg = 8 #average LAN latency
d = 0.1 #distance between waypoints [m]

#Velocity Scaling parameters
V_max = 0.26 #max robot speed [m/s]
lat_fact = 0.1
loss_fact = 0.973538734
Lmax = 500 #maximum tolerable latency?

#Assuming constant velocty
V = V_max
V_min = 0.1

#array representative of outcomes
pLoss = [0,1]

#Average packet loss rate 
# pRate = 0.0003587197575766985

#Percentile packet loss rates
pRate = 0.05

#repeat tests and store results
delay_list = []
oper_list = []

for i in range(0,sim_num):

    #Set constant times
    tOper = 0 #time to move the desired distance
    tStart = buf*tSnd #initial time to fill up the buffer [m]
    tDelay = 0 #time lost to network issues
    del_prev = 0
    vel_count = 0
    vel_sum = 0

    print(str(i)+"/"+str(sim_num))

    #For each waypoint
    for n in range(1,wp):
        #Sample latency offset
        Loff = lat_kde.sample(1)[0][0]
        LANoff = lat_lan.sample(1)[0][0]
        Lsum = (Loff + Lavg + LANoff + LANavg)
        #Sample initial packet loss 
        pL = np.random.choice(pLoss,p=[1-pRate,pRate])
        rt_count = 0
        while pL == 1: #Retransmit until successful
            pL = np.random.choice(pLoss,p=[1-pRate,pRate])
            rt_count += 1
            #Simulate the rare burst packet loss
            #rt_count += 500
        
        #Velocity Scaling
        klat = 1 - lat_fact*(Lsum/Lmax)
        kbuf = 1 - loss_fact*(2*(Lavg+LANavg))
        if(rt_count == 0):
            kbuf = 1
        V = V_max*klat*kbuf
        if V < V_min:
            V = V_min

        #Calculate the operational inefficiencies
        tOper += (d/V)-(d/V_max)

        del_thresh = (buf*d)/V
        del_trans = 2*(Lsum)*rt_count
        del_total = del_trans-del_thresh-del_prev
        #Calculate the new delay overflow, assuming simultaneous retransmission
        if del_total < 0:
            del_total = 0
        del_prev += del_total
        tDelay += del_total

    tDelay /= 1000
    tStart /= 1000

    totalTime = tOper + tDelay + tStart
    print("ITERATION: "+str(i)+"/"+str(sim_num)+"  Oper Cost: " + str(tOper) + "         Delay time: "+str(tDelay))
    oper_list.append(tOper)
    delay_list.append(tDelay)

# print(delay_list)
#Store results
#Distribution of delay
delayFile = open('delayPickle','wb')
pickle.dump(delay_list,delayFile)
delayFile.close()
#Distribution of operating time
operFile = open('operPickle','wb')
pickle.dump(oper_list,operFile)
operFile.close()