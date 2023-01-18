#Script to simulate the system

from email.utils import decode_rfc2231
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os

import pickle

import matplotlib.pyplot as plt

from SA import obj_funct

if __name__ == "__main__":
    
    #Load PDF for latency
    kdefile = open('kdePickle','rb')
    lat_kde = pickle.load(kdefile)
    kdefile.close()

    lanfile = open('LAN_Pickle','rb')
    lat_lan = pickle.load(lanfile)
    lanfile.close()

    pbadFile = open('pbadPickle','rb')
    p_bad = pickle.load(pbadFile)
    pbadFile.close()

    #CONSTANTS
    sim_num = 10000

    #CONSTRAINTS
    wp = 200
    bcrit = 3
    buf_min = bcrit+1
    buf_max = wp
    fact_min = 0
    fact_max = 1

    #Decision variables
    buf = 10
    a_lat = 0.0001974
    a_buf = 0.0007305

    #Optimised by GA
    # buf = 5
    # a_lat = 0.12664153292974867
    # a_buf = 0.0243406027410771
    
    #Optimised by GA with aggregation
    # buf = 5
    # a_lat = 0.007197591153989442
    # a_buf = 0.05322556233229219
    
    #Optimised using NSGA2 with aggregation
    buf = 27
    a_lat = 0.002072
    a_buf = 0.00001262

    #Random Guess
    # buf = 20
    # a_lat = 0.7
    # a_buf = 0.7

    params = (lat_kde,lat_lan,p_bad)
    variables = np.array([buf, a_lat, a_buf])
    
    speed_list = []
    smooth_list = []
    wait_list = []
    count_list = []

    for i in range(0,sim_num):
        print(str(i)+"/"+str(sim_num))
        cost_list= obj_funct(variables,*params)
        speed_list.append(cost_list[0])
        smooth_list.append(cost_list[1])
        wait_list.append(cost_list[2])
        count_list.append(cost_list[3])

    fix, [(ax1, ax2),(ax3, ax4)] = plt.subplots(2, 2)
    ax1.hist(speed_list, bins=100)
    ax1.set_title('Speed Cost Distribution Over 10000 Simulations')
    ax1.set(xlabel='Speed Cost Value', ylabel='Frequency (samples)')
    ax2.hist(smooth_list, bins=100)
    ax2.set_title('Smooth Cost Distribution Over 10000 Simulations')
    ax2.set(xlabel='Smooth Cost Value', ylabel='Frequency (samples)')
    ax3.hist(wait_list, bins=100)
    ax3.set_title('Wait Cost Distribution Over 10000 Simulations')
    ax3.set(xlabel='Wait Cost Value', ylabel='Frequency (samples)')
    ax4.hist(count_list, bins=100)
    ax4.set_title('Wait Count Cost Distribution Over 10000 Simulations')
    ax4.set(xlabel='Wait Count Cost Value', ylabel='Frequency (samples)')
    fix.tight_layout()
    plt.show()

############################################

# #Add to list
# speed_list.append(Z_Speed)
# smooth_list.append(Z_Smooth)
# wait_list.append(Z_Wait)
# count_list.append(Z_Count)
# cost_list.append(Z_System)

# print("ITERATION:   "+str(i)+"  SPEED: "+str(Z_Speed)+"  SMOOTH: "+str(Z_Smooth)+"  Wait: "+str(Z_Wait)+"  Count: "+str(Z_Count))
# print("TOTAL COST: "+str(Z_System))

# # Store results
# # Distribution of speed
# speedFile = open('speedPickle','wb')
# pickle.dump(speed_list,speedFile)
# speedFile.close()
# #Distribution of smooth
# smoothFile = open('smoothPickle','wb')
# pickle.dump(smooth_list,smoothFile)
# smoothFile.close()
# #Distribution of wait
# waitFile = open('waitPickle','wb')
# pickle.dump(wait_list,waitFile)
# waitFile.close()
# #Distribution of count
# countFile = open('countPickle','wb')
# pickle.dump(count_list,countFile)
# countFile.close()
# #Distribution of total cost
# costFile = open('costPickle','wb')
# pickle.dump(cost_list,costFile)
# costFile.close()