import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import scipy

import pickle

import matplotlib.pyplot as plt

# fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

kdeFile = open('kdePickle','rb')
kde = pickle.load(kdeFile)
kdeFile.close()

LAN_File = open('LAN_Pickle','rb')
lan = pickle.load(LAN_File)
LAN_File.close()

x2 = lan.sample(50000)
x1 = kde.sample(50000)


# nbins = 100

# ax1.hist(x1,bins=nbins)
# ax1.set_title('Probability Density of Latency Variation from Mean (WAN)')
# ax1.set(xlabel='Latency Difference From Mean (ms)', ylabel='Frequency (samples)')
# ax2.hist(x2,bins=nbins)
# ax2.set_title('Probability Density of Latency Variation from Mean (LAN)')
# ax2.set(xlabel='Latency Difference From Mean (ms)', ylabel='Frequency (samples)')
# plt.show()

pbadFile = open('pbadPickle','rb')
p_bad = pickle.load(pbadFile)
pbadFile.close()

plt.title('Markov Chain Probabilities')
x = range(len(p_bad))
y = p_bad
plt.bar(x,y)
print(p_bad)
plt.show()

# res_list = []

# #array representative of outcomes
# pLoss = [0,1]

# #Average packet loss rate 
# pRate = 0.0003587197575766985
# pRate = 0.05
# iters = 100000

# for i in range(0,iters):
#     print(str(i)+"/"+str(iters))
#     count = 0
#     #calculate random value
#     pL = np.random.choice(pLoss,p=[1-pRate,pRate])
#     while pL == 1:
#         count+= 1
#         print(count)
#         #reroll die
#         pL = np.random.choice(pLoss,p=[1-pRate,pRate])
        
#     res_list.append(count)


# PLFile = open('PLPickle','wb')
# pickle.dump(res_list,PLFile)
# PLFile.close()