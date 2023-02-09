import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import scipy

import pickle

import matplotlib.pyplot as plt

# fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

consecutiveFile = open('.\Markov_Analysis\consecutivePickle','rb')
consecutive = pickle.load(consecutiveFile)
consecutiveFile.close()

x1 = consecutive[0][0:500]#Array of loss frequencies
data_count = consecutive[1] #total number of data points
nbins = 100
ind = range(0,len(x1))
# plt.bar(ind, x1)
# plt.show()

#Calculating the probabilities for use in an extended markov chain
P_bad = []
P_bad.append(consecutive[0][0]/data_count)
for i in range(1,len(x1)):
    P_bad.append(consecutive[0][i]/consecutive[0][i-1])

pbadFile = open('pbadPickle','wb')
pickle.dump(P_bad,pbadFile)
pbadFile.close()
print(P_bad)