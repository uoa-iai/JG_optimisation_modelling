#Script to simulate the system

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import scipy

import pickle

import matplotlib.pyplot as plt

fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

delayFile = open('delayPickle','rb')
delay = pickle.load(delayFile)
delayFile.close()

operFile = open('operPickle','rb')
oper = pickle.load(operFile)
operFile.close()

# PLFile = open('PLPickle','rb')
# PLoss = pickle.load(PLFile)
# PLFile.close()

bin_n = 30
ax1.hist(delay, bins=bin_n)
ax1.set_title('Delay time')
ax1.set(xlabel='Stoppage Time (s)', ylabel='Frequency (samples)')
ax2.hist(oper, bins=bin_n)
ax2.set_title('Operation Time')
ax2.set(xlabel='Operation Time Cost (s)', ylabel='Frequency (samples)')
# ax3.hist(PLoss, bins='auto')
# ax3.set_title('Loss Count')
plt.show()

