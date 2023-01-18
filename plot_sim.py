import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import scipy

import pickle

import matplotlib.pyplot as plt

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 8))

speedFile = open('speedPickle','rb')
spd = pickle.load(speedFile)
speedFile.close()
#Distribution of smooth
smoothFile = open('smoothPickle','rb')
smt = pickle.load(smoothFile)
smoothFile.close()
#Distribution of wait
waitFile = open('waitPickle','rb')
wat = pickle.load(waitFile)
waitFile.close()
#Distribution of count
countFile = open('countPickle','rb')
cnt = pickle.load(countFile)
countFile.close()
#Distribution of total cost
costFile = open('costPickle','rb')
cst = pickle.load(costFile)
costFile.close()

x1 = spd
x2 = smt
x3 = wat
x4 = cnt
x5 = cst
nbins = 150

ax1.hist(x1,bins=nbins)
ax1.set_title('Speed Cost')
ax1.set(xlabel='Cost Value', ylabel='Frequency (samples)')
ax2.hist(x2,bins=nbins)
ax2.set_title('Smoothness Cost')
ax2.set(xlabel='Cost Value', ylabel='Frequency (samples)')
ax3.hist(x3,bins=nbins)
ax3.set_title('Wait Cost')
ax3.set(xlabel='Cost Value', ylabel='Frequency (samples)')
ax4.hist(x4,bins=nbins)
ax4.set_title('Count Cost')
ax4.set(xlabel='Cost Value', ylabel='Frequency (samples)')
ax5.hist(x5,bins=nbins)
ax5.set_title('Total Cost')
ax5.set(xlabel='Cost Value', ylabel='Frequency (samples)')
fig.tight_layout()
plt.show()