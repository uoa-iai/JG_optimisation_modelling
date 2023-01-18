import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.neighbors import KernelDensity
import os
import glob

import pickle

#Script to obtain data, downsample, and run an one-way ANOVA test to determine statistical difference between TCP and UDP results

#obtain absolute pathing for results
dirname = os.path.dirname(__file__)
filetype='raw_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'..','LAN_TCP','Output',filetype))
dirwan = os.path.abspath(os.path.join(dirname,'..','WAN_TCP','**'))


print('\n\nWAN TCP FILES\n')

#initialise stack
lat_stack = np.empty(0)

#list of means
meanList = []

#percentage of data to sample
smp_frac = 0.1

#loss data
loss_sum = 0
loss_count = 0

#loop through folders
for fpath in glob.glob(dirwan):
    lat_sum = 0
    lat_count = 0
    file_count = 0
    file_cur = 1
    wanfiles = os.path.abspath(os.path.join(fpath,'Output',filetype))
    for flpath in glob.glob(wanfiles):
        file_count += 1

    #loop through files to get mean for the region
    for flpath in glob.glob(wanfiles):
        print('Calculating mean for: ' + str(fpath) + '         Progress: '+str(file_cur)+'/'+str(file_count), end='\r')
        file_cur += 1
        df = pd.read_csv(flpath, usecols=[7]).sample(frac=smp_frac)
        df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.5)*500)|(df.iloc[:,0] == np.nan)]
        #drop NA values and reset indicies
        lat_df = df.iloc[:,0].dropna(axis=0).reset_index(drop=True)
        lat_sum += lat_df.sum(axis=0)
        lat_count += lat_df.size
        #Also get the loss percentage
        df = pd.read_csv(flpath, usecols=[3]).iloc[:,0].dropna(axis=0).reset_index(drop=True)
        loss_sum += df.sum(axis=0)
        loss_count += df.size

    #obtain mean
    lat_mean = lat_sum/lat_count
    meanList.append(lat_mean)
    print(str(fpath)+'                                          \nMean: ' + str(lat_mean))

    #loop again to stack 
    file_cur = 1
    for flpath in glob.glob(wanfiles):
        print('Stacking normalised data: ' + str(fpath) + '         Progress: '+str(file_cur)+'/'+str(file_count), end='\r')
        file_cur += 1
        df = pd.read_csv(flpath, usecols=[7]).sample(frac=smp_frac)
        #filter out outliers
        df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.5)*500)|(df.iloc[:,0] == np.nan)]
        if(lat_stack.size == 0):
            lat_stack = df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean
        else:
            lat_stack = np.vstack((lat_stack,df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean))

print(lat_stack)

#obtain loss rate
loss_rate = loss_sum/loss_count
meanList.append(0)
meanList.append(loss_rate)

meanFile = open('wanMeans','ab')
pickle.dump(meanList,meanFile)
meanFile.close()

#With the stack, now obtain the kde 
lat_k1 = KernelDensity(kernel = 'gaussian',bandwidth=0.05).fit(lat_stack.reshape(-1, 1))

#Save kde to pickle
kdefile = open('kdePickle','ab')
pickle.dump(lat_k1, kdefile)
kdefile.close()

# print('Mean:\nLatency: '+str(klat_mean)+'\n')