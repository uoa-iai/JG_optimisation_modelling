import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.neighbors import KernelDensity
import os
import glob
import matplotlib.pyplot as plt
import pickle

from scipy import stats

#A script to analyse the existing set of results and output KDEs of each individual test and all the tests aggregated

#percentage of data to sample
smp_frac = 0.01

#number of files to process
n = 10

#obtain absolute pathing for results
### RELATIVE TO CURRENT DIRECTORY
# dirname = os.path.dirname(__file__)
# filetype='raw_*.csv'
# dirtcp = os.path.abspath(os.path.join(dirname,'..','LAN_TCP','Output',filetype))
# dirwan = os.path.abspath(os.path.join(dirname,'..','WAN_TCP','**'))

### Absolute path to external drive
dirname = 'E:\\'
filetype='raw_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','LAN_TCP','Output',filetype))
dirwan = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','WAN_TCP','**'))

dirs = [dirwan, dirtcp]

for dir in dirs:
    #loop through regions and LAN 
    for fpath in glob.glob(dir):
        
        #init counters
        lat_sum = 0
        lat_count = 0
        file_count = 0
        file_cur = 1
        
        #init stack
        #initialise stack
        lat_stack = np.empty(0)
        
        #Append for wan 
        if dir.find("WAN_TCP") >= 0:
            #label
            label = fpath[fpath.find("WAN_TCP")+8:]
            fpath = os.path.abspath(os.path.join(fpath,'Output',filetype))
        else:
            label = "LAN"
            
        for flpath in glob.glob(fpath):
            file_count += 1

        #loop through files to get mean for the region
        for flpath in glob.glob(fpath):
            #report progress
            print('Calculating mean for: ' + str(flpath) + '         Progress: '+str(file_cur)+'/'+str(file_count), end='\r')
            df = pd.read_csv(flpath, usecols=[7]).sample(frac=smp_frac)
            #Filter outliers
            df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.5)*500)|(df.iloc[:,0] == np.nan)]
            #drop NA values and reset indicies
            lat_df = df.iloc[:,0].dropna(axis=0).reset_index(drop=True)
            #add entries to summation and count the entries
            lat_sum += lat_df.sum(axis=0)
            lat_count += lat_df.size
            
            #TEMP
            if file_cur > n:
                break
            
            file_cur += 1

        #obtain mean
        lat_mean = lat_sum/lat_count
        print(str(fpath)+'                                          \nMean: ' + str(lat_mean))

        #loop again to stack 
        file_cur = 1
        for flpath in glob.glob(fpath):
            print('Stacking normalised data: ' + str(flpath) + '         Progress: '+str(file_cur)+'/'+str(file_count), end='\r')
            df = pd.read_csv(flpath, usecols=[7]).sample(frac=smp_frac)
            #filter out outliers
            df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.5)*500)|(df.iloc[:,0] == np.nan)]
            if(lat_stack.size == 0):
                lat_stack = df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean
            else:
                lat_stack = np.vstack((lat_stack,df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean))
            if file_cur > n:
                break
            file_cur += 1
        
        #With the stack, now obtain the kde
        #scipy kde
        lat_skde = stats.gaussian_kde(lat_stack, 'scott')
        x1 = np.linspace(0, 1, 100)
        y1 = lat_skde.pdf(x1)
        
        lat_reshape = lat_stack.reshape(-1, 1)
        lat_k1 = KernelDensity(kernel = 'gaussian',bandwidth=0.1).fit(lat_reshape)
        lat_k2 = KernelDensity(kernel = 'gaussian',bandwidth=0.01).fit(lat_reshape)
        # log_dens = lat_k1.score_samples(lat_reshape)
        y2 = lat_k1.sample(50000)
        y3 = lat_k2.sample(50000)
                
        #plot the results
        print("PLOTTING")
        fig, ax = plt.subplots()
        ax.plot(x1,y1,c='b')
        ax.hist(lat_reshape,density = True, bins = 100)
        ax.hist(y2.ravel(),density=True,bins=100,histtype=u'step')
        ax.hist(y3.ravel(),density=True,bins=100,histtype=u'step')
        plt.show()
        
        #Store the data
        #Save kde to pickle
        kdefile = open(label+"_KDE",'wb')
        pickle.dump(lat_k1, kdefile)
        kdefile.close()
        
        rawfile = open(label+"_RAW",'wb')
        pickle.dump(lat_reshape, rawfile)
        rawfile.close()
        break

print(lat_stack)
