import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.neighbors import KernelDensity
import os
import glob
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV

from scipy import stats

#A script to analyse the existing set of results and output KDEs of each individual test and all the tests aggregated


#timing
import time

class TimerError(Exception):
    #Report timer errors
    print("TIMER ERROR")

class Timer:
    """A class to implement simple code timing"""
    def __init__(self):
        self._start_time = None
        
    def start(self):
        #Check if timer is not running by checking state of start time
        if self._start_time is not None:
            raise TimerError(f"Timer is running, use .stop() to stop it")
        
        self._start_time = time.perf_counter() # time.perf_counter() is the current value of an arbitrary timer
    
    def stop(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running, use .start() to start it")
        
        #calculate runtime
        runtime = time.perf_counter() - self._start_time
        self._start_time = None
        return runtime
    
    def report(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running, use .start() to start it")
        
        runtime = time.perf_counter() - self._start_time
        return runtime

#number of files to process
n = 1000

#obtain absolute pathing for results
### RELATIVE TO CURRENT DIRECTORY
# dirname = os.path.dirname(__file__)
# filetype='raw_*.csv'
# dirtcp = os.path.abspath(os.path.join(dirname,'..','LAN_TCP','Output',filetype))
# dirwan = os.path.abspath(os.path.join(dirname,'..','WAN_TCP','**'))

### Absolute path to external drive
dirname = 'E:\\'
filetype='raw_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','LAN_TCP'))
dirwan = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','WAN_TCP','**'))

dirs = [dirtcp, dirwan]

for dir in dirs:
    #loop through regions and LAN 
    for fpath in glob.glob(dir):
        #start timer
        tm = Timer()
        tm_prev = 0
        tm_sum = 0
        #init counters
        lat_sum = 0
        lat_count = 0
        file_count = 0
        file_cur = 1
        
        #TEMP
        temp = 1500
        #TEMP FIND THE DESIRED REGION
        if fpath.find("SYD_LDN") < 0 and fpath.find("LAN") < 0 :
            pass
        else:
            continue
        
        #init stack
        #initialise stack
        lat_stack = np.empty(0)
        
        #Append for wan 
        if dir.find("WAN_TCP") >= 0:
            #label
            label = fpath[fpath.find("WAN_TCP")+8:]
        else:
            label = "LAN"
            
        fpath = os.path.abspath(os.path.join(fpath,'Output',filetype))
            
        for flpath in glob.glob(fpath):
            file_count += 1

        #loop through files to get mean for the region
        for flpath in glob.glob(fpath):
            #report progress
            tm.start()
            print(f'Calculating mean for: {flpath}         Progress: {file_cur}/{file_count}  Time: {tm_sum}/{tm_prev*file_count}', end='\r')
            df = pd.read_csv(flpath, usecols=[7])
            #Filter outliers
            df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.95))|(df.iloc[:,0] == np.nan)]
            #drop NA values and reset indicies
            lat_df = df.iloc[:,0].dropna(axis=0).reset_index(drop=True)
            #add entries to summation and count the entries
            lat_sum += lat_df.sum(axis=0)
            lat_count += lat_df.size
            
            file_cur += 1
            tm_prev = tm.stop()
            tm_sum += tm_prev
            if file_cur > temp:
                break

        #obtain mean
        lat_mean = lat_sum/lat_count
        print(f'{fpath}                                          \nMean: {lat_mean}')

        #loop again to stack 
        file_cur = 1
        for flpath in glob.glob(fpath):
            print(f'Stacking normalised data: {flpath}         Progress: {file_cur}/{file_count}', end='\r')
            df = pd.read_csv(flpath, usecols=[7])
            #filter out outliers
            df = df[(df.iloc[:,0] <  df.iloc[:,0].quantile(0.95))|(df.iloc[:,0] == np.nan)]
            if(lat_stack.size == 0):
                lat_stack = df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean
            else:
                lat_stack = np.vstack((lat_stack,df.dropna(axis=0).reset_index(drop=True).to_numpy() - lat_mean))
            file_cur += 1
            if file_cur > temp:
                break
        
        #replace infs and nans
        lat_stack = lat_stack[~np.isnan(lat_stack)]
        lat_stack = lat_stack[~np.isinf(lat_stack)]
        
        #generate kde
        lat_reshape = lat_stack.reshape(-1, 1)
        
        #use cross validation to determine bandwidth
        # grid = GridSearchCV(KernelDensity(), {"bandwidth": np.linspace(0.1,1.0,30)}, cv=20)
        # grid.fit(lat_reshape)
        # bw = grid.best_estimator_.bandwidth
        
        print(f"Fitting {label}")
        
        lat_k2 = KernelDensity(kernel = 'gaussian',bandwidth=0.17).fit(lat_reshape)
        
        print(f"Saving {label}")
        
        # #TEMP PLOTTING
        # y1 = lat_k2.sample(100000)
        
        # #plot
        # fig, ax = plt.subplots()
        # ax.hist(lat_reshape, density = True, bins = 50, color='blue', label='raw')
        # ax2=ax.twinx()
        # ax2.hist(y1.ravel(), density = True, bins = 200, histtype=u'step', color='red', label='kde', linewidth=2)
        # plt.show()
        
        
        #Store the data
        #Save kde to pickle
        kdefile = open(f"{label}_LAT_KDE",'wb')
        pickle.dump(lat_k2, kdefile)
        kdefile.close()
        
        rawfile = open(f"{label}_LAT_RAW",'wb')
        pickle.dump(lat_reshape, rawfile)
        rawfile.close()
