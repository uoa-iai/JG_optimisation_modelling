import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Plot_pickle import plotMProb
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#A script to plot the latency and packet loss charts
def plotLat(rawpath, kdepath, gen):
    rawFile = open(rawpath,'rb')
    rawDist = pickle.load(rawFile)
    rawFile.close()
    
    print(f"path {rawpath}")
    fname = str(rawpath[rawpath.rfind("\\")+1:-4])
    print(f"plotting for {fname}")
    # shortDist = rawDist[:int(len(rawDist)/30)].copy()
    
    if(gen):    
        #use cross validation to determine bandwidth TOO DAMN SLOW
        # print("generating grid")
        # grid = GridSearchCV(KernelDensity(), {"bandwidth": np.linspace(0.1,0.2,5)}, verbose=3)
        # print("fitting grid")
        # grid.fit(rawDist)
        # print("getting bandwidth")
        # bw = grid.best_estimator_.bandwidth
        # print(f"{fname} best bandwidth: {bw}")
        #generate kde
        
        kdeDist = KernelDensity(kernel = 'gaussian',bandwidth='silverman').fit(rawDist)
        kdeFile = open(kdepath,'wb')
        pickle.dump(kdeDist, kdeFile)
        kdeFile.close()
    else:
        kdeFile = open(kdepath,'rb')
        kdeDist = pickle.load(kdeFile)
        kdeFile.close()
    
    y1 = kdeDist.sample(1000)
    nbins = 200
    
    
    
    #plot
    fig, ax = plt.subplots()
    ax.hist(rawDist, density = True, bins = nbins, color='blue', label='raw')
    ax.hist(y1.ravel(), density = True, bins = nbins, histtype=u'step', color='red', label='kde', linewidth=2)
    ax.legend(loc="upper right")
    ax.set(xlabel='Latency Variation (ms)', ylabel='Sample Density', title=f'Latency Distributions {fname[0:-4]}')
    plt.savefig(f'{fname}.png', bbox_inches='tight')

if __name__ == '__main__':
    '''main function to read the pickle files and call the plotting function'''
    dirname = os.path.dirname(__file__)
    filetype='*LAT_KDE'
    dirroot = os.path.abspath(os.path.join(dirname,'..'))
    dirs = os.path.abspath(os.path.join(dirname,'..',filetype))
    
    #Loop through KDE pickle files, getting the associated raw files and passing them to the plotting function
    for dirkde in glob.glob(dirs):
        fname = dirkde[dirkde.rfind('\\')+1:]
        dirraw = os.path.abspath(os.path.join(dirroot,fname[:-3]+"RAW"))
        
        # #TEMP FIND THE DESIRED REGION
        # if dirkde.find("SYD_LDN") < 0 and dirkde.find("LAN") < 0 :
        #     continue
        
        plotLat(dirraw, dirkde, True)