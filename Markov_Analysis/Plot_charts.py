import os
import glob
import pickle
import matplotlib.pyplot as plt
from Plot_pickle import plotMProb
from sklearn.neighbors import KernelDensity

#A script to plot the latency and packet loss charts
def plotLat(rawpath, kdepath):
    rawFile = open(rawpath,'rb')
    rawDist = pickle.load(rawFile)
    rawFile.close()
    
    kdeFile = open(kdepath,'rb')
    kdeDist = pickle.load(kdeFile)
    kdeFile.close()
    
    #generate kde
    kdeDist = KernelDensity(kernel = 'gaussian',bandwidth=0.17).fit(rawDist)
    
    y1 = kdeDist.sample(100000)
    nbins = 200
    
    #plot
    fig, ax = plt.subplots()
    ax.hist(rawDist, density = True, bins = nbins, color='blue', label='raw')
    ax.hist(y1.ravel(), density = True, bins = nbins, histtype=u'step', color='red', label='kde', linewidth=2)
    ax.legend(loc="upper right")
    ax.set(xlabel='Latency Variation (ms)', ylabel='Sample Density', title=f'Markov Probabilities in {fname[0:fname.rfind("_")]}')
    plt.show()

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
        
        #TEMP FIND THE DESIRED REGION
        if dirkde.find("SYD_LDN") < 0 and dirkde.find("LAN") < 0 :
            continue
        
        plotLat(dirraw, dirkde)
        pass