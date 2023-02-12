import os
import glob
import pickle
import matplotlib.pyplot as plt

#Script to process the consecutive loss pickle into transition probabilities

def plotMProb(pbad,fname):
    fig,ax = plt.subplots()
    indices = range(len(pbad))
    
    ax.bar(indices,pbad)
    ax.set(xlabel='Consecutive packet losses', ylabel='Probability', title=f'Markov Probabilities in {fname[0:fname.rfind("_")]}')
    plt.savefig(f'{fname}.png', bbox_inches='tight')

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    filetype='*ConLoss'
    dirroot = os.path.abspath(os.path.join(dirname,'..'))
    dirs = os.path.abspath(os.path.join(dirname,'..',filetype))
    
    for fpath in glob.glob(dirs):
        fname = fpath[fpath.rfind('\\')+1:]
        dirfile = os.path.abspath(os.path.join(dirname,'..',fname))
        consecutiveFile = open(dirfile,'rb')
        consecutive = pickle.load(consecutiveFile)
        consecutiveFile.close()

        x1 = consecutive[0][0:500] # Array of loss frequencies
        data_count = consecutive[1] # Total number of data points
        ind = range(0,len(x1))

        #Calculating the probabilities for use in an extended markov chain
        P_bad = []
        P_bad.append(consecutive[0][0]/data_count)
        for i in range(1,len(x1)):
            try:
                P_bad.append(consecutive[0][i]/consecutive[0][i-1])
            except ZeroDivisionError:
                P_bad.append(0)

        pbadFile = open(f'{fname}_pbad','wb')
        pickle.dump(P_bad,pbadFile)
        pbadFile.close()
        print(P_bad)
        plotMProb(P_bad,fname)