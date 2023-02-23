import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import os
import glob
import pickle

# Script to find the mean and std deviation for the loss values 

### Absolute path to external drive
dirname = 'E:\\'
filetype='out_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','LAN_TCP'))
dirwan = os.path.abspath(os.path.join(dirname,'Sfti_Network_Testing_Data','WAN_TCP','**'))

syd_sg1   = 0
syd_sg2   = 0
ldn_ore1  = 0
ldn_ore2  = 0
sg_ldn1   = 0
sg_ldn2   = 0
syd_ore1  = 0
syd_ore2  = 0
sg_ore1   = 0
sg_ore2   = 0
syd_ldn1  = 0
syd_ldn2  = 0


dirs = [dirtcp,dirwan]

# List of integers counting the number of consecutive packet losses before the encountered instance
# i.e. index 0 is the number of "first" packet losses, index 1 is the number of "second consecutive"
consecutive_list = [0]*1000000
total_count = 0

# for each of WAN and LAN
for dir in dirs:
    # Analysis of markov chain probabilities
    for fpath in glob.glob(dir): #for each region...
        file_count = 0
        file_cur = 1
        #Append for wan 
        if dir.find("WAN_TCP") >= 0:
            #label
            label = fpath[fpath.find("WAN_TCP")+8:]
            fpath = os.path.abspath(os.path.join(fpath,'Output',filetype))
        else:
            label = "LAN"
            
        fpath = os.path.abspath(os.path.join(fpath,'Output',filetype))
            
        for flpath in glob.glob(fpath):
            file_count += 1
        
        #loop through TCP files
        for flpath in glob.glob(fpath):
            con_counter = 0
            #report progress
            print(f'Calculating markov probabilities for: {flpath}         Progress: {file_cur}/{file_count}', end='\r')
            df = pd.read_csv(flpath)
            #drop NA values and reset indicies
            loss_df = df.iloc[:,3].dropna(axis=0).reset_index(drop=True)
            #loop through all rows and get the data
            for i in range(0,len(loss_df.index)):
                total_count += 1
                if loss_df.iat[i] > 0:
                    consecutive_list[con_counter] += 1
                    con_counter += 1
                    if loss_df.iat[i] > 1:
                        print(f"LOSS INDICATOR GREATER THAN 1 AT: {loss_df.iat[i]}   {i}   {flpath}")
                else:
                    con_counter = 0
            file_cur += 1

        #Save data for each region
        markov_data = [consecutive_list, total_count]
        consecutiveFile = open(f'{label}_ConLoss','wb')
        pickle.dump(markov_data,consecutiveFile)
        consecutiveFile.close()

print("LIST: "+ str(consecutive_list[0]) + " " + str(consecutive_list[1]) + " " + str(consecutive_list[2]))
print("COUNT: "+ str(total_count))