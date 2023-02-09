import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import os
import glob

import pickle

#Script to count the number of consecutive packet losses encountered during the testing process

#obtain absolute pathing for results
dirname = os.path.dirname(__file__)
filetype='raw_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','LAN_TCP','Output',filetype))
dirwan = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','WAN_TCP','**','Output',filetype))
dirudp = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','LAN_UDP','Output',filetype))

print('\n\nWAN TCP FILES\n')

loss_sum = 0
loss_count = 0

#DETERMINE MARKOV CHAIN PROBABILITIES

#List of integers counting the number of consecutive packet losses before the encountered instance 
#i.e. index 0 is the number of "first" packet losses, index 1 is the number of "second consecutive losses"
consecutive_list = [0]*1000000
total_count = 0

for fpath in glob.glob(dirwan): #for each TCP csv file...
    con_counter = 0 #reset counter
    print(fpath) #track files
    df = pd.read_csv(fpath) #drop NA values and reset indicies
    loss_df = df.iloc[:,3].dropna(axis=0).reset_index(drop=True)
    #loop through all rows and get the data
    for i in range(0,len(loss_df.index)):
        total_count += 1
        if loss_df.iat[i] > 0:
            consecutive_list[con_counter] += 1
            con_counter += 1
        else:
            con_counter = 0

#export data
markov_data = [consecutive_list, total_count]
consecutiveFile = open('consecutivePickle','wb')
pickle.dump(markov_data,consecutiveFile)
consecutiveFile.close()

print("LIST: "+ str(consecutive_list[0]) + " " + str(consecutive_list[1]) + " " + str(consecutive_list[2]))
print("COUNT: "+ str(total_count))