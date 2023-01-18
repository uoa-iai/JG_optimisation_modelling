import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import os
import glob

import pickle

#Script to obtain data, downsample, and run an one-way ANOVA test to determine statistical difference between TCP and UDP results

#obtain absolute pathing for results
dirname = os.path.dirname(__file__)
filetype='raw_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','LAN_TCP','Output',filetype))
dirwan = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','WAN_TCP','**','Output',filetype))
dirudp = os.path.abspath(os.path.join(dirname,'..','..','\Sfti_Network_Testing_Data','LAN_UDP','Output',filetype))

print('\n\nLAN TCP FILES\n')

# loss_sum = 0
# loss_count = 0

# #loop through and obtain data
# for fpath in glob.glob(dirtcp): #for each csv file...
#     print(fpath)
#     df = pd.read_csv(fpath)
#     #drop NA values and reset indicies
#     loss_df = df.iloc[:,3].dropna(axis=0).reset_index(drop=True)
#     loss_sum += loss_df.sum(axis=0)
#     loss_count += loss_df.size

# loss_mean = loss_sum/loss_count
# print(loss_mean)

print('\n\nWAN TCP FILES\n')

loss_sum = 0
loss_count = 0

#List of integers counting the number of consecutive packet losses before the encountered instance 
#i.e. index 0 is the number of "first" packet losses, index 1 is the number of "second consecutive"
consecutive_list = [0]*1000000
total_count = 0

#loop through and obtain data
#Analysis of markov chain probabilities
for fpath in glob.glob(dirwan): #for each TCP csv file...
    con_counter = 0
    print(fpath)
    df = pd.read_csv(fpath)
    #drop NA values and reset indicies
    loss_df = df.iloc[:,3].dropna(axis=0).reset_index(drop=True)
    #loop through all rows and get the data
    # loss_sum += loss_df.sum(axis=0)
    # loss_count += loss_df.size
    for i in range(0,len(loss_df.index)):
        total_count += 1
        if loss_df.iat[i] > 0:
            consecutive_list[con_counter] += 1
            con_counter += 1
        else:
            con_counter = 0

# loss_mean = loss_sum/loss_count
# print(loss_mean)
markov_data = [consecutive_list, total_count]
consecutiveFile = open('consecutivePickle','wb')
pickle.dump(markov_data,consecutiveFile)
consecutiveFile.close()

print("LIST: "+ str(consecutive_list[0]) + " " + str(consecutive_list[1]) + " " + str(consecutive_list[2]))
print("COUNT: "+ str(total_count))