import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import os
import glob

#Script to obtain data, downsample, and run an one-way ANOVA test to determine statistical difference between TCP and UDP results

#obtain absolute pathing for results
dirname = os.path.dirname(__file__)
filetype='out_*.csv'
dirtcp = os.path.abspath(os.path.join(dirname,'..','LAN_TCP','Output',filetype))
dirudp = os.path.abspath(os.path.join(dirname,'..','LAN_UDP','Output',filetype))

sample_size = 700000
sample_small = 30000

print('\n\nUDP FILES\n')

#Initialise lists
lat_list = []
jit_list = []
loss1_list = []
loss2_list = []

#loop through and obtain data
for fpath in glob.glob(dirudp): #for each csv file...
    print(fpath)
    df = pd.read_csv(fpath)
    #drop NA values and reset indicies
    lat_list.append(df.iloc[:,2].dropna(axis=0).reset_index(drop=True))
    jit_list.append(df.iloc[:,4].dropna(axis=0).reset_index(drop=True))
    loss1_list.append(df.iloc[:,7].dropna(axis=0).reset_index(drop=True))
    loss2_list.append(df.iloc[:,8].dropna(axis=0).reset_index(drop=True))

#convert to numpy arrays and sample
udp_lat = pd.concat(lat_list,ignore_index=True).sample(n=sample_size).to_numpy()
udp_jit = pd.concat(jit_list,ignore_index=True).sample(n=sample_size).to_numpy()
udp_loss1 = pd.concat(loss1_list,ignore_index=True).sample(n=sample_size).to_numpy()
udp_loss2 = pd.concat(loss2_list,ignore_index=True).sample(n=sample_small).to_numpy()

print('\n\nTCP FILES\n')
lat_list = []
jit_list = []
loss1_list = []
loss2_list = []
for fpath in glob.glob(dirtcp):
    print(fpath)
    df = pd.read_csv(fpath)
    #drop NA values and reset indicies
    lat_list.append(df.iloc[:,2].dropna(axis=0).reset_index(drop=True))
    jit_list.append(df.iloc[:,8].dropna(axis=0).reset_index(drop=True))
    loss1_list.append(df.iloc[:,13].dropna(axis=0).reset_index(drop=True))
    loss2_list.append(df.iloc[:,14].dropna(axis=0).reset_index(drop=True))

#convert to numpy arrays and sample
tcp_lat = pd.concat(lat_list,ignore_index=True).sample(n=sample_size).to_numpy()
tcp_jit = pd.concat(jit_list,ignore_index=True).sample(n=sample_size).to_numpy()
tcp_loss1 = pd.concat(loss1_list,ignore_index=True).sample(n=sample_size).to_numpy()
tcp_loss2 = pd.concat(loss2_list,ignore_index=True).sample(n=sample_small).to_numpy()

lat_list = []
jit_list = []
loss1_list = []
loss2_list = []

F, p = f_oneway(udp_lat, tcp_lat)
print("Latency: Fstat: "+str(F)+" pValue: "+str(p))
F, p = f_oneway(udp_jit, tcp_jit)
print("Jitter: Fstat: "+str(F)+" pValue: "+str(p))
F, p = f_oneway(udp_loss1, tcp_loss1)
print("Loss (200ms): Fstat: "+str(F)+" pValue: "+str(p))
F, p = f_oneway(udp_loss2, tcp_loss2)
print("Loss (5000ms): Fstat: "+str(F)+" pValue: "+str(p))