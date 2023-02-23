import os
import glob
import pickle
import matplotlib.pyplot as plt

#Script to process the consecutive loss pickle into transition probabilities

pbadFile = open('./SYD_LDN_ConLoss_pbad','rb')
p_bad = pickle.load(pbadFile)
pbadFile.close()

print(p_bad[0:10])
print(p_bad[0:-5])

pbadFile = open('./LAN_ConLoss_pbad','rb')
p_bad_lan = pickle.load(pbadFile)
pbadFile.close()

print(p_bad_lan[0:10])