import os
import glob
import pickle
import matplotlib.pyplot as plt

#Script to process the consecutive loss pickle into transition probabilities

fname = ''

pbad = open(f'{fname}_pbad','wb')
fig,ax = plt.subplots()
indices = range(len(pbad))

ax.bar(indices,pbad)
ax.set(xlabel='Consecutive packet losses', ylabel='Probability', title=f'Markov Probabilities in {fname[0:fname.rfind("_")]}')
plt.savefig(f'{fname}.png', bbox_inches='tight')