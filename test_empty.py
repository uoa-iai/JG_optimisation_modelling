import numpy as np
from matplotlib import pyplot as plt
import collections

d = collections.deque(maxlen=10)
for i in range(0,20):
    d.append(i)
    print(d)
    print(sum(d)/float(len(d)))
