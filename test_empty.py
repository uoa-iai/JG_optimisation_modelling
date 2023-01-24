import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x = np.linspace(-2, 2, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
f, axes = plt.subplots()
axes.plot(x, y1, c='r', label="sine")
axes.legend(loc='upper left')

plt.show()