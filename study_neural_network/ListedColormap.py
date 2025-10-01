import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)


colors = ['red', 'green', 'blue']
cmap = ListedColormap(colors)


plt.scatter(x, y, c=z, cmap=cmap, s=50)
plt.colorbar()
plt.show()