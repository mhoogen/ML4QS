##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 6 - Exemplary graphs                            #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import math
import copy
from scipy.stats import norm
from sklearn.decomposition import PCA
from Chapter4.FrequencyAbstraction import FourierTransformation
from matplotlib.patches import Rectangle
import re
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


np.random.seed(0)

# Figure 6.1

df = pd.DataFrame(columns=['x', 'y'])
x = np.random.normal(0, 0.5, 100)
df['x'] = x
y = 2.5 * x + 3
df['y'] = y

a = np.arange(0, 5, 0.1)
b = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(a, b)

result = np.empty((0,3))
for i in b:
    for j in a:
        y_calc = x * i + j
        error = sklearn.metrics.mean_squared_error(y, y_calc)
        result = np.vstack([result, [i, j, error]])

X, Y = np.meshgrid(a, b)
e_df = pd.DataFrame(result, columns=['b', 'a', 'error'])

Z = e_df['error'].values.reshape(len(X),len(Y))

fig = plot.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='brg_r')
#ax.scatter(e_df['a'], e_df['b'], e_df['error'])
ax.set_xlabel('$\\theta_{1}$')
ax.set_ylabel('$\\theta_{2}$')
ax.set_zlabel('$E_{in}(h)$')
fig.colorbar(surf, shrink=0.5, aspect=5)

plot.show()

# Figure 6.2

V, U = np.gradient(Z, .2, .2)
Q = plot.quiver(X, Y, -U, -V, pivot='mid', units='inches')
plot.xlabel('$\\theta_{1}$')
plot.ylabel('$\\theta_{2}$')
plot.show()

p = plot.contour(X, Y, Z,cmap='brg_r')
plot.clabel(p, fontsize=9, inline=1)
current_value = np.array([0,0])
x_values = [0]
y_values = [0]
V, U = np.gradient(Z, .1, .1)
steps = 1000
for i in range(0, steps):
    current_value = current_value - [0.1*V[int(current_value[0]/0.1), int(current_value[1]/0.1)],
                                     0.1*U[int(current_value[0]/0.1), int(current_value[1]/0.1)]]
    x_values.append(current_value[0])
    y_values.append(current_value[1])
plot.plot(x_values, y_values, 'k:')
plot.gca().arrow(x_values[-1]-0.1, y_values[-1], +0.0001, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
plot.xlabel('$\\theta_{1}$')
plot.ylabel('$\\theta_{2}$')
plot.legend(['$gradient$ $descent$ $path$'], loc=1, fontsize='small')

plot.show()