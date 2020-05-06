##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7 - Exemplary graphs                            #
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
import random
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)


# Figure 7.2

df1 = pd.DataFrame(np.random.normal(30,5,size=(50, 2)), columns=list('XY'))
df2 = pd.DataFrame(np.random.normal(70,5,size=(50, 2)), columns=list('XY'))
df1 = df1 / float(100)
df2 = df2 / float(100)
# Set the initial random centers to values such that we have a nice example.
plot.plot(df1['X'], df1['Y'], 'ro')
plot.plot(df2['X'], df2['Y'], 'bo')
plot.plot([0,1],[1,0],'k:')
plot.legend(['$class$ $1$ $(active)$', '$class$ $2$ $(inactive)$', '$decision$ $boundary$'], loc=4, fontsize='small', numpoints=1)
plot.xlim([0,1])
plot.ylim([0,1])
plot.xlabel('$X_{1}$')
plot.ylabel('$X_{2}$')
plot.show()

# Figure 7.3

df1 = pd.DataFrame(np.vstack([np.hstack([np.random.normal(50,5,size=(50, 1)), np.random.normal(80,5,size=(50, 1))]),
                              np.hstack([np.random.normal(50,5,size=(50, 1)), np.random.normal(20,5,size=(50, 1))])]),
                              columns=list('XY'))
df1 = df1 / float(100)
df2 = pd.DataFrame(np.vstack([np.hstack([np.random.normal(20,5,size=(50, 1)), np.random.normal(50,5,size=(50, 1))]),
                              np.hstack([np.random.normal(80,5,size=(50, 1)), np.random.normal(50,5,size=(50, 1))])]),
                              columns=list('XY'))
df2 = df2 / float(100)
# Set the initial random centers to values such that we have a nice example.
plot.plot(df1['X'], df1['Y'], 'ro')
plot.plot(df2['X'], df2['Y'], 'bo')
x = np.arange(0,1.1,0.1)
y1 = x
y2 = 1-x
plot.plot(x,y1,'k:')
plot.plot(x,y2,'k:')
ax = plot.gca()
ax.fill_between(x, y1, y2, where=y1<=y2, facecolor='grey', linewidth=0.0)
ax.fill_between(x, y2, y1, where=y1>=y2, facecolor='grey', linewidth=0.0)
ax.annotate('$P_{1}$', xy=(0.2, 0.2), xytext=(0.3, 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3.5, headlength=3), fontsize=15)
ax.annotate('$P_{2}$', xy=(0.2, 0.8), xytext=(0.3, 0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3.5, headlength=3), fontsize=15)

plot.legend(['$class$ $1$ $(active)$', '$class$ $2$ $(inactive)$', '$decision$ $boundary$'], loc=4, fontsize='small', numpoints=1)
plot.xlim([0,1])
plot.ylim([0,1])
plot.xlabel('$X_{1}$')
plot.ylabel('$X_{2}$')
plot.show()

# Figure 7.6
df1 = pd.DataFrame(np.random.normal(30,5,size=(50, 2)), columns=list('XY'))
df2 = pd.DataFrame(np.random.normal(70,5,size=(50, 2)), columns=list('XY'))
df1 = df1 / float(100)
df2 = df2 / float(100)

# Given the way we have generate out data, the line
# y = 1-x is the best separating line. This can be
# written as -x-y+1 = 0

# Let us computer the distance of our data points to this line...
result1 = ((-1 * df1['X']) + (-1 * df1['Y']) + 1).abs() / float(math.sqrt(2))
df1['dist'] = result1
index_closest_point_1 = df1['dist'].idxmin(axis=0)
result2 = ((-1 * df2['X']) + (-1 * df2['Y']) + 1).abs() / float(math.sqrt(2))
df2['dist'] = result2
index_closest_point_2 = df2['dist'].idxmin(axis=0)

# And draw the two lines that go through this point:
b_1 = df1.loc[index_closest_point_1, 'X'] + df1.loc[index_closest_point_1, 'Y']
b_2 = df2.loc[index_closest_point_2, 'X'] + df2.loc[index_closest_point_2, 'Y']
x = np.arange(0,1.1, 0.1)
y_1 = -1 * x + b_1
y_2 = -1 * x + b_2
y = (y_1 + y_2) / 2


# Set the initial random centers to values such that we have a nice example.
plot.plot(df1['X'], df1['Y'], 'ro')
plot.plot(df2['X'], df2['Y'], 'bo')
plot.plot(x,y,'k-')
plot.plot(x,y_1,'k:')
plot.plot(x,y_2,'k:')
plot.legend(['$class$ $1$ $(active)$', '$class$ $2$ $(inactive)$', '$decision$ $boundary$'], loc=2, fontsize='small', numpoints=1)

plot.xlim([0,1])
plot.ylim([0,1])
plot.xlabel('$X_{1}$')
plot.ylabel('$X_{2}$')
ax = plot.gca()
ax.annotate('$w^{T}x + b=1$', xy=(x[9], y_2[9]), xytext=(x[9] - 0.08, y_2[9] + 0.08),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3.5, headlength=3), fontsize=15)
ax.annotate('$w^{T}x + b=0$', xy=(x[8], y[8]), xytext=(x[8] - 0.08, y[8] + 0.08),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3.5, headlength=3), fontsize=15)
ax.annotate('$w^{T}x + b=-1$', xy=(x[7], y_1[7]), xytext=(x[7] - 0.08, y_1[7] + 0.08),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3.5, headlength=3), fontsize=15)
ax.annotate('', (x[2], y_1[2]), (x[4]-0.02, y_2[4]+0.02), arrowprops={'arrowstyle':'<->'})
ax.text(0.25, 0.6, '$2/|W|$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
plot.show()

# Figure 7.7


x = np.arange(0, 1.05, 0.05)

# Radius of one cicle is 0.1, the other circle 0.3

x_1 = []
x_2 = []
y_1 = []
y_2 = []
for i in range(0,1000):
    angle = random.uniform(0,1)*(math.pi*2)
    x_1.append(math.cos(angle)/3 + 0.5 + random.gauss(0, 0.025))
    x_2.append(math.cos(angle)/10 + 0.5 + random.gauss(0, 0.025))
    y_1.append(math.sin(angle)/3 + 0.5 + random.gauss(0, 0.025))
    y_2.append(math.sin(angle)/10 + 0.5 + random.gauss(0, 0.025))

fig = plot.figure(figsize=plot.figaspect(2.))
ax = fig.add_subplot(1, 2, 1)

ax.plot(x_1,y_1,'ro')
ax.plot(x_2,y_2,'bo')
ax.legend(['$class$ $1$ $(other$ $activity)$', '$class$ $2$ $(walking)$'], loc=2, fontsize='small', numpoints=1)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('$X_{1}$')
ax.set_ylabel('$X_{2}$')

df1 = pd.DataFrame(columns=list('XY'))
df1['X'] = x_1
df1['Y'] = y_1
df2 = pd.DataFrame(columns=list('XY'))
df2['X'] = x_2
df2['Y'] = y_2
sigma = 1
z_1 = np.power(math.e, -(sklearn.metrics.pairwise.euclidean_distances(X=df1, Y=np.array([0.5, 0.5]).reshape(1, -1))/2 * math.pow(sigma, 2)))
z_2 = np.power(math.e, -(sklearn.metrics.pairwise.euclidean_distances(X=df2, Y=np.array([0.5, 0.5]).reshape(1, -1))/2 * math.pow(sigma, 2)))
ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(x_1, y_1, z_1, color='r', marker='o')
ax.scatter(x_2, y_2, z_2, color='b', marker='o')
ax.set_xlabel('$X_{1}$')
ax.set_ylabel('$X_{2}$')
ax.set_zlabel('$e^{||x-x''||^{2}/2\cdot\sigma^{2}}$')
ax.legend(['$class$ $1$ $(other$ $activity)$', '$class$ $2$ $(walking)$'], loc=2, fontsize='small', numpoints=1)

plot.show()

# Figure 7.8

df1 = pd.DataFrame(np.random.normal(30,5,size=(50, 2)), columns=list('XY'))
df2 = pd.DataFrame(np.random.normal(70,5,size=(50, 2)), columns=list('XY'))
df1 = df1 / float(100)
df2 = df2 / float(100)
# Set the initial random centers to values such that we have a nice example.
plot.plot(df1['X'], df1['Y'], 'ro')
plot.plot(df2['X'], df2['Y'], 'bo')
plot.plot([0.51],[0.51],'ko')
k = 3
df_full = pd.concat([df1, df2], ignore_index=True)
distances_df_full = sklearn.metrics.pairwise.euclidean_distances(X=df_full, Y=np.array([0.51, 0.51]).reshape(1, -1)).flatten()
ind = np.argsort(distances_df_full)[:k]
plot.plot(df_full.loc[ind, 'X'], df_full.loc[ind, 'Y'] ,'y*', markersize=12)


plot.legend(['$class$ $1$ $(active)$', '$class$ $2$ $(inactive)$', '$new$ $point$', '$nearest$ $neighbors$'], loc=4, fontsize='small', numpoints=1)
plot.xlim([0,1])
plot.ylim([0,1])
plot.xlabel('$X_{1}$')
plot.ylabel('$X_{2}$')
plot.show()
