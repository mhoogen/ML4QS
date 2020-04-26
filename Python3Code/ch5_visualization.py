##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5 - Exemplary graphs                            #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy.stats import norm
from sklearn.decomposition import PCA
from Chapter4.FrequencyAbstraction import FourierTransformation
from matplotlib.patches import Rectangle
import re
import sklearn

np.random.seed(0)

# Figure 5.1

points_x = [0.25, 0.75]
points_y = [0.25, 0.75]

plt.plot(points_x, points_y, 'ro')
manhattan_x = [points_x[0], points_x[0], points_x[1]]
manhattan_y = [points_y[0], points_y[1], points_y[1]]
euclidean_x = [points_x[0], points_x[1]]
euclidean_y = [points_y[0], points_y[1]]

plt.plot(manhattan_x, manhattan_y, 'b-')
plt.plot(euclidean_x, euclidean_y, 'r:')

plt.legend(['$measurements$','$manhattan$ $distance$', '$euclidean$ $distance$'], loc=4, fontsize='small')
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()

# Figure 5.2 (complicated figure....)

df = pd.DataFrame(np.arange(0, 1, 0.001), columns=list('X'))
mean = 0.5
sd = 0.1
p = pd.DataFrame(norm.pdf(df,mean,sd), columns=list('p'))
mean2 = 0.6
sd2 = 0.2
p2 = pd.DataFrame(norm.pdf(df,mean2,sd2), columns=list('p'))

f, axarr = plt.subplots(7, 3)
f.subplots_adjust(hspace=0.8)

axarr[0, 0].axes.set_axis_off()
axarr[0, 0].set_xlim([0,1])
axarr[0, 0].set_ylim([0,1])
axarr[0, 0].text(0, 0.65, '1', fontsize=12,
        bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})
axarr[0, 0].text(0.2, 0.8, '$x_{1,qs_{i}}^{1},\dots,x_{N_{qs_{i}},qs_{i}}^{1}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[0, 0].plot([0.4, 0.4], [0.55, 0.75], 'k:')
axarr[0, 0].text(0.2, 0.2, '$x_{1,qs_{i}}^{p},\dots,x_{N_{qs_{i}},qs_{i}}^{p}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[0, 0].arrow(0.75, 0.65, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[0, 1].axes.set_axis_off()
axarr[0, 1].set_xlim([0,1])
axarr[0, 1].set_ylim([0,1])
axarr[0, 1].text(0.2, 0.8, '$x\_mean_{qs_{i}}^{1}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[0, 1].plot([0.4, 0.4], [0.55, 0.75], 'k:')
axarr[0, 1].text(0.2, 0.2, '$x\_mean_{qs_{i}}^{p}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[0, 2].arrow(0, 0.7, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[0, 2].axes.set_axis_off()
axarr[0, 2].set_xlim([0,1])
axarr[0, 2].set_ylim([0,1])
axarr[0, 2].text(0.3, 0.55, '$cluster$ $on$ $mean$ $values$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=12)
axarr[1, 0].axes.set_axis_off()
axarr[1, 1].plot(df['X'], p, 'b-')
axarr[1, 1].xaxis.set_ticklabels([])
axarr[1, 1].yaxis.set_ticklabels([])
axarr[1, 1].set_xlabel('$X_{1}$')
axarr[1, 1].set_ylabel('$P(X_{1})$')
axarr[1, 2].axes.set_axis_off()
axarr[2, 0].axes.set_axis_off()
axarr[2, 0].set_xlim([0,1])
axarr[2, 0].set_ylim([0,1])
axarr[2, 0].text(0, 0.65, '2', fontsize=12,
        bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})
axarr[2, 0].text(0.2, 0.8, '$x_{1,qs_{i}}^{1},\dots,x_{N_{qs_{i}},qs_{i}}^{1}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[2, 0].plot([0.4, 0.4], [0.55, 0.75], 'k:')
axarr[2, 0].text(0.2, 0.2, '$x_{1,qs_{i}}^{p},\dots,x_{N_{qs_{i}},qs_{i}}^{p}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[2, 0].arrow(0.75, 0.65, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[2, 1].axes.set_axis_off()
axarr[2, 1].set_xlim([0,1])
axarr[2, 1].set_ylim([0,1])
axarr[2, 1].plot([0.5,0.5], [0,1], 'k:')
axarr[2, 2].axes.set_axis_off()
axarr[2, 2].set_xlim([0,1])
axarr[2, 2].set_ylim([0,1])
axarr[2, 2].arrow(0, 0.7, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[2, 2].text(0.3, 0.55, '$cluster$ $on$ $distribution$ $parameters$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=12)
axarr[3, 0].axes.set_axis_off()
axarr[3, 0].set_xlim([0,1])
axarr[3, 0].set_ylim([0,1])
axarr[3, 1].plot(df['X'], p, 'b-')
axarr[3, 1].xaxis.set_ticklabels([])
axarr[3, 1].yaxis.set_ticklabels([])
axarr[3, 1].set_xlabel('$X_{p}$')
axarr[3, 1].set_ylabel('$P(X_{p})$')
axarr[3, 2].axes.set_axis_off()

axarr[4, 0].axes.set_axis_off()
axarr[4, 0].set_xlim([0,1])
axarr[4, 0].set_ylim([0,1])
axarr[4, 0].text(0, 0.65, '3', fontsize=12,
        bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})
axarr[4, 0].text(0.2, 0.8, '$x_{1,qs_{i}}^{1},\dots,x_{N_{qs_{i}},qs_{i}}^{1}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[4, 0].plot([0.4, 0.4], [0.55, 0.75], 'k:')
axarr[4, 0].text(0.2, 0.2, '$x_{1,qs_{i}}^{p},\dots,x_{N_{qs_{i}},qs_{i}}^{p}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[4, 0].plot([0.4, 0.4], [-0.2, 0.2], 'k:')
axarr[4, 1].plot(df['X'], p, 'b-')
axarr[4, 1].plot(df['X'], p2, 'r-')
axarr[4, 1].legend(['$i$', '$j$'], loc=2, fontsize='xx-small')
axarr[4, 1].xaxis.set_ticklabels([])
axarr[4, 1].yaxis.set_ticklabels([])
axarr[4, 1].set_xlabel('$X_{1}$')
axarr[4, 1].set_ylabel('$P(X_{1})$')
axarr[4, 2].axes.set_axis_off()
axarr[5, 0].axes.set_axis_off()
axarr[5, 0].set_xlim([0,1])
axarr[5, 0].set_ylim([0,1])
axarr[5, 0].text(0.2, 0.8, '$x_{1,qs_{j}}^{1},\dots,x_{N_{qs_{j}},qs_{j}}^{1}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[5, 0].plot([0.4, 0.4], [0.55, 0.75], 'k:')
axarr[5, 0].text(0.2, 0.2, '$x_{1,qs_{j}}^{p},\dots,x_{N_{qs_{j}},qs_{j}}^{p}$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=10)
axarr[5, 0].arrow(0.75, 0.65, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[5, 1].axes.set_axis_off()
axarr[5, 1].set_xlim([0,1])
axarr[5, 1].set_ylim([0,1])
axarr[5, 1].plot([0.5,0.5], [0,1], 'k:')
axarr[5, 2].axes.set_axis_off()
axarr[5, 2].set_xlim([0,1])
axarr[5, 2].set_ylim([0,1])
axarr[5, 2].arrow(0, 0.7, 0.15, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
axarr[5, 2].text(0.3, 0.55, '$cluster$ $on$ $p$ $values$ $between$ $i$ $and$ $j$',
        verticalalignment='bottom', horizontalalignment='left',
        color='black', fontsize=12)
axarr[6, 0].axes.set_axis_off()
axarr[6, 0].set_xlim([0,1])
axarr[6, 0].set_ylim([0,1])
axarr[6, 1].plot(df['X'], p, 'b-')
axarr[6, 1].plot(df['X'], p2, 'r-')
axarr[6, 1].legend(['$i$', '$j$'], loc=2, fontsize='xx-small')
axarr[6, 1].xaxis.set_ticklabels([])
axarr[6, 1].yaxis.set_ticklabels([])
axarr[6, 1].set_xlabel('$X_{p}$')
axarr[6, 1].set_ylabel('$P(X_{p})$')
axarr[6, 2].axes.set_axis_off()
plt.show()

# Figure 5.3

time = np.array([1,2,3,4,5,6,7])
y_arnold = np.array([0.2,0.2,0.5,0.2,0.2,0.2,0.2])
y_eric = np.array([0.18,0.18,0.18,0.34,0.5,0.34,0.18])

plt.plot(time, y_arnold, 'b-o')
plt.plot(time, y_eric, 'r:*')

plt.legend(['$Arnold$','$Eric$'], loc=1, fontsize='small')
plt.xlabel('time')
plt.ylabel('$X_{1}$')
plt.ylim([0,1])
plt.show()

# Figure 5.4

f, axarr = plt.subplots(2, 2)
f.subplots_adjust(hspace=0)
f.subplots_adjust(wspace=0)

axarr[0, 0].axes.set_axis_off()
axarr[0, 0].set_xlim([0,max(1-y_arnold)+0.05])
axarr[0, 0].set_ylim([1,8])
axarr[0, 0].plot(1-y_arnold, time+0.5, 'b-o')
axarr[1, 0].axes.set_axis_off()
axarr[0, 1].xaxis.set_ticklabels([])
axarr[0, 1].xaxis.set_ticks(time)
axarr[0, 1].set_xlim([1,8])
axarr[0, 1].yaxis.set_ticklabels([])
axarr[0, 1].yaxis.set_ticks(time)
for t in time:
    axarr[0, 1].plot([t,t], [min(time), max(time)+1], 'k:')
for t in time:
    axarr[0, 1].plot([min(time), max(time)+1], [t,t], 'k:')
axarr[0, 1].add_patch(Rectangle((1, 1), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((2, 2), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((3, 2), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((4, 3), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((5, 3), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((6, 3), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((7, 4), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((7, 5), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((7, 6), 1, 1,alpha=1))
axarr[0, 1].add_patch(Rectangle((7, 7), 1, 1,alpha=1))

axarr[0, 1].set_ylim([1,8])
axarr[1, 1].axes.set_axis_off()
axarr[1, 1].set_xlim([1,8])
axarr[1, 1].set_ylim([0,max(1-y_eric)+0.05])
axarr[1, 1].plot(time+0.5, 1-y_eric, 'r-*')
plt.show()

# Figure 5.5

np.random.seed(0)
f, axarr = plt.subplots(2, 2)
f.subplots_adjust(hspace=0.4)
f.subplots_adjust(wspace=0.4)

# Generate random data points.
numbers = np.vstack([np.random.randint(10,20,size=(10, 2)),np.random.randint(70,90,size=(10, 2))])
df = pd.DataFrame(numbers, columns=list('XY'))
centers = pd.DataFrame(np.vstack([[0.25, 0.35], [0.02, 0.02]]), columns=list('XY'))
df = df / float(100)
# Set the initial random centers to values such that we have a nice example.
axarr[0, 0].plot(centers['X'], centers['Y'], 'ko')
axarr[0, 0].plot(df['X'], df['Y'], 'ro')
axarr[0, 0].legend(['$centers$', '$data$ $points$'], loc=4, fontsize='small', numpoints=1)
axarr[0, 0].set_xlim([0,1])
axarr[0, 0].set_ylim([0,1])
axarr[0, 0].set_xlabel('$X_{1}$')
axarr[0, 0].set_ylabel('$X_{2}$')
axarr[0, 0].set_title('$step$ $1:$ $random$ $centers$')

# Determine the cluster for each of the data points

cluster = np.argmin(sklearn.metrics.pairwise.euclidean_distances(X=df, Y=centers), axis=1)
df['cluster'] = cluster

axarr[0, 1].plot(df[df['cluster']==0]['X'], df[df['cluster']==0]['Y'], 'ro')
axarr[0, 1].plot(df[df['cluster']==1]['X'], df[df['cluster']==1]['Y'], 'bo')
axarr[0, 1].plot(centers['X'], centers['Y'], 'ko')
axarr[0, 1].legend(['$cluster$ $1$', '$cluster$ $2$', '$centers$'], loc=4, fontsize='small', numpoints=1)
axarr[0, 1].set_xlim([0,1])
axarr[0, 1].set_ylim([0,1])
axarr[0, 1].set_xlabel('$X_{1}$')
axarr[0, 1].set_ylabel('$X_{2}$')
axarr[0, 1].set_title('$step$ $2:$ $cluster$ $assignment$')

# Update the centers

centers.iloc[0,:] = df[df['cluster']==0].mean(axis=0)[['X','Y']]
centers.iloc[1,:] = df[df['cluster']==1].mean(axis=0)[['X','Y']]

axarr[1, 0].plot(df[df['cluster']==0]['X'], df[df['cluster']==0]['Y'], 'ro')
axarr[1, 0].plot(df[df['cluster']==1]['X'], df[df['cluster']==1]['Y'], 'bo')
axarr[1, 0].plot(centers['X'], centers['Y'], 'ko')
axarr[1, 0].legend(['$cluster$ $1$', '$cluster$ $2$', '$centers$'], loc=4, fontsize='small', numpoints=1)
axarr[1, 0].set_xlim([0,1])
axarr[1, 0].set_ylim([0,1])
axarr[1, 0].set_xlabel('$X_{1}$')
axarr[1, 0].set_ylabel('$X_{2}$')
axarr[1, 0].set_title('$step$ $3:$ $update$ $centers$')

# And determine the cluster for each of the data points:

cluster = np.argmin(sklearn.metrics.pairwise.euclidean_distances(X=df[['X', 'Y']], Y=centers), axis=1)
df['cluster'] = cluster

axarr[1, 1].plot(df[df['cluster']==0]['X'], df[df['cluster']==0]['Y'], 'ro')
axarr[1, 1].plot(df[df['cluster']==1]['X'], df[df['cluster']==1]['Y'], 'bo')
axarr[1, 1].plot(centers['X'], centers['Y'], 'ko')
axarr[1, 1].legend(['$cluster$ $1$', '$cluster$ $2$', '$centers$'], loc=4, fontsize='small', numpoints=1)
axarr[1, 1].set_xlim([0,1])
axarr[1, 1].set_ylim([0,1])
axarr[1, 1].set_xlabel('$X_{1}$')
axarr[1, 1].set_ylabel('$X_{2}$')
axarr[1, 1].set_title('$step$ $4:$ $cluster$ $assignment$')

plt.show()

# Figure 5.7

np.random.seed(0)
numbers = np.vstack([np.random.randint(0,20,size=(20, 2)),
                     np.random.randint(60,100,size=(30, 2)),
                     np.random.randint(40,60,size=(20, 2))])
numbers = pd.DataFrame(numbers, columns=list('XY'))
numbers = numbers / float(100)

values = np.arange(0,1,0.2)
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(numbers['X'], numbers['Y'], 'ro')
plt.legend(['$data$ $points$'], loc=4, fontsize='small', numpoints=1)
for v in values:
    plt.plot([v,v], [min(values), max(values)+1], 'k:')
for v in values:
    plt.plot([min(values), max(values)+1], [v,v], 'k:')
    ax = plt.gca()

ax.add_patch(Rectangle((0.0, 0), 0.2, 0.2,alpha=0.5, color='grey'))
ax.add_patch(Rectangle((0.4, 0.4), 0.2, 0.2,alpha=0.5, color='grey'))
ax.add_patch(Rectangle((0.6, 0.6), 0.4, 0.4,alpha=0.5, color='grey'))
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')

plt.show()
