##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3 - Exemplary graphs                            #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from Chapter3.DataTransformation import LowPassFilter
from sklearn.decomposition import PCA

np.random.seed(0)

# Figure 3.1

df = pd.DataFrame(np.arange(0, 1, 0.001), columns=list('X'))
mean = 0.5
sd = 0.1
p = pd.DataFrame(norm.pdf(df,mean,sd), columns=list('p'))

plt.plot(df, p)
plt.xlabel('$X_{1}$')
plt.ylabel('$P(X_{1})$')
ax = plt.gca()
ax.fill_between(df['X'], 0, p['p'], where=df['X']<=0.3, facecolor='red')
ax.fill_between(df['X'], 0, p['p'], where=df['X']>=0.7, facecolor='red')
ax.annotate('outliers', xy=(0.3, 0.25), xytext=(0.45, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('outliers', xy=(0.7, 0.25), xytext=(0.45, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# Figure 3.2

df = pd.DataFrame(np.random.random_sample(size=(100, 2)), columns=list('XY'))
plt.plot(df['X'], df['Y'], 'ro')
plt.xlabel('X$_{1}$')
plt.ylabel('X$_{2}$')

plt.plot([0.5], [0.5], 'ko')

# draw the circle with the arrow
# http://stackoverflow.com/questions/34823886/plotting-circle-diagram-with-rotary-arrow

radius = 0.2
angle = 20
angle_rad = angle * math.pi / 180  # degrees to radians
# Draw circle
circle = plt.Circle((0.5,0.5), radius, color='black', fill=False)
fig = plt.gcf()
fig.gca().add_artist(circle)

ax = plt.gca()
ax.arrow(0.5, 0.5,
         (radius - 0.02) * math.cos(angle_rad),
         (radius - 0.02) * math.sin(angle_rad),
         head_width=0.02, head_length=0.02, fc='k', ec='k')
ax.annotate('$d_{min}$', xy=(.6, .5),  xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')
plt.show()


# Figure 3.3

np.random.seed(0)
df1 = pd.DataFrame(np.random.randint(10,20,size=(40, 2)), columns=list('XY'))
df2 = pd.DataFrame(np.random.randint(70,90,size=(5, 2)), columns=list('XY'))
df1 = df1 / 100
df2 = df2 / 100

plt.plot(df1['X'], df1['Y'], 'ro')
plt.plot(0.7, 0.7, 'ro')
plt.plot(df2['X'], df2['Y'], 'ro')
plt.plot(0.2, 0.2, 'ro')
plt.xlabel('X$_{1}$')
plt.ylabel('X$_{2}$')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0.25], [0.25], 'ko')
plt.plot([0.65], [0.65], 'ko')
plt.show()

# Figure 3.4

# Sample frequency (Hz)
fs = 100

# Create time points....
t = pd.DataFrame(np.arange(0, 16, float(1)/fs), columns=list('X'))
c1 = 3 * np.sin(2 * math.pi * 0.1 * t)
c2 = 2 * np.sin(2 * math.pi * t)

plt.plot(t, c1, 'b--')
plt.plot(t, c2, 'b:')
plt.plot(t, c1+c2, 'b-')
LowPass = LowPassFilter()
new_dataset = LowPass.low_pass_filter(c1+c2, 'X', fs, 0.5, order=3, phase_shift=True)
plt.plot(t, new_dataset['X_lowpass'], 'r-')
plt.legend(['$3 \cdot sin(2 \cdot \pi \cdot 0.1 \cdot t))$', '$2 \cdot sin(2 \cdot \pi \cdot t))$', '$combined$', '$combined$ $after$ $filter (f_{c}=0.5Hz, n=3)$'],
            loc=4, fontsize='small')
plt.xlabel('time')
plt.ylabel('$X_{1}$')
plt.show()

# Figure 3.5

df = pd.DataFrame(np.arange(0, 1, 0.1), columns=list('X'))
df['Y'] = pd.DataFrame(np.random.normal(0, 0.1, size=(10,1)), columns=list('Y'))
df['Y'] = df['Y'] + df['X']

pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
first_component = pca.components_[0]
second_component = pca.components_[1]

factor_1 = first_component[0]/first_component[1]
factor_2 = second_component[0]/second_component[1]


plt.plot(df['X'], df['Y'], 'ro')
plt.plot(df['X'], df['X']*factor_1, 'r-')
plt.plot(df['X'], df['X']*factor_2+0.5, 'b-')
plt.legend(['$data$', '$first$ $component$', '$second$ $component$'], loc=2)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')

plt.show()

# Figure 3.6

transformed_dataset = np.inner(first_component, df)

plt.plot(transformed_dataset, [0]*transformed_dataset.shape[0], 'ro')
plt.ylim([-0.05,1])
plt.xlabel('$X\'_{1}$')
ax = plt.gca()
ax.get_yaxis().set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position('zero')
plt.legend(['$transformed$ $data$'], loc=(0.5, 0.1))

plt.show()

# Figure 3.7

transformed_dataset = np.inner(pca.components_, df)

plt.plot(transformed_dataset[0], transformed_dataset[1], 'ro')
plt.xlabel('$X\'_{1}$')
plt.ylabel('$X\'_{2}$')
plt.legend(['$transformed$ $data$'], loc=4)

plt.show()
