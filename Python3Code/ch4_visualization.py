##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4 - Exemplary graphs                            #
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
import re

np.random.seed(0)

# Figure 4.1

# Sample frequency (Hz)
fs = 10

# Create time points....
df = pd.DataFrame(np.arange(0, 16.1, float(1)/fs), columns=list('X'))
c1 = 3 * np.sin(2 * math.pi * 0.2 * df['X'])
c2 = 2 * np.sin(2 * math.pi * 0.25 * (df['X']-2)) + 5
df['Y'] = c1 + c2

plt.plot(df['X'], df['Y'], 'b-')
plt.legend(['$example$ $measurement$ $sequence$'], loc=3, fontsize='small')
plt.xlabel('time')
plt.ylabel('$X_{1}$')
plt.show()

# Figure 4.2

FreqAbs = FourierTransformation()
data_table = FreqAbs.abstract_frequency(copy.deepcopy(df), ['Y'], 160, fs)
# Get the frequencies from the columns....
frequencies = []
values = []
for col in data_table.columns:
    val = re.findall(r'freq_\d+\.\d+_Hz', col)
    if len(val) > 0:
        frequency = float((val[0])[5:len(val)-4])
        frequencies.append(frequency)
        values.append(data_table.loc[data_table.index, col])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlim([0, 5])
ax1.plot(frequencies, values, 'b+')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('$a$')
plt.show()

