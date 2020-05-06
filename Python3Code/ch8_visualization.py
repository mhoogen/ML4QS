##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8 - Exemplary graphs                            #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.dates as md
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
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
import pyflux as pf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels

np.random.seed(0)


# Figure 8.1

time = np.arange(1,15,1)

pred_mood = [0.5]
pred_al = [0.4]
x = [1]
delta_t = 1
gamma_1 = 5
gamma_2 = 0.8
gamma_3 = 0.25
gamma_4 = 1
gamma_5 = 1

for t in range(1, len(time)):
    pred_mood.append(pred_mood[-1] + x[-1]*(gamma_1*(1-pred_mood[-1])*max((pred_al[-1] - pred_mood[-1]), 0) + gamma_2*pred_mood[-1]*(min((pred_al[-1] - pred_mood[-1]), 0)))*delta_t)
    pred_al.append(pred_al[-1] + gamma_3 * (pred_al[-1] * min(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0) + (1-pred_al[-1])* max(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0)))
    x.append(x[-1]);

values_mood = pred_mood
activity_level = pred_al

plot.plot(time, pred_mood, 'ro-')
plot.plot(time, pred_al, 'bo:')
plot.ylim([0,1])
plot.xlabel('time')
plot.ylabel('value')
plot.legend(['$mood$', '$activity$ $level$'], loc=4, fontsize='small', numpoints=1)
plot.show()

# Figure 8.2

f, axarr = plot.subplots(3, 2)
f.subplots_adjust(hspace=0.5)
f.subplots_adjust(wspace=0.5)
random.seed(0)
random_time_series = pd.DataFrame(np.random.normal(0.1, 2, 5200), index=range(0, 5200), columns=['value'])
axarr[0,0].plot(random_time_series.index, random_time_series['value'])
axarr[0,0].set_xlim([101, 500])
axarr[0,0].set_ylim([-10, 10])
axarr[0,0].set_xlabel('time')
axarr[0,0].set_ylabel('value')

autocorrelation_plot(random_time_series['value'], ax=axarr[0, 1])
axarr[0, 1].set_xlim([0, 30])
axarr[0, 1].set_ylim([-1.1, 1.1])

#rolling_window_data = pd.rolling_mean(random_time_series['value'], 10)
rolling_window_data = pd.Series(random_time_series['value']).rolling(window=10).mean()
axarr[1,0].plot(random_time_series.index, rolling_window_data)
axarr[1,0].set_xlim([101, 500])
axarr[1,0].set_ylim([-10, 10])
axarr[1,0].set_xlabel('time')
axarr[1,0].set_ylabel('value')

autocorrelation_plot(rolling_window_data[10:], ax=axarr[1, 1])
axarr[1, 1].set_xlim([0, 30])
axarr[1, 1].set_ylim([-1.1, 1.1])

cumsum_data = random_time_series.cumsum(axis=0)
axarr[2,0].plot(random_time_series.index, cumsum_data)
axarr[2,0].set_xlim([0, 5000])
axarr[2,0].set_xlabel('time')
axarr[2,0].set_ylabel('value')

autocorrelation_plot(cumsum_data, ax=axarr[2, 1])
axarr[2, 1].set_xlim([0, 30])
axarr[2, 1].set_ylim([-1.1, 1.1])
plot.show()

# Figure 8.3

plot.plot(random_time_series.index, cumsum_data, 'b-')
plot.plot(random_time_series.index, cumsum_data.ewm(alpha=0.2).mean(), 'k:')
plot.plot(random_time_series.index, cumsum_data.ewm(alpha=0.05).mean(), 'r:')
plot.xlim([0, 1000])
plot.ylim([-50, 100])
plot.xlabel('time')
plot.ylabel('value')
plot.legend(['$original$ $series$', '$\\alpha=0.2$', '$\\alpha=0.05$'], fontsize='small')
plot.show()

# Figure 8.4

random_time_series = pd.DataFrame(np.random.normal(0, 1, 5200), index=range(0, 5200), columns=['value'])
random_time_series['value'] = random_time_series['value'] + 5
random_time_series.iloc[1000:3999, random_time_series.columns.get_loc('value')] = random_time_series.iloc[1000:3999, random_time_series.columns.get_loc('value')] + 20
linear_deduction = np.arange(0, -20, -(float(20)/2000))
random_time_series.iloc[2000:3999, random_time_series.columns.get_loc('value')] = random_time_series.iloc[2000:3999, random_time_series.columns.get_loc('value')] + linear_deduction[0:1999]

plot.plot(random_time_series.index, random_time_series['value'], 'b-')
plot.plot(random_time_series.index, random_time_series['value'].ewm(alpha=0.05).mean(), 'r:')
plot.plot(random_time_series.index, random_time_series['value'].diff(periods=1), 'k-')

plot.xlim([0, 5200])
plot.xlabel('time')
plot.ylabel('value')
plot.legend(['$original$ $series$', '$trend$ $using $ $\\alpha=0.05$', '$detrended$'], fontsize='small')
plot.show()

# Figure 8.5

dataset_path = './intermediate_datafiles/'
dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
dataset.index = pd.to_datetime(dataset.index)

f, axarr = plot.subplots(1, 2)
f.subplots_adjust(hspace=0.5)
f.subplots_adjust(wspace=0.5)
xfmt = md.DateFormatter('%H:%M')

axarr[0].plot(dataset.index, dataset['acc_phone_x'], 'b-')
axarr[0].set_xlabel('time')
axarr[0].set_ylabel('value')
axarr[0].xaxis.set_major_formatter(xfmt)
axarr[1].plot(dataset.index, dataset['acc_phone_x'].diff(periods=1), 'k-')
axarr[1].set_xlabel('time')
axarr[1].set_ylabel('value')
axarr[1].xaxis.set_major_formatter(xfmt)
plot.show()

# Figure 8.6

dataset_path = './intermediate_datafiles/'
dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
dataset.index = pd.to_datetime(dataset.index)

xfmt = md.DateFormatter('%H:%M')

plot.plot(dataset.index, dataset['acc_phone_x'], color='0.75')
plot.xlabel('time')
plot.ylabel('value')
plot.gca().xaxis.set_major_formatter(xfmt)
dataset['filtered_acc_x'] = dataset['acc_phone_x'].rolling(400).mean().shift(-200)
dataset['filtered_acc_y'] = dataset['acc_phone_y'].rolling(400).mean().shift(-200)
dataset['filtered_acc_z'] = dataset['acc_phone_z'].rolling(400).mean().shift(-200)
plot.plot(dataset.index, dataset['filtered_acc_x'], 'b-')
dataset['radius'] = dataset['filtered_acc_x'].pow(2) + dataset['filtered_acc_y'].pow(2) + dataset['filtered_acc_z'].pow(2)
dataset['radius'] = dataset['radius'].pow(0.5)
plot.plot(dataset.index, dataset['radius'], 'r-')
plot.legend(['$original$ $series$', '$filtered$', '$||a||(filtered)$'], fontsize='small')
plot.show()

# Figure 8.7

f, axarr = plot.subplots(1, 2)
dataset_path = 'datasets/crowdsignals/csv-participant-one/'
dataset = pd.read_csv(dataset_path + 'accelerometer_phone.csv', index_col=0)
dataset.index = pd.to_datetime(dataset['timestamps']).values
del dataset['timestamps']
dataset = dataset.iloc[400000:404000, dataset.columns.get_loc('x')]
dataset = dataset.resample('10L').mean()


temp_ts = dataset
xfmt = md.DateFormatter('%H:%M')

autocorrelation_plot(temp_ts, ax=axarr[0])
axarr[0].set_xlim([0,1000])
axarr[0].set_ylim([-1,1])

pacf_x, confint = pacf(temp_ts, nlags=100, alpha=.05)
df = pd.DataFrame(confint, columns=['lower', 'upper'])
df['lower'] = df['lower'] - np.array(pacf_x)
df['upper'] = df['upper'] - np.array(pacf_x)
axarr[1].plot(range(0, 101), pacf_x, 'b-')
axarr[1].plot(range(1, 101), df.iloc[1:, df.columns.get_loc('lower')], color='0.5')
axarr[1].plot(range(0, 101), [0]*101, color='0')
axarr[1].plot(range(1, 101), df.iloc[1:, df.columns.get_loc('upper')], color='0.5')
axarr[1].grid()
axarr[1].set_ylim([-1,1])
axarr[1].set_xlabel('Lag')
axarr[1].set_ylabel('Partial Autocorrelation')
plot.show()

# Figure 8.8

# Note: with the package used in this Python3 code this Figure is formatted slgihtly different.

df = pd.DataFrame(temp_ts[0:500], index=temp_ts.index[0:500], columns=['x'])
model = ARIMA(df, order=(3,1,2))
results = model.fit(disp=-1)
fig = results.plot_predict(200,500)
xfmt = md.DateFormatter('%H:%M:%S')
plot.gca().xaxis.set_major_formatter(xfmt)

plot.legend(['$predicted$', '$original$ $series$'], fontsize='small')
plot.xlabel('time')
plot.ylabel('value')
plot.show()


#Figure 8.9

df = pd.DataFrame(temp_ts[0:400], index=temp_ts.index[0:400], columns=['x'])

model = ARIMA(df, order=(3,1,2))
results = model.fit()
#fig = results.plot_predict(400,500)

fc, se, conf = results.forecast(100, alpha=0.05)

plot.plot(temp_ts.index[400:500], fc, 'r:')
y1 = conf[:, 0]
y2 = conf[:, 1]
plot.fill_between(temp_ts.index[400:500], y1, y2, where=y2 >= y1, facecolor='grey', interpolate=True)
plot.plot(temp_ts.index[200:500], temp_ts[200:500], 'b')

plot.legend(['$predicted$', '$original$ $series$'], fontsize='small')
plot.xlabel('time')
plot.ylabel('value')
plot.gca().xaxis.set_major_formatter(xfmt)
plot.show()

#Figure 8.10

xfmt = md.DateFormatter('%H:%M:%S')
df = pd.DataFrame(temp_ts[0:500], index=pd.to_datetime(temp_ts.index[0:500]), columns=['x'])
decomposition = seasonal_decompose(np.array(df['x'].values),freq=107)

seasonal = decomposition.seasonal
residual = df['x'] - decomposition.seasonal

f, axarr = plot.subplots(3, 1)
f.subplots_adjust(hspace=0.5)
f.subplots_adjust(wspace=0.5)
axarr[0].xaxis.set_major_formatter(xfmt)
axarr[0].plot(temp_ts.index[0:500], temp_ts[0:500], 'b')
axarr[0].legend(['$original$ $data$'], fontsize='small')
axarr[0].set_ylabel('value')
axarr[1].xaxis.set_major_formatter(xfmt)
axarr[1].plot(temp_ts.index[0:500], seasonal, 'b:')
axarr[1].legend(['$seasonality$'], fontsize='small')
axarr[1].set_ylabel('value')
axarr[2].xaxis.set_major_formatter(xfmt)
axarr[2].plot(temp_ts.index[0:500], residual, 'k:')
axarr[2].legend(['$residual$'], fontsize='small')
axarr[2].set_ylabel('value')
axarr[2].set_xlabel('time')


plot.show()

#Figure 8.11

df = pd.DataFrame(residual[0:400], index=temp_ts.index[0:400], columns=['x'])
model = ARIMA(df, order=(3,1,2))
results = model.fit()
fc, se, conf = results.forecast(100, alpha=0.05)

plot.plot(temp_ts.index[400:500], fc + seasonal[400:500], 'r:')
y1 = conf[:, 0] + seasonal[400:500]
y2 = conf[:, 1] + seasonal[400:500]
plot.fill_between(temp_ts.index[400:500], y1, y2, where=y2 >= y1, facecolor='grey', interpolate=True)
plot.plot(temp_ts.index[200:500], temp_ts[200:500], 'b')

plot.legend(['$predicted$', '$original$ $series$'], fontsize='small')
plot.xlabel('time')
plot.ylabel('value')
plot.gca().xaxis.set_major_formatter(xfmt)
plot.show()


# Figure 8.16


f, axarr = plot.subplots(1, 2)

pred_mood = [0.5]
pred_al = [0.4]
x = [1]
delta_t = 1
gamma_1 = gamma_2 = gamma_3 = gamma_4 = gamma_5 = 1

for t in range(1, len(time)):
    pred_mood.append(pred_mood[-1] + x[-1]*(gamma_1*(1-pred_mood[-1])*max((pred_al[-1] - pred_mood[-1]), 0) + gamma_2*pred_mood[-1]*(min((pred_al[-1] - pred_mood[-1]), 0)))*delta_t)
    pred_al.append(pred_al[-1] + gamma_3 * (pred_al[-1] * min(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0) + (1-pred_al[-1])* max(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0)))
    x.append(x[-1]);

axarr[0].plot(time, values_mood, 'ro')
axarr[0].plot(time, activity_level, 'bo')
axarr[0].plot(time, pred_mood, 'r-')
axarr[0].plot(time, pred_al, 'b:')

axarr[0].set_ylim([0,1])
axarr[0].set_xlabel('time')
axarr[0].set_ylabel('value')
axarr[0].legend(['$mood$', '$activity$ $level$', '$predicted$ $mood$ $with$ $\gamma_{1}=\gamma_{2}=\gamma_{3}=\gamma_{4}=\gamma_{5}=1$',
             '$predicted$ $mood$ $with$ $\gamma_{1}=\gamma_{2}=\gamma_{3}=\gamma_{4}=\gamma_{5}=1$'], loc=4, fontsize='small', numpoints=1)

pred_mood = [0.5]
pred_al = [0.4]
x = [1]
delta_t = 1
gamma_1 = 5
gamma_2 = 0.75
gamma_3 = 0.3
gamma_4 = 1
gamma_5 = 1

for t in range(1, len(time)):
    pred_mood.append(pred_mood[-1] + x[-1]*(gamma_1*(1-pred_mood[-1])*max((pred_al[-1] - pred_mood[-1]), 0) + gamma_2*pred_mood[-1]*(min((pred_al[-1] - pred_mood[-1]), 0)))*delta_t)
    pred_al.append(pred_al[-1] + gamma_3 * (pred_al[-1] * min(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0) + (1-pred_al[-1])* max(math.sin(((t-(gamma_4*math.pi))/gamma_5)), 0)))
    x.append(x[-1]);

axarr[1].plot(time, values_mood, 'ro')
axarr[1].plot(time, activity_level, 'bo')
axarr[1].plot(time, pred_mood, 'r-')
axarr[1].plot(time, pred_al, 'b:')

axarr[1].set_ylim([0,1])
axarr[1].set_xlabel('time')
axarr[1].set_ylabel('value')
axarr[1].legend(['$mood$', '$activity$ $level$', '$predicted$ $mood$ $with$ $\gamma_{1}=5, \gamma_{2}=0.75, \gamma_{3}=0.3, \gamma_{4}=\gamma_{5}=1$',
             '$predicted$ $mood$ $with$ $\gamma_{1}=5, \gamma_{2}=0.75, \gamma_{3}=0.3, \gamma_{4}=\gamma_{5}=1$'], loc=4, fontsize='small', numpoints=1)


plot.show()

# Figure 8.19

x = np.arange(0.05, 1.01, 0.01)
y = 0.05/x
plot.plot(x, y, 'r-')
plot.plot([x[1], x[6], x[20], x[40], x[80]], [y[1], y[6], y[20], y[40], y[80]], 'ro')
plot.plot([0.4, 0.6], [0.6, 0.4], 'bo')
plot.legend(['$pareto$ $front$', '$non-dominated$ $instance$','$dominated$ $instance$'], loc=1, fontsize='small', numpoints=1)
plot.xlim([0,1])
plot.ylim([0,1])
plot.xlabel('$E_{X_{1}}$')
plot.ylabel('$E{X_{2}}$')
plot.show()
