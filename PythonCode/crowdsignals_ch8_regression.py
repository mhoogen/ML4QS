##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot

import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split


# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter5_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate', '2016-02-08 18:29:56',
#                                                                                   '2016-02-08 18:29:58','2016-02-08 18:29:59')
                                                                                   '2016-02-08 19:34:07', '2016-02-08 20:07:50')

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z',
                  'labelOnTable','labelSitting','labelWashingHands','labelWalking','labelStanding','labelDriving','labelEating','labelRunning',
                  'light_phone_lux','mag_phone_x','mag_phone_y','mag_phone_z','mag_watch_x','mag_watch_y','mag_watch_z','press_phone_pressure']
pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
time_features = [name for name in dataset.columns if ('temp_' in name and not 'hr_watch' in name)]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print '#basic features: ', len(basic_features)
print '#PCA features: ', len(pca_features)
print '#time features: ', len(time_features)
print '#frequency features: ', len(freq_features)
cluster_features = ['cluster']
print '#cluster features: ', len(cluster_features)
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

selected_features = ['temp_pattern_labelOnTable','labelOnTable', 'temp_pattern_labelOnTable(b)labelOnTable', 'cluster',
                     'pca_1_temp_mean_ws_120','pca_2_temp_mean_ws_120','pca_2','acc_watch_y_temp_mean_ws_120','gyr_watch_y_pse',
                     'gyr_watch_x_pse']
possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

# Let us first study whether the time series is stationary and what the autocorrelations are.

dftest = adfuller(dataset['hr_watch_rate'], autolag='AIC')
print dftest

autocorrelation_plot(dataset['hr_watch_rate'])
plot.show()

# Now let us focus on the learning part.

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.

repeats = 5

# we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

washout_time = 10

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):

    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_res = 0
    performance_tr_res_std = 0
    performance_te_res = 0
    performance_te_res_std = 0
    performance_tr_rnn = 0
    performance_tr_rnn_std = 0
    performance_te_rnn = 0
    performance_te_rnn_std = 0

    for repeat in range(0, repeats):
        print '----', repeat
        regr_train_y, regr_test_y = learner.reservoir_computing(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True, per_time_step=False)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_res += mean_tr
        performance_tr_res_std += std_tr
        performance_te_res += mean_te
        performance_te_res_std += std_te

        regr_train_y, regr_test_y = learner.recurrent_neural_network(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_rnn += mean_tr
        performance_tr_rnn_std += std_tr
        performance_te_rnn += mean_te
        performance_te_rnn_std += std_te


    # We only apply the time series in case of the basis features.
    if (feature_names[i] == 'initial set'):
        regr_train_y, regr_test_y = learner.time_series(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        overall_performance_tr_ts = mean_tr
        overall_performance_tr_ts_std = std_tr
        overall_performance_te_ts = mean_te
        overall_performance_te_ts_std = std_te
    else:
        overall_performance_tr_ts = 0
        overall_performance_tr_ts_std = 0
        overall_performance_te_ts = 0
        overall_performance_te_ts_std = 0

    overall_performance_tr_res = performance_tr_res/repeats
    overall_performance_tr_res_std = performance_tr_res_std/repeats
    overall_performance_te_res = performance_te_res/repeats
    overall_performance_te_res_std = performance_te_res_std/repeats
    overall_performance_tr_rnn = performance_tr_rnn/repeats
    overall_performance_tr_rnn_std = performance_tr_rnn_std/repeats
    overall_performance_te_rnn = performance_te_rnn/repeats
    overall_performance_te_rnn_std = performance_te_rnn_std/repeats

    scores_with_sd = [(overall_performance_tr_res, overall_performance_tr_res_std, overall_performance_te_res, overall_performance_te_res_std),
                      (overall_performance_tr_rnn, overall_performance_tr_rnn_std, overall_performance_te_rnn, overall_performance_te_rnn_std),
                      (overall_performance_tr_ts, overall_performance_tr_ts_std, overall_performance_te_ts, overall_performance_te_ts_std)]
    print scores_with_sd
    util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_regression(['Reservoir', 'RNN', 'Time series'], feature_names, scores_over_all_algs)

regr_train_y, regr_test_y = learner.reservoir_computing(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'], test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')
regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X[basic_features], train_y, test_X[basic_features], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'], test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')
regr_train_y, regr_test_y = learner.time_series(train_X[basic_features], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'], test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')
