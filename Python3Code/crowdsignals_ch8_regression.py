##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8: Predictive modeling with notion of time      #
#                                                            #
##############################################################

import argparse
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plt
from pathlib import Path

# Set up filenames and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter5_result.csv'


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    # Read the result from the previous chapter and convert the index to datetime
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(__file__)

    # Consider the second task, namely the prediction of the heart rate. Therefore create a dataset with the heart
    # rate as target and split using timestamps, because this is considered as a temporal task
    print('\n- - - Loading dataset - - -')
    prepare = PrepareDatasetForLearning()
    train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate',
                                                                                       '2016-02-08 18:29:56',
                                                                                       '2016-02-08 19:34:07',
                                                                                       '2016-02-08 20:07:50')
    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features
    print('\n- - - Selecting subsets - - -')
    basic_features = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z',
                      'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z',
                      'labelOnTable', 'labelSitting', 'labelWashingHands', 'labelWalking', 'labelStanding',
                      'labelDriving', 'labelEating', 'labelRunning', 'light_phone_lux', 'mag_phone_x', 'mag_phone_y',
                      'mag_phone_z', 'mag_watch_x', 'mag_watch_y', 'mag_watch_z', 'press_phone_pressure']
    pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7']
    time_features = [name for name in dataset.columns if ('temp_' in name and 'hr_watch' not in name)]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    cluster_features = ['cluster']
    print('#basic features: ', len(basic_features))
    print('#PCA features: ', len(pca_features))
    print('#time features: ', len(time_features))
    print('#frequency features: ', len(freq_features))
    print('#cluster features: ', len(cluster_features))
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(features_after_chapter_3, time_features, freq_features))
    features_after_chapter_5 = list(set().union(features_after_chapter_4, cluster_features))

    selected_features = ['temp_pattern_labelOnTable', 'labelOnTable', 'temp_pattern_labelOnTable(b)labelOnTable',
                         'cluster', 'pca_1_temp_mean_ws_120', 'pca_2_temp_mean_ws_120', 'pca_2',
                         'acc_watch_y_temp_mean_ws_120', 'gyr_watch_y_pse', 'gyr_watch_x_pse']
    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4,
                             features_after_chapter_5, selected_features]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    if FLAGS.mode == 'correlation' or FLAGS.mode == 'all':
        # First study whether the time series is stationary and what the autocorrelations are
        adfuller(dataset['hr_watch_rate'], autolag='AIC')
        plt.Figure()
        autocorrelation_plot(dataset['hr_watch_rate'])
        DataViz.save(plt)
        plt.show()

    # Now focus on the learning part
    learner = TemporalRegressionAlgorithms()
    evaluate = RegressionEvaluation()

    if FLAGS.mode == 'overall' or FLAGS.mode == 'all':
        # Repeat the experiment a number of times to get a bit more robust data as the initialization of e.g. the NN is
        # random
        repeats = FLAGS.repeats

        # Set a washout time to give the NN's the time to stabilize (so don't compute the error during the washout time)
        washout_time = FLAGS.washout
        scores_over_all_algs = []

        for i in range(0, len(possible_feature_sets)):
            print(f'Evaluating for features {possible_feature_sets[i]}')
            selected_train_X = train_X[possible_feature_sets[i]]
            selected_test_X = test_X[possible_feature_sets[i]]

            # First run non deterministic classifiers a number of times to average their score
            performance_tr_res, performance_tr_res_std = 0, 0
            performance_te_res, performance_te_res_std = 0, 0
            performance_tr_rnn, performance_tr_rnn_std = 0, 0
            performance_te_rnn, performance_te_rnn_std = 0, 0

            for repeat in range(0, repeats):
                print(f'--- run {repeat} ---')
                regr_train_y, regr_test_y = learner.reservoir_computing(selected_train_X, train_y, selected_test_X,
                                                                        test_y,
                                                                        gridsearch=True, per_time_step=False)

                mean_tr, std_tr = evaluate.mean_squared_error_with_std(train_y.iloc[washout_time:, ],
                                                                       regr_train_y.iloc[washout_time:, ])
                mean_te, std_te = evaluate.mean_squared_error_with_std(test_y.iloc[washout_time:, ],
                                                                       regr_test_y.iloc[washout_time:, ])

                performance_tr_res += mean_tr
                performance_tr_res_std += std_tr
                performance_te_res += mean_te
                performance_te_res_std += std_te

                regr_train_y, regr_test_y = learner.recurrent_neural_network(selected_train_X, train_y, selected_test_X,
                                                                             test_y,
                                                                             gridsearch=True)

                mean_tr, std_tr = evaluate.mean_squared_error_with_std(train_y.iloc[washout_time:, ],
                                                                       regr_train_y.iloc[washout_time:, ])
                mean_te, std_te = evaluate.mean_squared_error_with_std(test_y.iloc[washout_time:, ],
                                                                       regr_test_y.iloc[washout_time:, ])

                performance_tr_rnn += mean_tr
                performance_tr_rnn_std += std_tr
                performance_te_rnn += mean_te
                performance_te_rnn_std += std_te

            # Only apply the time series in case of the basis features
            if feature_names[i] == 'initial set':
                regr_train_y, regr_test_y = learner.time_series(selected_train_X, train_y, selected_test_X, test_y,
                                                                gridsearch=True)

                mean_tr, std_tr = evaluate.mean_squared_error_with_std(train_y.iloc[washout_time:, ],
                                                                       regr_train_y.iloc[washout_time:, ])
                mean_te, std_te = evaluate.mean_squared_error_with_std(test_y.iloc[washout_time:, ],
                                                                       regr_test_y.iloc[washout_time:, ])

                overall_performance_tr_ts = mean_tr
                overall_performance_tr_ts_std = std_tr
                overall_performance_te_ts = mean_te
                overall_performance_te_ts_std = std_te
            else:
                overall_performance_tr_ts = 0
                overall_performance_tr_ts_std = 0
                overall_performance_te_ts = 0
                overall_performance_te_ts_std = 0

            overall_performance_tr_res = performance_tr_res / repeats
            overall_performance_tr_res_std = performance_tr_res_std / repeats
            overall_performance_te_res = performance_te_res / repeats
            overall_performance_te_res_std = performance_te_res_std / repeats
            overall_performance_tr_rnn = performance_tr_rnn / repeats
            overall_performance_tr_rnn_std = performance_tr_rnn_std / repeats
            overall_performance_te_rnn = performance_te_rnn / repeats
            overall_performance_te_rnn_std = performance_te_rnn_std / repeats

            scores_with_sd = [(overall_performance_tr_res, overall_performance_tr_res_std, overall_performance_te_res,
                               overall_performance_te_res_std),
                              (overall_performance_tr_rnn, overall_performance_tr_rnn_std, overall_performance_te_rnn,
                               overall_performance_te_rnn_std),
                              (overall_performance_tr_ts, overall_performance_tr_ts_std, overall_performance_te_ts,
                               overall_performance_te_ts_std)]
            util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index),
                                                         len(selected_test_X.index), scores_with_sd)
            scores_over_all_algs.append(scores_with_sd)

        DataViz.plot_performances_regression(['Reservoir', 'RNN', 'Time series'], feature_names, scores_over_all_algs)

    if FLAGS.mode == 'detail' or FLAGS.mode == 'all':
        regr_train_y, regr_test_y = learner.reservoir_computing(train_X[features_after_chapter_5], train_y,
                                                                test_X[features_after_chapter_5], test_y,
                                                                gridsearch=False)
        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'],
                                                      test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')

        regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X[basic_features], train_y,
                                                                     test_X[basic_features], test_y, gridsearch=True)
        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'],
                                                      test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')

        regr_train_y, regr_test_y = learner.time_series(train_X[basic_features], train_y, test_X[basic_features],
                                                        test_y, gridsearch=True)
        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['hr_watch_rate'],
                                                      test_X.index, test_y, regr_test_y['hr_watch_rate'], 'heart rate')

    if FLAGS.mode == 'dynamical' or FLAGS.mode == 'all':
        # And now some example code for using the dynamical systems model with parameter tuning (note: focus on
        # predicting accelerometer data):
        train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(copy.deepcopy(dataset),
                                                                                   ['acc_phone_x', 'acc_phone_y'], 0.9,
                                                                                   filter_data=False, temporal=True)
        output_sets = learner. \
            dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y,
                                           ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                           ['self.a * self.acc_phone_x + self.b * self.acc_phone_y',
                                            'self.c * self.acc_phone_y + self.d * self.acc_phone_z',
                                            'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                           ['self.acc_phone_x', 'self.acc_phone_y'],
                                           ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                           pop_size=10, max_generations=10, per_time_step=True)
        DataViz.plot_pareto_front(output_sets)

        DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y,
                                                                output_sets, 0, 'acc_phone_x')

        regr_train_y, regr_test_y = learner. \
            dynamical_systems_model_ga(train_X, train_y, test_X, test_y,
                                       ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                       ['self.a * self.acc_phone_x + self.b * self.acc_phone_y',
                                        'self.c * self.acc_phone_y + self.d * self.acc_phone_z',
                                        'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                       ['self.acc_phone_x', 'self.acc_phone_y'],
                                       ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                       pop_size=5, max_generations=10, per_time_step=True)

        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['acc_phone_x'],
                                                      regr_train_y['acc_phone_x'], test_X.index, test_y['acc_phone_x'],
                                                      regr_test_y['acc_phone_x'], 'acc_phone_x')

        regr_train_y, regr_test_y = learner. \
            dynamical_systems_model_sa(train_X, train_y, test_X, test_y,
                                       ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                       ['self.a * self.acc_phone_x + self.b * self.acc_phone_y',
                                        'self.c * self.acc_phone_y + self.d * self.acc_phone_z',
                                        'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                       ['self.acc_phone_x', 'self.acc_phone_y'],
                                       ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                       max_generations=10, per_time_step=True)

        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['acc_phone_x'],
                                                      regr_train_y['acc_phone_x'], test_X.index, test_y['acc_phone_x'],
                                                      regr_test_y['acc_phone_x'], 'acc_phone_x')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        help="Select what version to run: Autocorrelation, Overall, Detail, Dynamical or All. \
                             'correlation' to check the autocorrelation \
                             'overall' to train all models specified in learning algorithms \
                             'detail' to take a closer look at results \
                             'dynamical' to train and evaluate the results of dynamical models \
                             'all' to run all parts of the script.",
                        choices=['selection', 'overall', 'detail', 'all'])
    parser.add_argument('--repeats', type=int, default=10,
                        help="Number of repeats to use for overall training.")
    parser.add_argument('--washout', type=int, default=10,
                        help="Number of iterations to use as washout.")

    FLAGS, unparsed = parser.parse_known_args()

    # Prints args und run main script
    print_flags()
    main()
