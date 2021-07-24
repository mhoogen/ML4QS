##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7: Predictive modeling without notion of time   #
#                                                            #
##############################################################

import argparse
from pathlib import Path
import pandas as pd
from util import util
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionRegression

# Set up filenames and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FILENAME = 'chapter5_result.csv'
EXPORT_TREE_PATH = Path('figures/example_graphs/Chapter7/')
EXPORT_TREE_PATH.mkdir(exist_ok=True, parents=True)


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    # Read the result from the previous chapter and convert the index to datetime
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FILENAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(__file__)

    # Consider the second task, namely the prediction of the heart rate. Therefore create a dataset with the heart
    # rate as target and split using timestamps, because this is considered as a temporal task.
    print('\n- - - Loading dataset - - -')
    prepare = PrepareDatasetForLearning()
    learner = RegressionAlgorithms()
    evaluation = RegressionEvaluation()
    train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate',
                                                                                       '2016-02-08 18:28:56',
                                                                                       '2016-02-08 19:34:07',
                                                                                       '2016-02-08 20:07:50')
    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features
    print('- - - Selecting subsets - - -')
    basic_features = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z',
                      'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z',
                      'labelOnTable', 'labelSitting', 'labelWashingHands', 'labelWalking', 'labelStanding',
                      'labelDriving',
                      'labelEating', 'labelRunning', 'light_phone_lux', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z',
                      'mag_watch_x', 'mag_watch_y', 'mag_watch_z', 'press_phone_pressure']
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

    if FLAGS.mode == 'selection' or FLAGS.mode == 'all':
        # First, consider the Pearson correlations and see whether features can be selected based on them
        fs = FeatureSelectionRegression()
        print('\n- - - Running feature selection - - -')
        features, correlations = fs.pearson_selection(10, train_X[features_after_chapter_5], train_y)
        util.print_pearson_correlations(correlations)

    # Select the 10 features with the highest correlation.
    selected_features = ['temp_pattern_labelOnTable', 'labelOnTable', 'temp_pattern_labelOnTable(b)labelOnTable',
                         'pca_2_temp_mean_ws_120', 'pca_1_temp_mean_ws_120', 'acc_watch_y_temp_mean_ws_120', 'pca_2',
                         'acc_phone_z_temp_mean_ws_120', 'gyr_watch_y_pse', 'gyr_watch_x_pse']
    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4,
                             features_after_chapter_5, selected_features]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    if FLAGS.mode == 'overall' or FLAGS.mode == 'all':
        print('\n- - - Running test of all different regression algorithms - - -')
        # First study the importance of the parameter settings. Therefore repeat the experiment a number of times to get
        # a bit more robust data as the initialization of e.g. the NN is random
        REPEATS = FLAGS.repeats
        scores_over_all_algs = []

        for i in range(0, len(possible_feature_sets)):
            selected_train_X = train_X[possible_feature_sets[i]]
            selected_test_X = test_X[possible_feature_sets[i]]

            performance_tr_nn, performance_tr_nn_std = 0, 0
            performance_tr_rf, performance_tr_rf_std = 0, 0
            performance_te_nn, performance_te_nn_std = 0, 0
            performance_te_rf, performance_te_rf_std = 0, 0

            # First run non deterministic classifiers a number of times to average their score
            for repeat in range(0, REPEATS):
                print(f'Training NeuralNetwork run {repeat + 1}/{REPEATS} ... ')
                regr_train_y, regr_test_y = learner.\
                    feedforward_neural_network(selected_train_X, train_y, selected_test_X, gridsearch=True)
                mean_tr, std_tr = evaluation.mean_squared_error_with_std(train_y, regr_train_y)
                mean_te, std_te = evaluation.mean_squared_error_with_std(test_y, regr_test_y)
                performance_tr_nn += mean_tr
                performance_tr_nn_std += std_tr
                performance_te_nn += mean_te
                performance_te_nn_std += std_te

                print(f'Training RandomForest run {repeat + 1}/{REPEATS} ... ')
                regr_train_y, regr_test_y = learner.random_forest(selected_train_X, train_y, selected_test_X,
                                                                  gridsearch=True)
                mean_tr, std_tr = evaluation.mean_squared_error_with_std(train_y, regr_train_y)
                mean_te, std_te = evaluation.mean_squared_error_with_std(test_y, regr_test_y)
                performance_tr_rf += mean_tr
                performance_tr_rf_std += std_tr
                performance_te_rf += mean_te
                performance_te_rf_std += std_te

            overall_performance_tr_nn = performance_tr_nn / REPEATS
            overall_performance_tr_nn_std = performance_tr_nn_std / REPEATS
            overall_performance_te_nn = performance_te_nn / REPEATS
            overall_performance_te_nn_std = performance_te_nn_std / REPEATS
            overall_performance_tr_rf = performance_tr_rf / REPEATS
            overall_performance_tr_rf_std = performance_tr_rf_std / REPEATS
            overall_performance_te_rf = performance_te_rf / REPEATS
            overall_performance_te_rf_std = performance_te_rf_std / REPEATS

            # Run deterministic algorithms:
            print("Support Vector Regressor run 1/1 ... ")
            # Convergence of the SVR does not always occur (even adjusting tolerance and iterations does not help)
            regr_train_y, regr_test_y = learner.\
                support_vector_regression_without_kernel(selected_train_X, train_y, selected_test_X, gridsearch=False)
            mean_tr, std_tr = evaluation.mean_squared_error_with_std(train_y, regr_train_y)
            mean_te, std_te = evaluation.mean_squared_error_with_std(test_y, regr_test_y)
            performance_tr_svm = mean_tr
            performance_tr_svm_std = std_tr
            performance_te_svm = mean_te
            performance_te_svm_std = std_te

            print("Training Nearest Neighbor run 1/1 ... ")
            regr_train_y, regr_test_y = learner.k_nearest_neighbor(selected_train_X, train_y, selected_test_X,
                                                                   gridsearch=True)
            mean_tr, std_tr = evaluation.mean_squared_error_with_std(train_y, regr_train_y)
            mean_te, std_te = evaluation.mean_squared_error_with_std(test_y, regr_test_y)
            performance_tr_knn = mean_tr
            performance_tr_knn_std = std_tr
            performance_te_knn = mean_te
            performance_te_knn_std = std_te

            print("Training Decision Tree run 1/1 ... ")
            regr_train_y, regr_test_y = learner.\
                decision_tree(selected_train_X, train_y, selected_test_X, gridsearch=True,
                              export_tree_path=EXPORT_TREE_PATH)
            mean_tr, std_tr = evaluation.mean_squared_error_with_std(train_y, regr_train_y)
            mean_te, std_te = evaluation.mean_squared_error_with_std(test_y, regr_test_y)
            performance_tr_dt = mean_tr
            performance_tr_dt_std = std_tr
            performance_te_dt = mean_te
            performance_te_dt_std = std_te

            scores_with_sd = [(overall_performance_tr_nn, overall_performance_tr_nn_std, overall_performance_te_nn,
                               overall_performance_te_nn_std),
                              (overall_performance_tr_rf, overall_performance_tr_rf_std, overall_performance_te_rf,
                               overall_performance_te_rf_std),
                              (performance_tr_svm, performance_tr_svm_std, performance_te_svm, performance_te_svm_std),
                              (performance_tr_knn, performance_tr_knn_std, performance_te_knn, performance_te_knn_std),
                              (performance_tr_dt, performance_tr_dt_std, performance_te_dt, performance_te_dt_std)]
            util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index),
                                                         len(selected_test_X.index), scores_with_sd)
            scores_over_all_algs.append(scores_with_sd)

        # Plot the results
        DataViz.plot_performances_regression(['NN', 'RF', 'SVM', 'KNN', 'DT'], feature_names, scores_over_all_algs)

    if FLAGS.mode == 'detail' or FLAGS.mode == 'all':
        print('\n- - - Running visualization of results - - -')
        regr_train_y, regr_test_y = learner.random_forest(train_X[features_after_chapter_5], train_y,
                                                          test_X[features_after_chapter_5], gridsearch=False,
                                                          print_model_details=True)
        DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y, test_X.index, test_y,
                                                      regr_test_y, 'heart rate')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        help="Select what version to run: Feature Selection, Overall, Detail or All. \
                             'selection' to run the process of feature selection \
                             'overall' to train all models specified in learning algorithms \
                             'detail' to take a closer look at results \
                             'all' to run all parts of the script.",
                        choices=['selection', 'overall', 'detail', 'all'])
    parser.add_argument('--nnrepeat', type=int, default=5,
                        help="Number of repeats to use for overall training.")

    FLAGS, unparsed = parser.parse_known_args()

    # Prints args und run main script
    print_flags()
    main()
