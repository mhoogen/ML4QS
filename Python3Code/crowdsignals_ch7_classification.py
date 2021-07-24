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
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up filenames and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FILENAME = 'chapter5_result.csv'
RESULT_FILENAME = 'chapter7_classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')


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

    # Consider the first task, namely the prediction of the label. Therefore create a single column with the categorical
    # attribute representing the class. Furthermore, use 70% of the data for training and the remaining 30% as an
    # independent test set. Select the sets based on stratified sampling and remove cases where the label is unknown.
    print('\n- - - Loading dataset - - -')
    prepare = PrepareDatasetForLearning()
    learner = ClassificationAlgorithms()
    evaluation = ClassificationEvaluation()
    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7,
                                                                                   filter_data=True, temporal=False)

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features
    print('- - - Selecting subsets - - -')
    basic_features = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z',
                      'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z',
                      'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'mag_watch_x',
                      'mag_watch_y', 'mag_watch_z', 'press_phone_pressure']
    pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7']
    time_features = [name for name in dataset.columns if '_temp_' in name]
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
        # First, consider the performance over a selection of features
        N_FORWARD_SELECTION = FLAGS.nfeatures
        fs = FeatureSelectionClassification()
        print('\n- - - Running feature selection - - -')
        features, ordered_features, ordered_scores = fs.forward_selection(max_features=N_FORWARD_SELECTION,
                                                                          X_train=train_X[features_after_chapter_5],
                                                                          y_train=train_y)
        DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION + 1)], y=[ordered_scores],
                        xlabel='number of features', ylabel='accuracy')

    # Select the most important features (based on python2 features)
    selected_features = ['acc_phone_y_freq_0.0_Hz_ws_40', 'press_phone_pressure_temp_mean_ws_120',
                         'gyr_phone_x_temp_std_ws_120', 'mag_watch_y_pse', 'mag_phone_z_max_freq',
                         'gyr_watch_y_freq_weighted', 'gyr_phone_y_freq_1.0_Hz_ws_40', 'acc_phone_x_freq_1.9_Hz_ws_40',
                         'mag_watch_z_freq_0.9_Hz_ws_40', 'acc_watch_y_freq_0.5_Hz_ws_40']

    if FLAGS.mode == 'regularization' or FLAGS.mode == 'all':
        print('\n- - - Running regularization and model complexity test - - -')
        # Study the impact of regularization and model complexity: does regularization prevent overfitting?
        # Due to runtime constraints run the experiment 3 times, for even more robust data increase the repetitions
        N_REPEATS_NN = FLAGS.nnrepeat
        reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        performance_training = []
        performance_test = []

        for reg_param in reg_parameters:
            performance_tr = 0
            performance_te = 0
            for i in range(0, N_REPEATS_NN):
                class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                    train_X, train_y,
                    test_X, hidden_layer_sizes=(250,), alpha=reg_param, max_iter=500,
                    gridsearch=False
                )
                performance_tr += evaluation.accuracy(train_y, class_train_y)
                performance_te += evaluation.accuracy(test_y, class_test_y)
            performance_training.append(performance_tr / N_REPEATS_NN)
            performance_test.append(performance_te / N_REPEATS_NN)
        DataViz.plot_xy(x=[reg_parameters, reg_parameters], y=[performance_training, performance_test],
                        method='semilogx', xlabel='regularization parameter value', ylabel='accuracy',
                        ylim=[0.95, 1.01], names=['training', 'test'], line_styles=['r-', 'b:'])

    if FLAGS.mode == 'tree' or FLAGS.mode == 'all':
        print('\n- - - Running leaf size test of decision tree - - -')
        # Consider the influence of certain parameter settings for the tree model. (very related to the
        # regularization) and study the impact on performance.
        leaf_settings = [1, 2, 5, 10]
        performance_training = []
        performance_test = []

        for no_points_leaf in leaf_settings:
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
                train_X[selected_features], train_y, test_X[selected_features], min_samples_leaf=no_points_leaf,
                gridsearch=False, print_model_details=False)

            performance_training.append(evaluation.accuracy(train_y, class_train_y))
            performance_test.append(evaluation.accuracy(test_y, class_test_y))

        DataViz.plot_xy(x=[leaf_settings, leaf_settings], y=[performance_training, performance_test],
                        xlabel='Minimum number of points per leaf', ylabel='Accuracy',
                        names=['training', 'test'], line_styles=['r-', 'b:'])

    if FLAGS.mode == 'overall' or FLAGS.mode == 'all':
        print('\n- - - Running test of all different classification algorithms - - -')
        # Perform grid searches over the most important parameters and do so by means of cross validation upon the
        # training set
        possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4,
                                 features_after_chapter_5, selected_features]
        feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']
        N_KCV_REPEATS = FLAGS.kcvrepeat

        scores_over_all_algs = []

        for i in range(0, len(possible_feature_sets)):
            selected_train_X = train_X[possible_feature_sets[i]]
            selected_test_X = test_X[possible_feature_sets[i]]

            # First run non deterministic classifiers a number of times to average their score
            performance_tr_nn, performance_te_nn = 0, 0
            performance_tr_rf, performance_te_rf = 0, 0
            performance_tr_svm, performance_te_svm = 0, 0

            for repeat in range(0, N_KCV_REPEATS):
                print(
                    f'Training NeuralNetwork run {repeat + 1} / {N_KCV_REPEATS}, featureset is {feature_names[i]} ... ')
                class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                    selected_train_X, train_y, selected_test_X, gridsearch=True)

                print(
                    f'Training RandomForest run {repeat + 1} / {N_KCV_REPEATS}, featureset is {feature_names[i]} ... ')
                performance_tr_nn += evaluation.accuracy(train_y, class_train_y)
                performance_te_nn += evaluation.accuracy(test_y, class_test_y)

                class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                    selected_train_X, train_y, selected_test_X, gridsearch=True)
                performance_tr_rf += evaluation.accuracy(train_y, class_train_y)
                performance_te_rf += evaluation.accuracy(test_y, class_test_y)

                print(f'Training SVM run {repeat + 1} / {N_KCV_REPEATS}, featureset is {feature_names[i]} ...')

                class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner. \
                    support_vector_machine_with_kernel(selected_train_X, train_y, selected_test_X, gridsearch=True)
                performance_tr_svm += evaluation.accuracy(train_y, class_train_y)
                performance_te_svm += evaluation.accuracy(test_y, class_test_y)

            overall_performance_tr_nn = performance_tr_nn / N_KCV_REPEATS
            overall_performance_te_nn = performance_te_nn / N_KCV_REPEATS
            overall_performance_tr_rf = performance_tr_rf / N_KCV_REPEATS
            overall_performance_te_rf = performance_te_rf / N_KCV_REPEATS
            overall_performance_tr_svm = performance_tr_svm / N_KCV_REPEATS
            overall_performance_te_svm = performance_te_svm / N_KCV_REPEATS

            # Run deterministic classifiers:
            print("Deterministic Classifiers:")

            print(f'Training Nearest Neighbor run 1 / 1, featureset {feature_names[i]}')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
                selected_train_X, train_y, selected_test_X, gridsearch=True)
            performance_tr_knn = evaluation.accuracy(train_y, class_train_y)
            performance_te_knn = evaluation.accuracy(test_y, class_test_y)

            print(f'Training Decision Tree run 1 / 1  featureset {feature_names[i]}')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
                selected_train_X, train_y, selected_test_X, gridsearch=True)
            performance_tr_dt = evaluation.accuracy(train_y, class_train_y)
            performance_te_dt = evaluation.accuracy(test_y, class_test_y)

            print(f'Training Naive Bayes run 1/1 featureset {feature_names[i]}')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
                selected_train_X, train_y, selected_test_X)
            performance_tr_nb = evaluation.accuracy(train_y, class_train_y)
            performance_te_nb = evaluation.accuracy(test_y, class_test_y)

            scores_with_sd = util. \
                print_table_row_performances(feature_names[i], len(selected_train_X.index),
                                             len(selected_test_X.index), [
                                                 (overall_performance_tr_nn, overall_performance_te_nn),
                                                 (overall_performance_tr_rf, overall_performance_te_rf),
                                                 (overall_performance_tr_svm, overall_performance_te_svm),
                                                 (performance_tr_knn, performance_te_knn),
                                                 (performance_tr_knn, performance_te_knn),
                                                 (performance_tr_dt, performance_te_dt),
                                                 (performance_tr_nb, performance_te_nb)])
            scores_over_all_algs.append(scores_with_sd)

        DataViz.plot_performances_classification(['NN', 'RF', 'SVM', 'KNN', 'DT', 'NB'], feature_names,
                                                 scores_over_all_algs)

    if FLAGS.mode == 'detail' or FLAGS.mode == 'all':
        print('\n- - - Running detail test of promising classification algorithms - - -')
        # Study two promising ones in more detail, namely decision tree and random forest algorithm
        learner.decision_tree(train_X[selected_features], train_y, test_X[selected_features],
                              gridsearch=True, print_model_details=True, export_tree_path=EXPORT_TREE_PATH)

        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
            train_X[selected_features], train_y, test_X[selected_features],
            gridsearch=True, print_model_details=True)

        test_cm = evaluation.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)
        DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        help="Select what version to run: Feature Selection, Regularization, Leaf Size, Overall, "
                             "Detail or All. \
                             'selection' to run the process of feature selection \
                             'regularization' to study the impact of regularization and model complexity \
                             'tree' to study the impact of different leaf sizes \
                             'overall' to train all models specified in learning algorithms \
                             'detail' to take a look at promising results \
                             'all' to run all parts of the script.",
                        choices=['selection', 'regularization', 'tree', 'overall', 'detail', 'all'])
    parser.add_argument('--nfeatures', type=int, default=50,
                        help="The number of features to select in forward feature selection.")
    parser.add_argument('--nnrepeat', type=int, default=3,
                        help="Number of repeats to use for regularization.")
    parser.add_argument('--kcvrepeat', type=int, default=5,
                        help="Number of repeats to use for overall training.")

    FLAGS, unparsed = parser.parse_known_args()

    # Prints args und run main script
    print_flags()
    main()
