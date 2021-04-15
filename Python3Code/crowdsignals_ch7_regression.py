##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from pathlib import Path
import pandas as pd

from util import util
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset(__file__)

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter5_result.csv'
EXPORT_TREE_PATH = Path('figures/example_graphs/Chapter7/')

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

EXPORT_TREE_PATH.mkdir(exist_ok=True, parents=True)

# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'hr_watch_rate', '2016-02-08 18:28:56',
                                                                                   '2016-02-08 19:34:07', '2016-02-08 20:07:50')
#                                                                                   '2016-02-08 18:28:58','2016-02-08 18:28:59')

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z',
                  'labelOnTable','labelSitting','labelWashingHands','labelWalking','labelStanding','labelDriving','labelEating','labelRunning',
                  'light_phone_lux','mag_phone_x','mag_phone_y','mag_phone_z','mag_watch_x','mag_watch_y','mag_watch_z','press_phone_pressure']
pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
time_features = [name for name in dataset.columns if ('temp_' in name and not 'hr_watch' in name)]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#basic features: ', len(basic_features))
print('#PCA features: ', len(pca_features))
print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))
cluster_features = ['cluster']
print('#cluster features: ', len(cluster_features))
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

fs = FeatureSelectionRegression()

# First, let us consider the Pearson correlations and see whether we can select based on them.
features, correlations = fs.pearson_selection(10, train_X[features_after_chapter_5], train_y)
util.print_pearson_correlations(correlations)

# We select the 10 features with the highest correlation.

selected_features = ['temp_pattern_labelOnTable','labelOnTable','temp_pattern_labelOnTable(b)labelOnTable','pca_2_temp_mean_ws_120',
                     'pca_1_temp_mean_ws_120','acc_watch_y_temp_mean_ws_120','pca_2','acc_phone_z_temp_mean_ws_120',
                     'gyr_watch_y_pse','gyr_watch_x_pse']

possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

# Let us first study the importance of the parameter settings.

learner = RegressionAlgorithms()
eval = RegressionEvaluation()

# We repeat the experiment a number of times to get a bit more robust data as the initialization of e.g. the NN is random.

REPEATS = 5

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):

    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_nn = 0
    performance_tr_nn_std = 0
    performance_tr_rf = 0
    performance_tr_rf_std = 0
    performance_tr_svm = 0
    performance_tr_svm_std = 0
    performance_te_nn = 0
    performance_te_nn_std = 0
    performance_te_rf = 0
    performance_te_rf_std = 0
    performance_te_svm = 0
    performance_te_svm_std = 0

    for repeat in range(0, REPEATS):
        print("Training NeuralNetwork run {} / {} ... ".format(repeat, REPEATS))
        regr_train_y, regr_test_y = learner.feedforward_neural_network(selected_train_X, train_y, selected_test_X, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
        mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
        mean_training = eval.mean_squared_error(train_y, regr_train_y)
        performance_tr_nn += mean_tr
        performance_tr_nn_std += std_tr
        performance_te_nn += mean_te
        performance_te_nn_std += std_te
        print("Training RandomForest run {} / {} ... ".format(repeat, REPEATS))
        regr_train_y, regr_test_y = learner.random_forest(selected_train_X, train_y, selected_test_X, gridsearch=True)
        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
        mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
        performance_tr_rf += mean_tr
        performance_tr_rf_std += std_tr
        performance_te_rf += mean_te
        performance_te_rf_std += std_te

    overall_performance_tr_nn = performance_tr_nn/REPEATS
    overall_performance_tr_nn_std = performance_tr_nn_std/REPEATS
    overall_performance_te_nn = performance_te_nn/REPEATS
    overall_performance_te_nn_std = performance_te_nn_std/REPEATS
    overall_performance_tr_rf = performance_tr_rf/REPEATS
    overall_performance_tr_rf_std = performance_tr_rf_std/REPEATS
    overall_performance_te_rf = performance_te_rf/REPEATS
    overall_performance_te_rf_std = performance_te_rf_std/REPEATS

    # And we run our deterministic algorithms:

    print("Support Vector Regressor run 1 / 1 ... ")
    # Convergence of the SVR does not always occur (even adjusting tolerance and iterations does not help)
    regr_train_y, regr_test_y = learner.support_vector_regression_without_kernel(selected_train_X, train_y, selected_test_X, gridsearch=False)
    mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
    mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
    performance_tr_svm = mean_tr
    performance_tr_svm_std = std_tr
    performance_te_svm = mean_te
    performance_te_svm_std = std_te


    print("Training Nearest Neighbor run 1 / 1 ... ")
    regr_train_y, regr_test_y = learner.k_nearest_neighbor(selected_train_X, train_y, selected_test_X, gridsearch=True)
    mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
    mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
    performance_tr_knn = mean_tr
    performance_tr_knn_std = std_tr
    performance_te_knn = mean_te
    performance_te_knn_std = std_te

    print("Training Decision Tree run 1 / 1 ... ")
    regr_train_y, regr_test_y = learner.decision_tree(selected_train_X, train_y, selected_test_X, gridsearch=True, export_tree_path=EXPORT_TREE_PATH)

    mean_tr, std_tr = eval.mean_squared_error_with_std(train_y, regr_train_y)
    mean_te, std_te = eval.mean_squared_error_with_std(test_y, regr_test_y)
    performance_tr_dt = mean_tr
    performance_tr_dt_std = std_tr
    performance_te_dt = mean_te
    performance_te_dt_std = std_te

    scores_with_sd = [(overall_performance_tr_nn, overall_performance_tr_nn_std, overall_performance_te_nn, overall_performance_te_nn_std),
                      (overall_performance_tr_rf, overall_performance_tr_rf_std, overall_performance_te_rf, overall_performance_te_rf_std),
                          (performance_tr_svm, performance_tr_svm_std, performance_te_svm, performance_te_svm_std),
                      (performance_tr_knn, performance_tr_knn_std, performance_te_knn, performance_te_knn_std),
                      (performance_tr_dt, performance_tr_dt_std, performance_te_dt, performance_te_dt_std)]
    #util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_regression(['NN', 'RF','SVM', 'KNN', 'DT'], feature_names, scores_over_all_algs)


regr_train_y, regr_test_y = learner.random_forest(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], gridsearch=False, print_model_details=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y, test_X.index, test_y, regr_test_y, 'heart rate')
