##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import RegressionEvaluation
from scipy.stats import pearsonr
import sys
import copy
import numpy as np
from operator import itemgetter
import pandas as pd

# Specifies feature selection approaches for classification to identify the most important features.
class FeatureSelectionClassification:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            print i

            #Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ''

            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = ce.accuracy(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f
            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)
        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):
        # First select all features.
        selected_features = X_train.columns.tolist()
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = 0
            worst_feature = ''

            # Select from the features that are still in the selection.
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = ce.accuracy(y_train, pred_y_train)

                # If we score better without the feature than what we have seen so far
                # this is the worst feature.
                if perf > best_perf:
                    best_perf = perf
                    worst_feature = f

            # Remove the worst feature.
            selected_features.remove(worst_feature)
        return selected_features

# Specifies feature selection approaches for classification to identify the most important features.
class FeatureSelectionRegression:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        ordered_features = []
        ordered_scores = []

        # Start with no features.
        selected_features = []
        ra = RegressionAlgorithms()
        re = RegressionEvaluation()
        prev_best_perf = sys.float_info.max

        # Select the appropriate number of features.
        for i in range(0, max_features):

            #Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = sys.float_info.max
            best_feature = ''

            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the mse of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test = ra.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = re.mean_squared_error(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for low mse)
                # we set the current feature to the best feature and the same for the best performance.
                if perf < best_perf:
                    best_perf = perf
                    best_feature = f
            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)
        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):

        # First select all features.
        selected_features = X_train.columns.tolist()
        ra = RegressionAlgorithms()
        re = RegressionEvaluation()

        # Select from the features that are still in the selection.
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = sys.float_info.max
            worst_feature = ''
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test = ra.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = re.mean_squared_error(y_train, pred_y_train)
                # If we score better (i.e. a lower mse) without the feature than what we have seen so far
                # this is the worst feature.
                if perf < best_perf:
                    best_perf = perf
                    worst_feature = f
            # Remove the worst feature.
            selected_features.remove(worst_feature)
        return selected_features

    # Select features based upon the correlation through the Pearson coefficient.
    # It return the max_features best features.
    def pearson_selection(self, max_features, X_train, y_train):
        correlations = []
        full_columns_and_corr = []
        abs_columns_and_corr = []

        # Compute the absolute correlations per column.
        for i in range(0, len(X_train.columns)):
            corr, p = pearsonr(X_train[X_train.columns[i]], y_train)
            correlations.append(abs(corr))
            if np.isfinite(corr):
                full_columns_and_corr.append((X_train.columns[i], corr))
                abs_columns_and_corr.append((X_train.columns[i], abs(corr)))

        sorted_attributes = sorted(abs_columns_and_corr,key=itemgetter(1), reverse=True)
        res_list = [x[0] for x in sorted_attributes[0:max_features]]

        # And return the most correlated ones.
        return res_list, sorted(full_columns_and_corr,key=itemgetter(1), reverse=True)