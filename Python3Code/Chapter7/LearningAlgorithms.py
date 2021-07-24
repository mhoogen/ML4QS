##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os
from typing import Tuple, List


class ClassificationAlgorithms:
    """
    This class provides different machine learning algorithms for classification tasks.
    """

    @staticmethod
    def feedforward_neural_network(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                   hidden_layer_sizes: Tuple[int] = (100,), max_iter: int = 500,
                                   activation: str = 'logistic', alpha: float = 0.0001, learning_rate: str = 'adaptive',
                                   gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a neural network for classification upon the training data with the specified composition of hidden
        layers and number of iterations and use the created network to predict the outcome for both the test and
        training set. It returns the categorical predictions for the training and test set as well as the probabilities
        associated with each class, each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param hidden_layer_sizes: Number of neurons in hidden layers.
        :param max_iter: Maximum number of iterations when training the neural network.
        :param activation: Activation function to use for neurons.
        :param alpha: Regularization parameter for training penalties.
        :param learning_rate: Adjust step size of weight fitting in training process.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached,
            # therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100, 5,), (100, 10,), ],
                                 'activation': [activation],
                                 'learning_rate': [learning_rate], 'max_iter': [2000, 3000], 'alpha': [alpha]}]
            nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            # Create the model
            nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
                               learning_rate=learning_rate, alpha=alpha, random_state=42)

        # Fit the model
        nn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def support_vector_machine_with_kernel(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                           C: float = 1, kernel: str = 'rbf', gamma: float = 1e-3,
                                           gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a support vector machine for classification upon the training data with the specified values for C,
        gamma and the kernel function and use the created model to predict the outcome for both the test and
        training set. It returns the categorical predictions for the training and test set as well as the
        probabilities associated with each class, each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param C: Hyperparamter C to set the penalty for wrong classified points.
        :param kernel: Kernel function to use when training the SVM.
        :param gamma: Hyperparamter gamma to set the precision of the decision boundary.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def support_vector_machine_without_kernel(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                              C: float = 1, tol: float = 1e-3, max_iter: int = 1000,
                                              gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a support vector machine for classification upon the training data with the specified values for C,
        gamma and the kernel function and use the created model to predict the outcome for both the test and
        training set. It returns the categorical predictions for the training and test set as well as the
        probabilities associated with each class, each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param C: Hyperparamter C to set the penalty for wrong classified points.
        :param tol: Hyperparamter to set for stopping tolerance.
        :param max_iter: Number of maximum iterations.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        distance_training_platt = 1 / (1 + np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:, None]
        distance_test_platt = 1 / (1 + np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:, None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def k_nearest_neighbor(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, n_neighbors: int = 5,
                           gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a nearest neighbor approach for classification upon the training data with the specified value for k
        and use the created model to predict the outcome for both the test and training set. It returns the
        categorical predictions for the training and test set as well as the probabilities associated with each class,
        each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param n_neighbors: Number of neighbours to consider when predicting new data points.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def decision_tree(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, min_samples_leaf: int = 50,
                      criterion: str = 'gini', export_tree_path: str = './figures/crowdsignals_ch7_classification/',
                      export_tree_name: str = 'tree.dot', print_model_details: bool = False, gridsearch: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a decision tree approach for classification upon the training data with the specified value for the
        minimum samples in the leaf and use the created model to predict the outcome for both the test and training
        set. It returns the categorical predictions for the training and test set as well as the probabilities
        associated with each class, each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param min_samples_leaf: Hyperparameter to set the minimum number of samples per leaf when building the tree.
        :param criterion: Hyperparameter to set the criterion to use during training.
        :param export_tree_path: Path to to directory to save the tree files in.
        :param export_tree_name: Name of the tree file.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion': ['gini', 'entropy']}]
            dec_tree = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            dec_tree = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf)

        # Fit the model

        dec_tree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dec_tree.best_params_)

        if gridsearch:
            dec_tree = dec_tree.best_estimator_

        # Apply the model
        pred_prob_training_y = dec_tree.predict_proba(train_X)
        pred_prob_test_y = dec_tree.predict_proba(test_X)
        pred_training_y = dec_tree.predict(train_X)
        pred_test_y = dec_tree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dec_tree.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dec_tree.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in
                               sorted(enumerate(dec_tree.feature_importances_), key=lambda x: x[1], reverse=True)]
            print('Feature importance decision tree:')
            for i in range(0, len(dec_tree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dec_tree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dec_tree, out_file=str(export_tree_path) + '/' + export_tree_name,
                                 feature_names=train_X.columns, class_names=dec_tree.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def naive_bayes(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a naive bayes approach for classification upon the training data and use the created model to predict the
        outcome for both the test and training set. It returns the categorical predictions for the training and test
        set as well as the probabilities associated with each class, each class being represented as a column in the
        data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        nb = GaussianNB()

        train_y = train_y.values.ravel()
        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    @staticmethod
    def random_forest(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, n_estimators: int = 10,
                      min_samples_leaf: int = 5, criterion: str = 'gini', print_model_details: bool = False,
                      gridsearch: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Apply a random forest approach for classification upon the training data with the specified values for the
        minimum samples in the leaf, the number of trees and use the created model to predict the outcome for both
        the test and training set. It returns the categorical predictions for the training and test set as well as
        the probabilities associated with each class, each class being represented as a column in the data frame.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param n_estimators: Hyperparameter to set the number of trees to build.
        :param min_samples_leaf: Hyperparameter to set the minimum number of samples per leaf when building the trees.
        :param criterion: Hyperparameter to set the criterion to use during training.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Categorical predictions and class probabilities of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators': [10, 50, 100],
                                 'criterion': ['gini', 'entropy']}]
            rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)
        if gridsearch:
            rf = rf.best_estimator_

        # Apply the model
        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in
                               sorted(enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True)]
            print('Feature importance random forest:')
            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y


class RegressionAlgorithms:
    """
    This class provides different machine learning algorithms for regression tasks.
    """

    @staticmethod
    def feedforward_neural_network(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                   hidden_layer_sizes: Tuple[int] = (100,), max_iter: int = 500,
                                   activation: str = 'identity', learning_rate: str = 'adaptive',
                                   gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a neural network for regression upon the training data with the specified composition of hidden layers
        and number of iterations and use the created network to predict the outcome for both the test and training
        set. It returns the numerical predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param hidden_layer_sizes: Number of neurons in hidden layers.
        :param max_iter: Maximum number of iterations when training the neural network.
        :param activation: Activation function to use for neurons.
        :param learning_rate: Adjust step size of weight fitting in training process.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached,
            # therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100, 5,), (100, 10,), ],
                                 'activation': ['identity'],
                                 'learning_rate': ['adaptive'], 'max_iter': [4000, 10000]}]
            nn = GridSearchCV(MLPRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
                              learning_rate=learning_rate)

        # Fit the model
        nn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)

        return pred_training_y, pred_test_y

    @staticmethod
    def support_vector_regression_with_kernel(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                              kernel: str = 'rbf', C: float = 1, gamma: float = 1e-3,
                                              gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a support vector machine with a given kernel function for regression upon the training data with the
        specified values for C, gamma and the kernel function and use the created model to predict the outcome for
        both the test and training set. It returns the numerical predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param C: Hyperparamter C to set the penalty for wrong classified points.
        :param kernel: Kernel function to use when training the SVM.
        :param gamma: Hyperparamter gamma to set the precision of the decision boundary.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf', 'poly'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svr = GridSearchCV(SVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            svr = SVR(C=C, kernel=kernel, gamma=gamma)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    @staticmethod
    def support_vector_regression_without_kernel(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
                                                 C: float = 1, tol: float = 1e-3, max_iter: int = 1000,
                                                 gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a support vector machine without a complex kernel function for regression upon the training data with
        the specified values for C, tolerance and max iterations and use the created model to predict the outcome for
        both the test and training set. It returns the numerical predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param C: Hyperparamter C to set the penalty for wrong predicted points.
        :param tol: Hyperparamter to set for stopping tolerance.
        :param max_iter: Number of maximum iterations.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached,
            # with increased iterations/tolerance often still fails to converge.
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                                 'C': [1, 10, 100]}]
            svr = GridSearchCV(LinearSVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            svr = LinearSVR(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    @staticmethod
    def k_nearest_neighbor(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, n_neighbors: int = 5,
                           gridsearch: bool = True, print_model_details: bool = False) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a nearest neighbor approach for regression upon the training data with the specified value for k and use
        the created model to predict the outcome for both the test and training set. It returns the numerical
        predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param n_neighbors: Number of neighbours to consinder when predicting new data points.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)

        return pred_training_y, pred_test_y

    @staticmethod
    def decision_tree(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, min_samples_leaf: int = 50,
                      criterion: str = 'mse', export_tree_path: str = './figures/crowdsignals_ch7_regression/',
                      export_tree_name: str = 'tree.dot', gridsearch=True, print_model_details: bool = False) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a decision tree approach for regression upon the training data with the specified values for the minimum
        samples in the leaf and use the created model to predict the outcome for both the test and training set. It
        returns the numerical predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param min_samples_leaf: Hyperparameter to set the minimum number of samples per leaf when building the tree.
        :param criterion: Hyperparameter to set the criterion to use during training.
        :param export_tree_path: Path to to directory to save the tree files in.
        :param export_tree_name: Name of the tree file.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
        :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion': ['mse']}]
            dtree = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            dtree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)

        if print_model_details:
            print('Feature importance decision tree:')
            ordered_indices = [i[0] for i in
                               sorted(enumerate(dtree.feature_importances_), key=lambda x: x[1], reverse=True)]
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name,
                                 feature_names=train_X.columns, class_names=dtree.classes_)

        return pred_training_y, pred_test_y

    @staticmethod
    def random_forest(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, n_estimators: int = 10,
                      min_samples_leaf: int = 5, criterion: str = 'mse', print_model_details: bool = False,
                      gridsearch: bool = True) \
            -> Tuple[List[float], List[float]]:
        """
        Apply a random forest approach for regression upon the training data with the specified values for the minimum
        samples in the leaf and the number of trees and use the created model to predict the outcome for both the test
        and training set. It returns the numerical predictions for the training and test set.

        :param train_X: Features to use in training.
        :param train_y: True labels corresponding of training set.
        :param test_X: Features of test set.
        :param n_estimators: Hyperparameter to set the number of trees to build.
        :param min_samples_leaf: Hyperparameter to set the minimum number of samples per leaf when building the trees.
        :param criterion: Hyperparameter to set the criterion to use during training.
        :param gridsearch: Set to True to find best parameters using gridsearch.
        :param print_model_details: Set to True to print out model details.
         :return: Predictions of both training and test set.
        """

        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators': [10, 50, 100],
                                 'criterion': ['mse']}]
            rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                       criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        # Apply the model
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)

        if print_model_details:
            print('Feature importance random forest:')
            ordered_indices = [i[0] for i in
                               sorted(enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True)]

            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y
