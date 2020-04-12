##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from sklearn import metrics
import pandas as pd
import numpy as np
import math

# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    # Returns the accuracy given the true and predicted values.
    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    # Returns the recall given the true and predicted values.
    # Note that it returns the recall per class.
    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    # Returns the f1 given the true and predicted values.
    # Note that it returns the recall per class.
    def f1(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average=None)

    # Returns the area under the curve given the true and predicted values.
    # Note: we expect a binary classification problem here(!)
    def auc(self, y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    # Returns the confusion matrix given the true and predicted values.
    def confusion_matrix(self, y_true, y_pred, labels):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

# Class for evaluation metrics of regression problems.
class RegressionEvaluation:

    # Returns the mean squared error between the true and predicted values.
    def mean_squared_error(self, y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred)

    # Returns the mean squared error between the true and predicted values.
    def mean_squared_error_with_std(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.square(y_true-y_pred)
        mse = errors.mean()
        std = errors.std()
        return mse.mean(), std.mean()

    # Returns the mean absolute error between the true and predicted values.
    def mean_absolute_error(self, y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    # Return the mean absolute error between the true and predicted values
    # as well as its standard deviation.
    def mean_absolute_error_with_std(self, y_true, y_pred):
        errors = np.absolute((y_pred - y_true))
        return errors.mean(), errors.std()
