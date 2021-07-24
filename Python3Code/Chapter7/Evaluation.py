##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from sklearn import metrics
import numpy as np
from typing import Tuple, Iterable, List, Union


class ClassificationEvaluation:
    """
    Class for evaluation metrics of classification problems.
    """

    @staticmethod
    def accuracy(y_true: Union[List, np.ndarray, Iterable], y_pred: Union[List, np.ndarray, Iterable]) -> float:
        """
        Calculate the accuracy given the true and predicted values.

        :param y_true: True labels of the classes.
        :param y_pred: Predicted labels of the classes.
        :return: Percentage of correctly classified classes.
        """

        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    @staticmethod
    def precision(y_true: Union[List, np.ndarray, Iterable], y_pred: Union[List, np.ndarray, Iterable]) -> np.ndarray:
        """
        Calculate the precision given the true and predicted values. In case of multiclass classification the precision
        of each class is computed.

        :param y_true: True labels of the classes.
        :param y_pred: Predicted labels of the classes.
        :return: Precision for each class.
        """

        return metrics.precision_score(y_true, y_pred, average=None)

    @staticmethod
    def recall(y_true: Union[List, np.ndarray, Iterable], y_pred: Union[List, np.ndarray, Iterable]) -> np.ndarray:
        """
        Calculate the recall given the true and predicted values. In case of multiclass classification the recall of
        each class is computed.

        :param y_true: True labels of the classes.
        :param y_pred: Predicted labels of the classes.
        :return: Recall for each class.
        """

        return metrics.recall_score(y_true, y_pred, average=None)

    @staticmethod
    def f1(y_true: Union[List, np.ndarray, Iterable], y_pred: Union[List, np.ndarray, Iterable]) -> np.ndarray:
        """
        Calculate the f1 score given the true and predicted values. In case of multiclass classification the f1 score of
        each class is computed.

        :param y_true: True labels of the classes.
        :param y_pred: Predicted labels of the classes.
        :return: f1 score for each class.
        """

        return metrics.f1_score(y_true, y_pred, average=None)

    @staticmethod
    def auc(y_true: Union[List, np.ndarray, Iterable], y_pred_prob: Union[List, np.ndarray, Iterable]) -> np.ndarray:
        """
        Calculate the area under the curve given the true and predicted values.
        Note: This method expects a binary classification problem!

        :param y_true: True labels of the classes.
        :param y_pred_prob: Predicted probabilities of the classes.
        :return: ROC-AUC-Score for all predictions.
        """

        return metrics.roc_auc_score(y_true, y_pred_prob)

    @staticmethod
    def confusion_matrix(y_true: Union[List, np.ndarray, Iterable], y_pred: Union[List, np.ndarray, Iterable],
                         labels: List[str]) -> np.ndarray:
        """
        Compute the confusion matrix given the true and predicted values.

        :param y_true: True labels of the classes.
        :param y_pred: Predicted labels of the classes.
        :param labels: List of class labels.
        :return: Confusion matrix in form of 2-dimensional Numpy array.
        """

        return metrics.confusion_matrix(y_true, y_pred, labels=labels)


class RegressionEvaluation:
    """
    Class for evaluation metrics of regression problems.
    """

    @staticmethod
    def mean_squared_error(y_true: Union[List, np.ndarray, Iterable],
                           y_pred: Union[List, np.ndarray, Iterable]) -> float:
        """
        Compute the mean squared error between the true and predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Mean squared error.
        """

        return metrics.mean_squared_error(y_true, y_pred)

    @staticmethod
    def mean_squared_error_with_std(y_true: Union[List, np.ndarray, Iterable],
                                    y_pred: Union[List, np.ndarray, Iterable]) -> Tuple[float, float]:
        """
        Compute the mean squared error as well as the standard deviation of the squared errors between the true and
        predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Mean and standard deviation of squared errors.
        """

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.square(y_true - y_pred)
        return errors.mean(), errors.mean()

    @staticmethod
    def mean_absolute_error(y_true: Union[List, np.ndarray, Iterable],
                            y_pred: Union[List, np.ndarray, Iterable]) -> float:
        """
        Compute the mean absolute error between the true and predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Mean squared error.
        """

        return metrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mean_absolute_error_with_std(y_true: Union[List, np.ndarray, Iterable],
                                     y_pred: Union[List, np.ndarray, Iterable]) -> Tuple[float, float]:
        """
        Compute the mean absolute error as well as the standard deviation of the absolute errors between the true and
        predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: Mean and standard deviation of absolute errors.
        """

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.absolute((y_pred - y_true))
        return errors.mean(), errors.std()
