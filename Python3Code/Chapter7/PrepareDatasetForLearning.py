##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from sklearn.model_selection import train_test_split
import numpy as np
import random
import copy
import pandas as pd
from typing import List, Tuple


class PrepareDatasetForLearning:
    """
    This class creates datasets that can be used by the learning algorithms in LearningAlgorithms.py code. Up till now
    binary columns are assumed for each class, but approaches to create a single categorical attribute will be
    introduced.
    """

    default_label = 'undefined'
    class_col = 'class'
    person_col = 'person'

    def assign_label(self, dataset: pd.DataFrame, class_labels: List[str]) -> pd.DataFrame:
        """
        Create a single class column based on a set of binary class columns. The given class columns are merged and
        removed after.

        :param dataset: DataFrame with class columns to merge.
        :param class_labels: List of class labels.
        :return: Dataframe with merged class column instead of binary class columns.
        """

        # Find which columns are relevant based on the possibly partial class_label specification
        labels = []
        for class_label in class_labels:
            labels.extend([col for col in list(dataset.columns) if col.startswith(class_label)])

        # Determine how many class values are label as 'true' in class columns
        sum_values = dataset[labels].sum(axis=1)
        # Create a new 'class' column and set the value to the default class
        dataset['class'] = self.default_label
        for i in range(0, len(dataset.index)):
            # Assign a class case of exactly one true class column, otherwise we keep the default class.
            if sum_values[i] == 1:
                dataset.iloc[i, dataset.columns.get_loc(self.class_col)] = dataset[labels].iloc[i].idxmax(axis=1)
        # Remove our old binary columns
        dataset = dataset.drop(labels, axis=1)
        return dataset

    def split_single_dataset_classification(self, dataset: pd.DataFrame, class_labels: List[str], matching: str,
                                            training_frac: float, filter_data: bool = True, temporal: bool = False,
                                            random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                            pd.Series, pd.Series]:
        """
        Split a dataset of a single person for a classification problem with the the specified class columns. If 'like'
        is specified for matching parameter, class columns are merged into a single column. If filter is set to True,
        rows without a unique class and rows with missing values are dropped. Set temporal to True, if a temporal
        dataset is present and set a fraction to use as training data. Training and test set with their corresponding
        labels are returned.

        :param dataset: DataFrame with data to split into train and test set.
        :param class_labels: List of class labels.
        :param matching: Merge class labels into a single column or not.
        :param training_frac: Fraction of data to use for training set.
        :param filter_data: Choose to filter out undesired rows.
        :param temporal: Set if temporal data is present.
        :param random_state: Set the random seed to make the split reproducible.
        :return: Training and test set in form of (training_set_X, test_set_X, training_set_y, test_set_y).
        """
        # Create a single class column if 'like' is matching option
        if matching == 'like':
            dataset = self.assign_label(dataset, class_labels)
            class_labels = self.class_col
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # Drop instances with missing values if desired and those for which the class cannot be determined
        if filter_data:
            dataset = dataset.dropna()
            dataset = dataset[dataset['class'] != self.default_label]

        # Get features that are the ones not in the class label
        features = [dataset.columns.get_loc(x) for x in dataset.columns if x not in class_labels]
        class_label_indices = [dataset.columns.get_loc(x) for x in dataset.columns if x in class_labels]

        # Select the set fraction of training data from the first part and use the rest as test set for temporal data
        if temporal:
            end_training_set = int(training_frac * len(dataset.index))
            training_set_X = dataset.iloc[0:end_training_set, features]
            training_set_y = dataset.iloc[0:end_training_set, class_label_indices]
            test_set_X = dataset.iloc[end_training_set:len(dataset.index), features]
            test_set_y = dataset.iloc[end_training_set:len(dataset.index), class_label_indices]
        # Use a standard function for non temporal data to randomly split the dataset
        else:
            training_set_X, test_set_X, training_set_y, test_set_y = \
                train_test_split(dataset.iloc[:, features], dataset.iloc[:, class_label_indices],
                                 test_size=(1 - training_frac), stratify=dataset.iloc[:, class_label_indices],
                                 random_state=random_state)
        return training_set_X, test_set_X, training_set_y, test_set_y

    @staticmethod
    def split_single_dataset_regression_by_time(dataset: pd.DataFrame, target: str, start_training: str,
                                                end_training: str, end_test: str) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                                           pd.Series, pd.Series]:
        """
        Split a single dataset by time into training and test set.

        :param dataset: DataFrame with data to split.
        :param target: Target column for regression.
        :param start_training: Start for training set in datetime format.
        :param end_training: End of training set in datetime format.
        :param end_test: End of test set in datetime format.
        :return: Train and test set in form of (training_set_X, test_set_X, training_set_y, test_set_y).
        """

        training_instances = dataset[start_training:end_training]
        test_instances = dataset[end_training:end_test]
        train_y = copy.deepcopy(training_instances[target])
        test_y = copy.deepcopy(test_instances[target])
        train_X = training_instances
        del train_X[target]
        test_X = test_instances
        del test_X[target]
        return train_X, test_X, train_y, test_y

    def split_single_dataset_regression(self, dataset: pd.DataFrame, targets: List[str], training_frac: float,
                                        filter_data: bool = False, temporal: bool = False, random_state: int = 0) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split a dataset of a single person for a regression with the specified targets. Multiple targets are possible
        and need to be passed as a list of column names. Select whether the data is temporal or not and set the
        fraction to use for training. The function returns training and test set in the form of (training_set_X,
        test_set_X, training_set_y, test_set_y).

        :param dataset: DataFrame with the data to split.
        :param targets: List of target columns.
        :param training_frac: Fraction to use as training data (float between 0 and 1).
        :param filter_data: Set to True to drop undesired rows.
        :param temporal: Set to true for temporal data.
        :param random_state: Set the random seed to make the split reproducible.
        :return: Train and test set in form of (training_set_X, test_set_X, training_set_y, test_set_y).
        """

        # Temporarily change some attribute values associated with the classification algorithm for numerical values
        # Then apply the classification variant of the function
        temp_default_label = self.default_label
        self.default_label = np.nan
        training_set_X, test_set_X, training_set_y, test_set_y = \
            self.split_single_dataset_classification(dataset, targets, 'exact', training_frac, filter_data=filter_data,
                                                     temporal=temporal, random_state=random_state)
        self.default_label = temp_default_label
        return training_set_X, test_set_X, training_set_y, test_set_y

    @staticmethod
    def update_set(source_set: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
        """
        Append to dataframes and check if the indices are overlapping ((e.g. user 1 and user 2 have the same time
        stamps). In case of overlapping indices create a new unique index.

        :param source_set: First DataFrame to append the second one to.
        :param addition: DataFrame to append to the first one.
        :return: Merged dataframes with unique index.
        """

        if source_set is None:
            return addition
        else:
            # Check if the index is unique and create a new if not
            if len(set(source_set.index) & set(addition.index)) > 0:
                return source_set.append(addition).reset_index(drop=True)
            else:
                return source_set.append(addition)

    def split_multiple_datasets_classification(self, datasets: List[pd.DataFrame], class_labels: List[str],
                                               matching: str, training_frac: float, filter_data: bool = False,
                                               temporal: bool = False, unknown_users: bool = False,
                                               random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                               pd.Series, pd.Series]:
        """
        Split multiple datasets representing different users into training and test set to perform classification. In
        addition select what should be predicted: Classification for unknown use (unknown_user=True) or for unseen
        data over all users. In the former, the method returns a training set containing all data of training_frac
        users and test data for the remaining users. If the later, it returns the training_frac data of each user as
        a training set, and 1-training_frac data as a test set. Train and test set are returned in the form of
        (training_set_X, test_set_X, training_set_y, test_set_y).

        :param datasets: List of DataFrames.
        :param class_labels: List of class labels to use for classification task.
        :param matching: Set to merge class labels to single columns or keep different columns.
        :param training_frac: Part of data to use for training.
        :param filter_data: Set to True to drop undesired rows.
        :param temporal: Set to true for temporal data.
        :param unknown_users: Set to True for classification with unknown user.
        :param random_state: Set the random seed to make the split reproducible.
        :return: Train and test set in form of (training_set_X, test_set_X, training_set_y, test_set_y).
        """

        # Initialize empty training and test sets
        training_set_X = None
        training_set_y = None
        test_set_X = None
        test_set_y = None

        if unknown_users:
            # Shuffle the users
            random.seed(random_state)
            indices = range(0, len(datasets))
            random.shuffle(indices)
            training_len = int(training_frac * len(datasets))

            # Select the data of the first fraction of users as the training set and remaining data as test set
            for i in range(0, training_len):
                # Use the single dataset function for classification and add it to the training data
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = \
                    self.split_single_dataset_classification(datasets[indices[i]], class_labels, matching, 1,
                                                             filter_data=filter_data, temporal=temporal,
                                                             random_state=random_state)
                # Add a person column
                training_set_X_person[self.person_col] = indices[i]
                training_set_X = self.update_set(training_set_X, training_set_X_person)
                training_set_y = self.update_set(training_set_y, training_set_y_person)

            for j in range(training_len, len(datasets)):
                # Use the single dataset function for classification and add it to the test data
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = \
                    self.split_single_dataset_classification(datasets[indices[j]], class_labels, matching, 1,
                                                             filter_data=filter_data, temporal=temporal,
                                                             random_state=random_state)
                # Add a person column.
                training_set_X_person[self.person_col] = indices[j]
                test_set_X = self.update_set(test_set_X, training_set_X_person)
                test_set_y = self.update_set(test_set_y, training_set_y_person)
        else:
            # Otherwise split each dataset individually in a training and test set and add them
            for i in range(0, len(datasets)):
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = \
                    self.split_single_dataset_classification(datasets[i], class_labels, matching, training_frac,
                                                             filter_data=filter_data, temporal=temporal,
                                                             random_state=random_state)
                # Add a person column
                training_set_X_person[self.person_col] = i
                test_set_X_person[self.person_col] = i
                training_set_X = self.update_set(training_set_X, training_set_X_person)
                training_set_y = self.update_set(training_set_y, training_set_y_person)
                test_set_X = self.update_set(test_set_X, test_set_X_person)
                test_set_y = self.update_set(test_set_y, test_set_y_person)
        return training_set_X, test_set_X, training_set_y, test_set_y

    def split_multiple_datasets_regression(self, datasets: List[pd.DataFrame], targets: List[str], training_frac: float,
                                           filter_data: bool = False, temporal: bool = False,
                                           unknown_users: bool = False, random_state: int = 0) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split multiple datasets representing different users into training and test set to perform regression. In
        addition select what should be predicted: Classification for unknown use (unknown_user=True) or for unseen
        data over all users. In the former, the method returns a training set containing all data of training_frac
        users and test data for the remaining users. If the later, it returns the training_frac data of each user as
        a training set, and 1-training_frac data as a test set. Train and test set are returned in the form of
        (training_set_X, test_set_X, training_set_y, test_set_y).

        :param datasets: List of DataFrames.
        :param targets: List of targets to use for regression task.
        :param training_frac: Part of data to use for training.
        :param filter_data: Set to True to drop undesired rows.
        :param temporal: Set to true for temporal data.
        :param unknown_users: Set to True for classification with unknown user.
        :param random_state: Set the random seed to make the split reproducible.
        :return: Train and test set in form of (training_set_X, test_set_X, training_set_y, test_set_y).
        """

        # Temporarily change some attribute values associated with the regression algorithm for numerical values
        # Then apply the classification variant of the function
        temp_default_label = self.default_label
        self.default_label = np.nan
        training_set_X, test_set_X, training_set_y, test_set_y = \
            self.split_multiple_datasets_classification(datasets, targets, 'exact', training_frac,
                                                        filter_data=filter_data,
                                                        temporal=temporal, unknown_users=unknown_users,
                                                        random_state=random_state)
        self.default_label = temp_default_label
        return training_set_X, test_set_X, training_set_y, test_set_y
