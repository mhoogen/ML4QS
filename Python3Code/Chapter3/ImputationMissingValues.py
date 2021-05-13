##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################
import pandas as pd


class ImputationMissingValues:

    @staticmethod
    def impute_mean(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Impute the mean value in case if missing data.

        :param dataset: Dataset with missing values.
        :param col: Column containing missing values that will be filled.
        :return: Dataframe with imputed values in col.
        """

        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    @staticmethod
    def impute_median(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Impute the median value in case if missing data.

        :param dataset: Dataset with missing values.
        :param col: Column containing missing values that will be filled.
        :return: Dataframe with imputed values in col.
        """

        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    @staticmethod
    def impute_interpolate(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Interpolate the dataset based on previous and next values in case of missing values.

        :param dataset: Dataset with missing values.
        :param col: Column containing missing values that will be filled.
        :return:
        """

        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset
