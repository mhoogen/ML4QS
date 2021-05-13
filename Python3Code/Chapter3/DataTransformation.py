##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################
import pandas as pd
from sklearn.decomposition import PCA
import util.util as util
from scipy.signal import butter, lfilter, filtfilt
from typing import List


class LowPassFilter:
    # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    @staticmethod
    def low_pass_filter(data_table: pd.DataFrame, col: str, sampling_frequency: float, cutoff_frequency: float,
                        order: int = 5, phase_shift: bool = True) -> pd.DataFrame:
        """
        Apply lowpass filter on the data to remove high frequency data (that might be considered noise) from the data.

        :param data_table: DataFrame with the data to apply the lowpass filter on.
        :param col: Column name to the relevant data.
        :param sampling_frequency: Recording frequency of the data.
        :param cutoff_frequency: Limit value above which the frequencies are removed.
        :param order: Value that determines how sharp from limit value is cut off.
        :param phase_shift: Choose to apply phase shift or not.
        :return: Dataframe with a new column containing the filtered data.
        """

        # Cutoff frequencies are expressed as a fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype='low', output='ba', analog=False)
        if phase_shift:
            data_table[col + '_lowpass'] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + '_lowpass'] = lfilter(b, a, data_table[col])
        return data_table


class PrincipalComponentAnalysis:

    @staticmethod
    def determine_pc_explained_variance(data_table: pd.DataFrame, cols: List[str]) -> List[float]:
        """
        Perform the PCA on the selected columns and return the explained variance. The function can only be applied if
        the data does not have missing values (i.E. NaN), so rows with missing values have to be deleted before calling.

        :param data_table: Dataframe with the data to fit PCA on.
        :param cols: Columns in data_table to use for fitting the PCA.
        :return: List with explained variances of the principle components.
        """

        # Normalize the data
        dt_norm = util.normalize_dataset(data_table, cols)

        # Perform the PCA and return the explained variances
        pca = PCA(n_components=len(cols))
        pca.fit(dt_norm[cols])
        return pca.explained_variance_ratio_

    @staticmethod
    def apply_pca(data_table: pd.DataFrame, cols: List[str], number_comp: int) -> pd.DataFrame:
        """
        Fit PCA to the data and calculate the principal components. The transformed values are added to the dataframe
        in separate columns name 'pca_X', where X is the component number. The function can only be applied if the data
        does not have missing values (i.E. NaN), so rows with missing values have to be deleted before calling.

        :param data_table: Dataframe containing the data to apply the PCA on.
        :param cols: Columns to use for transforming the data to principal components.
        :param number_comp: Number of components to add to calculate. Can only be smaller or equal to the number of
        columns.
        :return: Dataframe with the original data and the transformed components.
        """

        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)

        # Perform the PCA
        pca = PCA(n_components=number_comp)
        pca.fit(dt_norm[cols])

        # Transform old values and add them to the original dataframe
        new_values = pca.transform(dt_norm[cols])
        for comp in range(0, number_comp):
            data_table['pca_' + str(comp + 1)] = new_values[:, comp]
        return data_table
