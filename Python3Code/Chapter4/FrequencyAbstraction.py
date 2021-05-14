##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import numpy as np
import pandas as pd
from typing import Union, Iterable, Tuple, List
from tqdm import tqdm


class FourierTransformation:
    """
    This class performs a Fourier transformation on the data to find frequencies that occur often and filter noise.
    """

    @staticmethod
    def find_fft_transformation(data: Union[np.ndarray, Iterable]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the amplitudes of the different frequencies using a fast fourier transformation.

        :param data: Iterable or Numpy array to perform the fourier transform on.
        :return: Tuple with real and imaginary part of fourier transformation.
        """

        # Create the transformation including the amplitudes of both the real and imaginary part
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    def abstract_frequency(self, data_table: pd.DataFrame, cols: List[str], window_size: int, sampling_rate: float) \
            -> pd.DataFrame:
        """
        Calculate frequencies over certain floating time windows for the specified columns.

        :param data_table: DataFrame with the data to calculate the frequencies of.
        :param cols: Columns to use for frequency abstraction.
        :param window_size: Size of the floating window.
        :param sampling_rate: Sampling rate of the data in Hz.
        :return: Original DataFrame with added columns for each frequency.
        """

        # Create new columns for the frequency data
        freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)
        for col in cols:
            data_table[f'{col}_max_freq'] = np.nan
            data_table[f'{col}_freq_weighted'] = np.nan
            data_table[f'{col}_pse'] = np.nan
            for freq in freqs:
                data_table[f'{col}_freq_{freq}_Hz_ws_{window_size}'] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history) and compute the values
        for i in tqdm(range(window_size, len(data_table.index))):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col][i - window_size:min(i + 1, len(data_table.index))])
                # Only use the real part in this implementation
                for j in range(0, len(freqs)):
                    data_table.iloc[i, data_table.columns.get_loc(f'{col}_freq_{freqs[j]}_Hz_ws_{window_size}')] = \
                        real_ampl[j]
                # Select the dominant frequency considering the positive frequencies for now
                data_table.iloc[i, data_table.columns.get_loc(f'{col}_max_freq')] = freqs[
                    np.argmax(real_ampl[0:len(real_ampl)])]
                data_table.iloc[i, data_table.columns.get_loc(f'{col}_freq_weighted')] = float(
                    np.sum(freqs * real_ampl)) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))

                # Make sure there are no zeros
                if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
                    data_table.iloc[i, data_table.columns.get_loc(f'{col}_pse')] = -np.sum(np.log(PSD_pdf) * PSD_pdf)
                else:
                    data_table.iloc[i, data_table.columns.get_loc(f'{col}_pse')] = 0

        return data_table
