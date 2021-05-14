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
import scipy.stats as stats
from typing import List, Tuple
from tqdm import tqdm


class NumericalAbstraction:
    """
    Class to abstract a history of numerical values that can used as an attribute.
    """

    @staticmethod
    def aggregate_value(data: pd.Series, aggregation_function: str) -> float:
        """
        Aggregate a list of values in form of Pandas Series using the specified aggregation function.

        :param data: Series object to calculate the aggregated value on.
        :param aggregation_function: Aggregation function to use. Supported values are 'mean', 'max', 'min', 'median',
            'std', 'slope'.
        :return: Aggregated value.
        :raises ValueError In case of unknown aggregation function.
        """

        # Compute the value depending on aggregation function and return the result
        if aggregation_function == 'mean':
            return data.mean(skipna=True)
        elif aggregation_function == 'max':
            return data.max(skipna=True)
        elif aggregation_function == 'min':
            return data.min(skipna=True)
        elif aggregation_function == 'median':
            return data.median(skipna=True)
        elif aggregation_function == 'std':
            return data.std(skipna=True)
        elif aggregation_function == 'slope':
            # Create time points, assuming discrete time steps with fixed delta t
            times = np.array(range(0, len(data.index)))
            data = data.as_matrix().astype(np.float32)
            # Check for NaN's
            mask = ~np.isnan(data)
            # If data contains no data but NaN's return NaN
            if len(data[mask]) == 0:
                return np.nan
            # Otherwise return the slope
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times[mask], data[mask])
                return slope
        else:
            raise ValueError(f'Unknown aggregation function {aggregation_function}.')

    def abstract_numerical(self, data_table: pd.DataFrame, cols: List[str], window_size: int,
                           aggregation_function: str) -> pd.DataFrame:
        """
        Abstract numerical columns specified given a floating window size (i.e. the number of time points from the past
        considered) and an aggregation function.

        :param data_table: DataFrame with the data to abstract numerical features from.
        :param cols: Column names to use for feature abstraction.
        :param window_size: Winddow size to use for abstraction in seconds.
        :param aggregation_function: Aggregation function to use. Supported values are 'mean', 'max', 'min', 'median',
            'std', 'slope'.
        :return: DataFrame with original data and added colums with abstracted features.
        """

        # Create new columns for the temporal data
        for col in cols:
            data_table[col + '_temp_' + aggregation_function + '_ws_' + str(window_size)] = np.nan
        # Process the dataset starting at the window size (since there is not enough history before) to the last
        # value and compute the aggregated values for each window.
        for i in tqdm(range(window_size, len(data_table.index))):
            for col in cols:
                data_table.iloc[i, data_table.columns.get_loc(
                    f'{col}_temp_{aggregation_function}_ws_{window_size}')] = self.aggregate_value(
                    data_table[col].iloc[i - window_size:min(i + 1, len(data_table.index))], aggregation_function)

        return data_table


class CategoricalAbstraction:
    """
    Class to perform categorical abstraction obtaining patterns of categorical attributes that occur frequently over
    time.
    """

    def __init__(self):
        self.pattern_prefix = 'temp_pattern_'
        self.before = '(b)'
        self.co_occurs = '(c)'
        self.cache = {}

    def determine_pattern_times(self, data_table: pd.DataFrame, pattern: List[str], window_size: int) -> List[int]:
        """
        Determine the time points a pattern occurs in the dataset given a window size.

        :param data_table: DataFrame with the data to detect the pattern in.
        :param pattern: Pattern to detect.
        :param window_size: Window size to use for pattern detection.
        :return: List of indices where the pattern occurs.
        """
        times = []

        # If the pattern length is one
        if len(pattern) == 1:
            # If it is in the cache, get the times from the cache
            if self.to_string(pattern) in self.cache:
                times = self.cache[self.to_string(pattern)]
            # Otherwise identify the time points at which the value is observed
            else:
                timestamp_rows = data_table[data_table[pattern[0]] > 0].index.values.tolist()
                times = [data_table.index.get_loc(pd.to_datetime(i)) for i in timestamp_rows]
                self.cache[self.to_string(pattern)] = times

        # If we have a complex pattern (<n> (b) <m> or <n> (c) <m>)
        elif len(pattern) == 3:
            # Compute the time points of <n> and <m>
            time_points_first_part = self.determine_pattern_times(data_table, pattern[0], window_size)
            time_points_second_part = self.determine_pattern_times(data_table, pattern[2], window_size)

            # If it co-occurs we take the intersection.
            if pattern[1] == self.co_occurs:
                # No use for co-occurences of the same patterns...
                if pattern[0] == pattern[2]:
                    times = []
                else:
                    times = list(set(time_points_first_part) & set(time_points_second_part))
            # Otherwise we take all time points from <m> at which we observed <n> within the given
            # window size.
            elif pattern[1] == self.before:
                for t in time_points_second_part:
                    if len([i for i in time_points_first_part if ((i >= t - window_size) & (i < t))]):
                        times.append(t)
        return times

    def to_string(self, pattern: List[str]) -> str:
        """
        Create a string representation of a pattern.

        :param pattern: Pattern to create string representation for.
        :return: String of the pattern.
        """

        # In case of just have one component, return the string
        if len(pattern) == 1:
            return str(pattern[0])
        # Otherwise, return the merger of the strings of all components
        else:
            name = ''
            for p in pattern:
                name = name + self.to_string(p)
            return name

    def select_k_patterns(self, data_table: pd.DataFrame, patterns: List[List[str]], min_support: float,
                          window_size: int) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Select the patterns from 'patterns' that meet the minimum support in the dataset given the window size.

        :param data_table: DataFrame the pattern appear in.
        :param patterns: List of patterns to check minimum support.
        :param min_support: Minimum support for pattern to be selected.
        :param window_size: Windows size to detect patterns in.
        :return: DataFrame with added columns with pattern occurance and list of the selected patterns.
        """

        selected_patterns = []
        for pattern in patterns:
            # Determine the times at which the pattern occurs
            times = self.determine_pattern_times(data_table, pattern, window_size)
            # Compute the support
            support = float(len(times)) / len(data_table.index)
            # If we meet the minimum support, append the selected patterns and set the
            # value to 1 at which it occurs.
            if support >= min_support:
                selected_patterns.append(pattern)
                print(self.to_string(pattern))
                # Set the occurrence of the pattern in the row to 0
                data_table[self.pattern_prefix + self.to_string(pattern)] = 0
                # data_table[self.pattern_prefix + self.to_string(pattern)][times] = 1
                data_table.iloc[times, data_table.columns.get_loc(self.pattern_prefix + self.to_string(pattern))] = 1
        return data_table, selected_patterns

    def extend_k_patterns(self, k_patterns: List[List[str]], one_patterns: List[List[str]]) -> List[List[str]]:
        """
        Extend a set of k-patterns with the 1-patterns that have sufficient support.

        :param k_patterns: List of patterns with k elements.
        :param one_patterns: List of patterns with one element.
        :return: List of extended patterns.
        """

        new_patterns = []
        for k_p in k_patterns:
            for one_p in one_patterns:
                # Add a before relationship
                new_patterns.append([k_p, self.before, one_p])
                # Add a co-occurs relationship
                new_patterns.append([k_p, self.co_occurs, one_p])
        return new_patterns

    def abstract_categorical(self, data_table: pd.DataFrame, cols: List[str], match: List[str], min_support: float,
                             window_size: int, max_pattern_size: int) -> pd.DataFrame:
        """
        Abstract categorical data assuming a list of binary columns representing the different categories. Set
        whether the column names should match exactly 'exact' or should include the specified name 'like'. Express a
        minimum support, a window size between succeeding patterns and a maximum size for the number of patterns.

        :param data_table: DataFrame with the categorical columns to abstract.
        :param cols: Column names containing the categorical values.
        :param match: Set whether column names should match 'exact' or 'like'.
        :param min_support: Minimum support for patterns.
        :param window_size: Window size to use for abstraction.
        :param max_pattern_size: Maximum length of patterns to search for.
        :return:
        """

        # Find all the relevant columns of binary attributes
        col_names = list(data_table.columns)
        selected_patterns = []

        relevant_dataset_cols = []
        for i in range(0, len(cols)):
            if match[i] == 'exact':
                relevant_dataset_cols.append(cols[i])
            else:
                relevant_dataset_cols.extend([name for name in col_names if cols[i] in name])

        # Generate the one patterns first
        potential_1_patterns = [[pattern] for pattern in relevant_dataset_cols]

        new_data_table, one_patterns = self.select_k_patterns(data_table, potential_1_patterns, min_support,
                                                              window_size)
        selected_patterns.extend(one_patterns)
        print(f'Number of patterns of size 1 is {len(one_patterns)}')

        k = 1
        k_patterns = one_patterns

        # And generate all following patterns
        while (k < max_pattern_size) & (len(k_patterns) > 0):
            k += 1
            potential_k_patterns = self.extend_k_patterns(k_patterns, one_patterns)
            new_data_table, selected_new_k_patterns = self.select_k_patterns(new_data_table, potential_k_patterns,
                                                                             min_support, window_size)
            selected_patterns.extend(selected_new_k_patterns)
            print(f'Number of patterns of size {k} is {len(selected_new_k_patterns)}')

        return new_data_table
