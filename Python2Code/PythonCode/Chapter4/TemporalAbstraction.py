##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import numpy as np
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:


    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(self, data, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == 'mean':
            return data.mean(skipna = True)
        elif aggregation_function == 'max':
            return data.max(skipna = True)
        elif aggregation_function == 'min':
            return data.min(skipna = True)
        elif aggregation_function == 'median':
            return data.median(skipna = True)
        elif aggregation_function == 'std':
            return data.std(skipna = True)
        elif aggregation_function == 'slope':
            # For the slope we need a bit more work.
            # We create time points, assuming discrete time steps with fixed delta t:
            times = np.array(range(0, len(data.index)))
            data = data.as_matrix().astype(np.float32)

            # Check for NaN's
            mask = ~np.isnan(data)

            # If we have no data but NaN we return NaN.
            if (len(data[mask]) == 0):
                return np.nan
            # Otherwise we return the slope.
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times[mask], data[mask])
                return slope
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):

        # Create new columns for the temporal data.
        for col in cols:
            data_table[col + '_temp_' + aggregation_function + '_ws_' + str(window_size)] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                data_table.ix[i, col + '_temp_' + aggregation_function + '_ws_' +str(window_size)] = self.aggregate_value(data_table[col][i-window_size:min(i+1, len(data_table.index))], aggregation_function)

        return data_table

# Class to perform categorical abstraction. We obtain patterns of categorical attributes that occur frequently
# over time.
class CategoricalAbstraction:

    pattern_prefix = 'temp_pattern_'
    before = '(b)'
    co_occurs = '(c)'
    cache = {}

    # Determine the time points a pattern occurs in the dataset given a windows size.
    def determine_pattern_times(self, data_table, pattern, window_size):
        times = []

        # If we have a pattern of length one
        if len(pattern) == 1:
            # If it is in the cache, we get the times from the cache.
            if self.cache.has_key(self.to_string(pattern)):
                times = self.cache[self.to_string(pattern)]
            # Otherwise we identify the time points at which we observe the value.
            else:
                timestamp_rows = data_table[data_table[pattern[0]] > 0].index.values.tolist()
                times = [data_table.index.get_loc(i) for i in timestamp_rows]
                self.cache[self.to_string(pattern)] = times

        # If we have a complex pattern (<n> (b) <m> or <n> (c) <m>)
        elif len(pattern) == 3:
            # We computer the time points of <n> and <m>
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

    # Create a string representation of a pattern.
    def to_string(self, pattern):
        # If we just have one component, return the string.
        if len(pattern) == 1:
            return str(pattern[0])
        # Otherwise, return the merger of the strings of all
        # components.
        else:
            name = ''
            for p in pattern:
                name = name + self.to_string(p)
            return name

    # Selects the patterns from 'patterns' that meet the minimum support in the dataset
    # given the window size.
    def select_k_patterns(self, data_table, patterns, min_support, window_size):
        selected_patterns = []
        for pattern in patterns:
            # Determine the times at which the pattern occurs.
            times = self.determine_pattern_times(data_table, pattern, window_size)
            # Compute the support
            support = float(len(times))/len(data_table.index)
            # If we meet the minum support, append the selected patterns and set the
            # value to 1 at which it occurs.
            if support >= min_support:
                selected_patterns.append(pattern)
                print self.to_string(pattern)
                # Set the occurrence of the pattern in the row to 0.
                data_table[self.pattern_prefix + self.to_string(pattern)] = 0
                data_table.ix[times, self.pattern_prefix + self.to_string(pattern)] = 1
        return data_table, selected_patterns


    # extends a set of k-patterns with the 1-patterns that have sufficient support.
    def extend_k_patterns(self, k_patterns, one_patterns):
        new_patterns = []
        for k_p in k_patterns:
            for one_p in one_patterns:
                # Add a before relationship
                new_patterns.append([k_p, self.before, one_p])
                # Add a co-occurs relationship.
                new_patterns.append([k_p, self.co_occurs, one_p])
        return new_patterns


    # Function to abstract our categorical data. Note that we assume a list of binary columns representing
    # the different categories. We set whether the column names should match exactly 'exact' or should include the
    # specified name 'like'. We also express a minimum support,a windows size between succeeding patterns and a
    # maximum size for the number of patterns.
    def abstract_categorical(self, data_table, cols, match, min_support, window_size, max_pattern_size):

        # Find all the relevant columns of binary attributes.
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

        new_data_table, one_patterns = self.select_k_patterns(data_table, potential_1_patterns, min_support, window_size)
        selected_patterns.extend(one_patterns)
        print 'Number of patterns of size 1 is ' + str(len(one_patterns))

        k = 1
        k_patterns = one_patterns

        # And generate all following patterns.
        while (k < max_pattern_size) & (len(k_patterns) > 0):
            k = k + 1
            potential_k_patterns = self.extend_k_patterns(k_patterns, one_patterns)
            new_data_table, selected_new_k_patterns = self.select_k_patterns(new_data_table, potential_k_patterns, min_support, window_size)
            selected_patterns.extend(selected_new_k_patterns)
            print 'Number of patterns of size ' + str(k) + ' is ' + str(len(selected_new_k_patterns))

        return new_data_table



