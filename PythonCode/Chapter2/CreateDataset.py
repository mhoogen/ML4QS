##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md


class CreateDataset():

    base_dir = ''
    granulairity = 0
    data_table = None

    def __init__(self, base_dir, granularity):
        self.base_dir = base_dir
        self.granularity = granularity

    # Create an initial data table with entries from start till end time, with steps
    # of size granularity. Granularity is specified in milliseconds
    def create_timestamps(self, start_time, end_time):
        return pd.date_range(start_time, end_time, freq=str(self.granularity)+'ms')

    def create_dataset(self, start_time, end_time, cols, prefix):
        c = copy.deepcopy(cols)
        if not prefix == '':
            for i in range(0, len(c)):
                c[i] = str(prefix) + str(c[i])
        timestamps = self.create_timestamps(start_time, end_time)
        self.data_table = pd.DataFrame(index=timestamps, columns=c)

    # Add numerical data, we assume timestamps in the form of nanoseconds from the epoch
    def add_numerical_dataset(self, file, timestamp_col, value_cols, aggregation='avg', prefix=''):
        dataset = pd.read_csv(self.base_dir + file, skipinitialspace=True)

        # Convert timestamps to dates
        dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        # Create a table based on the times found in the dataset
        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            for col in value_cols:
                self.data_table[str(prefix) + str(col)] = np.nan

        # Over all rows in the new table
        for i in range(0, len(self.data_table.index)):
            # Select the relevant measurements.
            relevant_rows = dataset[
                (dataset[timestamp_col] >= self.data_table.index[i]) &
                (dataset[timestamp_col] < (self.data_table.index[i] +
                                           timedelta(milliseconds=self.granularity)))
            ]
            for col in value_cols:
                # Take the average value
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[self.data_table.index[i], str(prefix)+str(col)] = np.average(relevant_rows[col])
                    else:
                        raise ValueError("Unknown aggregation '" + aggregation + "'")
                else:
                    self.data_table.loc[self.data_table.index[i], str(prefix)+str(col)] = np.nan

    # Remove undesired value from the names.
    def clean_name(self, name):
        return re.sub('[^0-9a-zA-Z]+', '', name)

    # Add data in which we have rows that indicate the occurrence of a certain event with a given start and end time.
    # 'aggregation' can be 'sum' or 'binary'.
    def add_event_dataset(self, file, start_timestamp_col, end_timestamp_col, value_col, aggregation='sum'):
        dataset = pd.read_csv(self.base_dir + file)

        # Convert timestamps to datetime.
        dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col])
        dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col])

        # Clean the event values in the dataset
        dataset[value_col] = dataset[value_col].apply(self.clean_name)
        event_values = dataset[value_col].unique()

        # Add columns for all possible values (or create a new dataset if empty), set the default to 0 occurrences
        if self.data_table is None:
            self.create_dataset(min(dataset[start_timestamp_col]), max(dataset[end_timestamp_col]), event_values, value_col)
        for col in event_values:
            self.data_table[(str(value_col) + str(col))] = 0

        # Now we need to start counting by passing along the rows....
        for i in range(0, len(dataset.index)):
            # identify the time points of the row in our dataset and the value
            start = dataset[start_timestamp_col][i]
            end = dataset[end_timestamp_col][i]
            value = dataset[value_col][i]
            border = (start - timedelta(milliseconds=self.granularity))

            # get the right rows from our data table
            relevant_rows = self.data_table[(start <= (self.data_table.index +timedelta(milliseconds=self.granularity))) & (end > self.data_table.index)]

            # and add 1 to the rows if we take the sum
            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] += 1
            # or set to 1 if we just want to know it happened
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, str(value_col) + str(value)] = 1
            else:
                raise ValueError("Unknown aggregation '" + aggregation + "'")

    # This function returns the column names that have one of the strings expressed by 'ids' in the column name.
    def get_relevant_columns(self, ids):
        relevant_dataset_cols = []
        cols = list(self.data_table.columns)

        for id in ids:
            relevant_dataset_cols.extend([col for col in cols if id in col])

        return relevant_dataset_cols
