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
from tqdm import tqdm
import pathlib


class CreateDataset:
    def __init__(self, base_dir: pathlib.Path, granularity: int):
        self.base_dir = base_dir
        self.granularity = granularity
        self.data_table = None

    def create_timestamps(self, start_time: datetime, end_time: datetime) -> pd.DatetimeIndex:
        """
        Create DatetimeIndex between start end end time with steps of instance granularity.

        :param start_time: Time to start the timestamps in datetime format.
        :param end_time: Time to end the timestamps in datetime format.
        :return: Pandas DatetimeIndex with the timestamps.
        """

        return pd.date_range(start_time, end_time, freq=f'{self.granularity}ms')

    def create_dataset(self, start_time: datetime, end_time: datetime, cols: list, prefix: str = ''):
        """
        Create empty pandas dataframe with cols as columns and timestamps as index and save it to instance attribute.

        :param start_time: Start time for the index in datetime format.
        :param end_time: End time for the index in datetime format.
        :param cols: Columns that will be created in dataframe.
        :param prefix: String that will be added to column names.
        """

        if prefix != '':
            cols = [f'{prefix}{col}' for col in cols]
        timestamps = self.create_timestamps(start_time, end_time)
        self.data_table = pd.DataFrame(index=timestamps, columns=cols)

    def add_numerical_dataset(self, file: str, timestamp_col: str, value_cols: list, aggregation: str = 'avg',
                              prefix: str = ''):
        """
        Add numerical dataset to the instance dataset with defined granularity. Timestamps are assumed to be in the
        format of nanoseconds.

        :param file: Name of the file in base directory with the numerical data.
        :param timestamp_col: Name of the column containing timestamps in nanoseconds.
        :param value_cols: Columns of the dataset that should be used for aggregation.
        :param aggregation: Aggregation type. By now only 'avg' is supported.
        :param prefix: String to add before column names in aggregated dataset.
        """

        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file, skipinitialspace=True)

        # Convert timestamps to dates
        dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        # Create a table based on the time found in the dataset
        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            for col in value_cols:
                self.data_table[f'{prefix}{col}'] = np.nan

        # Iterate over all rows in the new table
        for timestamp in tqdm(self.data_table.index):
            # Select the relevant measurements
            relevant_rows = dataset[
                (dataset[timestamp_col] >= timestamp) &
                (dataset[timestamp_col] < (timestamp + timedelta(milliseconds=self.granularity)))
                ]
            for col in value_cols:
                # Take the average value if rows are present
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[timestamp, f'{prefix}{col}'] = np.average(relevant_rows[col])
                    else:
                        raise ValueError(f"Unknown aggregation {aggregation}")
                else:
                    self.data_table.loc[timestamp, f'{prefix}{col}'] = np.nan

    @staticmethod
    def clean_name(name: str) -> str:
        """
        Remove undesired values from string.

        :param name: String that has to be cleaned.
        :return: Cleaned string.
        """

        return re.sub('[^0-9a-zA-Z]+', '', name)

    def add_event_dataset(self, file: str, start_timestamp_col: str, end_timestamp_col: str, value_col: str,
                          aggregation: str = 'sum'):
        """
        Add data which indicates the occurrence of a certain event between a given start and end time.

        :param file: Name of the file in base directory with the numerical data.
        :param start_timestamp_col: Name of the column containing the start timestamp for a event period.
        :param end_timestamp_col: Name of the column containing the end timestamp for a event period.
        :param value_col: Column name that cotains the event name / description.
        :param aggregation: Aggregation type. Supported types are 'sum' and 'binary'.
        """

        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.base_dir / file)

        # Convert timestamps to datetime
        dataset[start_timestamp_col] = pd.to_datetime(dataset[start_timestamp_col])
        dataset[end_timestamp_col] = pd.to_datetime(dataset[end_timestamp_col])

        # Clean the event values in the dataset
        dataset[value_col] = dataset[value_col].apply(self.clean_name)
        event_values = dataset[value_col].unique()

        # Add columns for all possible values (or create a new dataset if empty), set the default to 0 occurrences
        if self.data_table is None:
            self.create_dataset(min(dataset[start_timestamp_col]), max(dataset[end_timestamp_col]), event_values,
                                value_col)
        for col in event_values:
            self.data_table[f'{value_col}{col}'] = 0

        # Start counting by passing along the rows
        for i in tqdm(range(0, len(dataset.index))):
            # Identify the time points of the row in the dataset and get the event value
            start = dataset[start_timestamp_col][i]
            end = dataset[end_timestamp_col][i]
            value = dataset[value_col][i]

            # Get the right rows from our data table
            relevant_rows = self.data_table[
                (start <= (self.data_table.index + timedelta(milliseconds=self.granularity))) &
                (end > self.data_table.index)]

            # and add 1 to the rows if we take the sum
            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, f'{value_col}{value}'] += 1
            # or set to 1 if we just want to know it happened
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, f'{value_col}{value}'] = 1
            else:
                raise ValueError(f'Unknown aggregation {aggregation}')
