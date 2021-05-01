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
        self.BASE_DIR = base_dir
        self.GRANULARITY = granularity
        self.data_table = None

    def create_timestamps(self, start_time: datetime, end_time: datetime) -> pd.DatetimeIndex:
        """
        Create DatetimeIndex between start end end time with steps of instance granularity.

        :param start_time: Time to start the timestamps in datetime format.
        :param end_time: Time to end the timestamps in datetime format.
        :return: Pandas DatetimeIndex with the timestamps.
        """

        return pd.date_range(start_time, end_time, freq=str(self.GRANULARITY) + 'ms')

    def create_dataset(self, start_time: datetime, end_time: datetime, cols: list, prefix: str = '') -> None:
        """
        Create empty pandas dataframe with cols as columns and timestamps as index.

        :param start_time: Start time for the index in datetime format.
        :param end_time: End time for the index in datetime format.
        :param cols: Columns that will be created in dataframe.
        :param prefix: String that will be added to column names.
        :return: Pandas dataframe with timestamps as index and columns.
        """

        if prefix != '':
            cols = [f'{prefix}{col}' for col in cols]
        timestamps = self.create_timestamps(start_time, end_time)
        self.data_table = pd.DataFrame(index=timestamps, columns=cols)

    # Add numerical data, we assume timestamps in the form of nanoseconds from the epoch
    def add_numerical_dataset(self, file: str, timestamp_col: str, value_cols: list, aggregation: str = 'avg',
                              prefix: str = ''):
        """
        Add numerical dataset to the aggregated dataset with instance granularity.

        :param file: Name of the file in base directory with the numerical data.
        :param timestamp_col: Name of the column containing timestamps in nanoseconds.
        :param value_cols: Columns of the dataset that should be used for aggregation.
        :param aggregation: Aggregation type. By now only 'avg' is supported.
        :param prefix: String to add before column names in new aggregated dataset.
        """

        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.BASE_DIR / file, skipinitialspace=True)

        # Convert timestamps to dates
        dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])

        # Create a table based on the times found in the dataset
        if self.data_table is None:
            self.create_dataset(min(dataset[timestamp_col]), max(dataset[timestamp_col]), value_cols, prefix)
        else:
            for col in value_cols:
                self.data_table[f'{prefix}{col}'] = np.nan

        # Over all rows in the new table
        for timestamp in tqdm(self.data_table.index):
            # Select the relevant measurements.
            relevant_rows = dataset[
                (dataset[timestamp_col] >= timestamp) &
                (dataset[timestamp_col] < (timestamp + timedelta(milliseconds=self.GRANULARITY)))
            ]
            for col in value_cols:
                # Take the average value
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

    #
    # 'aggregation' can be 'sum' or 'binary'.
    def add_event_dataset(self, file: str, start_timestamp_col: str, end_timestamp_col: str, value_col: str,
                          aggregation: str = 'sum'):
        """
        Add data which have rows indicating the occurrence of a certain event with a given start and end time.

        :param file: Name of the file in base directory with the numerical data.
        :param start_timestamp_col: Name of the column containing the start timestamp for a event period.
        :param end_timestamp_col: Name of the column containing the end timestamp for a event period.
        :param value_col: Column name that cotains the event name / description.
        :param aggregation: Aggregation type. Supported types are 'sum' and 'binary'.
        """

        print(f'Reading data from {file}')
        dataset = pd.read_csv(self.BASE_DIR / file)

        # Convert timestamps to datetime.
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

        # Now we need to start counting by passing along the rows....
        for i in tqdm(range(0, len(dataset.index))):
            # Identify the time points of the row in our dataset and the value
            start = dataset[start_timestamp_col][i]
            end = dataset[end_timestamp_col][i]
            value = dataset[value_col][i]

            # Get the right rows from our data table
            relevant_rows = self.data_table[
                (start <= (self.data_table.index + timedelta(milliseconds=self.GRANULARITY))) &
                (end > self.data_table.index)]

            # and add 1 to the rows if we take the sum
            if aggregation == 'sum':
                self.data_table.loc[relevant_rows.index, f'{value_col}{value}'] += 1
            # or set to 1 if we just want to know it happened
            elif aggregation == 'binary':
                self.data_table.loc[relevant_rows.index, f'{value_col}{value}'] = 1
            else:
                raise ValueError(f'Unknown aggregation {aggregation}')

    # This function returns the column names that have one of the strings expressed by 'ids' in the column name.
    # TODO: Function is not used. May be deleted?!?
    def get_relevant_columns(self, ids: list) -> list:
        relevant_dataset_cols = []
        cols = list(self.data_table.columns)
        for i in ids:
            relevant_dataset_cols.extend([col for col in cols if i in col])
        return relevant_dataset_cols
