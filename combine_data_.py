import pandas as pd
import numpy as np
from tqdm import tqdm

def fix_weather_data(dir: str, rows_skipped=30):
    print('fixing weather data')
    weather_raw = pd.read_csv(dir, dtype=str, skiprows=rows_skipped)

    # replace confusing names with better names derived from .txt file
    weather = weather_raw.rename(columns={'   HH': 'hour',
                                          '   DD': 'direction_wind',
                                          '   FH': 'windspeed_avg_hour',
                                          '   FF': 'windspeed_avg_10min',
                                          '   FX': 'max_wind_gust',
                                          '    T': 'temp_celsius',
                                          ' T10N': 'temp_min_6h',
                                          '   TD': 'temp_dewpoint',
                                          '   SQ': 'sunshine_duration',
                                          '    Q': 'glob_radiation',
                                          '   DR': 'precipitation_duration',
                                          '   RH': 'precipitation_amount_hourly',
                                          '    P': 'air_pressure',
                                          '   VV': 'horizontal_visibility',
                                          '    N': 'cloud_cover',
                                          '    U': 'relative_humidity',
                                          '   WW': 'weather_code',
                                          '   IX': 'indicator_present_weather_code',
                                          '    M': 'fog',
                                          '    R': 'rain',
                                          '    S': 'snow',
                                          '    O': 'thunder',
                                          '    Y': 'ice_formation'}, inplace=False)

    # remove whitespace from values
    for i in weather.columns:
        weather[i] = weather[i].str.strip()

    weather = weather.replace('', np.nan, regex=True)

    # change datatype
    for i in weather.columns:
        if i == 'precipitation_amount_hourly':
            weather[i] = weather[i].astype(float)
        else:
            weather[i] = weather[i].astype(float).astype("Int32")

    # fix datetimes
    weather['hour'] = pd.to_datetime(weather['hour'] - 1, format='%H', exact=False).dt.strftime(
        '%H:%M:%S')  # hour minus one because somehow 24:00:00 gets converted to 02:00:00, needs to be fixed
    weather['YYYYMMDD'] = pd.to_datetime(weather['YYYYMMDD'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    weather['datetime'] = weather[['YYYYMMDD', 'hour']].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
    weather['datetime'] = pd.to_datetime(pd.to_datetime(weather['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'))
    weather['start_weather'] = weather['datetime']
    weather['end_weather'] = pd.to_datetime(weather['start_weather']) + pd.Timedelta('1H')
    cols = list(weather)
    cols.insert(0, cols.pop(cols.index('end_weather')))
    cols.insert(0, cols.pop(cols.index('start_weather')))
    cols.pop(cols.index('hour'))
    cols.pop(cols.index('YYYYMMDD'))
    weather = weather.loc[:, cols]
    return weather


def merge_steps_weather(dir_weather: str, dir_steps: str, steps_column_datetime='startDate'):
    steps = pd.read_csv(dir_steps)
    weather = fix_weather_data(dir_weather)
    # decide on column to merge by, change steps_column_datetime to choose another datetime column
    weather.rename(columns={'datetime': steps_column_datetime}, inplace=True)

    steps[steps_column_datetime] = pd.to_datetime(
        pd.to_datetime(steps[steps_column_datetime]).dt.strftime('%Y-%m-%d %H:%M:%S'))
    steps = steps[steps['startDate'].dt.strftime('%Y') == '2023']
    steps.sort_values(steps_column_datetime, inplace=True)
    weather.sort_values(steps_column_datetime, inplace=True)

    merge = pd.merge_asof(steps, weather, on=steps_column_datetime, tolerance=pd.Timedelta("60m"))

    return merge


def construct_time_intervals(df, Tdelta='15min'):
    dat_int = df.resample(Tdelta, on='startDate', convention='end').agg({
        'value': 'sum',
        'windspeed_avg_hour': 'mean',
        'temp_celsius': 'mean',
        'sunshine_duration': 'mean',
        'precipitation_duration': 'mean',
        'fog': 'mean',
        'rain': 'mean',
        'snow': 'mean',
        'thunder': 'mean',
        'ice_formation': 'mean'
    }).reset_index()

    dat_int['datetime'] = dat_int['datetime'] + pd.Timedelta('15min')
    dat_int.rename({'startDate': 'datetime'}, inplace=True)
    return dat_int


def fix_steps_time_intervals(dir, start_date, end_date, delta_t):
    print('fixing steps data')
    steps = pd.read_csv(dir)
    steps = steps[pd.to_datetime(steps['startDate']).dt.strftime('%Y-%m') == pd.to_datetime(start_date).strftime('%Y-%m')].reset_index(drop=True)
    steps_start = pd.to_datetime(pd.to_datetime(steps['startDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))
    steps_end = pd.to_datetime(pd.to_datetime(steps['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))

    range_start = pd.to_datetime(pd.date_range(start=start_date,
                                               end=end_date,
                                               freq='S'))
    range_end = pd.to_datetime(pd.date_range(start=pd.to_datetime(start_date) + pd.Timedelta(seconds=1),
                                             end=pd.to_datetime(end_date) + pd.Timedelta(seconds=1),
                                             freq='S'))
    new_steps = pd.DataFrame({'start': range_start,
                              'end': range_end,
                              'steps': [0] * len(range_start)})

    for i in tqdm(range(len(steps_start))):
        mask = (new_steps['start'] >= steps_start[i]) & (new_steps['end'] <= steps_end[i])
        trues = len(mask[mask == True])
        if trues != 0:
            res = steps.loc[i, 'value'] / trues
            new_steps.loc[mask, 'steps'] = res
    new_steps = new_steps.resample(delta_t, on='end').steps.sum().reset_index().rename(columns = {'end': 'start'})
    new_steps['end'] = new_steps['start'] + pd.Timedelta(delta_t)

    new_steps = new_steps[['start','end','steps']]

    return new_steps

def merge_steps_weather(dir_weather: str, dir_steps: str, start_date, end_date, delta_t, steps_column_datetime='start'):
    steps = fix_steps_time_intervals(dir_steps, start_date, end_date, delta_t)
    weather = fix_weather_data(dir_weather)

    print('merging weather and steps data')

    # decide on column to merge by, change steps_column_datetime to choose another datetime column
    weather.rename(columns={'datetime': steps_column_datetime}, inplace=True)

    steps[steps_column_datetime] = pd.to_datetime(
        pd.to_datetime(steps[steps_column_datetime]).dt.strftime('%Y-%m-%d %H:%M:%S'))
    steps.sort_values(steps_column_datetime, inplace=True)
    weather.sort_values(steps_column_datetime, inplace=True)

    merge = pd.merge_asof(steps, weather, on=steps_column_datetime, tolerance=pd.Timedelta("60m"))

    return merge

res = merge_steps_weather(dir_weather='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather.txt',
                          dir_steps='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\StepCount.csv',
                          start_date='2023-01-01 00:00:00',
                          end_date='2023-06-06 00:00:00',
                          delta_t='10min')

res.to_csv("New_weather_steps.csv")


