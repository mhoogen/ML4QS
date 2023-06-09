import pandas as pd
import numpy as np
import sys

def fix_weather_data(dir : str, rows_skipped=30):
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

    return weather


def merge_steps_weather(dir_weather: str, dir_steps: str, steps_column_datetime = 'startDate'):
    steps = pd.read_csv(dir_steps)
    weather = fix_weather_data(dir_weather)
    # decide on column to merge by, change steps_column_datetime to choose another datetime column
    weather.rename(columns={'datetime': steps_column_datetime}, inplace=True)

    steps[steps_column_datetime] = pd.to_datetime(pd.to_datetime(steps[steps_column_datetime]).dt.strftime('%Y-%m-%d %H:%M:%S'))
    steps = steps[steps['startDate'].dt.strftime('%Y') == '2023']
    steps.sort_values(steps_column_datetime, inplace=True)
    weather.sort_values(steps_column_datetime, inplace=True)

    merge = pd.merge_asof(steps, weather, on=steps_column_datetime, tolerance=pd.Timedelta("60m"))

    return merge

def create_weather_steps(dir_weather: str, dir_steps: str):
    #Create usefull dataframe
    weather_steps = merge_steps_weather(dir_weather, dir_steps)
    New_steps = weather_steps[['creationDate', 'hour', 'value']]
    New_weather = weather_steps.iloc[:, 13:35]
    New_weather_steps = pd.concat([New_steps, New_weather], axis=1)

    return(New_weather_steps)

if __name__ == '__main__':
    merge = create_weather_steps('./data_used/weather.txt', './data_used/StepCount.csv')
    merge.to_csv('weather_steps.csv')


