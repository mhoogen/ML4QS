import pandas as pd
import numpy as np

#import data
New_weather_steps = pd.read_csv("Transformed_weather_steps.csv")
New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)

#interpolate
New_weather_steps['steps'] = New_weather_steps['steps'].interpolate()

