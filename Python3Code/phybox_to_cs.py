from pathlib import Path
import pandas 
from datetime import datetime

BASE_DIR = './datasets/'
PROJECT_DIR = 'example/' # update this when running
RUN_DIR = BASE_DIR + PROJECT_DIR

OUT_DIR = BASE_DIR + 'translations/' # update this too

[path.mkdir(exist_ok=True, parents=True) for path in [Path(RUN_DIR), Path(OUT_DIR)]]

def get_start_ts():
    dataset = pandas.read_csv(RUN_DIR + 'meta/time.csv', skipinitialspace=True)

    return dataset["system time"][0]

def get_unix_timestamp_ns(input_ts):
    # ts = datetime.fromtimestamp(input_ts)
    nums = str(input_ts).split(".")
    ts = nums[0] + nums[1]
    # Creates timestamp with length 19
    # don't ask me why, it's the length in the crowdsignal dataset
    if len(ts) < 19:
        for _ in range(19-len(ts)):
            ts += "0"
    print(ts)
    return int(ts)

def convert_filename(filename):
    return filename.replace(' ', '_').lower()

def mutate_phybox_csv(filename, typecasts=None):
    start_time = get_start_ts()
    time_field = "Time (s)"
    dataset = pandas.read_csv(RUN_DIR + filename, skipinitialspace=True)

    size = len(dataset[time_field])
    index = 0
    for t in dataset[time_field]:
        start_time += t
        time = start_time
        dataset[time_field][index] = get_unix_timestamp_ns(time)
        index += 1

    dataset[time_field] = dataset[time_field].astype(int)
    if typecasts:
        dataset = dataset.astype(typecasts)

    dataset.to_csv(OUT_DIR + convert_filename(filename), index=False)



if __name__ == "__main__":
    mutate_phybox_csv('Accelerometer.csv')
    mutate_phybox_csv('Gyroscope.csv')
    mutate_phybox_csv('Linear Accelerometer.csv')
    mutate_phybox_csv('Location.csv')
    mutate_phybox_csv('Magnetometer.csv')