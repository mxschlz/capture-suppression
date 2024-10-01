import mne
import pandas as pd
from mne.viz.eyetracking import plot_gaze
import numpy as np
from datetime import datetime, timezone


# get start time from .VideoConfig
start_time = 20240926175836189030

# Convert to datetime format
ts_datetime = datetime.strptime(str(start_time), '%Y%m%d%H%M%S%f')

# Load the CSV file into a DataFrame
df = pd.read_csv('/home/max/data/behavior/SPACEPRIME/tracking_log_data/sub-99_eye_tracking_log_27-09-2024_11-29-37.csv')

# Extract the data into a NumPy array
data = df.drop(columns=['Timestamp (ms)']).values.T

# Extract the timestamps and convert them to seconds
timestamps = df['Timestamp (ms)'].values

# Calculate the sampling frequency
sfreq = 25

# Create an MNE Info object
ch_names = df.columns.tolist()
ch_names.remove('Timestamp (ms)')  # Remove the timestamp column from channel names
ch_types = ['misc'] * len(ch_names)  # Set channel types to 'misc' for eye-tracking data
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create an MNE RawArray object
raw = mne.io.RawArray(data, info)

# Define the function `create_continuous_time_series`
def create_continuous_time_series(start_timestamp, sampling_frequency):
    # Convert the sampling frequency to the time between samples
    time_between_samples = 1 / sampling_frequency

    # Create a time series with the given sampling frequency
    time_series_data = start_timestamp + (np.arange(len(df)) * time_between_samples)

    return time_series_data

# Call the function `create_continuous_time_series`
time_series_data = create_continuous_time_series(start_time, 25)
# Set the measurement date
raw.set_meas_date(ts_datetime.astimezone(timezone.utc))

plot_gaze()
