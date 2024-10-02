import mne
import pandas as pd
from mne.viz.eyetracking import plot_gaze
import numpy as np


# get headgaze data
# Load the CSV file into a DataFrame
df = pd.read_csv('/home/max/data/behavior/SPACEPRIME/tracking_log_data/sub-99_eye_tracking_log_27-09-2024_11-29-37.csv')
# load raw eeg data
raw_eeg = mne.io.read_raw_brainvision("/home/max/data/eeg/raw/sub-99_block_1.vhdr", preload=True)
# get start time from .VideoConfig
start_time_hg = 20240926175836189030
start_time_eeg = 20240926175836348218
time_offset = start_time_eeg - start_time_hg
# Get sampling frequency of headgaze data
sfreq_hg = 25
sfreq_eeg = 1000
# --- Reconstruct eye-tracking timestamps ---
num_samples_hg = len(df)
num_samples_eeg = raw_eeg.get_data().shape[1]
hg_timestamps = np.arange(num_samples_hg) / sfreq_hg
hg_timestamps = hg_timestamps + start_time_hg  # Add the start time
# --- Align timestamps ---
hg_timestamps = hg_timestamps + time_offset
# --- Determine padding needs ---
pad_start_samples = int(abs(hg_timestamps[0] - start_time_eeg))  # in samples
end_time_eeg = start_time_eeg + (raw_eeg._data.shape[1]/sfreq_eeg)
pad_end_samples = int(abs(end_time_eeg - hg_timestamps[-1]))  # in samples
# Extract the data into a NumPy array
data = df.drop(columns=['Timestamp (ms)']).values.T
data_padded = np.pad(data, (pad_start_samples, pad_end_samples), 'constant')
# Create an MNE Info object
ch_names = df.columns.tolist()
ch_names.remove('Timestamp (ms)')  # Remove the timestamp column from channel names
ch_types = ['misc'] * len(ch_names)  # Set channel types to 'misc' for eye-tracking data
info = mne.create_info(ch_names=ch_names, sfreq=sfreq_hg, ch_types=ch_types)
# Create an MNE RawArray object
raw_hg = mne.io.RawArray(data, info)
# Set the measurement date
raw_hg.set_meas_date(raw_eeg.info["meas_date"])
# resample
raw_hg.resample(sfreq_eeg)
# concatenate the data
concat = raw_eeg.add_channels([raw_hg])
