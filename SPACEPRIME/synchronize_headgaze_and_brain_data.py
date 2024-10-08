import mne
import pandas as pd
from mne.viz.eyetracking import plot_gaze
import numpy as np
from matplotlib import pyplot as plt
from encoding import EEG_TRIGGER_MAP


# get headgaze data
# Load the CSV file into a DataFrame
df = pd.read_csv('/home/max/data/behavior/SPACEPRIME/tracking_log_data/sub-99_eye_tracking_log_07-10-2024_15-21-03.csv')
num_rows_to_add = 3727 - len(df)
zero_data = {col: [0] * num_rows_to_add for col in df.columns}
df_zero = pd.DataFrame(zero_data)
df = pd.concat([df, df_zero], ignore_index=True)
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
hg_timestamps = df["Timestamp (ms)"]
for i, row in enumerate(hg_timestamps):
	hg_timestamps[i] = int(row)
# --- Align timestamps ---
hg_timestamps = hg_timestamps + time_offset
# Extract the data into a NumPy array
data = df.drop(columns=['Timestamp (ms)', "Frame Nr"]).values.T
# Create an MNE Info object
ch_names = df.columns.tolist()
ch_names.remove('Timestamp (ms)')  # Remove the timestamp column from channel names
ch_names.remove("Frame Nr")
ch_types = ['eeg'] * len(ch_names)  # Set channel types to 'misc' for eye-tracking data
info = mne.create_info(ch_names=ch_names, sfreq=sfreq_hg, ch_types=ch_types)
# Create an MNE RawArray object
raw_hg = mne.io.RawArray(data, info)
#picks_hg = mne.pick_types(raw_hg.info, eeg=True, meg=False, eog=False, stim=False, exclude='bads')
# resample
raw_hg.resample(sfreq_eeg)
raw_eeg.filter(l_freq=None, h_freq=raw_hg.info["lowpass"])

# Set the measurement date
raw_hg.set_meas_date(raw_eeg.info["meas_date"])

missing_samples = len(raw_eeg) - len(raw_hg)
# Erstelle eine Null-Matrix mit den fehlenden Samples und den gleichen Anzahl an Kanälen
zero_data = np.zeros((raw_hg.info['nchan'], missing_samples))
# Erstelle ein neues RawArray-Objekt mit den Null-Daten
zero_raw = mne.io.RawArray(zero_data, raw_hg.info)
# Hänge die Null-Daten an das kürzere Raw-Objekt an
raw_short = mne.concatenate_raws([raw_hg, zero_raw])
# Jetzt kannst du die Daten zusammenführen
raw = raw_eeg.add_channels([raw_short])

events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events=events, tmin=-0.3, tmax=1.0, baseline=None, preload=True)
gaze_picks = ["Left Eye Center X", "Right Eye Center X", "Right Eye Center Y", "Left Eye Center Y"]
head_picks = ["Roll", "Pitch", "Yaw"]
epochs.apply_baseline().average().plot(picks=gaze_picks)
#plot_gaze(epochs, width=1920, height=1080)
data = epochs.apply_baseline().average().get_data(picks=head_picks)
r = np.linspace(-0.3, 1, data.shape[1])
theta = data[2]
theta_rad = np.deg2rad(theta)
# plot the stuff
ax = plt.subplot(polar=True)
ax.set_theta_zero_location('N', offset=0)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.plot(np.deg2rad(data[0]), r, 'r', label="Rotation", color="r")
ax.plot(np.deg2rad(data[1]), r, 'r', label="Elevation", color="g")
ax.plot(np.deg2rad(data[2]), r, 'r', label="Azimuth", color="b")
ax.legend()
