import mne
import mne_icalabel
from matplotlib import pyplot as plt
from meegkit.detrend import detrend
from meegkit.dss import dss_line
from pyprep.ransac import find_bad_by_ransac
import numpy as np
plt.ion()


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
raw.set_montage(montage)
# get events from annotations
events, event_id = mne.events_from_annotations(raw)
# Downsample because the computer crashes if sampled with 1000 Hz :-(
raw, events = raw.resample(sfreq=250, events=events)
# plot some channels
plt.plot(raw.times, raw.get_data()[[4, 5]].T*1e6, linewidth=0.4)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [µV]')
plt.title('Before detrending')
# detrend
X = raw.get_data().T # transpose so the data is organized time-by-channels
X, _, _ = detrend(X, order=1)
X, _, _ = detrend(X, order=6)
raw._data = X.T  # overwrite raw data
plt.plot(raw.times, X[:, [4, 5]], linewidth=0.4)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [muV]')
plt.title('After detrending')
# compute raw psd
psd = raw.compute_psd()
psd.plot()
# get the noise only
X, noise = dss_line(X, fline=50, sfreq=raw.info['sfreq'], nremove=1)
raw._data = X.T
# plot the noise only
noise = mne.io.RawArray(noise.T, raw.info)
noise.compute_psd().plot()
# interpolate bad channels
bads, _ = find_bad_by_ransac(
     data = raw.get_data(),
     sample_rate = raw.info['sfreq'],
     complete_chn_labs = np.asarray(raw.info['ch_names']),
     chn_pos = np.stack([ch['loc'][0:3] for ch in raw.info['chs']]),
     exclude = [],
     corr_thresh = 0.9
     )
raw_clean = raw.copy()
raw_clean.info['bads'] = bads
raw_clean.interpolate_bads()
raw_clean.set_eeg_reference('average', projection=True)  #compute the reference
raw.add_proj(raw_clean.info['projs'][0])
del raw_clean  # delete the copy
raw.apply_proj()  # apply the reference
# rename the keys in the event_id
renamed_event_id = {}
for key, value in event_id.items():
	new_key = key.replace('Stimulus/S', '').strip()  # Remove "Stimulus/S" and strip any spaces
	renamed_event_id[new_key] = value
event_id = renamed_event_id
# filter for values above 10
event_dict = {key: value for key, value in event_id.items() if value >= 9}
# epoch the data
epochs = mne.Epochs(raw, events, event_dict, tmin=-0.1, tmax=1.5, baseline=(None, 0))
# plot the data
epochs.average().plot()
