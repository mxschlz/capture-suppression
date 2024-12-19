import mne
import mne_icalabel
import autoreject
import os
import pandas as pd
from SPACEPRIME.encoding import encoding, encoding_sub_101
import numpy as np


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 104
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/eeg/"
settings_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + f"sub-{subject_id}_task-spaceprime_raw.fif", preload=True)
# get events from annotations
events, event_id = mne.events_from_annotations(raw)
# Downsample because the computer crashes if sampled with 1000 Hz :-(
raw, events = raw.resample(250, events=events)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
raw.set_montage(montage)
# interpolate bad channels
if subject_id == 101:
    bad_chs = ["TP9"]
    raw.info["bads"] = bad_chs
    raw.interpolate_bads()
if subject_id == 104:
    bad_chs = ["P2"]
    raw.info["bads"] = bad_chs
    raw.interpolate_bads()
# average reference
raw.set_eeg_reference(ref_channels="average")
# Filter the data. These values are needed for the CNN to label the ICs effectively
raw_filt = raw.copy().filter(1, 100)
# apply ICA
ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))
ica.fit(raw_filt)
ic_labels = mne_icalabel.label_components(raw_filt, ica, method="iclabel")
exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > 0.9]
print(f"Excluding these ICA components: {exclude_idx}")
# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw, exclude=exclude_idx)
# band pass filter
reconst_raw_filt = reconst_raw.copy().filter(1, 40)
try:
    os.makedirs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/")
except FileExistsError:
    print("EEG preproc directory already exists")
reconst_raw_filt.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_raw.fif",
                      overwrite=True)
# cut epochs
# flat = dict(eeg=1e-6)
# reject=dict(eeg=200e-6)
if subject_id == 101:
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_101, preload=True, tmin=-0.5, tmax=2.0,
                        baseline=None)
else:
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=-0.5, tmax=2.0,
                        baseline=None)
# append behavior to metadata attribute in epochs for later analyses
beh = pd.read_csv(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/beh/sub-{subject_id}_clean.csv")
metadata = beh[(beh["event_type"]=="mouse_click") & (beh["phase"]==1)]
# get absolute trial_nr count
metadata['absolute_trial_nr'] = (metadata['block']) * 180 + metadata['trial_nr'] - 1
# Find duplicate trial numbers
duplicates = metadata[metadata.duplicated(subset='absolute_trial_nr', keep=False)]
print("Duplicate trial numbers:\n", duplicates)
# drop duplicates
metadata = metadata.drop_duplicates(subset='absolute_trial_nr', keep='first')  # or 'last'
# Create a new DataFrame to store aligned behavioral data
metadata_to_append = metadata.set_index('absolute_trial_nr')
# Create a complete range of trial numbers
all_trials = pd.RangeIndex(start=0, stop=len(epochs), step=1, name='absolute_trial_nr')
# Reindex the DataFrame with the complete range
metadata_to_append = metadata_to_append.reindex(all_trials)
# append metadata to epochs
epochs.metadata = metadata_to_append
# run AutoReject
ar = autoreject.AutoReject(n_jobs=-1)
epochs_ar, log = ar.fit_transform(epochs, return_log=True)
# save epochs
epochs_ar.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
            overwrite=True)
# save the drop log
log.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo_log.npz",
         overwrite=True)
#epochs.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
            #overwrite=True)
