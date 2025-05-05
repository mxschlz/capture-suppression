#import matplotlib
#matplotlib.use('TkAgg')
import mne
import autoreject
import os
import pandas as pd
from SPACEPRIME.encoding import *
from SPACEPRIME.rename_events import add_to_events
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
import glob


n_jobs = -1
mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="Fz",
    ica_reject_threshold=0.9,
    highpass=1,
    lowpass=40,
    epoch_tmin=-1.0,
    epoch_tmax=1.0
)
settings_path = f"{get_data_path()}settings/"
# get subject id and settings path
subject_ids = subject_ids[-1:]
for subject_id in subject_ids:
    if subject_id in []:  # already processed
        continue
    raw = mne.io.read_raw_fif(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_raw.fif",
                              preload=True)
    events, event_id = mne.events_from_annotations(raw)
    if subject_id == 101:
        epochs = mne.Epochs(raw, events=events, event_id=encoding_sub_101, preload=True, tmin=params["epoch_tmin"]+0.3, tmax=params["epoch_tmax"]+0.3,
                            baseline=None)
    elif subject_id in [102]:
        epochs = mne.Epochs(raw, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"]+0.08, tmax=params["epoch_tmax"]+0.08,
                            baseline=None)
    elif subject_id in [166]:
        epochs = mne.Epochs(raw, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
    elif subject_id in [103, 104, 105, 112, 116, 118, 120]:
        epochs = mne.Epochs(raw, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
    elif subject_id in [106, 107, 108, 110, 114, 124, 126, 128, 130, 132, 136, 138, 140]:
        epochs = mne.Epochs(raw, events=events, event_id=encoding_sub_106, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
        epochs = add_to_events(epochs, new_encoding=encoding, change_by=1)
    elif subject_id in [122, 134]:
        epochs = mne.Epochs(raw, events=events, event_id=encoding_sub_122, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
        epochs = add_to_events(epochs, new_encoding=encoding, change_by=-1)    # append behavior to metadata attribute in epochs for later analyses
    beh = pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/beh/sub-{subject_id}_clean*.csv")[0])
    # append metadata to epochs
    epochs.metadata = beh
    # run AutoReject
    ar = autoreject.AutoReject(n_jobs=n_jobs, random_state=42)
    epochs_ar, log = ar.fit_transform(epochs, return_log=True)
    # save epochs
    try:
        os.makedirs(f"{get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")
    epochs_ar.save(f"{get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
                overwrite=True)
    # save the drop log
    log.save(f"{get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo_log.npz",
             overwrite=True)
    del raw, epochs, epochs_ar, log, beh, ar
