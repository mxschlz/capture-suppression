#import matplotlib
#matplotlib.use('TkAgg')
import mne
import mne_icalabel
import autoreject
import os
import pandas as pd
from SPACEPRIME.encoding import *
from SPACEPRIME.rename_events import add_to_events
from SPACEPRIME import get_data_path
import glob



mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="Fz",
    ica_reject_threshold=0.9,
    highpass=1,
    lowpass=40,
    epoch_tmin=-0.2,
    epoch_tmax=1.0
)
settings_path = f"{get_data_path()}settings/"
# get subject id and settings path
subject_ids = [116]
for subject_id in subject_ids:
    if subject_id in []:  # already processed
        continue
    data_path = f"{get_data_path()}sourcedata/raw/sub-{subject_id}/eeg/"
    # read raw fif
    raw_orig = mne.io.read_raw_fif(data_path + f"sub-{subject_id}_task-spaceprime_raw.fif", preload=True)
    # Downsample because the computer crashes if sampled with 1000 Hz :-(
    raw = raw_orig.copy().resample(params['resampling_freq'])
    # get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    # add reference channel
    raw.add_reference_channels([params["add_ref_channel"]])
    # Add a montage to the data
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    raw.set_montage(montage)
    # interpolate bad channels
    if subject_id == 101:
        bad_chs = ["TP9"]
    elif subject_id in [103, 104]:
        bad_chs = ["P2"]
    elif subject_id in [106]:
        bad_chs = ["P2", "P7"]
    elif subject_id in [116]:
        bad_chs = ["P3", "TP10"]
    else:
        bad_chs = None
    if bad_chs:
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
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > params["ica_reject_threshold"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    # ica.plot_properties(raw_filt, picks=exclude_idx)  # inspect the identified IC
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    # band pass filter
    reconst_raw_filt = reconst_raw.copy().filter(params["highpass"], params["lowpass"])
    try:
        os.makedirs(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/")
    except FileExistsError:
        print("EEG derivatives preprocessing directory already exists")
    # save the preprocessed raw file
    reconst_raw_filt.save(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_raw.fif",
                          overwrite=True)
    # save the ica fit
    ica.save(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_ica.fif",
             overwrite=True)
    # save the indices that were excluded
    with open(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_ica_labels.txt", "w") as file:
        for item in exclude_idx:
            file.write(f"{item}\n")
    # cut epochs
    # flat = dict(eeg=1e-6)
    # reject=dict(eeg=200e-6)
    if subject_id == 101:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_101, preload=True, tmin=params["epoch_tmin"]+0.3, tmax=params["epoch_tmax"]+0.3,
                            baseline=None)
    elif subject_id == 102:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"]+0.08, tmax=params["epoch_tmax"]+0.08,
                            baseline=None)
    elif subject_id in [103, 104, 105, 112, 116]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
    elif subject_id not in [106, 107]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_106, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
        epochs = add_to_events(epochs, new_encoding=encoding)
    # append behavior to metadata attribute in epochs for later analyses
    beh = pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/beh/sub-{subject_id}_clean*.csv")[0])
    # append metadata to epochs
    epochs.metadata = beh
    # run AutoReject
    ar = autoreject.AutoReject(n_jobs=-1)
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
    del raw_orig, raw, raw_filt, reconst_raw_filt, reconst_raw, epochs, epochs_ar, log, beh, ar, ica
