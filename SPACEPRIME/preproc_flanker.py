#import matplotlib
#matplotlib.use('TkAgg')
import mne
import mne_icalabel
import autoreject
import os
import pandas as pd
from SPACEPRIME.encoding import *
from SPACEPRIME import get_data_path
import glob
from SPACEPRIME.bad_chs import bad_chs


mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="Fz",
    ica_reject_threshold=0.9,
    highpass=1,
    lowpass=40,
    epoch_tmin=-0.2,
    epoch_tmax=0.8
)
settings_path = f"{get_data_path()}settings/"
# get subject id and settings path
subject_ids = [118, 120]
for subject_id in subject_ids:
    data_path = f"{get_data_path()}sourcedata/raw/sub-{subject_id}/eeg/"
    # read raw fif
    raw_orig = mne.io.read_raw_fif(data_path + f"sub-{subject_id}_task-flanker_raw.fif", preload=True)
    # Downsample because the computer crashes if sampled with 1000 Hz :-(
    raw = raw_orig.resample(params['resampling_freq'])
    # get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    # add reference channel
    raw.add_reference_channels([params["add_ref_channel"]])
    # Add a montage to the data
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    raw.set_montage(montage)
    # interpolate bad channels
    bads = bad_chs[subject_id]
    if bads:
        raw.info["bads"] = bads
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
    reconst_raw_filt.save(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker_raw.fif",
                          overwrite=True)
    # save the ica fit
    ica.save(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker_ica.fif",
             overwrite=True)
    # save the indices that were excluded
    with open(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker_ica_labels.txt", "w") as file:
        for item in exclude_idx:
            file.write(f"{item}\n")
    if subject_id in [103, 104, 106]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=FLANKER_MAP, preload=True,
                            tmin=params["epoch_tmin"]+0.05, tmax=params["epoch_tmax"]+0.05,
                            baseline=None)
    else:
        # cut epochs
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=FLANKER_MAP, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
    # append behavior to metadata attribute in epochs for later analyses
    beh = pd.read_csv(glob.glob(f"{get_data_path()}sourcedata/raw/sub-{subject_id}/beh/flanker_data_{subject_id}*.csv")[0])
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
    epochs_ar.save(f"{get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker-epo.fif",
                overwrite=True)
    # save the drop log
    log.save(f"{get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker-epo_log.npz",
             overwrite=True)
    del raw_orig, raw, raw_filt, reconst_raw_filt, reconst_raw, epochs, epochs_ar, log, beh, ar, ica
