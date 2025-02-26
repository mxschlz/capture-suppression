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
from SPACEPRIME.bad_chs import bad_chs
import glob


mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="TP9",
    ica_reject_threshold=0.9,
    highpass=0.1,
    lowpass=30,
    epoch_tmin=-2.0,
    epoch_tmax=2.0
)
settings_path = f"{get_data_path()}settings/"
data_path = f"G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\raw"
sub_ids = os.listdir(data_path)[2:]
for subject_id in sub_ids:
    raw_fp = os.path.join(data_path, subject_id)
    raw_filenames = sorted([x for x in os.listdir(raw_fp) if ".vhdr" in x])
    for raw_filename in raw_filenames:
        raw_brainvision = mne.io.read_raw_brainvision(raw_fp + "\\" + raw_filename, preload=True)
    # read raw fif
    raw_orig = raw_brainvision
    # Downsample because the computer crashes if sampled with 1000 Hz :-(
    raw = raw_orig.copy().resample(params['resampling_freq'])
    # get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    # add reference channel
    raw.add_reference_channels([params["add_ref_channel"]])
    # Add a montage to the data
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    raw.set_montage(montage)
    """# interpolate bad channels
    bads = bad_chs[subject_id]
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads()"""
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
        os.makedirs(f"{data_path}\\{subject_id}\\derivatives\\preprocessing")
    except FileExistsError:
        print("EEG derivatives preprocessing directory already exists")
    # save the preprocessed raw file
    reconst_raw_filt.save(f"{data_path}\\{subject_id}\\derivatives/preprocessing/{subject_id}_task-supratyp_raw.fif",
                          overwrite=True)
    # save the ica fit
    ica.save(f"{data_path}\\{subject_id}\\derivatives/preprocessing/{subject_id}_task-supratyp_ica.fif",
             overwrite=True)
    # save the indices that were excluded
    with open(f"{data_path}\\{subject_id}\\derivatives/preprocessing/{subject_id}_task-supratyp_ica_labels.txt", "w") as file:
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
    elif subject_id in [103, 104, 105, 112, 116, 118, 120]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
    elif subject_id not in [106, 107]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_106, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
        epochs = add_to_events(epochs, new_encoding=encoding, change_by=1)
    elif subject_id in [122]:
        epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_122, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                            baseline=None)
        epochs = add_to_events(epochs, new_encoding=encoding, change_by=-1)    # append behavior to metadata attribute in epochs for later analyses
    #beh = pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/{subject_id}_clean*.csv")[0])
    # append metadata to epochs
    #epochs.metadata = beh
    # run AutoReject
    ar = autoreject.AutoReject(n_jobs=-1)
    epochs_ar, log = ar.fit_transform(epochs, return_log=True)
    # save epochs
    try:
        os.makedirs(f"{data_path}\\{subject_id}\\derivatives\\epoching")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")
    epochs_ar.save(f"{data_path}\\{subject_id}\\derivatives\\epoching/{subject_id}_task-spaceprime-epo.fif",
                overwrite=True)
    # save the drop log
    log.save(f"{data_path}\\{subject_id}\\derivatives\\epoching/{subject_id}_task-spaceprime-epo_log.npz",
             overwrite=True)
    del raw_orig, raw, raw_filt, reconst_raw_filt, reconst_raw, epochs, epochs_ar, log, ar, ica
