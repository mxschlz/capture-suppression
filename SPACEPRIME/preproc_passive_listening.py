import mne
import mne_icalabel
import autoreject
import os
from SPACEPRIME.encoding import PASSIVE_LISTENING_MAP
import glob


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 103
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/eeg/"
settings_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + f"sub-{subject_id}_task-passive_raw.fif", preload=True)
# get events from annotations
events, event_id = mne.events_from_annotations(raw)
# Downsample because the computer crashes if sampled with 1000 Hz :-(
raw, events = raw.resample(250, events=events)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
raw.set_montage(montage)
# interpolate bad channels
if subject_id == 101:
    bad_chs = ["TP9"]
    raw.interpolate_bads()
if subject_id == 104:
    bad_chs = ["P2"]
if subject_id == 103:
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
exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")
# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw, exclude=exclude_idx)
# band pass filter
reconst_raw_filt = reconst_raw.copy().filter(1, 40)
try:
    os.makedirs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/")
except FileExistsError:
    print("EEG derivatives preprocessing directory already exists")
# save the preprocessed raw file
reconst_raw_filt.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-passive_raw.fif",
                      overwrite=True)
# save the ica fit
ica.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-passive_ica.fif",
         overwrite=True)
# save the indices that were excluded
with open(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-passive_ica_labels.txt", "w") as file:
    for item in exclude_idx:
        file.write(f"{item}\n")
# cut epochs
# flat = dict(eeg=1e-6)
# reject=dict(eeg=200e-6)
epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=PASSIVE_LISTENING_MAP, preload=True, tmin=-0.1, tmax=0.5,
                    baseline=None)
# run AutoReject
ar = autoreject.AutoReject(n_jobs=-1)
epochs_ar, log = ar.fit_transform(epochs, return_log=True)
# save epochs
try:
    os.makedirs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/")
except FileExistsError:
    print("EEG derivatives epoching directory already exists")
epochs_ar.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-passive-epo.fif",
            overwrite=True)
# save the drop log
log.save(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-passive-epo_log.npz",
         overwrite=True)
