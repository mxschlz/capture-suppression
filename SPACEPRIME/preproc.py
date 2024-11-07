import mne
import mne_icalabel
import autoreject
from SPACEPRIME.encoding import encoding, encoding_sub_101


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/raw/sub-{subject_id}/eeg/"
settings_path = "/home/max/Insync/schulz.max5@gmail.com/Google Drive/PhD/data/SPACEPRIME/settings/"
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
bad_chs = ["TP9"]
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
reconst_raw_filt.save(f"/home/max/Insync/schulz.max5@gmail.com/Google Drive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
                      overwrite=True)
# cut epochs
if subject_id == 101:
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding_sub_101, preload=True, tmin=-0.2, tmax=1.5,
                        baseline=None)
else:
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=-0.2, tmax=1.5,
                        baseline=None)
ar = autoreject.AutoReject(n_jobs=-1)
epochs_ar, log = ar.fit_transform(epochs, return_log=True)
# save epochs
epochs_ar.save(f"/home/max/Insync/schulz.max5@gmail.com/Google Drive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
            overwrite=True)
# save the drop log
log.save(f"/home/max/Insync/schulz.max5@gmail.com/Google Drive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo_log.npz",
         overwrite=True)