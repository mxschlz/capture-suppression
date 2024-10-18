import mne
import mne_icalabel
from mne_icalabel.gui import label_ica_components


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
# crop the data to for illustration pursposes
raw = raw.crop(tmax=60)
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
# Downsample because the computer crashes if sampled with 1000 Hz :-(
raw = raw.resample(500)
# Filter the data
raw_filt = raw.copy().filter(1.0, 100)
# apply ICA
ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))
ica.fit(raw_filt)
ic_labels = mne_icalabel.label_components(raw_filt, ica, method="iclabel")
# gui = label_ica_components(raw, ica)
print(ica.labels_)
exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")
# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw, exclude=exclude_idx)
# plot original and reconstructed data
raw.plot(order=exclude_idx, n_channels=len(exclude_idx), show_scrollbars=False)
reconst_raw.plot(order=exclude_idx, n_channels=len(exclude_idx), show_scrollbars=True)
