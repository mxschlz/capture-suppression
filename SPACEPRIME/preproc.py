import mne
import mne_icalabel
import autoreject
from SPACEPRIME.encoding import encoding

mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
# get events from annotations
events, event_id = mne.events_from_annotations(raw)
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
raw, events = raw.resample(250, events=events)
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
reconst_raw_filt = reconst_raw.copy().filter(0.1, 40)
# cut epochs
# get rejection criteria:
#reject = reject_based_on_snr(reconst_raw_filt, signal_interval=(0.35, 0.6), epoch_interval=(-0.2, 1.5),
                             #event_dict=event_dict)
epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=encoding, preload=True, tmin=-0.2, tmax=1.5,
                    baseline=None)
# reject epochs
ar = autoreject.AutoReject(n_jobs=-1)
epochs, log = ar.fit_transform(epochs, return_log=True)
# plot epochs
epochs.plot_image(picks="Cz")
# save epochs
epochs.save("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif", overwrite=True)
