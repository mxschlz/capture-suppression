import mne
from SPACEPRIME.encoding import EEG_TRIGGER_MAP


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "concatenated_raw.fif", preload=True)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
raw.set_montage(montage)
# get events
events, event_id = mne.events_from_annotations(raw)
event_dict = EEG_TRIGGER_MAP
# Extract some info
sample_rate = raw.info["sfreq"]
# Make a copy of the data
raw_copy = raw.copy()
# filter raw
raw_filt = raw.filter(1, 40, n_jobs=-1)
# cut epochs
epochs = mne.Epochs(raw_filt, events=events, preload=True, tmin=-0.2, tmax=1.0)
bad_chs = ["TP9"]
epochs.info["bads"] = bad_chs
# epochs.interpolate_bads()
# average reference
epochs.set_eeg_reference(ref_channels="average", projection=True)
epochs.apply_proj()
# apply ICA
ica = mne.preprocessing.ICA(n_components=5)
ica.fit(epochs)
