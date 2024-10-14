import mne
from SPACEPRIME.encoding import EEG_TRIGGER_MAP


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "concatenated_raw.fif", preload=True)
# Downsample to 250 Hz
raw = raw.resample(250)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
raw.set_montage(montage)
# get events
events, event_id = mne.events_from_annotations(raw)
event_dict = EEG_TRIGGER_MAP
# Make a copy of the data
raw_copy = raw.copy()
# filter raw
raw_filt = raw.filter(1, 40, n_jobs=-1)
# cut epochs
epochs = mne.Epochs(raw_filt, events=events, preload=True, tmin=-0.2, tmax=1.0, event_repeated="merge", on_missing="warn")
bad_chs = ["TP9"]
epochs.info["bads"] = bad_chs
epochs.interpolate_bads()
# average reference
epochs.set_eeg_reference(ref_channels="average", projection=True)
epochs.apply_proj()
# apply ICA
ica = mne.preprocessing.ICA(n_components=95)
ica.fit(epochs)
ica.plot_components()
ica.plot_sources(epochs, start=0, stop=15)
exclude = [0, 1, 2]
ica.plot_properties(epochs, picks=exclude)
ica.apply(epochs, exclude=[0])
