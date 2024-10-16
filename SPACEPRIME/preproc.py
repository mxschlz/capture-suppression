import mne


mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
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
raw.set_eeg_reference(ref_channels="average", projection=True)
raw.apply_proj()
# Downsample because the computer crashes if sampled with 1000 Hz :-(
raw = raw.resample(250)
# Filter the data
raw_filt = raw.filter(0.1, 40)
# apply ICA
ica = mne.preprocessing.ICA()
ica.fit(raw_filt)
ica.plot_components(picks=range(20))
ica.plot_sources(raw_filt, picks=range(20))
exclude = [0]
# ica.plot_properties(raw_filt, picks=exclude)
ica.apply(raw_filt, exclude=exclude)
# get events from annotations
events, event_id = mne.events_from_annotations(raw)
# rename the keys in the event_id
renamed_event_id = {}
for key, value in event_id.items():
  new_key = key.replace('Stimulus/S', '').strip()  # Remove "Stimulus/S" and strip any spaces
  renamed_event_id[new_key] = value
event_id = renamed_event_id
# filter for values above 10
event_dict = {key: value for key, value in event_id.items() if value >= 9}

# cut epochs
epochs = mne.Epochs(raw_filt, events=events, event_id=event_dict, preload=True, tmin=-0.2, tmax=1.5, baseline=(None, 0))
evokeds = epochs.average(by_event_type=True)
np_trials = [x for x in evokeds if int(x.comment) > 200 ]
pp_trials = [x for x in evokeds if 200 > int(x.comment) > 100 ]
c_trials = [x for x in evokeds if int(x.comment) < 100 ]

