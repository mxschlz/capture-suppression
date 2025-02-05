import mne


mne.set_log_level("DEBUG")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
raw.resample(250)
events, event_id = mne.events_from_annotations(raw)
# rename the keys in the event_id
renamed_event_id = {}
for key, value in event_id.items():
  new_key = key.replace('Stimulus/S', '').strip()  # Remove "Stimulus/S" and strip any spaces
  renamed_event_id[new_key] = value
event_id = renamed_event_id
# filter for values above 10
event_dict = {key: value for key, value in event_id.items() if value >= 9}

mne.viz.plot_events(events, sfreq=raw.info['sfreq'], event_id=event_dict)
