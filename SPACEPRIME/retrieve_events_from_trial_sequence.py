import pandas as pd
import mne
from SPACEPRIME.encoding import EEG_TRIGGER_MAP


# priming encoding
PRIMING = {
    0: "C",
    1: "PP",
    -1: "NP"
}
# get trial types from results file
df = pd.read_excel("/home/max/data/SPACEPRIME/sub-101/beh/sub-101_task-spaceprime.xlsx", index_col=0)
df = df[(df['event_type'] == 'stim')]
trial_types = []
for i, row in df.iterrows():
	trigger_name = f'Target-{int(row["TargetLoc"])}-Singleton-{int(row["SingletonLoc"])}-{PRIMING[row["Priming"]]}'
	trial_types.append(trigger_name)
# get events from eeg data
mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/SPACEPRIME/settings/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "sub-101_task-spaceprime_raw.fif", preload=True)
# Downsample to 250 Hz
raw = raw.resample(500)
# add reference channel
raw.add_reference_channels(["Fz"])
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
raw.set_montage(montage)
# get events
events, event_id = mne.events_from_annotations(raw)
# loop through events
event_occurences = []
trial_ends = []
events_from_trial_seq = []
fixed_events = events.copy()
trial_count = -1  # start with trial 0
# iterate over events
for i, e in enumerate(events):
	# if event is 8 (trial offset) and the next trial is 7 (trial_onset), this means we reached a trial boundary
	if e[2] == 8 and events[i+1][2] == 7:
		# if we found a trial boundary, add 1 to the trial counter
		trial_count += 1
		# save the trial end just to be sure
		trial_ends.append(i)
		# if the preceding trial >= 9, this means an stimulus was played
		if events[i-1][2] >= 9:
			fixed_events[i-1][2] = EEG_TRIGGER_MAP[trial_types[trial_count]]

mne.write_events(filename=data_path + f"{subject_id}_fixed-eve.txt", events=fixed_events)