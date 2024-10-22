import mne
from mne.channels import make_1020_channel_selections


# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True)
# get channels left and right
channels = make_1020_channel_selections(epochs.info, midline="z", return_ch_names=False)
# channel names
channel_names = make_1020_channel_selections(epochs.info, midline="z", return_ch_names=True)
# get all conditions from epochs
all_conds = list(epochs.event_id.keys())
# now, only get the relevant conditions (lateral targets/distractors and frontal targets/distractors)
relevant_conds = [x for x in all_conds if "Target-1-Singleton-2" in x or "Target-3-Singleton-2" in x or "Target-2-Singleton-1" in x or "Target-2-Singleton-3" in x]
# equalize number of events in all conditions to prevent biases
# epochs.equalize_event_counts(relevant_conds)
# get lateral targets and central singletons
lateral_targets = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x or "Target-3-Singleton-2" in x]].average(by_event_type=True)
# get lateral singletons and central targets
lateral_singletons = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x or "Target-2-Singleton-3" in x]].average(by_event_type=True)
# combine all left targets
left_targets = mne.combine_evoked(lateral_targets[:3], "equal")
right_targets = mne.combine_evoked(lateral_targets[3:], "equal")
evokeds = dict(left_targets=lateral_targets[:3], right_targets=lateral_targets[3:])
# plot some data
mne.viz.plot_compare_evokeds(evokeds, picks=["C3", "C4"], combine="mean")
lateral_singletons.average().plot(picks=["C3", "C4"])