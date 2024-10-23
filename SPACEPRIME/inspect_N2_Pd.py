import mne
import numpy as np
import matplotlib.pyplot as plt


# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True)
all_conds = list(epochs.event_id.keys())
# Separate epochs based on target/distractor location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
# get the contralateral evoked response and average
contra_data = np.mean([left_target_epochs.copy().apply_baseline().average(picks=["C4"]).get_data(),
                                right_target_epochs.copy().apply_baseline().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_data = np.mean([left_target_epochs.copy().apply_baseline().average(picks=["C3"]).get_data(),
                                right_target_epochs.copy().apply_baseline().average(picks=["C4"]).get_data()], axis=0)
# plot the data
times = left_target_epochs.average().times
plt.plot(times, contra_data[0])
plt.plot(times, ipsi_data[0])
plt.legend(["Contra", "Ipsi"])



































# get the data for the evoked responses
left_target_evoked_data = left_target_evoked.get_data()
right_target_evoked_data = right_target_evoked.get_data()
# For left targets, C4 is contralateral and C3 is ipsilateral
left_target_diff = mne.EvokedArray((left_target_evoked_data[1] - left_target_evoked_data[0]).reshape([1, 551]),
                                   info=mne.create_info(["Left_Diff"], sfreq=250, ch_types="eeg"))
# For right targets, C3 is contralateral and C4 is ipsilateral
right_target_diff = mne.EvokedArray((right_target_evoked_data[0] - right_target_evoked_data[1]).reshape([1, 551]),
                                   info=mne.create_info(["Right_Diff"], sfreq=250, ch_types="eeg"))
# Plot the difference waves
left_target_diff.plot(titles="Left Target (C4-C3)")
right_target_diff.plot(titles="Right Target (C3-C4)")
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