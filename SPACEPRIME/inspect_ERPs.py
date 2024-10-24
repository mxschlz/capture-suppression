import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True).crop(None, 1.5)
all_conds = list(epochs.event_id.keys())
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
# get the contralateral evoked response and average
contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                    right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C3"]).get_data(),
                                  right_singleton_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
# get the contralateral evoked response and average
contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_target_data = np.mean([left_target_epochs.copy().average(picks=["C3"]).get_data(),
                               right_target_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
# plot the data
times = epochs.average().times
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(times, contra_target_data[0], color="r")
ax[0].plot(times, ipsi_target_data[0], color="b")
ax[0].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[0].legend(["Contra", "Ipsi"])
ax[0].set_title("Target lateral")
ax[0].set_ylabel("Amplitude [µV]")
ax[1].plot(times, contra_singleton_data[0], color="r")
ax[1].plot(times, ipsi_singleton_data[0], color="b")
ax[1].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[1].legend(["Contra", "Ipsi"])
ax[1].set_title("Singleton lateral")
ax[1].set_ylabel("Amplitude [µV]")
ax[1].set_xlabel("Time [s]")
plt.tight_layout()
# compute power density spectrum for evoked response and look for frequency tagging
freqs = np.arange(1, 35, 1)
method = "multitaper"
epochs.average().compute_psd(method=method, n_jobs=-1).plot(average=True)
# do some t statistics
from mne.stats import ttest_ind_no_p
ttest_ind_no_p(contra_target_data[0], ipsi_target_data)