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
mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")

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
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get the contralateral evoked response and average
contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_target_data = np.mean([left_target_epochs.copy().average(picks=["C3"]).get_data(),
                               right_target_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
# do some t statistics
contra_target_epochs = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                 right_target_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
# get the ipsilateral evoked response and average
ipsi_target_epochs = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                               right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
contra_singleton_epochs = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C4"),
                                 right_singleton_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
# get the ipsilateral evoked response and average
ipsi_singleton_epochs = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C3"),
                               right_singleton_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
from scipy.stats import ttest_ind
result_target = ttest_ind(contra_target_epochs, ipsi_target_epochs, axis=0)
result_singleton = ttest_ind(contra_singleton_epochs, ipsi_singleton_epochs, axis=0)
# plot the data
times = epochs.average().times
fig, ax = plt.subplots(2, 2, sharex=True)
ax[0][0].plot(times, contra_target_data[0], color="r")
ax[0][0].plot(times, ipsi_target_data[0], color="b")
ax[0][0].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[0][0].legend(["Contra", "Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][1].plot(times, contra_singleton_data[0], color="r")
ax[0][1].plot(times, ipsi_singleton_data[0], color="b")
ax[0][1].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[0][1].legend(["Contra", "Ipsi"])
ax[0][1].set_title("Singleton lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
ax[1][0].plot(times, result_target[0])
ax[1][1].plot(times, result_singleton[0])
plt.tight_layout()

# compute power density spectrum for evoked response and look for frequency tagging
freqs = np.arange(1, 35, 1)
method = "morlet"
n_cycles = freqs / 2  # different number of cycle per frequency
power = epochs.compute_(method=method, n_jobs=-1, decim=1, n_cycles=n_cycles, freqs=freqs,)
power.plot(picks=["Cz"])