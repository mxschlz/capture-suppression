import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


subject_id = 102
# load epochs
epochs = mne.read_epochs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
                         preload=True)
# epochs.apply_baseline()
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
# get the trial-wise data for targets
contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                 right_target_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                               right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
# get the trial-wise data for singletons
contra_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C4"),
                                 right_singleton_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C3"),
                               right_singleton_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
from scipy.stats import ttest_ind, ttest_rel
result_target = ttest_ind(contra_target_epochs_data, ipsi_target_epochs_data, axis=0)
result_singleton = ttest_ind(contra_singleton_epochs_data, ipsi_singleton_epochs_data, axis=0)
# plot the data
times = epochs.average().times
fig, ax = plt.subplots(2, 2)
# first plot
ax[0][0].plot(times, contra_target_data[0], color="r")
ax[0][0].plot(times, ipsi_target_data[0], color="b")
ax[0][0].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[0][0].axvspan(0.35, 0.55, color='gray', alpha=0.3)  # Shade the area
ax[0][0].legend(["Contra", "Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# second plot
ax[0][1].plot(times, contra_singleton_data[0], color="r")
ax[0][1].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[0][1].axvspan(0.35, 0.55, color='gray', alpha=0.3)  # Shade the area
ax[0][1].plot(times, ipsi_singleton_data[0], color="b")
ax[0][1].set_title("Singleton lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# third plot
ax[1][0].plot(times, result_target[0])
ax[1][0].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[1][0].axvspan(0.35, 0.55, color='gray', alpha=0.3)  # Shade the area
# fourth plot
ax[1][1].plot(times, result_singleton[0])
ax[1][1].axvspan(0.65, 0.85, color='gray', alpha=0.3)  # Shade the area
ax[1][1].axvspan(0.35, 0.55, color='gray', alpha=0.3)  # Shade the area
plt.tight_layout()
# compute power density spectrum for evoked response and look for frequency tagging
# first, create epochs objects from numpy arrays computed above for ipsi and contra targets
ipsi_target_epochs = mne.EpochsArray(data=ipsi_target_epochs_data.reshape(127, 1, 551),
                                     info=mne.create_info(ch_names=["Ipsi Target"], sfreq=250, ch_types="eeg"))
contra_target_epochs = mne.EpochsArray(data=contra_target_epochs_data.reshape(127, 1, 551),
                                     info=mne.create_info(ch_names=["Contra Target"], sfreq=250, ch_types="eeg"))
# now, compute the power spectra for both ipsi and contra targets
target_diff = mne.EpochsArray(data=(contra_target_epochs.get_data() - ipsi_target_epochs.get_data()),
                              info=mne.create_info(ch_names=["Contra - Ipsi Target"], sfreq=250, ch_types="eeg"))
# subtract the ipsi from the contralateral target power
powerdiff_target = target_diff.compute_psd(method="welch")
powerdiff_target.plot()
# another approach on all epochs together
psd = epochs.average().compute_psd("welch")
psd.plot(average=True)