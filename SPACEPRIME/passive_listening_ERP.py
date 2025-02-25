import mne
import matplotlib.pyplot as plt
import glob
import numpy as np
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
plt.ion()


def get_passive_listening_ERPs():
    # load epochs
    epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-passive-epo.fif")[0]) for subject in subject_ids[2:]])
    # get all conditions
    # all_conds = list(epochs.event_id.keys())
    # get target, distractor and control epochs
    # target_epochs = epochs[[x for x in all_conds if "target" in x]]
    # distractor_epochs = epochs[[x for x in all_conds if "distractor" in x]]
    # control_epochs = epochs[[x for x in all_conds if "control" in x]]
    # plot ERP
    # mne.viz.plot_compare_evokeds([target_epochs.average(), distractor_epochs.average(), control_epochs.average()], picks="FCz")

    # plot diff waves
    all_conds = list(epochs.event_id.keys())
    # Separate epochs based on distractor location
    left_singleton_epochs = epochs[[x for x in all_conds if "distractor-location-1" in x]]
    right_singleton_epochs = epochs[[x for x in all_conds if "distractor-location-3" in x]]
    mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
    # get the contralateral evoked response and average
    contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                        right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C3"]).get_data(),
                                      right_singleton_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
    # now, do the same for the lateral targets
    # Separate epochs based on target location
    left_target_epochs = epochs[[x for x in all_conds if "target-location-1" in x]]
    right_target_epochs = epochs[[x for x in all_conds if "target-location-3" in x]]
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
    diff_singleton = contra_singleton_data - ipsi_singleton_data
    diff_target = contra_target_data - ipsi_target_data
    return contra_singleton_epochs_data, ipsi_singleton_epochs_data, contra_target_epochs_data, ipsi_target_epochs_data, diff_target, diff_singleton

#contra_singleton_epochs_data, ipsi_singleton_epochs_data, contra_target_epochs_data, ipsi_target_epochs_data, diff_target, diff_singleton = get_passive_listening_ERPs()

"""# get time points from epochs
times =  mne.concatenate_epochs([mne.read_epochs(
    glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-passive-epo.fif")[0]) for
                                 subject in subject_ids[2:]]).times

from scipy.stats import ttest_ind
result_target = ttest_ind(contra_target_epochs_data, ipsi_target_epochs_data, axis=0)
result_singleton = ttest_ind(contra_singleton_epochs_data, ipsi_singleton_epochs_data, axis=0)
# plot the data
fig, ax = plt.subplots(2, 2)
# first plot
ax[0][0].plot(times, contra_target_epochs_data.mean(axis=0), color="r")
ax[0][0].plot(times, ipsi_target_epochs_data.mean(axis=0), color="b")
ax[0][0].plot(times, diff_target[0], color="g")
ax[0][0].axvspan(0.2, 0.3, color='gray', alpha=0.3)  # Shade the area
ax[0][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# second plot
ax[0][1].plot(times, contra_singleton_epochs_data.mean(axis=0), color="r")
ax[0][1].plot(times, ipsi_singleton_epochs_data.mean(axis=0), color="b")
ax[0][1].plot(times, diff_singleton[0], color="g")
ax[0][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[0][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][1].set_title("Singleton lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# third plot
ax[1][0].plot(times, result_target[0])
ax[1][0].axvspan(0.2, 0.3, color='gray', alpha=0.3)  # Shade the area
ax[1][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][0].hlines(y=0, xmin=times[0], xmax=times[-1])
# fourth plot
ax[1][1].plot(times, result_singleton[0])
ax[1][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[1][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][1].hlines(y=0, xmin=times[0], xmax=times[-1])
plt.tight_layout()
"""
