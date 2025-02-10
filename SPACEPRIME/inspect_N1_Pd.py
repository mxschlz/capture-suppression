import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import SPACEPRIME
from mne.stats import permutation_cluster_test
plt.ion()


# define data root dir
data_root = SPACEPRIME.get_data_path()+"derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{SPACEPRIME.get_data_path()}derivatives/epoching/{subject}/eeg/{subject}_task-spaceprime-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 107]])
#epochs = epochs["select_target==True"]
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

# get difference waves
diff_wave_target = contra_target_data - ipsi_target_data
diff_wave_distractor = contra_singleton_data - ipsi_singleton_data
# run ttests
from scipy.stats import ttest_ind
result_target = ttest_ind(contra_target_epochs_data, ipsi_target_epochs_data, axis=0)
result_singleton = ttest_ind(contra_singleton_epochs_data, ipsi_singleton_epochs_data, axis=0)
# plot the data
times = epochs.average().times
fig, ax = plt.subplots(2, 2)
# first plot
ax[0][0].plot(times, contra_target_data[0], color="r")
ax[0][0].plot(times, ipsi_target_data[0], color="b")
ax[0][0].plot(times, diff_wave_target[0], color="g")
ax[0][0].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[0][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# second plot
ax[0][1].plot(times, contra_singleton_data[0], color="r")
ax[0][1].plot(times, ipsi_singleton_data[0], color="b")
ax[0][1].plot(times, diff_wave_distractor[0], color="g")
ax[0][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[0][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][1].set_title("Singleton lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# third plot
ax[1][0].plot(times, result_target[0])
ax[1][0].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[1][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][0].hlines(y=0, xmin=times[0], xmax=times[-1])
# fourth plot
ax[1][1].plot(times, result_singleton[0])
ax[1][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[1][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][1].hlines(y=0, xmin=times[0], xmax=times[-1])
plt.tight_layout()


# number of permutations
n_permutations = 10000
# some stats
n_jobs = -1
pval = 0.05
threshold = dict(start=0, step=0.2)  # the smaller the step and the closer the start to 0, the better the approximation

# mne.viz.plot_ch_adjacency(epochs.info, adjacency, epochs.info["ch_names"])
X = [contra_target_epochs_data, ipsi_target_epochs_data]
t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(X, threshold=threshold, n_permutations=n_permutations,
                                                           n_jobs=n_jobs, out_type="mask", tail=0)
times = epochs.times
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title("Contra minus ipsi")
ax.plot(
    times,
    contra_target_epochs_data.mean(axis=0) - ipsi_target_epochs_data.mean(axis=0),
    label="ERP Contrast (Contra minus ipsi)",
)
ax.set_ylabel("EEG (µV)")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_pv[i_c] <= pval:
        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
    else:
        h = 0
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

hf = plt.plot(times, t_obs, "g")
ax2.legend((h,), ("cluster p-value < 0.05",))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")