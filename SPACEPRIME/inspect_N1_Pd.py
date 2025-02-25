import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from mne.stats import permutation_cluster_test
from scipy.stats import ttest_ind
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos
from SPACEPRIME.passive_listening_ERP import get_passive_listening_ERPs
plt.ion()


settings_path = f"{get_data_path()}settings/"
montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
ch_pos = montage.get_positions()["ch_pos"]
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0]) for subject in subject_ids[2:]])
epochs.crop(-0.1, 0.6)
#epochs = epochs["select_target==True"]
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# get all channels from epochs
all_chs = epochs.ch_names
# Separate epochs based on distractor location
left_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
mne.epochs.equalize_epoch_counts([left_distractor_epochs, right_distractor_epochs], method="random")

# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")

# get the trial-wise data for targets
contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                 right_target_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                               right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
# get the trial-wise data for distractors
contra_distractor_epochs_data = np.mean(np.concatenate([left_distractor_epochs.copy().get_data(picks="C4"),
                                 right_distractor_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_distractor_epochs_data = np.mean(np.concatenate([left_distractor_epochs.copy().get_data(picks="C3"),
                               right_distractor_epochs.copy().get_data(picks="C4")], axis=1), axis=1)

# get difference waves
diff_wave_target = contra_target_epochs_data.mean(axis=0) - ipsi_target_epochs_data.mean(axis=0)
diff_wave_distractor = contra_distractor_epochs_data.mean(axis=0) - ipsi_distractor_epochs_data.mean(axis=0)
# run ttests
result_target = ttest_ind(contra_target_epochs_data,
                          ipsi_target_epochs_data, axis=0)
result_distractor = ttest_ind(contra_distractor_epochs_data,
                              ipsi_distractor_epochs_data, axis=0)
# plot the data
times = epochs.times
fig, ax = plt.subplots(2, 2, sharey=False)
ax[1][1].sharey(ax[1][0])
# first plot
ax[0][0].plot(times, contra_target_epochs_data.mean(axis=0), color="r")
ax[0][0].plot(times, ipsi_target_epochs_data.mean(axis=0), color="b")
ax[0][0].plot(times, diff_wave_target, color="g")
ax[0][0].axvspan(0.2, 0.30, color='gray', alpha=0.3)  # Shade the area
ax[0][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# second plot
ax[0][1].plot(times, contra_distractor_epochs_data.mean(axis=0), color="r")
ax[0][1].plot(times, ipsi_distractor_epochs_data.mean(axis=0), color="b")
ax[0][1].plot(times, diff_wave_distractor, color="g")
ax[0][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[0][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][1].set_title("Distractor lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# third plot
ax[1][0].plot(times, result_target[0])
ax[1][0].axvspan(0.2, 0.3, color='gray', alpha=0.3)  # Shade the area
ax[1][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][0].hlines(y=0, xmin=times[0], xmax=times[-1])
# fourth plot
ax[1][1].plot(times, result_distractor[0])
ax[1][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[1][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][1].hlines(y=0, xmin=times[0], xmax=times[-1])
plt.tight_layout()

# extract all channel data to plot topographies
# get left and right channels
# Create ROIs by checking channel labels
selections = mne.channels.make_1020_channel_selections(epochs.info, midline="z")
# get the trial-wise data for targets
contra_target_epochs_data_all_chs = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks=selections["Right"]),
                                                            right_target_epochs.copy().get_data(picks=selections["Left"])], axis=0),
                                            axis=0)
ipsi_target_epochs_data_all_chs = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks=selections["Left"]),
                                                          right_target_epochs.copy().get_data(picks=selections["Right"])], axis=0),
                                          axis=0)
# get the trial-wise data for distractors
contra_distractor_epochs_data_all_chs = np.mean(np.concatenate([left_distractor_epochs.copy().get_data(picks=selections["Right"]),
                                                               right_distractor_epochs.copy().get_data(picks=selections["Left"])], axis=0),
                                               axis=0)
ipsi_distractor_epochs_data_all_chs = np.mean(np.concatenate([left_distractor_epochs.copy().get_data(picks=selections["Left"]),
                                                             right_distractor_epochs.copy().get_data(picks=selections["Right"])], axis=0),
                                             axis=0)
# create diff evoked objects to plot nice topographies
# Get the channel names corresponding to those indices
selected_ch_names = [epochs.info['ch_names'][i] for i in selections["Right"]]
diff_wave_target_evoked = mne.EvokedArray(data=contra_target_epochs_data_all_chs-ipsi_target_epochs_data_all_chs,
                                          info=mne.create_info(ch_names=28,
                                                               sfreq=250))
diff_wave_distractor_evoked = mne.EvokedArray(data=contra_distractor_epochs_data_all_chs-ipsi_distractor_epochs_data_all_chs,
                                              info=mne.create_info(ch_names=28,
                                                                   sfreq=250))
# number of permutations
n_permutations = 10000
# some stats
n_jobs = -1
pval = 0.05
threshold = dict(start=0, step=0.2)  # the smaller the step and the closer the start to 0, the better the approximation

# mne.viz.plot_ch_adjacency(epochs.info, adjacency, epochs.info["ch_names"])
X = [contra_target_epochs_data, ipsi_target_epochs_data]
t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(X, threshold=threshold, n_permutations=n_permutations,
                                                           n_jobs=n_jobs, out_type="mask", tail=0, stat_fun=mne.stats.ttest_ind_no_p)
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
ax2.set_ylabel("statistic value")  # which statistic?

# calculate difference waves
diff_waves_target, diff_waves_distractor = difference_topos(epochs=epochs, montage=montage)

# --- Topomap Sequence Figure ---
info = epochs.info
# 1. Define Time Range and Step
start_time = 0  # Example: Start 100ms *before* stimulus onset
end_time = 0.6  # Example: End 500ms after stimulus onset
time_step = 0.05  # 50ms step

# 2. Create Time Points
times_to_plot = np.arange(start_time, end_time + time_step, time_step)

# 3. Choose Difference Wave (Target or Distractor)
plot_title_prefix = "Target"  # if diff_waves is diff_waves_distractor
if plot_title_prefix == "Target":
    diff_waves = diff_waves_target  # For TARGET difference waves
elif plot_title_prefix == "Distractor":
    diff_waves = diff_waves_distractor  # For DISTRACTOR difference waves

# 4. Determine Figure Layout (Rows and Columns)
n_plots = len(times_to_plot)
n_cols = int(np.ceil(np.sqrt(n_plots)))
n_rows = int(np.ceil(n_plots / n_cols))

# 5. Create Figure and Subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
axes = axes.flatten()

# 6.  Find Consistent vmin and vmax
all_values = []
for ch in diff_waves:
    for time_idx in [epochs.time_as_index(t)[0] for t in times_to_plot]:
        all_values.append(diff_waves[ch][time_idx])
vmin = np.min(all_values)
vmax = np.max(all_values)
print(f'vmin = {vmin}, vmax = {vmax}')

# 7. Choose the colormap *explicitly*
cmap = 'RdBu_r'  # Or any other colormap you prefer

# 8. Loop and Plot
for i, time_point in enumerate(times_to_plot):
    time_idx = epochs.time_as_index(time_point)[0]
    data_for_plot = np.array([diff_waves[ch][time_idx] for ch in all_chs if ch in diff_waves])

    # Now pass the `cmap` to plot_topomap
    mne.viz.plot_topomap(data_for_plot, info, axes=axes[i], cmap=cmap)
    axes[i].set_title(f"{time_point * 1000:.0f} ms")

# 9. Remove Unused Subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 10. Create ScalarMappable with the correct colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# 11. Add Colorbar and Title (using the ScalarMappable)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
plt.colorbar(sm, cax=cbar_ax)  # Use the ScalarMappable here

fig.suptitle(f"{plot_title_prefix} Difference Wave Topomaps (50ms Steps)", fontsize=16)
