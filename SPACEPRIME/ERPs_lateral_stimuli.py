import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from mne.stats import permutation_cluster_test
from scipy.stats import ttest_ind
from scipy.stats import t
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos
import seaborn as sns
plt.ion()


settings_path = f"{get_data_path()}settings/"
montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
ch_pos = montage.get_positions()["ch_pos"]
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
epochs = epochs.crop(0, 0.7)
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
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax = ax.flatten()
# first plot
ax[0].plot(times, contra_target_epochs_data.mean(axis=0)*10e5, color="r")
ax[0].plot(times, ipsi_target_epochs_data.mean(axis=0)*10e5, color="b")
ax[0].plot(times, diff_wave_target*10e5, color="g")
ax[0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0].set_title("Target lateral")
ax[0].set_ylabel("Amplitude [µV]")
ax[0].set_xlabel("Time [s]")
# second plot
ax[1].plot(times, contra_distractor_epochs_data.mean(axis=0)*10e5, color="r")
ax[1].plot(times, ipsi_distractor_epochs_data.mean(axis=0)*10e5, color="b")
ax[1].plot(times, diff_wave_distractor*10e5, color="g")
ax[1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[1].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[1].set_title("Distractor lateral")
ax[1].set_xlabel("Time [s]")
# add t stats on same plot with different axis
twin1 = ax[0].twinx()
twin1.tick_params(axis='y', labelcolor="brown")
twin1.plot(times, result_target[0], color="brown", linestyle="dashed", alpha=0.5)
# fourth plot
twin2 = ax[1].twinx()
twin1.sharey(twin2)
twin2.set_ylabel("T-Value", color="brown")
twin2.tick_params(axis='y', labelcolor="brown")
twin2.plot(times, result_distractor[0], color="brown", linestyle="dashed", alpha=0.5)
# despine
sns.despine(fig=fig, right=False)

# --- STATISTICS ---
n_permutations = 10000  # number of permutations
# some stats
n_jobs = -1
alpha = 0.05
tail = 0
# Now we need to set the threshold parameter. For this time-series data (1 electrode pair over time) which is NOT SUITED
# FOR SPATIAL COMPARISON BUT TEMPORAL COMPARISON, we should use a single t-value. A reasonable starting point would be
# a t-value corresponding to an uncorrected p-value of 0.05 for a single comparison. We can calculate this using
# scipy.stats.f.ppf.
for run_on in ["Target", "Distractor"]:
    n1 = contra_target_epochs_data.shape[0] if run_on == "Target" else contra_distractor_epochs_data.shape[0]
    n2 = ipsi_target_epochs_data.shape[0] if run_on == "Target" else contra_distractor_epochs_data.shape[0]
    df = n1 + n2 - 2
    if tail == 0:
        threshold = t.ppf(1 - alpha / 2, df)  # Two-tailed
    else:  # tail == -1 or tail == 1
        threshold = t.ppf(alpha, df) if tail == -1 else t.ppf(1 - alpha, df)
    print(f"Using threshold: {threshold}")

    # mne.viz.plot_ch_adjacency(epochs.info, adjacency, epochs.info["ch_names"])
    X = [contra_target_epochs_data, ipsi_target_epochs_data] if run_on == "Target" else [contra_distractor_epochs_data, ipsi_distractor_epochs_data]
    t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(X, threshold=threshold, n_permutations=n_permutations,
                                                               n_jobs=n_jobs, out_type="mask", tail=tail, stat_fun=mne.stats.ttest_ind_no_p)
    # plot on existing axes
    plot_on_axis = 1 if run_on == "Distractor" else 0
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_pv[i_c] <= alpha:
            h = ax[plot_on_axis].axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
        else:
            h = 0
            ax[plot_on_axis].axvspan(times[c.start], times[c.stop - 1], color="grey", alpha=0.3)

# --- TOPOGRAPHIES ---
# calculate difference waves
diff_waves_target, diff_waves_distractor = difference_topos(epochs=epochs, montage=montage)

# --- Topomap Sequence Figure ---
info = epochs.info
# 1. Define Time Range and Step
start_time = 0  # Example: Start 100ms *before* stimulus onset
end_time = 0.7  # Example: End 500ms after stimulus onset
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
