import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os # Added for file checking
from SPACEPRIME import get_data_path
# Use the paired t-test or 1-sample t-test on differences for cluster stats
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_rel, t # ttest_rel for paired tests
from SPACEPRIME.subjects import subject_ids
# Assuming difference_topos can be adapted or we recalculate here
from SPACEPRIME.plotting import difference_topos
import seaborn as sns
plt.ion()


# --- Settings and Setup ---
settings_path = f"{get_data_path()}settings/"
montage_path = os.path.join(settings_path, "CACS-64_NO_REF.bvef")
if not os.path.exists(montage_path):
    raise FileNotFoundError(f"Montage file not found at: {montage_path}")
montage = mne.channels.read_custom_montage(montage_path)

# Define ROIs (adjust if needed, using C3/C4 as in original code)
left_roi_distractor = ['C3']
right_roi_distractor = ['C4']
left_roi_target = ['C3']
right_roi_target = ['C4']

# --- Data Storage ---
# Dictionaries to store subject-level results
subject_data = {
    'target_contra': [],
    'target_ipsi': [],
    'target_diff': [],
    'distractor_contra': [],
    'distractor_ipsi': [],
    'distractor_diff': [],
    # For topographies (store average difference wave per subject across all channels)
    'target_diff_topo': [],
    'distractor_diff_topo': []
}
processed_subjects = [] # Keep track of subjects successfully processed

# --- Subject Loop ---
for subject in subject_ids:
    print(f"\n--- Processing Subject: {subject} ---")
    try:
        epoch_file = glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0]
        epochs_sub = mne.read_epochs(epoch_file, preload=True) # Preload for easier manipulation
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except IndexError:
        print(f"  Epoch file not found for subject {subject}. Skipping.")
        continue
    except Exception as e:
        print(f"  Error loading data for subject {subject}: {e}. Skipping.")
        continue

    # --- Preprocessing per Subject (if needed) ---
    epochs_sub = epochs_sub.crop(0, 0.7)

    all_conds_sub = list(epochs_sub.event_id.keys())
    times = epochs_sub.times # Get time vector from the first subject (should be consistent)
    info = epochs_sub.info # Get info structure

    # --- Condition Splitting and Equalization (TARGETS) ---
    left_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
    right_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]].copy()
    print(f"  Target trials: {len(left_target_epochs)} left, {len(right_target_epochs)} right")
    if len(left_target_epochs) == 0 or len(right_target_epochs) == 0:
         raise ValueError("Zero trials in one of the target conditions.")

    # --- Condition Splitting and Equalization (DISTRACTORS) ---
    left_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
    right_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]].copy()
    print(f"  Distractor trials: {len(left_distractor_epochs)} left, {len(right_distractor_epochs)} right")
    if len(left_distractor_epochs) == 0 or len(right_distractor_epochs) == 0:
         raise ValueError("Zero trials in one of the distractor conditions.")

    # --- Calculate Subject-Level Ipsi/Contra ERPs (TARGETS) ---
    # Get data returns (n_epochs, n_channels, n_times)
    # Average across epochs (axis=0) and channels (axis=1, if multiple channels in ROI)
    contra_target_sub = np.mean(np.concatenate([
        left_target_epochs.get_data(picks=right_roi_target), # Left stim, Right ROI
        right_target_epochs.get_data(picks=left_roi_target)  # Right stim, Left ROI
    ], axis=0), axis=(0, 1)) # Average over concatenated trials and ROI channels

    ipsi_target_sub = np.mean(np.concatenate([
        left_target_epochs.get_data(picks=left_roi_target),  # Left stim, Left ROI
        right_target_epochs.get_data(picks=right_roi_target) # Right stim, Right ROI
    ], axis=0), axis=(0, 1)) # Average over concatenated trials and ROI channels

    diff_target_sub = contra_target_sub - ipsi_target_sub

    # --- Calculate Subject-Level Ipsi/Contra ERPs (DISTRACTORS) ---
    contra_distractor_sub = np.mean(np.concatenate([
        left_distractor_epochs.get_data(picks=right_roi_distractor),
        right_distractor_epochs.get_data(picks=left_roi_distractor)
    ], axis=0), axis=(0, 1))

    ipsi_distractor_sub = np.mean(np.concatenate([
        left_distractor_epochs.get_data(picks=left_roi_distractor),
        right_distractor_epochs.get_data(picks=right_roi_distractor)
    ], axis=0), axis=(0, 1))

    diff_distractor_sub = contra_distractor_sub - ipsi_distractor_sub

    # --- Calculate Subject-Level Difference Topographies ---
    # Get data for all channels, average over trials for each condition
    left_target_all_ch = left_target_epochs.get_data(picks='eeg').mean(axis=0)
    right_target_all_ch = right_target_epochs.get_data(picks='eeg').mean(axis=0)
    left_distractor_all_ch = left_distractor_epochs.get_data(picks='eeg').mean(axis=0)
    right_distractor_all_ch = right_distractor_epochs.get_data(picks='eeg').mean(axis=0)

    # get difference topography
    target_diff_topo_dict, distractor_diff_topo_dict = difference_topos(epochs_sub, montage)
    # transform the resulting subject topo dictionaries into arrays of shape n_elecs x n_time (64 x 176)
    # --- Convert Topo Dictionaries to Ordered NumPy Arrays ---
    n_channels = len(info.ch_names) # Get the number of channels from info
    n_times = len(times)            # Get the number of time points
    expected_shape = (n_channels, n_times)

    # Ensure the channel order matches the info object
    channel_order = info.ch_names  # Use the channel order from the info object

    # Convert target topo dict to array
    ordered_target_data = []
    for i, ch_name in enumerate(channel_order):
        if ch_name in target_diff_topo_dict:
            # Ensure value is numpy array and has correct length
            ch_data = np.asarray(target_diff_topo_dict[ch_name])
            if ch_data.shape == (n_times,):
                ordered_target_data.append(ch_data)
            else:
                print(
                    f"Warning: Channel {ch_name} data shape mismatch ({ch_data.shape}) for target topo, subject {subject}. Using NaNs.")
                ordered_target_data.append(np.full(n_times, np.nan))
        else:
            print(
                f"Warning: Channel {ch_name} from info not found in target_diff_topo_dict for subject {subject}. Using NaNs.")
            ordered_target_data.append(np.full(n_times, np.nan))
    target_diff_topo_sub = np.array(ordered_target_data)  # Shape: (n_channels, n_times)

    # Convert distractor topo dict to array
    ordered_distractor_data = []
    for i, ch_name in enumerate(channel_order):
        if ch_name in distractor_diff_topo_dict:
            ch_data = np.asarray(distractor_diff_topo_dict[ch_name])
            if ch_data.shape == (n_times,):
                ordered_distractor_data.append(ch_data)
            else:
                print(
                    f"Warning: Channel {ch_name} data shape mismatch ({ch_data.shape}) for distractor topo, subject {subject}. Using NaNs.")
                ordered_distractor_data.append(np.full(n_times, np.nan))
        else:
            print(
                f"Warning: Channel {ch_name} from info not found in distractor_diff_topo_dict for subject {subject}. Using NaNs.")
            ordered_distractor_data.append(np.full(n_times, np.nan))
    distractor_diff_topo_sub = np.array(ordered_distractor_data)  # Shape: (n_channels, n
    # --- Store Subject Results ---
    subject_data['target_contra'].append(contra_target_sub)
    subject_data['target_ipsi'].append(ipsi_target_sub)
    subject_data['target_diff'].append(diff_target_sub)
    subject_data['distractor_contra'].append(contra_distractor_sub)
    subject_data['distractor_ipsi'].append(ipsi_distractor_sub)
    subject_data['distractor_diff'].append(diff_distractor_sub)
    subject_data['target_diff_topo'].append(target_diff_topo_sub)
    subject_data['distractor_diff_topo'].append(distractor_diff_topo_sub)
    processed_subjects.append(subject)

print(f"\n--- Successfully processed {len(processed_subjects)} subjects: {processed_subjects} ---")

if not processed_subjects:
    raise RuntimeError("No subjects were successfully processed. Cannot continue.")

# --- Convert lists to numpy arrays for easier handling ---
for key in subject_data:
    subject_data[key] = np.array(subject_data[key]) # Shape: (n_subjects, n_times) or (n_subjects, n_channels, n_times) for topo

# --- Group Level Analysis ---

# 1. Calculate Grand Averages
ga_target_contra = subject_data['target_contra'].mean(axis=0)
ga_target_ipsi = subject_data['target_ipsi'].mean(axis=0)
ga_target_diff = subject_data['target_diff'].mean(axis=0)
ga_distractor_contra = subject_data['distractor_contra'].mean(axis=0)
ga_distractor_ipsi = subject_data['distractor_ipsi'].mean(axis=0)
ga_distractor_diff = subject_data['distractor_diff'].mean(axis=0)

# Grand average difference topographies
ga_target_diff_topo = subject_data['target_diff_topo'].mean(axis=0)
ga_distractor_diff_topo = subject_data['distractor_diff_topo'].mean(axis=0)


# 2. Plot Grand Average ERPs
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax = ax.flatten()
scale_factor = 1e6 # Convert V to µV

# Target Plot
ax[0].plot(times, ga_target_contra * scale_factor, color="r", label="Contra")
ax[0].plot(times, ga_target_ipsi * scale_factor, color="b", label="Ipsi")
ax[0].plot(times, ga_target_diff * scale_factor, color="g", label="Contra-Ipsi")
ax[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax[0].axvline(x=0, color='k', linestyle=':', linewidth=0.5)
ax[0].legend()
ax[0].set_title(f"Target Lateralization (N={len(processed_subjects)})")
ax[0].set_ylabel("Amplitude [µV]")
ax[0].set_xlabel("Time [s]")

# Distractor Plot
ax[1].plot(times, ga_distractor_contra * scale_factor, color="r", label="Contra")
ax[1].plot(times, ga_distractor_ipsi * scale_factor, color="b", label="Ipsi")
ax[1].plot(times, ga_distractor_diff * scale_factor, color="g", label="Contra-Ipsi")
ax[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax[1].axvline(x=0, color='k', linestyle=':', linewidth=0.5)
ax[1].legend()
ax[1].set_title(f"Distractor Lateralization (N={len(processed_subjects)})")
ax[1].set_xlabel("Time [s]")

# Add T-stats (Optional - Cluster stats are better)
# Perform paired t-test across subjects for each time point
t_target, p_target = ttest_rel(subject_data['target_contra'], subject_data['target_ipsi'], axis=0)
t_distractor, p_distractor = ttest_rel(subject_data['distractor_contra'], subject_data['distractor_ipsi'], axis=0)

# Plot T-stats on twin axes
twin1 = ax[0].twinx()
twin1.set_ylabel("T-Value (Contra vs Ipsi)", color="purple")
twin1.tick_params(axis='y', labelcolor="purple")
twin1.plot(times, t_target, color="purple", linestyle="dashed", alpha=0.6, label="Paired T-test")
# twin1.legend(loc='lower right') # Add legend if desired

twin2 = ax[1].twinx()
twin2.sharey(twin1) # Share y-axis limits for t-values
# twin2.set_ylabel("T-Value", color="purple") # Label already on twin1
twin2.tick_params(axis='y', labelcolor="purple")
twin2.plot(times, t_distractor, color="purple", linestyle="dashed", alpha=0.6, label="Paired T-test")
# twin2.legend(loc='lower right')

sns.despine(fig=fig, right=False) # Keep right axis for T-values
fig.tight_layout()

# 3. Group Level Statistics (Cluster Permutation Test)
n_permutations = 10000  # Increase for publication (e.g., 10000)
alpha_stat = 0.05
tail = 0 # Two-tailed test (contra != ipsi)
n_subs = len(processed_subjects)
df = n_subs - 1
# Use t-distribution threshold appropriate for the paired test (or 1-sample on difference)
t_thresh = t.ppf(1 - alpha_stat / 2, df)
print(f"\n--- Running Cluster Permutation Tests (N={n_subs}) ---")
print(f"Using t-threshold: {t_thresh:.3f} (df={df}, alpha={alpha_stat}, two-tailed)")

# Run test on the difference wave (Contra - Ipsi) testing against 0
# This is equivalent to a paired t-test between contra and ipsi
X_target_diff = subject_data['target_diff'] # Shape: (n_subjects, n_times)
X_distractor_diff = subject_data['distractor_diff'] # Shape: (n_subjects, n_times)

# Target Cluster Test
t_obs_t, clusters_t, cluster_pv_t, h0_t = permutation_cluster_1samp_test(
    X_target_diff,
    threshold=t_thresh,
    n_permutations=n_permutations,
    tail=tail,
    n_jobs=-1, # Use all cores
    out_type="mask",
    verbose=True
)

# Distractor Cluster Test
t_obs_d, clusters_d, cluster_pv_d, h0_d = permutation_cluster_1samp_test(
    X_distractor_diff,
    threshold=t_thresh,
    n_permutations=n_permutations,
    tail=tail,
    n_jobs=-1,
    out_type="mask",
    verbose=True
)

# Add cluster significance shading to the ERP plot
# Target plot (ax[0])
significant_clusters_found_t = False
for i_c, c in enumerate(clusters_t):
    c = c[0] # Cluster mask is over time dimension
    if cluster_pv_t[i_c] <= alpha_stat:
        significant_clusters_found_t = True
        ax[0].axvspan(times[c.start], times[c.stop - 1], color="grey", alpha=0.3,
                      label=f'Cluster p={cluster_pv_t[i_c]:.6f}' if i_c == 0 else "_nolegend_") # Label first significant cluster
if significant_clusters_found_t:
    # Update legend to show cluster info
    handles_t, labels_t = ax[0].get_legend_handles_labels()
    ax[0].legend(handles_t, labels_t, loc='best')
else:
    print("No significant clusters found for Target condition.")


# Distractor plot (ax[1])
significant_clusters_found_d = False
for i_c, c in enumerate(clusters_d):
    c = c[0]
    if cluster_pv_d[i_c] <= alpha_stat:
        significant_clusters_found_d = True
        ax[1].axvspan(times[c.start], times[c.stop - 1], color="grey", alpha=0.3,
                      label=f'Cluster p={cluster_pv_d[i_c]:.6f}' if i_c == 0 else "_nolegend_")
if significant_clusters_found_d:
    handles_d, labels_d = ax[1].get_legend_handles_labels()
    ax[1].legend(handles_d, labels_d, loc='best')
else:
    print("No significant clusters found for Distractor condition.")

fig.canvas.draw_idle() # Update the figure


# --- TOPOGRAPHIES ---
# Plot grand average difference topographies at selected time points

# 1. Define Time Range and Step for Topomaps
start_time_topo = 0.0  # Example: Start 100ms after stimulus onset
end_time_topo = 0.7    # Example: End 400ms after stimulus onset
time_step_topo = 0.05  # 50ms step
times_to_plot = np.arange(start_time_topo, end_time_topo + time_step_topo, time_step_topo)
scale_factor = 1e6 # Convert V to µV

# 2. Choose Difference Wave (Target or Distractor)
for plot_type in ["Target", "Distractor"]:
    if plot_type == "Target":
        ga_diff_topo = ga_target_diff_topo
        plot_title_prefix = "Target"
    else:
        ga_diff_topo = ga_distractor_diff_topo
        plot_title_prefix = "Distractor"

    # 3. Determine Figure Layout
    n_plots = len(times_to_plot)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    # 4. Create Figure and Subplots
    fig_topo, axes_topo = plt.subplots(n_rows, n_cols, figsize=(max(10, 3*n_cols), 3*n_rows))
    if n_plots > 1:
        axes_topo = axes_topo.flatten()
    else: # Handle case of single plot
        axes_topo = [axes_topo]


    # 5. Find Consistent vmin and vmax across the selected time points for this condition
    topo_data_all_times = []
    time_indices_to_plot = [np.argmin(np.abs(times - t)) for t in times_to_plot]
    for time_idx in time_indices_to_plot:
         topo_data_all_times.extend(ga_diff_topo[:, time_idx]) # Add data from all channels at this time point

    # Calculate symmetric limits around 0 if appropriate for difference waves
    max_abs_val = np.max(np.abs(topo_data_all_times)) * scale_factor
    vmin, vmax = -max_abs_val, max_abs_val
    print(f'{plot_title_prefix} Topomap Limits: vmin = {vmin:.2f} µV, vmax = {vmax:.2f} µV')

    # 6. Choose the colormap
    cmap = 'RdBu_r'

    # 7. Loop and Plot
    for i, time_point in enumerate(times_to_plot):
        time_idx = time_indices_to_plot[i]
        data_for_plot = ga_diff_topo[:, time_idx] * scale_factor # Select time slice, scale to µV

        # Use MNE's plotting function
        im, cn = mne.viz.plot_topomap(data_for_plot, info, axes=axes_topo[i], cmap=cmap,
                                      vlim=(vmin, vmax), show=False, sensors=False)  # sensors='k.' to show
        axes_topo[i].set_title(f"{time_point * 1000:.0f} ms")

    # 8. Remove Unused Subplots
    for j in range(i + 1, len(axes_topo)):
        fig_topo.delaxes(axes_topo[j])

    # 9. Add Colorbar
    fig_topo.subplots_adjust(right=0.85) # Make space for colorbar
    cbar_ax = fig_topo.add_axes([0.88, 0.15, 0.03, 0.7]) # Position colorbar
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Amplitude Difference [µV]')

    fig_topo.suptitle(f"Grand Average {plot_title_prefix} Difference Wave Topomaps (N={n_subs})", fontsize=16)
