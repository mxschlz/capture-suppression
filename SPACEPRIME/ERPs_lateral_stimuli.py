import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from SPACEPRIME import get_data_path, load_concatenated_epochs
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos # Assuming this function is available and works as expected
import seaborn as sns
from stats import remove_outliers # Added for preprocessing

plt.ion()

# --- Script Parameters ---

# --- 1. Preprocessing Parameters (adopted from LMM script) ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'

# --- 2. General Script Parameters ---
# Whether to plot the topographies or not (because that takes a while)
PLOT_TOPOS = False

# Paths
SETTINGS_PATH = os.path.join(get_data_path(), "settings")
MONTAGE_FNAME = "CACS-64_NO_REF.bvef" # Relative to SETTINGS_PATH

# ROIs for Contra/Ipsi Calculation
LEFT_ROI_DISTRACTOR = ["C3"]
RIGHT_ROI_DISTRACTOR = ["C4"]
LEFT_ROI_TARGET = ["C3"]
RIGHT_ROI_TARGET = ["C4"]

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = 0.0, 0.7  # Seconds
AMPLITUDE_SCALE_FACTOR = 1e6

# Cluster Permutation Test Parameters
N_JOBS = 5
SEED = 42
N_PERMUTATIONS_CLUSTER = 10000     # Number of permutations (e.g., 5000-10000 for publication)
ALPHA_STAT_CLUSTER = 0.05         # Significance level for cluster p-values and for forming clusters
CLUSTER_TAIL = 0                  # 0 for two-tailed, 1 for right-tailed, -1 for left-tailed

# Significance Line Plotting Parameters (for ERP plots)
SIG_LINE_Y_OFFSET = 0.0          # Y-offset in µV (data units) from y=0 for the significance lines
SIG_LINE_LW = 4                   # Linewidth for significance lines
SIG_LINE_ALPHA = 0.6              # Alpha for significance lines
# Color will be matched to the difference wave color

# Topomap Plotting Parameters
# Note: Topomaps are now plotted only for significant time intervals found by the ERP cluster test.
# TOPO_TIME_STEP determines the sampling rate for plots within those intervals.
TOPO_TIME_STEP = 0.01             # Time step for topomap sequence within significant intervals (seconds)
TOPO_CMAP = 'RdBu_r'              # Colormap for topographies

# --- Parameters for Sensor-Space Cluster Permutation Test ---
PLOT_SIGNIFICANT_SENSORS_TOPO = False # Whether to run and plot sensor cluster results
# Define time windows for averaging topo data for cluster analysis (in seconds)
N2AC_TOPO_CLUSTER_WINDOW = (0.55, 0.6) # Example: 220-380ms for N2ac-like activity
PD_TOPO_CLUSTER_WINDOW = (0.3, 0.4)   # Example: 290-380ms for Pd-like activity
# Alpha for sensor cluster p-values (can be same as ALPHA_STAT_CLUSTER)
ALPHA_SENSOR_CLUSTER = 0.05
# Mask parameters for plotting significant sensors
SENSOR_MASK_PARAMS = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                          linewidth=0, markersize=5, markeredgewidth=1.5)
# --- End of Parameters ---

montage_path = os.path.join(SETTINGS_PATH, MONTAGE_FNAME)
if not os.path.exists(montage_path):
    raise FileNotFoundError(f"Montage file not found at: {montage_path}")
montage = mne.channels.read_custom_montage(montage_path)

# --- Data Storage ---
subject_data = {
    'target_contra': [], 'target_ipsi': [], 'target_diff': [],
    'distractor_contra': [], 'distractor_ipsi': [], 'distractor_diff': [],
    'target_diff_topo': [], 'distractor_diff_topo': []
}
processed_subjects = []
times_vector = None # To be populated from the first subject
epochs_info = None  # To be populated from the first subject

# --- Load and Preprocess Data ---
print("--- Loading and Preprocessing Data ---")
epochs = load_concatenated_epochs("spaceprime").crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Original number of trials: {len(epochs)}")

# Get metadata for preprocessing
df = epochs.metadata.copy().reset_index(drop=True)

# 1. Filter by phase
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    print(f"Filtering out trials from phase {FILTER_PHASE}...")
    df = df[df[PHASE_COL] != FILTER_PHASE]
    print(f"  Trials remaining after phase filter: {len(df)}")

# 2. Remove RT outliers
if REACTION_TIME_COL in df.columns:
    print(f"Removing RT outliers (threshold: {OUTLIER_RT_THRESHOLD} SD)...")
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"  Trials remaining after RT outlier removal: {len(df)}")

# 3. Apply the filter back to the epochs object
# The index of the cleaned dataframe corresponds to the trials to keep
epochs = epochs[df.index]
print(f"Final number of trials after preprocessing: {len(epochs)}")
# --- End of Preprocessing ---

# --- Subject Loop ---
for subject_id_num in subject_ids:
    subject_str = f"sub-{subject_id_num}" # Assuming subject_ids are numbers like 1, 2, etc.
                                      # If subject_ids are already "sub-XX", then just use subject_id_num
    print(f"\n--- Processing Subject: {subject_str} ---")
    try:
        epochs_sub = epochs[f"subject_id=={subject_id_num}"]
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for {subject_str}: {e}. Skipping.")
        continue

    if times_vector is None:
        times_vector = epochs_sub.times.copy()
        epochs_info = epochs_sub.info.copy()

    all_conds_sub = list(epochs_sub.event_id.keys())

    try:
        # TARGETS
        left_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
        right_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]].copy()
        print(f"  Target trials: {len(left_target_epochs)} left, {len(right_target_epochs)} right")
        if len(left_target_epochs) == 0 or len(right_target_epochs) == 0:
            print(f"  Skipping {subject_str} due to zero trials in one of the target conditions.")
            continue

        # DISTRACTORS
        left_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
        right_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]].copy()
        print(f"  Distractor trials: {len(left_distractor_epochs)} left, {len(right_distractor_epochs)} right")
        if len(left_distractor_epochs) == 0 or len(right_distractor_epochs) == 0:
            print(f"  Skipping {subject_str} due to zero trials in one of the distractor conditions.")
            continue

        # Calculate Subject-Level Ipsi/Contra ERPs (TARGETS)
        contra_target_sub = np.mean(np.concatenate([
            left_target_epochs.get_data(picks=RIGHT_ROI_TARGET),
            right_target_epochs.get_data(picks=LEFT_ROI_TARGET)
        ], axis=0), axis=(0, 1))
        ipsi_target_sub = np.mean(np.concatenate([
            left_target_epochs.get_data(picks=LEFT_ROI_TARGET),
            right_target_epochs.get_data(picks=RIGHT_ROI_TARGET)
        ], axis=0), axis=(0, 1))
        diff_target_sub = contra_target_sub - ipsi_target_sub

        # Calculate Subject-Level Ipsi/Contra ERPs (DISTRACTORS)
        contra_distractor_sub = np.mean(np.concatenate([
            left_distractor_epochs.get_data(picks=RIGHT_ROI_DISTRACTOR),
            right_distractor_epochs.get_data(picks=LEFT_ROI_DISTRACTOR)
        ], axis=0), axis=(0, 1))
        ipsi_distractor_sub = np.mean(np.concatenate([
            left_distractor_epochs.get_data(picks=LEFT_ROI_DISTRACTOR),
            right_distractor_epochs.get_data(picks=RIGHT_ROI_DISTRACTOR)
        ], axis=0), axis=(0, 1))
        diff_distractor_sub = contra_distractor_sub - ipsi_distractor_sub

        # Calculate Subject-Level Difference Topographies using imported function
        # Assuming difference_topos returns dicts of {ch_name: time_series_array}
        if PLOT_TOPOS:
            target_diff_topo_dict_sub, distractor_diff_topo_dict_sub = difference_topos(epochs_sub, montage)

            # Convert Topo Dictionaries to Ordered NumPy Arrays
            n_channels = len(epochs_info.ch_names)
            n_times_current = len(times_vector) # Should be consistent due to cropping
            channel_order = epochs_info.ch_names

            target_diff_topo_sub_arr = np.full((n_channels, n_times_current), np.nan)
            for i, ch_name in enumerate(channel_order):
                if ch_name in target_diff_topo_dict_sub:
                    ch_data = np.asarray(target_diff_topo_dict_sub[ch_name])
                    if ch_data.shape == (n_times_current,):
                        target_diff_topo_sub_arr[i, :] = ch_data
                    else:
                        print(f"  Warning: Ch {ch_name} data shape mismatch ({ch_data.shape} vs {(n_times_current,)}) for target topo, {subject_str}.")
                # else: # Already NaN by default
                #     print(f"  Warning: Ch {ch_name} not in target_diff_topo_dict for {subject_str}.")

            distractor_diff_topo_sub_arr = np.full((n_channels, n_times_current), np.nan)
            for i, ch_name in enumerate(channel_order):
                if ch_name in distractor_diff_topo_dict_sub:
                    ch_data = np.asarray(distractor_diff_topo_dict_sub[ch_name])
                    if ch_data.shape == (n_times_current,):
                        distractor_diff_topo_sub_arr[i, :] = ch_data
                    else:
                        print(f"  Warning: Ch {ch_name} data shape mismatch ({ch_data.shape} vs {(n_times_current,)}) for distractor topo, {subject_str}.")
                # else: # Already NaN by default
                #     print(f"  Warning: Ch {ch_name} not in distractor_diff_topo_dict for {subject_str}.")

        # Store Subject Results
        subject_data['target_contra'].append(contra_target_sub)
        subject_data['target_ipsi'].append(ipsi_target_sub)
        subject_data['target_diff'].append(diff_target_sub)
        subject_data['distractor_contra'].append(contra_distractor_sub)
        subject_data['distractor_ipsi'].append(ipsi_distractor_sub)
        subject_data['distractor_diff'].append(diff_distractor_sub)
        processed_subjects.append(subject_id_num)
        if PLOT_TOPOS:
            subject_data['target_diff_topo'].append(target_diff_topo_sub_arr)
            subject_data['distractor_diff_topo'].append(distractor_diff_topo_sub_arr)

    except Exception as e_proc:
        print(f"  Error during processing for subject {subject_str}: {e_proc}. Skipping subject for data aggregation.")
        continue


print(f"\n--- Successfully processed {len(processed_subjects)} subjects: {processed_subjects} ---")

if not processed_subjects:
    raise RuntimeError("No subjects were successfully processed. Cannot continue.")
if times_vector is None or epochs_info is None:
    raise RuntimeError("Essential data (times_vector or epochs_info) not populated. Check subject loop.")

# Convert lists to numpy arrays
for key in subject_data:
    try:
        subject_data[key] = np.array(subject_data[key])
    except Exception as e_conv:
        print(f"Could not convert {key} to numpy array. Data: {subject_data[key]}. Error: {e_conv}")
        # Handle cases where lists might be empty or have inconsistent shapes if not caught earlier
        if not subject_data[key]: # If list is empty
            if key.endswith('_topo'):
                subject_data[key] = np.empty((0, len(epochs_info.ch_names), len(times_vector)))
            else:
                subject_data[key] = np.empty((0, len(times_vector)))
        else: # If list is not empty but conversion failed, re-raise or handle
            raise e_conv


# --- Group Level Analysis ---
ga_data = {}
for key in ['target_contra', 'target_ipsi', 'target_diff',
            'distractor_contra', 'distractor_ipsi', 'distractor_diff',
            'target_diff_topo', 'distractor_diff_topo']:
    if subject_data[key].shape[0] > 0: # Check if there are subjects for this key
        ga_data[key] = np.nanmean(subject_data[key], axis=0) # Use nanmean for topo data
    else:
        # Create empty array with correct dimensions if no subjects
        if key.endswith('_topo'):
            ga_data[key] = np.empty((len(epochs_info.ch_names), len(times_vector))) * np.nan
        else:
            ga_data[key] = np.empty(len(times_vector)) * np.nan


# --- Plot Grand Average ERPs with Cluster Significance ---
fig_erp, ax_erp = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
ax_erp = ax_erp.flatten()

erp_plot_params = {
    'Target': {'ax': ax_erp[0], 'contra': ga_data['target_contra'], 'ipsi': ga_data['target_ipsi'], 'diff': ga_data['target_diff'], 'color_diff': 'black'},
    'Distractor': {'ax': ax_erp[1], 'contra': ga_data['distractor_contra'], 'ipsi': ga_data['distractor_ipsi'], 'diff': ga_data['distractor_diff'], 'color_diff': 'black'}
}

# Group Level Statistics (Cluster Permutation Test)
n_subs = len(processed_subjects)
if n_subs >= 2: # Need at least 2 subjects for t-test based threshold and cluster test
    df = n_subs - 1
    t_thresh_cluster = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, df) if df > 0 else None # Threshold for forming clusters
    print(f"\n--- Running Cluster Permutation Tests (N={n_subs}) ---")
    if t_thresh_cluster:
        print(f"Using cluster-forming t-threshold: {t_thresh_cluster:.3f} (df={df}, alpha={ALPHA_STAT_CLUSTER}, two-tailed)")
    else:
        print("Cannot calculate t-threshold for cluster forming (df <=0). Skipping cluster tests.")

    cluster_results = {}
    if t_thresh_cluster:
        for cond_type in ['Target', 'Distractor']:
            X_diff = subject_data[f'{cond_type.lower()}_diff']
            if X_diff.shape[0] < 2 : # Not enough subjects for this specific condition
                print(f"Not enough data for {cond_type} cluster test (N={X_diff.shape[0]}). Skipping.")
                cluster_results[cond_type] = None
                continue

            t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
                X_diff, threshold=t_thresh_cluster, n_permutations=N_PERMUTATIONS_CLUSTER,
                tail=CLUSTER_TAIL, n_jobs=N_JOBS, out_type="mask", seed=SEED, verbose=False
            )
            cluster_results[cond_type] = {'t_obs': t_obs, 'clusters': clusters, 'p_values': cluster_pv}
            print(f"  {cond_type}: Found {np.sum(cluster_pv < ALPHA_STAT_CLUSTER)} significant cluster(s).")
else:
    print("\nSkipping cluster permutation tests: not enough subjects processed (N < 2).")
    cluster_results = {'Target': None, 'Distractor': None}
    t_thresh_cluster = None


# Define colors before the loop for clarity and easy modification
CONTRA_COLOR = "#C51B7D"  # Deep Orange
IPSI_COLOR = "#8C510A"    # Dark Teal

for cond_name, params in erp_plot_params.items():
    ax = params['ax']
    # Plot mean ERP waves with new, consistent colors
    ax.plot(times_vector, params['contra'] * AMPLITUDE_SCALE_FACTOR, color=CONTRA_COLOR, linestyle='-', label="Contra")
    ax.plot(times_vector, params['ipsi'] * AMPLITUDE_SCALE_FACTOR, color=IPSI_COLOR, linestyle='--', label="Ipsi")
    ax.plot(times_vector, params['diff'] * AMPLITUDE_SCALE_FACTOR, color=params['color_diff'], label="Contra-Ipsi", linewidth=2.5)

    # --- Add shaded error bands (SEM) that match the line colors ---
    if n_subs > 1:
        # Error band for Contra wave
        contra_data_all_subs = subject_data[f'{cond_name.lower()}_contra']
        sem_contra = np.std(contra_data_all_subs, axis=0, ddof=1) / np.sqrt(n_subs)
        mean_contra = params['contra']
        ax.fill_between(times_vector,
                        (mean_contra - sem_contra) * AMPLITUDE_SCALE_FACTOR,
                        (mean_contra + sem_contra) * AMPLITUDE_SCALE_FACTOR,
                        color=CONTRA_COLOR, alpha=0.2, label='_nolegend_')

        # Error band for Ipsi wave
        ipsi_data_all_subs = subject_data[f'{cond_name.lower()}_ipsi']
        sem_ipsi = np.std(ipsi_data_all_subs, axis=0, ddof=1) / np.sqrt(n_subs)
        mean_ipsi = params['ipsi']
        ax.fill_between(times_vector,
                        (mean_ipsi - sem_ipsi) * AMPLITUDE_SCALE_FACTOR,
                        (mean_ipsi + sem_ipsi) * AMPLITUDE_SCALE_FACTOR,
                        color=IPSI_COLOR, alpha=0.2, label='_nolegend_')

        # Error band for Difference wave
        diff_data_all_subs = subject_data[f'{cond_name.lower()}_diff']
        sem_diff = np.std(diff_data_all_subs, axis=0, ddof=1) / np.sqrt(n_subs)
        mean_diff = params['diff']
        ax.fill_between(times_vector,
                        (mean_diff - sem_diff) * AMPLITUDE_SCALE_FACTOR,
                        (mean_diff + sem_diff) * AMPLITUDE_SCALE_FACTOR,
                        color=params['color_diff'], alpha=0.2, label='_nolegend_')

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.set_title(f"{cond_name} Lateralization (N={n_subs})")
    ax.set_ylabel("Amplitude [µV]")
    ax.set_xlabel("Time [s]")
    sns.despine(ax=ax)

    # Add significance lines for clusters
    if cluster_results.get(cond_name) and t_thresh_cluster:
        res = cluster_results[cond_name]
        significant_cluster_plotted = False
        for i_c, cl_mask in enumerate(res['clusters']):
            # cl_mask is already the boolean mask from out_type="mask"
            if res['p_values'][i_c] < ALPHA_STAT_CLUSTER:
                cluster_times_sig = times_vector[cl_mask]
                if len(cluster_times_sig) > 0:
                    ax.hlines(y=SIG_LINE_Y_OFFSET,
                              xmin=cluster_times_sig[0], xmax=cluster_times_sig[-1],
                              color=params['color_diff'], linewidth=SIG_LINE_LW,
                              alpha=SIG_LINE_ALPHA,
                              label='Sig. Cluster' if not significant_cluster_plotted else '_nolegend_')
                    significant_cluster_plotted = True
        if significant_cluster_plotted: # Update legend if sig lines were added
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right', frameon=True)

fig_erp.tight_layout()
fig_erp.canvas.draw_idle()

print("\n--- P-values for Significant ERP Time Clusters ---")
for cond_name in ['Target', 'Distractor']:
    if cluster_results.get(cond_name):
        res = cluster_results[cond_name]
        significant_clusters_found = False

        # Iterate through each cluster found for the condition
        for i, p_val in enumerate(res['p_values']):
            # Check if the cluster's p-value is below the significance threshold
            if p_val < ALPHA_STAT_CLUSTER:
                significant_clusters_found = True

                # Get the time points for this specific significant cluster
                cluster_mask = res['clusters'][i]
                cluster_times = times_vector[cluster_mask]

                start_time_ms = cluster_times[0] * 1000
                end_time_ms = cluster_times[-1] * 1000

                print(f"  {cond_name} -> "
                      f"Time Window: {start_time_ms:.0f}-{end_time_ms:.0f} ms, "
                      f"p-value: {p_val:.4f}")

        if not significant_clusters_found:
            print(f"  {cond_name}: No significant time clusters found.")
    else:
        print(f"  {cond_name}: No cluster analysis results available.")

# --- Sensor-Space Cluster Permutation Test ---
significant_sensors_masks = {'Target': None, 'Distractor': None}  # To store masks for plotting

if PLOT_SIGNIFICANT_SENSORS_TOPO and epochs_info and n_subs >= 2 and t_thresh_cluster is not None:
    print(f"\n--- Running Sensor-Space Cluster Permutation Tests (N={n_subs}) ---")
    try:
        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(epochs_info, ch_type='eeg')
        # Sanity check for channel names, though usually they match if epochs_info is consistent
        if not np.array_equal(ch_names_adj, epochs_info.ch_names):
            print("  Warning: Channel names from adjacency matrix differ from epochs_info.ch_names.")
            # This might happen if some channels in epochs_info have no position.
            # The adjacency matrix will be for channels with positions.
            # We need to ensure data_for_test aligns with ch_names_adj if they differ.
            # For now, assume they align or that permutation_cluster_1samp_test handles it via adjacency.
    except Exception as e_adj:
        print(f"  Could not create sensor adjacency matrix: {e_adj}. Skipping sensor cluster tests.")
        adjacency = None

    if adjacency is not None:
        for cond_type, time_window in [("Target", N2AC_TOPO_CLUSTER_WINDOW),
                                       ("Distractor", PD_TOPO_CLUSTER_WINDOW)]:
            print(
                f"  Processing {cond_type} difference topographies in window {time_window[0] * 1000:.0f}-{time_window[1] * 1000:.0f} ms")

            # subject_data[..._diff_topo] is (n_subs, n_channels, n_times)
            subject_topo_data_allchans = subject_data[f'{cond_type.lower()}_diff_topo']

            if subject_topo_data_allchans.shape[0] < 2:
                print(
                    f"    Not enough data for {cond_type} sensor cluster test (N={subject_topo_data_allchans.shape[0]}). Skipping.")
                continue
            if subject_topo_data_allchans.shape[1] != len(epochs_info.ch_names):
                print(
                    f"    Channel count mismatch for {cond_type} topo data ({subject_topo_data_allchans.shape[1]}) vs epochs_info ({len(epochs_info.ch_names)}). Skipping.")
                continue

            # Find time indices for the window
            t_start_idx = np.argmin(np.abs(times_vector - time_window[0]))
            t_end_idx = np.argmin(np.abs(times_vector - time_window[1]))

            if t_start_idx >= t_end_idx:  # Ensure window is valid and has some duration
                print(
                    f"    Invalid time window for {cond_type} ({time_window}), or window too small (indices: {t_start_idx}-{t_end_idx}). Skipping.")
                continue

            print(f"    Averaging data from time index {t_start_idx} to {t_end_idx}")

            # Average data within this time window for each subject
            # Data shape becomes (n_subjects, n_channels)
            data_for_test = np.mean(subject_topo_data_allchans[:, :, t_start_idx:t_end_idx + 1], axis=2)

            # Check for all-NaN slices which can cause issues with stats
            if np.all(np.isnan(data_for_test)):
                print(f"    Data for {cond_type} in window {time_window} is all NaN. Skipping cluster test.")
                continue

            try:
                # Note: permutation_cluster_1samp_test uses a t-test internally.
                # threshold=t_thresh_cluster (calculated earlier for 1D ERPs)
                t_obs_spatial, clusters_spatial, cluster_pv_spatial, H0_spatial = \
                    mne.stats.permutation_cluster_1samp_test(
                        data_for_test,
                        threshold=t_thresh_cluster,
                        n_permutations=N_PERMUTATIONS_CLUSTER,
                        tail=CLUSTER_TAIL,
                        adjacency=adjacency,
                        n_jobs=N_JOBS,
                        seed=SEED,
                        out_type='mask',  # Gives boolean masks for clusters
                        verbose=False
                    )

                # Combine masks of significant clusters
                significant_cluster_indices = np.where(cluster_pv_spatial < ALPHA_SENSOR_CLUSTER)[0]
                if len(significant_cluster_indices) > 0:
                    # final_sensor_mask is True for any sensor part of any significant cluster
                    final_sensor_mask = np.any(np.array(clusters_spatial)[significant_cluster_indices, :], axis=0)
                    significant_sensors_masks[cond_type] = final_sensor_mask

                    sig_ch_names = np.array(epochs_info.ch_names)[final_sensor_mask]
                    print(f"    {cond_type}: Found {len(significant_cluster_indices)} significant sensor cluster(s).")
                    print(f"      Significant sensors ({len(sig_ch_names)}): {', '.join(sig_ch_names)}")
                else:
                    significant_sensors_masks[cond_type] = None  # Explicitly set to None
                    print(f"    {cond_type}: No significant sensor clusters found.")

            except Exception as e_spatial_test:
                print(f"    Error during sensor cluster test for {cond_type}: {e_spatial_test}")
                significant_sensors_masks[cond_type] = None
else:
    if not PLOT_SIGNIFICANT_SENSORS_TOPO:
        print("\nSkipping sensor permutation tests as PLOT_SIGNIFICANT_SENSORS_TOPO is False.")
    elif epochs_info is None:
        print("\nSkipping sensor permutation tests as epochs_info is not available.")
    elif n_subs < 2:
        print("\nSkipping sensor permutation tests: not enough subjects processed (N < 2).")
    elif t_thresh_cluster is None:
        print("\nSkipping sensor permutation tests: t_thresh_cluster not available (likely due to N < 2).")

# --- TOPOGRAPHIES (PLOTTING ONLY SIGNIFICANT TIME INTERVALS) ---

# --- 1. Determine Time Windows for Plotting from Significant ERP Clusters ---
print("\n--- Determining time windows for topomap plotting from significant ERP clusters ---")
topo_plot_windows = {'Target': [], 'Distractor': []}

for cond_name in ['Target', 'Distractor']:
    significant_clusters_found = False
    if cluster_results.get(cond_name) and t_thresh_cluster:
        res = cluster_results[cond_name]
        for i_c, cl_mask in enumerate(res['clusters']):
            if res['p_values'][i_c] < ALPHA_STAT_CLUSTER:
                significant_clusters_found = True
                cluster_times = times_vector[cl_mask]
                if len(cluster_times) > 0:
                    # Add the (start, end) of this significant cluster
                    topo_plot_windows[cond_name].append((cluster_times[0], cluster_times[-1]))

    if significant_clusters_found:
        # Combine overlapping or adjacent windows for cleaner reporting
        # This part is for printing; the plotting logic handles overlaps correctly.
        from itertools import groupby, count
        combined_wins = []
        for win_start, win_end in topo_plot_windows[cond_name]:
            combined_wins.append(f"({win_start*1000:.0f}-{win_end*1000:.0f} ms)")
        print(f"  Found significant time windows for {cond_name}: {', '.join(combined_wins)}")
    else:
        print(f"  No significant time clusters found for {cond_name}.")


# --- 2. Generate Topomap Plots for the Determined Windows ---
# Define a highlight color for the ROI time window
ROI_HIGHLIGHT_COLOR = 'gold'
ROI_HIGHLIGHT_LW = 2.5

for plot_type in ["Target", "Distractor"]:
    windows = topo_plot_windows.get(plot_type, [])
    if not windows:
        print(f"\nSkipping {plot_type} topomaps as no significant time windows were found.")
        continue

    # --- Calculate a consistent vmin/vmax across all clusters for this condition ---
    all_times_for_cond = []
    for start, end in windows:
        times_in_win = np.arange(start, end + TOPO_TIME_STEP, TOPO_TIME_STEP)
        times_in_win = times_in_win[times_in_win <= end]
        all_times_for_cond.extend(list(times_in_win))
    all_times_for_cond = sorted(list(set(all_times_for_cond)))

    # Get grand average data for this condition
    ga_diff_topo_data = ga_data[f'{plot_type.lower()}_diff_topo']
    if np.all(np.isnan(ga_diff_topo_data)):
        print(f"Skipping {plot_type} topomaps as grand average data is all NaN.")
        continue

    all_time_indices = [np.argmin(np.abs(times_vector - t)) for t in all_times_for_cond]
    topo_data_all_times = ga_diff_topo_data[:, all_time_indices] * AMPLITUDE_SCALE_FACTOR
    if np.all(np.isnan(topo_data_all_times)):
        print(f"All selected topo data for {plot_type} is NaN. Setting default vlim for plots.")
        max_abs_val = 1.0
    else:
        max_abs_val = np.nanmax(np.abs(topo_data_all_times))
        if max_abs_val == 0: max_abs_val = 1.0

    vmin, vmax = -max_abs_val, max_abs_val
    print(f'\nCalculated consistent {plot_type} Topomap Limits: vmin = {vmin:.2f} µV, vmax = {vmax:.2f} µV')

    # --- Loop over each significant window (cluster) to create a separate plot ---
    for i_win, (start, end) in enumerate(windows):
        # Generate time points for THIS window only
        times_to_plot_topo = np.arange(start, end + TOPO_TIME_STEP, TOPO_TIME_STEP)
        times_to_plot_topo = times_to_plot_topo[times_to_plot_topo <= end]

        if len(times_to_plot_topo) == 0:
            continue

        n_plots = len(times_to_plot_topo)
        print(f"\nGenerating {n_plots} topomaps for {plot_type} cluster {i_win + 1} ({start*1000:.0f}-{end*1000:.0f} ms)...")

        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig_topo, axes_topo = plt.subplots(n_rows, n_cols, figsize=(max(10, 2.5 * n_cols), 2.5 * n_rows))
        axes_topo = np.array(axes_topo).flatten()

        time_indices_to_plot = [np.argmin(np.abs(times_vector - t)) for t in times_to_plot_topo]

        current_sensor_mask = significant_sensors_masks.get(plot_type, None)
        current_topo_cluster_window = N2AC_TOPO_CLUSTER_WINDOW if plot_type == "Target" else PD_TOPO_CLUSTER_WINDOW

        # This will hold the last plotted image for the colorbar
        im = None

        for i, time_point in enumerate(times_to_plot_topo):
            if i >= len(axes_topo): break
            ax_current = axes_topo[i]
            time_idx = time_indices_to_plot[i]
            data_for_plot = ga_diff_topo_data[:, time_idx] * AMPLITUDE_SCALE_FACTOR

            im, cn = mne.viz.plot_topomap(data_for_plot, epochs_info, axes=ax_current, cmap=TOPO_CMAP,
                                          vlim=(vmin, vmax), show=False, sensors=False, outlines='head',
                                          mask=current_sensor_mask if PLOT_SIGNIFICANT_SENSORS_TOPO else None,
                                          mask_params=SENSOR_MASK_PARAMS if PLOT_SIGNIFICANT_SENSORS_TOPO else None
                                         )
            ax_current.set_title(f"{time_point * 1000:.0f} ms", fontsize=10)

            if PLOT_SIGNIFICANT_SENSORS_TOPO and \
               current_topo_cluster_window[0] <= time_point <= current_topo_cluster_window[1]:
                plt.setp(ax_current.spines.values(), color=ROI_HIGHLIGHT_COLOR, linewidth=ROI_HIGHLIGHT_LW)

        # Clean up unused axes
        for j in range(i + 1, len(axes_topo)):
            fig_topo.delaxes(axes_topo[j])

        # Add colorbar to the figure
        if im:
            fig_topo.subplots_adjust(right=0.85, top=0.88)
            cbar_ax = fig_topo.add_axes([0.88, 0.15, 0.03, 0.7])
            cbar = plt.colorbar(im, cax=cbar_ax, format='%.1f')
            cbar.set_label('Amplitude Difference [µV]')

        # Add title to the figure
        title = (f"Grand Average {plot_type} Difference Wave Topomaps (N={n_subs})\n"
                 f"Significant Cluster {i_win + 1}: {start*1000:.0f} - {end*1000:.0f} ms")
        if PLOT_SIGNIFICANT_SENSORS_TOPO:
            title += f"\n(Time window for sensor cluster test highlighted in {ROI_HIGHLIGHT_COLOR.lower()})"
        fig_topo.suptitle(title, fontsize=14, y=0.98)

        fig_topo.tight_layout(rect=[0, 0, 0.85, 0.92])

# --- (New) Plot a single summary topomap for each condition's sensor test ---
print("\n--- Plotting Summary Topomaps for Sensor Cluster Analysis ---")
if PLOT_SIGNIFICANT_SENSORS_TOPO and n_subs >= 2:
    fig_summary_topo, axes_summary = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    axes_summary = axes_summary.flatten()

    summary_plot_params = {
        'Target': {
            'ax': axes_summary[0],
            'window': N2AC_TOPO_CLUSTER_WINDOW,
            'mask': significant_sensors_masks.get('Target')
        },
        'Distractor': {
            'ax': axes_summary[1],
            'window': PD_TOPO_CLUSTER_WINDOW,
            'mask': significant_sensors_masks.get('Distractor')
        }
    }

    # Find a common color limit for both plots for better comparison
    vmax_summary = 0
    for cond_type, params in summary_plot_params.items():
        ga_topo_data = ga_data[f'{cond_type.lower()}_diff_topo']
        if np.any(np.isnan(ga_topo_data)): continue
        t_start_idx = np.argmin(np.abs(times_vector - params['window'][0]))
        t_end_idx = np.argmin(np.abs(times_vector - params['window'][1]))

        # Average GA data over the window
        data_to_plot = np.mean(ga_topo_data[:, t_start_idx:t_end_idx + 1], axis=1) * AMPLITUDE_SCALE_FACTOR
        current_max = np.nanmax(np.abs(data_to_plot))
        if current_max > vmax_summary:
            vmax_summary = current_max

    if vmax_summary == 0: vmax_summary = 1.0  # Avoid vlim=(0,0)

    im = None  # To hold the last plotted image for the colorbar
    for cond_type, params in summary_plot_params.items():
        ax = params['ax']
        window = params['window']
        mask = params['mask']

        # Get the grand-average data averaged over the specified window
        ga_topo_data = ga_data[f'{cond_type.lower()}_diff_topo']
        if np.all(np.isnan(ga_topo_data)):
            ax.set_title(f"{cond_type} Difference\n(Data not available)")
            ax.axis('off')
            continue

        t_start_idx = np.argmin(np.abs(times_vector - window[0]))
        t_end_idx = np.argmin(np.abs(times_vector - window[1]))
        data_to_plot = np.mean(ga_topo_data[:, t_start_idx:t_end_idx + 1], axis=1) * AMPLITUDE_SCALE_FACTOR

        im, _ = mne.viz.plot_topomap(
            data_to_plot,
            epochs_info,
            axes=ax,
            cmap=TOPO_CMAP,
            vlim=(-vmax_summary, vmax_summary),
            show=False,
            sensors=False,
            outlines='head',
            mask=None,
            mask_params=SENSOR_MASK_PARAMS
        )

        title = f"{cond_type} Difference\n({window[0] * 1000:.0f}-{window[1] * 1000:.0f} ms)"
        ax.set_title(title)

    # Add a single colorbar for the summary figure
    if im:
        fig_summary_topo.subplots_adjust(right=0.85)
        cbar_ax = fig_summary_topo.add_axes([0.88, 0.25, 0.03, 0.5])
        cbar = plt.colorbar(im, cax=cbar_ax)
        # Corrected the unit from [mV/m²] to [µV] to match the data scaling
        cbar.set_label('Amplitude Difference [µV]')
    fig_summary_topo.suptitle(f"Summary of Sensor-Space Cluster Analysis (N={n_subs})", fontsize=14, y=0.98)
    fig_summary_topo.tight_layout(rect=[0, 0, 0.85, 0.92])
