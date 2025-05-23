import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from SPACEPRIME import get_data_path
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_rel, t
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos # Assuming this function is available and works as expected
import seaborn as sns

plt.ion()

# --- Script Parameters ---

# Whether to plot the topographies or not (because that takes a while)
PLOT_TOPOS = True

# Paths
SETTINGS_PATH = os.path.join(get_data_path(), "settings")
MONTAGE_FNAME = "CACS-64_NO_REF.bvef" # Relative to SETTINGS_PATH

# ROIs for Contra/Ipsi Calculation
# Using C3/C4 as in the original script. Adjust if more channels are needed.
# Example: LEFT_ROI_DISTRACTOR = ['P7', 'PO7', 'O1']
LEFT_ROI_DISTRACTOR = ['C3']
RIGHT_ROI_DISTRACTOR = ['C4']
LEFT_ROI_TARGET = ['C3']
RIGHT_ROI_TARGET = ['C4']

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = 0.0, 0.7  # Seconds
AMPLITUDE_SCALE_FACTOR = 1e6      # Volts to Microvolts for plotting

# Cluster Permutation Test Parameters
N_JOBS = 5
SEED = 42
N_PERMUTATIONS_CLUSTER = 10000     # Number of permutations (e.g., 5000-10000 for publication)
ALPHA_STAT_CLUSTER = 0.05         # Significance level for cluster p-values and for forming clusters
CLUSTER_TAIL = 0                  # 0 for two-tailed, 1 for right-tailed, -1 for left-tailed

# Significance Line Plotting Parameters (for ERP plots)
SIG_LINE_Y_OFFSET = -0.01          # Y-offset in µV (data units) from y=0 for the significance lines
SIG_LINE_LW = 4                   # Linewidth for significance lines
SIG_LINE_ALPHA = 0.6              # Alpha for significance lines
# Color will be matched to the difference wave color

# Topomap Plotting Parameters
TOPO_START_TIME = 0.0             # Start time for topomap sequence (seconds)
TOPO_END_TIME = 0.7               # End time for topomap sequence (seconds)
TOPO_TIME_STEP = 0.05             # Time step for topomap sequence (seconds)
TOPO_CMAP = 'RdBu_r'              # Colormap for topographies

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

# --- Subject Loop ---
for subject_id_num in subject_ids:
    subject_str = f"sub-{subject_id_num}" # Assuming subject_ids are numbers like 1, 2, etc.
                                      # If subject_ids are already "sub-XX", then just use subject_id_num
    print(f"\n--- Processing Subject: {subject_str} ---")
    try:
        # Adjust glob pattern if subject_ids are already "sub-XX"
        epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject_id_num}", "eeg", f"sub-{subject_id_num}_task-spaceprime-epo.fif")
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for {subject_str} using pattern: {epoch_file_pattern}. Skipping.")
            continue
        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for {subject_str}: {e}. Skipping.")
        continue

    epochs_sub.crop(EPOCH_TMIN, EPOCH_TMAX)

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
        subject_data['target_diff_topo'].append(target_diff_topo_sub_arr)
        subject_data['distractor_diff_topo'].append(distractor_diff_topo_sub_arr)
        processed_subjects.append(subject_id_num)

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
    'Target': {'ax': ax_erp[0], 'contra': ga_data['target_contra'], 'ipsi': ga_data['target_ipsi'], 'diff': ga_data['target_diff'], 'color_diff': 'green'},
    'Distractor': {'ax': ax_erp[1], 'contra': ga_data['distractor_contra'], 'ipsi': ga_data['distractor_ipsi'], 'diff': ga_data['distractor_diff'], 'color_diff': 'darkorange'}
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


for cond_name, params in erp_plot_params.items():
    ax = params['ax']
    ax.plot(times_vector, params['contra'] * AMPLITUDE_SCALE_FACTOR, color="r", label="Contra")
    ax.plot(times_vector, params['ipsi'] * AMPLITUDE_SCALE_FACTOR, color="b", label="Ipsi")
    ax.plot(times_vector, params['diff'] * AMPLITUDE_SCALE_FACTOR, color=params['color_diff'], label="Contra-Ipsi", linewidth=2)
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


# Optional: T-stats on twin axes (can be commented out if plots are too busy)
if n_subs >=2:
    t_target_paired, _ = ttest_rel(subject_data['target_contra'], subject_data['target_ipsi'], axis=0, nan_policy='omit')
    t_distractor_paired, _ = ttest_rel(subject_data['distractor_contra'], subject_data['distractor_ipsi'], axis=0, nan_policy='omit')

    twin1 = ax_erp[0].twinx()
    twin1.set_ylabel("Paired T-Value", color="purple", alpha=0.7)
    twin1.tick_params(axis='y', labelcolor="purple", color="purple", labelsize='small')
    twin1.plot(times_vector, t_target_paired, color="purple", linestyle=":", alpha=0.5, linewidth=1)
    sns.despine(ax=twin1, right=False, left=True) # Despine original right, keep new one
    ax_erp[0].spines['right'].set_visible(False) # Hide original right spine

    twin2 = ax_erp[1].twinx()
    twin2.sharey(twin1)
    twin2.tick_params(axis='y', labelcolor="purple", color="purple", labelsize='small')
    twin2.plot(times_vector, t_distractor_paired, color="purple", linestyle=":", alpha=0.5, linewidth=1)
    sns.despine(ax=twin2, right=False, left=True)
    ax_erp[1].spines['right'].set_visible(False)

fig_erp.tight_layout()
fig_erp.canvas.draw_idle()


# --- TOPOGRAPHIES ---
times_to_plot_topo = np.arange(TOPO_START_TIME, TOPO_END_TIME + TOPO_TIME_STEP, TOPO_TIME_STEP)

for plot_type in ["Target", "Distractor"]:
    ga_diff_topo_data = ga_data[f'{plot_type.lower()}_diff_topo']
    if np.all(np.isnan(ga_diff_topo_data)): # Skip if all data is NaN
        print(f"Skipping {plot_type} topomaps as grand average data is all NaN.")
        continue

    n_plots = len(times_to_plot_topo)
    if n_plots == 0: continue

    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig_topo, axes_topo = plt.subplots(n_rows, n_cols, figsize=(max(10, 2.5 * n_cols), 2.5 * n_rows))
    axes_topo = np.array(axes_topo).flatten() # Ensure it's always an array

    time_indices_to_plot = [np.argmin(np.abs(times_vector - t)) for t in times_to_plot_topo]

    # Determine consistent vmin and vmax
    topo_data_all_selected_times = ga_diff_topo_data[:, time_indices_to_plot] * AMPLITUDE_SCALE_FACTOR
    max_abs_val = np.nanmax(np.abs(topo_data_all_selected_times)) if not np.all(np.isnan(topo_data_all_selected_times)) else 1.0
    vmin, vmax = -max_abs_val, max_abs_val
    print(f'{plot_type} Topomap Limits: vmin = {vmin:.2f} µV, vmax = {vmax:.2f} µV')

    for i, time_point in enumerate(times_to_plot_topo):
        if i >= len(axes_topo): break # Should not happen with correct n_rows/n_cols
        time_idx = time_indices_to_plot[i]
        data_for_plot = ga_diff_topo_data[:, time_idx] * AMPLITUDE_SCALE_FACTOR

        im, cn = mne.viz.plot_topomap(data_for_plot, epochs_info, axes=axes_topo[i], cmap=TOPO_CMAP,
                                      vlim=(vmin, vmax), show=False, sensors=False, outlines='head')
        axes_topo[i].set_title(f"{time_point * 1000:.0f} ms", fontsize=10)

    for j in range(i + 1, len(axes_topo)):
        fig_topo.delaxes(axes_topo[j])

    fig_topo.subplots_adjust(right=0.85, top=0.90)
    cbar_ax = fig_topo.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, format='%.1f')
    cbar.set_label('Amplitude Difference [µV]')
    fig_topo.suptitle(f"Grand Average {plot_type} Difference Wave Topomaps (N={n_subs})", fontsize=14)
    fig_topo.tight_layout(rect=[0, 0, 0.85, 0.92]) # Adjust rect for suptitle and colorbar

plt.show(block=True)