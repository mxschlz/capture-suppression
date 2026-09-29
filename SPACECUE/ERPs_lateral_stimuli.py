import sys
import os
import mne
import numpy as np
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
from scipy.stats import t
import seaborn as sns
from mne.channels import find_ch_adjacency

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from SPACECUE import load_concatenated_epochs
from SPACECUE.subjects import subject_ids
from plotting import difference_topos # Assuming this function is available and works as expected

sns.set_theme(context="talk", style="ticks")
plt.ion()

PLOTS_DIR = os.path.join(project_root, "spacecue", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 2. General Script Parameters ---
# Whether to plot the topographies or not (because that takes a while)
PLOT_TOPOS = False

# ROIs for Contra/Ipsi Calculation
LEFT_ROI_DISTRACTOR = ["C3"]
RIGHT_ROI_DISTRACTOR = ["C4"]
LEFT_ROI_TARGET = ["C3"]
RIGHT_ROI_TARGET = ["C4"]

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.7  # Seconds
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
N2AC_TOPO_CLUSTER_WINDOW = (0.2, 0.4) # Example: 220-380ms for N2ac-like activity
PD_TOPO_CLUSTER_WINDOW = (0.3, 0.4)   # Example: 290-380ms for Pd-like activity
# Alpha for sensor cluster p-values (can be same as ALPHA_STAT_CLUSTER)
ALPHA_SENSOR_CLUSTER = 0.05
# Mask parameters for plotting significant sensors
SENSOR_MASK_PARAMS = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                          linewidth=0, markersize=5, markeredgewidth=1.5)
# --- End of Parameters ---
# --- Data Storage ---
conditions = ['high-probability', 'low-probability']
subject_data = {
    prob_cond: {
        'target_contra': [], 'target_ipsi': [], 'target_diff': [],
        'distractor_contra': [], 'distractor_ipsi': [], 'distractor_diff': [],
        'target_diff_topo': [], 'distractor_diff_topo': []
    } for prob_cond in conditions
}
processed_subjects = {prob_cond: [] for prob_cond in conditions}
subject_ages = {}
times_vector = None # To be populated from the first subject
epochs_info = None  # To be populated from the first subject

# --- Load and Preprocess Data ---
print("--- Loading and Preprocessing Data ---")
epochs = load_concatenated_epochs("spacecue").crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Original number of trials: {len(epochs)}")

print(f"Final number of trials after preprocessing: {len(epochs)}")
# --- End of Preprocessing ---
# use standard montage (the existing montage of the epochs is wrong)
montage = mne.channels.make_standard_montage("easycap-M1")
epochs.set_montage(montage)

fig_init = epochs.average().plot_joint(show=False)
out_init = os.path.join(PLOTS_DIR, "erp_lateral_initial_joint.png")
fig_init.savefig(out_init, dpi=300, bbox_inches='tight')
print(f"Saved initial joint plot to: {out_init}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_init)

# --- Subject Loop ---
for subject_id_num in subject_ids:
    subject_str = f"sci-{subject_id_num}" # Assuming subject_ids are numbers like 1, 2, etc.
                                      # If subject_ids are already "sub-XX", then just use subject_id_num
    print(f"\n--- Processing Subject: {subject_str} ---")
    try:
        epochs_sub = epochs[f"`Subject ID` == {subject_id_num}"]
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for {subject_str}: {e}. Skipping.")
        continue

    if times_vector is None:
        times_vector = epochs_sub.times.copy()
        epochs_info = epochs_sub.info.copy()

    if epochs_sub.metadata is not None and 'Age' in epochs_sub.metadata.columns:
        subject_ages[subject_id_num] = float(epochs_sub.metadata['Age'].dropna().iloc[0])
    else:
        print(f"  Warning: 'Age' not found in metadata for {subject_str}. Skipping.")
        continue

    if 'DistractorProb' not in epochs_sub.metadata.columns:
        print(f"  Warning: 'DistractorProb' not found in metadata for {subject_str}. Skipping.")
        continue

    for prob_cond in conditions:
        try:
            epochs_prob = epochs_sub[epochs_sub.metadata['DistractorProb'] == prob_cond]
            print(f"  [{prob_cond}] Loaded {len(epochs_prob)} epochs.")
            if len(epochs_prob) == 0:
                print(f"  No trials for {prob_cond} condition in {subject_str}. Skipping.")
                continue

            all_conds_prob = list(epochs_prob.event_id.keys())

            # TARGETS
            if prob_cond == 'high-probability':
                target_left_pattern = "Target-1-Singleton-3"
                target_right_pattern = "Target-3-Singleton-1"
            else:
                target_left_pattern = "Target-1-Singleton-3"
                target_right_pattern = "Target-3-Singleton-1"

            left_target_epochs = epochs_prob[[x for x in all_conds_prob if target_left_pattern in x]].copy()
            right_target_epochs = epochs_prob[[x for x in all_conds_prob if target_right_pattern in x]].copy()
            print(f"    Target trials: {len(left_target_epochs)} left, {len(right_target_epochs)} right")
            if len(left_target_epochs) == 0 or len(right_target_epochs) == 0:
                print(f"    Skipping {subject_str} - {prob_cond} due to zero trials in one of the target conditions.")
                continue

            # DISTRACTORS
            left_distractor_epochs = epochs_prob[[x for x in all_conds_prob if "Target-2-Singleton-1" in x]].copy()
            right_distractor_epochs = epochs_prob[[x for x in all_conds_prob if "Target-2-Singleton-3" in x]].copy()
            print(f"    Distractor trials: {len(left_distractor_epochs)} left, {len(right_distractor_epochs)} right")
            if len(left_distractor_epochs) == 0 or len(right_distractor_epochs) == 0:
                print(f"    Skipping {subject_str} - {prob_cond} due to zero trials in one of the distractor conditions.")
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
            if PLOT_TOPOS:
                target_diff_topo_dict_sub, distractor_diff_topo_dict_sub = difference_topos(epochs_prob, montage)

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
                            print(f"    Warning: Ch {ch_name} data shape mismatch ({ch_data.shape} vs {(n_times_current,)}) for target topo, {subject_str} - {prob_cond}.")

                distractor_diff_topo_sub_arr = np.full((n_channels, n_times_current), np.nan)
                for i, ch_name in enumerate(channel_order):
                    if ch_name in distractor_diff_topo_dict_sub:
                        ch_data = np.asarray(distractor_diff_topo_dict_sub[ch_name])
                        if ch_data.shape == (n_times_current,):
                            distractor_diff_topo_sub_arr[i, :] = ch_data
                        else:
                            print(f"    Warning: Ch {ch_name} data shape mismatch ({ch_data.shape} vs {(n_times_current,)}) for distractor topo, {subject_str} - {prob_cond}.")

            # Store Subject Results
            subject_data[prob_cond]['target_contra'].append(contra_target_sub)
            subject_data[prob_cond]['target_ipsi'].append(ipsi_target_sub)
            subject_data[prob_cond]['target_diff'].append(diff_target_sub)
            subject_data[prob_cond]['distractor_contra'].append(contra_distractor_sub)
            subject_data[prob_cond]['distractor_ipsi'].append(ipsi_distractor_sub)
            subject_data[prob_cond]['distractor_diff'].append(diff_distractor_sub)
            processed_subjects[prob_cond].append(subject_id_num)
            if PLOT_TOPOS:
                subject_data[prob_cond]['target_diff_topo'].append(target_diff_topo_sub_arr)
                subject_data[prob_cond]['distractor_diff_topo'].append(distractor_diff_topo_sub_arr)

        except Exception as e_proc:
            print(f"    Error during processing for subject {subject_str} - {prob_cond}: {e_proc}. Skipping condition for this subject.")
            continue


for prob_cond in conditions:
    print(f"\n--- Successfully processed {len(processed_subjects[prob_cond])} subjects for {prob_cond}: {processed_subjects[prob_cond]} ---")

if not any(processed_subjects[prob_cond] for prob_cond in conditions):
    raise RuntimeError("No subjects were successfully processed for any condition. Cannot continue.")
if times_vector is None or epochs_info is None:
    raise RuntimeError("Essential data (times_vector or epochs_info) not populated. Check subject loop.")

# Convert lists to numpy arrays
for prob_cond in conditions:
    for key in subject_data[prob_cond]:
        if len(subject_data[prob_cond][key]) == 0:  # If list is empty
            if key.endswith('_topo'):
                subject_data[prob_cond][key] = np.empty((0, len(epochs_info.ch_names), len(times_vector)))
            else:
                subject_data[prob_cond][key] = np.empty((0, len(times_vector)))
        else:
            try:
                subject_data[prob_cond][key] = np.array(subject_data[prob_cond][key])
            except Exception as e_conv:
                print(f"Could not convert {key} for {prob_cond} to numpy array. Error: {e_conv}")
                raise e_conv


# --- Group Level Analysis ---
ga_data = {prob_cond: {} for prob_cond in conditions}
for prob_cond in conditions:
    for key in ['target_contra', 'target_ipsi', 'target_diff',
                'distractor_contra', 'distractor_ipsi', 'distractor_diff',
                'target_diff_topo', 'distractor_diff_topo']:
        if subject_data[prob_cond][key].shape[0] > 0: # Check if there are subjects for this key
            ga_data[prob_cond][key] = np.nanmean(subject_data[prob_cond][key], axis=0) # Use nanmean for topo data
        else:
            # Create empty array with correct dimensions if no subjects
            if key.endswith('_topo'):
                ga_data[prob_cond][key] = np.empty((len(epochs_info.ch_names), len(times_vector))) * np.nan
            else:
                ga_data[prob_cond][key] = np.empty(len(times_vector)) * np.nan


# --- Plot Grand Average ERPs with Cluster Significance ---
fig_erp, ax_erp = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

erp_plot_params = []
for cond_idx, cond_name in enumerate(['Target', 'Distractor']):
    color_waves = 'green' if cond_name == 'Target' else 'red'
    for prob_idx, prob_cond in enumerate(conditions):
        ax = ax_erp[cond_idx, prob_idx]
        erp_plot_params.append({
            'cond_name': cond_name,
            'prob_cond': prob_cond,
            'ax': ax,
            'contra': ga_data[prob_cond][f'{cond_name.lower()}_contra'],
            'ipsi': ga_data[prob_cond][f'{cond_name.lower()}_ipsi'],
            'diff': ga_data[prob_cond][f'{cond_name.lower()}_diff'],
            'color_diff': 'black',
            'color_waves': color_waves
        })

# Group Level Statistics (Cluster Permutation Test)
cluster_results = {prob_cond: {'Target': None, 'Distractor': None} for prob_cond in conditions}
t_thresh_cluster = {prob_cond: None for prob_cond in conditions}

for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    if n_subs >= 2: # Need at least 2 subjects for t-test based threshold and cluster test
        deg_f = n_subs - 1
        t_thresh_cluster[prob_cond] = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, deg_f) if deg_f > 0 else None # Threshold for forming clusters
        print(f"\n--- Running Cluster Permutation Tests for {prob_cond} (N={n_subs}) ---")
        if t_thresh_cluster[prob_cond]:
            print(f"Using cluster-forming t-threshold: {t_thresh_cluster[prob_cond]:.3f} (df={deg_f}, alpha={ALPHA_STAT_CLUSTER}, two-tailed)")
        else:
            print(f"Cannot calculate t-threshold for cluster forming for {prob_cond} (df <=0). Skipping cluster tests.")

        if t_thresh_cluster[prob_cond]:
            for cond_type in ['Target', 'Distractor']:
                X_diff = subject_data[prob_cond][f'{cond_type.lower()}_diff']
                if X_diff.shape[0] < 2 : # Not enough subjects for this specific condition
                    print(f"Not enough data for {cond_type} ({prob_cond}) cluster test (N={X_diff.shape[0]}). Skipping.")
                    continue

                t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
                    X_diff, threshold=t_thresh_cluster[prob_cond], n_permutations=N_PERMUTATIONS_CLUSTER,
                    tail=CLUSTER_TAIL, n_jobs=N_JOBS, out_type="mask", seed=SEED, verbose=False
                )
                cluster_results[prob_cond][cond_type] = {'t_obs': t_obs, 'clusters': clusters, 'p_values': cluster_pv}
                print(f"  {cond_type} ({prob_cond}): Found {np.sum(cluster_pv < ALPHA_STAT_CLUSTER)} significant cluster(s).")
    else:
        print(f"\nSkipping cluster permutation tests for {prob_cond}: not enough subjects processed (N < 2).")


for params in erp_plot_params:
    cond_name = params['cond_name']
    prob_cond = params['prob_cond']
    ax = params['ax']
    n_subs = len(processed_subjects[prob_cond])

    # Plot mean ERP waves with new, consistent colors
    ax.plot(times_vector, params['contra'] * AMPLITUDE_SCALE_FACTOR, color=params['color_waves'], linestyle='-', label="Contra")
    ax.plot(times_vector, params['ipsi'] * AMPLITUDE_SCALE_FACTOR, color=params['color_waves'], linestyle='--', label="Ipsi")
    ax.plot(times_vector, params['diff'] * AMPLITUDE_SCALE_FACTOR, color=params['color_diff'], label="Contra-Ipsi", linewidth=2.5)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.set_title(f"{cond_name} lateral - {prob_cond} (N={n_subs})")
    ax.set_ylabel("Amplitude [µV]")
    ax.set_xlabel("Time [s]")
    sns.despine(ax=ax)

    # Add significance lines for clusters
    res = cluster_results[prob_cond].get(cond_name)
    t_thresh = t_thresh_cluster[prob_cond]
    if res and t_thresh:
        significant_cluster_plotted = False
        for i_c, cl_mask in enumerate(res['clusters']):
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
out_erp = os.path.join(PLOTS_DIR, "erp_lateral_grand_average.png")
fig_erp.savefig(out_erp, dpi=300, bbox_inches='tight')
print(f"Saved grand average ERP plot to: {out_erp}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_erp)

# --- Plot Age Cohort Comparison of Lateralized Potentials (Young vs. Old) ---
print("\n--- Running Age Cohort Comparisons for Lateralized Potentials (Young vs. Old) ---")
fig_age, ax_age = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

# Colors for cohorts (consistent with EEG age-split plots)
cohort_colors = {'Young': '#1F77B4', 'Old': '#FF7F0E'}

for cond_idx, cond_name in enumerate(['Target', 'Distractor']):
    for prob_idx, prob_cond in enumerate(conditions):
        ax = ax_age[cond_idx, prob_idx]
        
        sub_ids = processed_subjects[prob_cond]
        young_indices = [i for i, sub in enumerate(sub_ids) if subject_ages.get(sub, 0) < 36]
        old_indices = [i for i, sub in enumerate(sub_ids) if subject_ages.get(sub, 0) >= 36]
        
        young_diffs = subject_data[prob_cond][f'{cond_name.lower()}_diff'][young_indices]
        old_diffs = subject_data[prob_cond][f'{cond_name.lower()}_diff'][old_indices]
        
        n_young = len(young_indices)
        n_old = len(old_indices)
        
        mean_young = np.nanmean(young_diffs, axis=0) * AMPLITUDE_SCALE_FACTOR
        sem_young = (np.nanstd(young_diffs, axis=0) / np.sqrt(n_young)) * AMPLITUDE_SCALE_FACTOR if n_young > 1 else np.zeros_like(mean_young)
        
        mean_old = np.nanmean(old_diffs, axis=0) * AMPLITUDE_SCALE_FACTOR
        sem_old = (np.nanstd(old_diffs, axis=0) / np.sqrt(n_old)) * AMPLITUDE_SCALE_FACTOR if n_old > 1 else np.zeros_like(mean_old)
        
        # Run cluster permutation test between Young vs. Old if both groups have enough subjects
        sig_windows = []
        if n_young >= 2 and n_old >= 2:
            from mne.stats import permutation_cluster_test
            F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
                [young_diffs, old_diffs], n_permutations=N_PERMUTATIONS_CLUSTER, tail=0,
                threshold=None, out_type='indices', verbose=False, seed=SEED
            )
            
            is_sig = np.zeros(len(times_vector), dtype=bool)
            if clusters:
                for cl, p_val in zip(clusters, cluster_pv):
                    if p_val < ALPHA_STAT_CLUSTER:
                        is_sig[cl[0]] = True
                        sig_windows.append(f"{times_vector[cl[0][0]]*1000:.0f}-{times_vector[cl[0][-1]]*1000:.0f} ms (p={p_val:.4f})")
            
            # Shade significant difference windows (p < 0.05)
            for i in range(len(times_vector)):
                if is_sig[i]:
                    ax.axvspan(times_vector[i], times_vector[i] + (times_vector[1] - times_vector[0]),
                               color='purple', alpha=0.08, lw=0)
                        
        # Plot young
        ax.plot(times_vector, mean_young, color=cohort_colors['Young'], lw=2.5, label=f"Young (N={n_young})")
        ax.fill_between(times_vector, mean_young - sem_young, mean_young + sem_young, color=cohort_colors['Young'], alpha=0.15)
        
        # Plot old
        ax.plot(times_vector, mean_old, color=cohort_colors['Old'], lw=2.5, label=f"Old (N={n_old})")
        ax.fill_between(times_vector, mean_old - sem_old, mean_old + sem_old, color=cohort_colors['Old'], alpha=0.15)
        
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle=':', linewidth=0.8)
        ax.legend(loc='upper right')
        ax.set_title(f"{cond_name} Diff - {prob_cond} (Young vs Old)")
        ax.set_ylabel("Amplitude Difference [µV]")
        ax.set_xlabel("Time [s]")
        sns.despine(ax=ax)
        
        if sig_windows:
            print(f"  {cond_name} ({prob_cond}) Young vs Old Sig. Windows: {', '.join(sig_windows)}")
            
fig_age.suptitle("Age Cohort Comparison of Lateralized Potentials (Young vs. Old)", fontsize=16, fontweight='bold', y=0.98)
fig_age.tight_layout()
fig_age.canvas.draw_idle()
out_age = os.path.join(PLOTS_DIR, "erp_lateral_age_split.png")
fig_age.savefig(out_age, dpi=300, bbox_inches='tight')
print(f"Saved age-split ERP plot to: {out_age}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_age)

print("\n--- P-values for Significant ERP Time Clusters ---")
for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    print(f"\nCondition: {prob_cond} (N={n_subs})")
    for cond_name in ['Target', 'Distractor']:
        res = cluster_results[prob_cond].get(cond_name)
        if res:
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

                    # --- Calculate Effect Sizes for Reporting ---
                    # 1. T-values within the cluster
                    t_obs_cluster = res['t_obs'][cluster_mask]
                    mean_t = np.mean(t_obs_cluster)
                    peak_t = t_obs_cluster[np.argmax(np.abs(t_obs_cluster))]

                    # 2. Cohen's d from T-value (d = t / sqrt(N))
                    cohens_d = mean_t / np.sqrt(n_subs)

                    print(f"  {cond_name} -> "
                          f"Time Window: {start_time_ms:.0f}-{end_time_ms:.0f} ms, "
                          f"p-value: {p_val:.4f}, "
                          f"Mean t: {mean_t:.2f}, Peak t: {peak_t:.2f}, Cohen's d: {cohens_d:.2f}")

            if not significant_clusters_found:
                print(f"  {cond_name}: No significant time clusters found.")
        else:
            print(f"  {cond_name}: No cluster analysis results available.")

# --- Sensor-Space Cluster Permutation Test ---
significant_sensors_masks = {prob_cond: {'Target': None, 'Distractor': None} for prob_cond in conditions}

for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    t_thresh = t_thresh_cluster[prob_cond]
    if PLOT_SIGNIFICANT_SENSORS_TOPO and epochs_info and n_subs >= 2 and t_thresh is not None:
        print(f"\n--- Running Sensor-Space Cluster Permutation Tests for {prob_cond} (N={n_subs}) ---")
        try:
            adjacency, ch_names_adj = mne.channels.find_ch_adjacency(epochs_info, ch_type='eeg')
            # Sanity check for channel names, though usually they match if epochs_info is consistent
            if not np.array_equal(ch_names_adj, epochs_info.ch_names):
                print(f"  Warning: Channel names from adjacency matrix differ from epochs_info.ch_names for {prob_cond}.")
        except Exception as e_adj:
            print(f"  Could not create sensor adjacency matrix for {prob_cond}: {e_adj}. Skipping sensor cluster tests.")
            adjacency = None

        if adjacency is not None:
            for cond_type, time_window in [("Target", N2AC_TOPO_CLUSTER_WINDOW),
                                           ("Distractor", PD_TOPO_CLUSTER_WINDOW)]:
                print(
                    f"  Processing {cond_type} ({prob_cond}) difference topographies in window {time_window[0] * 1000:.0f}-{time_window[1] * 1000:.0f} ms")

                # subject_data[..._diff_topo] is (n_subs, n_channels, n_times)
                subject_topo_data_allchans = subject_data[prob_cond][f'{cond_type.lower()}_diff_topo']

                if subject_topo_data_allchans.shape[0] < 2:
                    print(
                        f"    Not enough data for {cond_type} ({prob_cond}) sensor cluster test (N={subject_topo_data_allchans.shape[0]}). Skipping.")
                    continue
                if subject_topo_data_allchans.shape[1] != len(epochs_info.ch_names):
                    print(
                        f"    Channel count mismatch for {cond_type} ({prob_cond}) topo data ({subject_topo_data_allchans.shape[1]}) vs epochs_info ({len(epochs_info.ch_names)}). Skipping.")
                    continue

                # Find time indices for the window
                t_start_idx = np.argmin(np.abs(times_vector - time_window[0]))
                t_end_idx = np.argmin(np.abs(times_vector - time_window[1]))

                if t_start_idx >= t_end_idx:  # Ensure window is valid and has some duration
                    print(
                        f"    Invalid time window for {cond_type} ({prob_cond}) ({time_window}), or window too small. Skipping.")
                    continue

                print(f"    Averaging data from time index {t_start_idx} to {t_end_idx}")

                # Average data within this time window for each subject
                # Data shape becomes (n_subjects, n_channels)
                data_for_test = np.mean(subject_topo_data_allchans[:, :, t_start_idx:t_end_idx + 1], axis=2)

                # Check for all-NaN slices which can cause issues with stats
                if np.all(np.isnan(data_for_test)):
                    print(f"    Data for {cond_type} ({prob_cond}) in window {time_window} is all NaN. Skipping cluster test.")
                    continue

                try:
                    t_obs_spatial, clusters_spatial, cluster_pv_spatial, H0_spatial = \
                        mne.stats.permutation_cluster_1samp_test(
                            data_for_test,
                            threshold=t_thresh,
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
                        significant_sensors_masks[prob_cond][cond_type] = final_sensor_mask

                        sig_ch_names = np.array(epochs_info.ch_names)[final_sensor_mask]
                        print(f"    {cond_type} ({prob_cond}): Found {len(significant_cluster_indices)} significant sensor cluster(s).")
                        print(f"      Significant sensors ({len(sig_ch_names)}): {', '.join(sig_ch_names)}")
                    else:
                        significant_sensors_masks[prob_cond][cond_type] = None  # Explicitly set to None
                        print(f"    {cond_type} ({prob_cond}): No significant sensor clusters found.")

                except Exception as e_spatial_test:
                    print(f"    Error during sensor cluster test for {cond_type} ({prob_cond}): {e_spatial_test}")
                    significant_sensors_masks[prob_cond][cond_type] = None

# --- TOPOGRAPHIES (PLOTTING ONLY SIGNIFICANT TIME INTERVALS) ---

for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    t_thresh = t_thresh_cluster[prob_cond]
    # --- 1. Determine Time Windows for Plotting from Significant ERP Clusters ---
    print(f"\n--- Determining time windows for {prob_cond} topomap plotting from significant ERP clusters ---")
    topo_plot_windows = {'Target': [], 'Distractor': []}

    for cond_name in ['Target', 'Distractor']:
        significant_clusters_found = False
        res = cluster_results[prob_cond].get(cond_name)
        if res and t_thresh:
            for i_c, cl_mask in enumerate(res['clusters']):
                if res['p_values'][i_c] < ALPHA_STAT_CLUSTER:
                    significant_clusters_found = True
                    cluster_times = times_vector[cl_mask]
                    if len(cluster_times) > 0:
                        # Add the (start, end) of this significant cluster
                        topo_plot_windows[cond_name].append((cluster_times[0], cluster_times[-1]))

        if significant_clusters_found:
            # Combine overlapping or adjacent windows for cleaner reporting
            from itertools import groupby, count
            combined_wins = []
            for win_start, win_end in topo_plot_windows[cond_name]:
                combined_wins.append(f"({win_start*1000:.0f}-{win_end*1000:.0f} ms)")
            print(f"  Found significant time windows for {cond_name} ({prob_cond}): {', '.join(combined_wins)}")
        else:
            print(f"  No significant time clusters found for {cond_name} ({prob_cond}).")


    # --- 2. Generate Topomap Plots for the Determined Windows ---
    # Define a highlight color for the ROI time window
    ROI_HIGHLIGHT_COLOR = 'gold'
    ROI_HIGHLIGHT_LW = 2.5

    for plot_type in ["Target", "Distractor"]:
        windows = topo_plot_windows.get(plot_type, [])
        if not windows:
            print(f"\nSkipping {plot_type} ({prob_cond}) topomaps as no significant time windows were found.")
            continue

        # --- Calculate a consistent vmin/vmax across all clusters for this condition ---
        all_times_for_cond = []
        for start, end in windows:
            times_in_win = np.arange(start, end + TOPO_TIME_STEP, TOPO_TIME_STEP)
            times_in_win = times_in_win[times_in_win <= end]
            all_times_for_cond.extend(list(times_in_win))
        all_times_for_cond = sorted(list(set(all_times_for_cond)))

        # Get grand average data for this condition
        ga_diff_topo_data = ga_data[prob_cond][f'{plot_type.lower()}_diff_topo']
        if np.all(np.isnan(ga_diff_topo_data)):
            print(f"Skipping {plot_type} ({prob_cond}) topomaps as grand average data is all NaN.")
            continue

        all_time_indices = [np.argmin(np.abs(times_vector - t)) for t in all_times_for_cond]
        topo_data_all_times = ga_diff_topo_data[:, all_time_indices] * AMPLITUDE_SCALE_FACTOR
        if np.all(np.isnan(topo_data_all_times)):
            print(f"All selected topo data for {plot_type} ({prob_cond}) is NaN. Setting default vlim for plots.")
            max_abs_val = 1.0
        else:
            max_abs_val = np.nanmax(np.abs(topo_data_all_times))
            if max_abs_val == 0: max_abs_val = 1.0

        vmin, vmax = -max_abs_val, max_abs_val
        print(f'\nCalculated consistent {plot_type} ({prob_cond}) Topomap Limits: vmin = {vmin:.2f} µV, vmax = {vmax:.2f} µV')

        # --- Loop over each significant window (cluster) to create a separate plot ---
        for i_win, (start, end) in enumerate(windows):
            # Generate time points for THIS window only
            times_to_plot_topo = np.arange(start, end + TOPO_TIME_STEP, TOPO_TIME_STEP)
            times_to_plot_topo = times_to_plot_topo[times_to_plot_topo <= end]

            if len(times_to_plot_topo) == 0:
                continue

            n_plots = len(times_to_plot_topo)
            print(f"\nGenerating {n_plots} topomaps for {plot_type} ({prob_cond}) cluster {i_win + 1} ({start*1000:.0f}-{end*1000:.0f} ms)...")

            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))

            fig_topo, axes_topo = plt.subplots(n_rows, n_cols, figsize=(max(10, 2.5 * n_cols), 2.5 * n_rows))
            axes_topo = np.array(axes_topo).flatten()

            time_indices_to_plot = [np.argmin(np.abs(times_vector - t)) for t in times_to_plot_topo]

            current_sensor_mask = significant_sensors_masks[prob_cond].get(plot_type, None)
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
            title = (f"Grand Average {plot_type} ({prob_cond}) Difference Wave Topomaps (N={n_subs})\n"
                     f"Significant Cluster {i_win + 1}: {start*1000:.0f} - {end*1000:.0f} ms")
            if PLOT_SIGNIFICANT_SENSORS_TOPO:
                title += f"\n(Time window for sensor cluster test highlighted in {ROI_HIGHLIGHT_COLOR.lower()})"
            fig_topo.suptitle(title, fontsize=14, y=0.98)

            fig_topo.tight_layout(rect=[0, 0, 0.85, 0.92])
            cond_str = prob_cond.replace("-", "_")
            out_topo = os.path.join(PLOTS_DIR, f"erp_lateral_topo_{plot_type.lower()}_{cond_str}_cluster_{i_win + 1}.png")
            fig_topo.savefig(out_topo, dpi=200, bbox_inches='tight')
            print(f"Saved topography plot to: {out_topo}")
            if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                plt.show()
            else:
                plt.close(fig_topo)

# --- Plot summary topomaps for sensor cluster analysis ---
print("\n--- Plotting Summary Topomaps for Sensor Cluster Analysis ---")
if PLOT_SIGNIFICANT_SENSORS_TOPO:
    for prob_cond in conditions:
        n_subs = len(processed_subjects[prob_cond])
        if n_subs >= 2:
            fig_summary_topo, axes_summary = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
            axes_summary = axes_summary.flatten()

            summary_plot_params = {
                'Target': {
                    'ax': axes_summary[0],
                    'window': N2AC_TOPO_CLUSTER_WINDOW,
                    'mask': significant_sensors_masks[prob_cond].get('Target')
                },
                'Distractor': {
                    'ax': axes_summary[1],
                    'window': PD_TOPO_CLUSTER_WINDOW,
                    'mask': significant_sensors_masks[prob_cond].get('Distractor')
                }
            }

            # Find a common color limit for both plots for better comparison
            vmax_summary = 0
            for cond_type, params in summary_plot_params.items():
                ga_topo_data = ga_data[prob_cond][f'{cond_type.lower()}_diff_topo']
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
                ga_topo_data = ga_data[prob_cond][f'{cond_type.lower()}_diff_topo']
                if np.all(np.isnan(ga_topo_data)):
                    ax.set_title(f"{cond_type} Difference ({prob_cond})\n(Data not available)")
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
                    mask=mask,
                    mask_params=SENSOR_MASK_PARAMS
                )

                title = f"{cond_type} Difference ({prob_cond})\n({window[0] * 1000:.0f}-{window[1] * 1000:.0f} ms)"
                ax.set_title(title)

            # Add a single colorbar for the summary figure
            if im:
                fig_summary_topo.subplots_adjust(right=0.85)
                cbar_ax = fig_summary_topo.add_axes([0.88, 0.25, 0.03, 0.5])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label('Amplitude Difference [µV]')
            fig_summary_topo.suptitle(f"Summary of Sensor-Space Cluster Analysis for {prob_cond} (N={n_subs})", fontsize=14, y=0.98)
            fig_summary_topo.tight_layout(rect=[0, 0, 0.85, 0.92])
            cond_str = prob_cond.replace("-", "_")
            out_summary_topo = os.path.join(PLOTS_DIR, f"erp_lateral_sensor_cluster_summary_{cond_str}.png")
            fig_summary_topo.savefig(out_summary_topo, dpi=200, bbox_inches='tight')
            print(f"Saved sensor cluster summary plot to: {out_summary_topo}")
            if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                plt.show()
            else:
                plt.close(fig_summary_topo)

# Run mass univariate analysis for each condition
for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    dist_diff_topo = subject_data[prob_cond]["distractor_diff_topo"]
    if n_subs >= 2 and dist_diff_topo.shape[0] >= 2:
        print(f"\n--- Running Mass Univariate Spatio-Temporal Analysis for {prob_cond} (N={n_subs}) ---")
        # 1. define reshaped data
        X = dist_diff_topo.transpose([0, 2, 1])
        # 2. Define the Spatial Connectivity
        connectivity, ch_names = find_ch_adjacency(epochs.info, ch_type='eeg')

        # 3. Run the Mass Univariate Test
        t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=connectivity,
            n_permutations=1000,
            tail=0,  # Two-tailed test
            threshold=None # Uses a standard t-threshold based on p < 0.05
        )

        # 4. Identify Significant Clusters
        significant_points = np.where(cluster_pv < 0.05)[0]

        if len(significant_points) > 0:
            print(f"\n--- Visualizing {len(significant_points)} Significant Spatio-Temporal Clusters for {prob_cond} ---")

            # Determine common scale for all cluster plots based on T-values
            v_max_abs = np.nanmax(np.abs(t_obs))

            for i_clust, clust_idx in enumerate(significant_points):
                # clusters[clust_idx] is a tuple of arrays (time_indices, channel_indices)
                time_inds, ch_inds = clusters[clust_idx]

                # Get time window
                t_min = times_vector[time_inds.min()]
                t_max = times_vector[time_inds.max()]

                # Create mask for significant sensors (union across time)
                mask = np.zeros(len(epochs_info.ch_names), dtype=bool)
                mask[np.unique(ch_inds)] = True

                # Average T-values over the cluster time window
                t_map = np.nanmean(t_obs[time_inds.min():time_inds.max() + 1, :], axis=0)

                # --- Calculate Effect Sizes for Spatio-Temporal Cluster ---
                t_vals_cluster = t_obs[time_inds, ch_inds]
                mean_t_cluster = np.mean(t_vals_cluster)
                peak_t_cluster = t_vals_cluster[np.argmax(np.abs(t_vals_cluster))]

                # Cohen's d from T-value
                cohens_d_st = mean_t_cluster / np.sqrt(X.shape[0])

                fig, ax = plt.subplots(figsize=(5, 5))
                im, _ = mne.viz.plot_topomap(
                    t_map,
                    epochs_info,
                    mask=mask,
                    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8),
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-v_max_abs, v_max_abs)
                )
                plt.colorbar(im, ax=ax, label='T-value')
                ax.set_title(f"Cluster {i_clust + 1} ({prob_cond}, p={cluster_pv[clust_idx]:.3f})\n{t_min * 1000:.0f} - {t_max * 1000:.0f} ms\nMean t={mean_t_cluster:.2f}, d={cohens_d_st:.2f}")
                cond_str = prob_cond.replace("-", "_")
                out_st_cluster = os.path.join(PLOTS_DIR, f"erp_lateral_spatiotemporal_cluster_{cond_str}_{i_clust + 1}.png")
                fig.savefig(out_st_cluster, dpi=200, bbox_inches='tight')
                print(f"Saved spatio-temporal cluster plot to: {out_st_cluster}")
                if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                    plt.show()
                else:
                    plt.close(fig)
        else:
            print(f"No significant spatio-temporal clusters found for {prob_cond}.")

        # 5. Butterfly Plot of T-values (MNE Joint Plot)
        print(f"\n--- Visualizing Temporal Evolution of T-values for {prob_cond} (MNE Joint Plot) ---")

        # Create Evoked object from T-values
        t_obs_plot = t_obs.copy()
        t_obs_plot[np.isnan(t_obs_plot)] = 0
        t_evoked = mne.EvokedArray(t_obs_plot.T, epochs_info, tmin=times_vector[0], nave=n_subs, comment='T-values')

        # Calculate threshold for visualization
        n_samples = X.shape[0]
        deg_f = n_samples - 1
        t_thresh = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, deg_f)

        # Plot joint (Butterfly + Topomaps)
        fig_joint = t_evoked.plot_joint(times='peaks', title=f'Mass Univariate T-values ({prob_cond})', show=False)

        # Add threshold lines to the butterfly plot axes
        for ax in fig_joint.axes:
            if ax.get_xlabel().startswith('Time'):
                ax.axhline(t_thresh, color='red', linestyle='--', alpha=0.5, label=f'p<{ALPHA_STAT_CLUSTER}')
                ax.axhline(-t_thresh, color='red', linestyle='--', alpha=0.5)
        cond_str = prob_cond.replace("-", "_")
        out_joint = os.path.join(PLOTS_DIR, f"erp_lateral_spatiotemporal_joint_{cond_str}.png")
        fig_joint.savefig(out_joint, dpi=200, bbox_inches='tight')
        print(f"Saved spatio-temporal joint plot to: {out_joint}")
        if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
            plt.show()
        else:
            plt.close(fig_joint)

        # 6. Image Plot of T-values
        print(f"\n--- Visualizing T-values for {prob_cond} (Image Plot) ---")

        # Create mask for significant clusters
        sig_mask = np.zeros(t_obs.shape, dtype=bool)
        if len(significant_points) > 0:
            for clust_idx in significant_points:
                time_inds, ch_inds = clusters[clust_idx]
                sig_mask[time_inds, ch_inds] = True

        # Transpose to (n_channels, n_times) for Evoked
        mask_to_plot = sig_mask.T

        # Create ROIs by checking channel labels
        selections = mne.channels.make_1020_channel_selections(epochs.info, midline="12z")

        fig_image = t_evoked.plot_image(
            picks='all',
            units=dict(eeg='T-value'), # Label colorbar correctly
            cmap='RdBu_r',
            show_names='all',          # Show electrode names
            mask=mask_to_plot,         # Mask non-significant values
            show=False,
            group_by=selections
        )
        cond_str = prob_cond.replace("-", "_")
        out_image = os.path.join(PLOTS_DIR, f"erp_lateral_spatiotemporal_image_{cond_str}.png")
        fig_image.savefig(out_image, dpi=200, bbox_inches='tight')
        print(f"Saved spatio-temporal image plot to: {out_image}")
        if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
            plt.show()
        else:
            plt.close(fig_image)
    else:
        print(f"\nSkipping mass univariate spatio-temporal analysis for {prob_cond}: not enough subjects (N < 2) or topography data missing (PLOT_TOPOS=False).")
