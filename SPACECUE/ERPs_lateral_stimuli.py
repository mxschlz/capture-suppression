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
from SPACECUE.colors import (
    COLOR_TARGET, COLOR_TARGET_YOUNG, COLOR_TARGET_OLD,
    COLOR_DISTRACTOR, COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD,
    COLOR_CONTROL, COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD,
    COLOR_DIFF, COHORT_COLORS, STIMULUS_COLORS, get_color
)
# NOTE: plotting.difference_topos() is deliberately NOT used for the
# topographies below: its internal display-pattern selection switches on the
# DistractorProb block factor and can select display types that do not exist
# within the location-level-filtered epochs, silently yielding all-NaN
# difference waves. The topographies are computed in-script from the same
# trial selections as the C3/C4 ROI waves instead.

sns.set_theme(context="talk", style="ticks")
plt.ion()

PLOTS_DIR = os.path.join(project_root, "spacecue", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 2. General Script Parameters ---
# Whether to compute and plot the topographies (adds runtime: per-subject
# full-channel difference topographies + spatio-temporal cluster tests)
PLOT_TOPOS = True

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
# Generic factor levels ('hp' vs 'lp') with stimulus-specific meaning:
# - N2ac (Target): target at the current HP distractor location ('hp') vs.
#   target at a lateral low-probability location ('lp'), computed on PURE
#   target displays (Target-1/3-Singleton-2). NOTE: on pure target displays
#   the singleton is frontal and therefore ALWAYS 'low-probability' by design,
#   so the distractor-probability factor is structurally undefined for N2ac.
# - Pd (Distractor): singleton at the HP location ('hp') vs. at a lateral
#   low-probability location ('lp'), computed on PURE distractor displays
#   (Target-2-Singleton-1/3).
conditions = ['hp', 'lp']
STIM_LEVEL_DESC = {
    'Target': {'hp': 'Target at HP loc', 'lp': 'Target at LP loc'},
    'Distractor': {'hp': 'Distractor at HP loc', 'lp': 'Distractor at LP loc'},
}
STIM_PATTERNS = {
    'Target': ("Target-1-Singleton-2", "Target-3-Singleton-2"),
    'Distractor': ("Target-2-Singleton-1", "Target-2-Singleton-3"),
}


def _loc_label(value):
    """Map a location code (numeric or string coding) to 'Left'/'Right'/'Front'."""
    s = str(value).strip()
    if s in ('1', '1.0'):
        return 'Left'
    if s in ('3', '3.0'):
        return 'Right'
    return s.title()


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

# Channel-hemisphere classification from the standard montage (needed for the
# per-channel Contra-Ipsi difference topographies below). Channels absent
# from the montage are treated as non-lateral (their topo rows stay 0).
_ch_pos = montage.get_positions()["ch_pos"]
CH_IS_LEFT = np.array([_ch_pos[ch][0] < 0 if ch in _ch_pos else False
                       for ch in epochs.info["ch_names"]])
CH_IS_MIDLINE = np.array([abs(_ch_pos[ch][0]) < 1e-4 if ch in _ch_pos else False
                          for ch in epochs.info["ch_names"]])

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

    # Required per-trial metadata columns for the factor definitions
    required_cols = ['TargetLoc', 'SingletonLoc', 'HP_Distractor_Loc']
    missing_cols = [c for c in required_cols if c not in epochs_sub.metadata.columns]
    if missing_cols:
        print(f"  Warning: metadata columns {missing_cols} not found for {subject_str}. Skipping.")
        continue

    meta_sub = epochs_sub.metadata

    # Per-trial location labels (robust to numeric or string coding)
    hp_loc_labels = meta_sub['HP_Distractor_Loc'].map(_loc_label)
    target_loc_labels = meta_sub['TargetLoc'].map(_loc_label)
    singleton_loc_labels = meta_sub['SingletonLoc'].map(_loc_label)

    # Factor masks per generic level ('hp' / 'lp'):
    # - Target (N2ac): target at the current HP distractor location vs. away.
    #   NOTE: on pure target displays the singleton is frontal and therefore
    #   ALWAYS 'low-probability' by design, so the distractor-probability
    #   factor is structurally undefined for N2ac.
    # - Distractor (Pd): singleton at the HP location vs. lateral LP location.
    is_target_hp = (target_loc_labels == hp_loc_labels).values
    is_distractor_hp = (singleton_loc_labels == hp_loc_labels).values

    # Row masks for the pure-display event patterns
    event_codes = epochs_sub.events[:, 2]
    code_to_name = {v: k for k, v in epochs_sub.event_id.items()}
    row_names = np.array([code_to_name.get(c, '') for c in event_codes])
    row_masks = {
        stim: (np.array([pat_l in n for n in row_names]),
               np.array([pat_r in n for n in row_names]))
        for stim, (pat_l, pat_r) in STIM_PATTERNS.items()
    }

    for prob_cond in conditions:  # generic levels: 'hp' / 'lp'
        try:
            stim_results = {}
            side_means = {}  # full-channel per-stimulus-side trial averages (topos)
            level_complete = True
            for stim_type in ['Target', 'Distractor']:
                factor = is_target_hp if stim_type == 'Target' else is_distractor_hp
                level_mask = factor if prob_cond == 'hp' else ~factor
                left_rows, right_rows = row_masks[stim_type]
                roi_left = LEFT_ROI_TARGET if stim_type == 'Target' else LEFT_ROI_DISTRACTOR
                roi_right = RIGHT_ROI_TARGET if stim_type == 'Target' else RIGHT_ROI_DISTRACTOR

                left_epochs = epochs_sub[np.where(left_rows & level_mask)[0]].copy()
                right_epochs = epochs_sub[np.where(right_rows & level_mask)[0]].copy()
                print(f"    {STIM_LEVEL_DESC[stim_type][prob_cond]} trials: "
                      f"{len(left_epochs)} left, {len(right_epochs)} right")
                if len(left_epochs) == 0 or len(right_epochs) == 0:
                    print(f"    Skipping {subject_str} - {stim_type} ({prob_cond}) "
                          f"due to zero trials in one of the side conditions.")
                    level_complete = False
                    break

                # Subject-level ipsi/contra ERPs: contra = electrode contralateral
                # to the stimulus side, ipsi = electrode on the stimulus side.
                contra_sub = np.mean(np.concatenate([
                    left_epochs.get_data(picks=roi_right),
                    right_epochs.get_data(picks=roi_left)
                ], axis=0), axis=(0, 1))
                ipsi_sub = np.mean(np.concatenate([
                    left_epochs.get_data(picks=roi_left),
                    right_epochs.get_data(picks=roi_right)
                ], axis=0), axis=(0, 1))
                stim_results[stim_type] = (contra_sub, ipsi_sub, contra_sub - ipsi_sub)
                if PLOT_TOPOS:
                    # Full-channel trial averages per stimulus side, for the
                    # Contra-Ipsi difference topographies
                    side_means[stim_type] = (left_epochs.get_data().mean(axis=0),
                                             right_epochs.get_data().mean(axis=0))

            if not level_complete:
                continue

            # Store subject results for both stimuli (row-aligned per subject)
            for stim_type in ['Target', 'Distractor']:
                stim_key = stim_type.lower()
                contra_sub, ipsi_sub, diff_sub = stim_results[stim_type]
                subject_data[prob_cond][f'{stim_key}_contra'].append(contra_sub)
                subject_data[prob_cond][f'{stim_key}_ipsi'].append(ipsi_sub)
                subject_data[prob_cond][f'{stim_key}_diff'].append(diff_sub)
            processed_subjects[prob_cond].append(subject_id_num)

            # Subject-level lateralization difference topographies (Contra -
            # Ipsi per channel), computed from the SAME trial selections as
            # the C3/C4 ROI waves above (see NOTE at the imports for why
            # plotting.difference_topos() is not used here).
            if PLOT_TOPOS:
                for stim_type in ['Target', 'Distractor']:
                    left_mean, right_mean = side_means[stim_type]
                    # Contra - Ipsi per channel: for channels over the LEFT
                    # hemisphere the contralateral stimulus is the right one,
                    # and vice versa. Midline channels carry no lateralization
                    # and are set to zero.
                    diff_topo_sub = np.where(CH_IS_LEFT[:, None],
                                             right_mean - left_mean,
                                             left_mean - right_mean)
                    diff_topo_sub[CH_IS_MIDLINE, :] = 0.0
                    stim_key = stim_type.lower()
                    subject_data[prob_cond][f'{stim_key}_diff_topo'].append(diff_topo_sub)

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
    color_waves = COLOR_TARGET if cond_name == 'Target' else COLOR_DISTRACTOR
    for prob_idx, prob_cond in enumerate(conditions):
        ax = ax_erp[cond_idx, prob_idx]
        erp_plot_params.append({
            'cond_name': cond_name,
            'prob_cond': prob_cond,
            'ax': ax,
            'contra': ga_data[prob_cond][f'{cond_name.lower()}_contra'],
            'ipsi': ga_data[prob_cond][f'{cond_name.lower()}_ipsi'],
            'diff': ga_data[prob_cond][f'{cond_name.lower()}_diff'],
            'color_diff': COLOR_DIFF,
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

    diff_arr = subject_data[prob_cond][f'{cond_name.lower()}_diff']
    sem_diff = (np.nanstd(diff_arr, axis=0) / np.sqrt(n_subs)) * AMPLITUDE_SCALE_FACTOR if n_subs > 1 else np.zeros_like(params['diff'])

    # Plot mean ERP waves with consistent, publication-quality styling
    ax.plot(times_vector, params['contra'] * AMPLITUDE_SCALE_FACTOR, color=params['color_waves'], linestyle='-', label="Contra", lw=2.0)
    ax.plot(times_vector, params['ipsi'] * AMPLITUDE_SCALE_FACTOR, color=params['color_waves'], linestyle='--', label="Ipsi", lw=2.0)
    ax.plot(times_vector, params['diff'] * AMPLITUDE_SCALE_FACTOR, color=params['color_diff'], label="Contra-Ipsi", linewidth=2.5)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', frameon=True)
    ax.set_title(f"{cond_name}: {STIM_LEVEL_DESC[cond_name][prob_cond]} (N={n_subs})")
    ax.set_ylabel("Amplitude (μV)")
    ax.set_xlabel("Time (s)")
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

# ==============================================================================
# --- AGE COHORT ANALYSES OF LATERALIZED POTENTIALS (Young vs. Old) ---
# ==============================================================================
print("\n" + "="*80)
print("--- RUNNING AGE COHORT ANALYSES FOR LATERALIZED POTENTIALS (Young vs. Old) ---")
print("="*80)

cohort_defs = {
    'Young': {'label': 'Younger Adults', 'filter': lambda age: age < 36},
    'Old': {'label': 'Older Adults', 'filter': lambda age: age >= 36}
}

cohort_data = {prob_cond: {'Young': {}, 'Old': {}} for prob_cond in conditions}
cohort_cluster_results = {prob_cond: {'Target': {'Young': None, 'Old': None},
                                      'Distractor': {'Young': None, 'Old': None}} for prob_cond in conditions}
age_cluster_results = {prob_cond: {'Target': None, 'Distractor': None} for prob_cond in conditions}

for prob_cond in conditions:
    sub_ids = processed_subjects[prob_cond]
    for c_name, c_info in cohort_defs.items():
        c_idx = [i for i, sub in enumerate(sub_ids) if c_info['filter'](subject_ages.get(sub, 0))]
        cohort_data[prob_cond][c_name]['indices'] = c_idx
        n_c = len(c_idx)
        cohort_data[prob_cond][c_name]['n'] = n_c
        for stim in ['target', 'distractor']:
            for meas in ['contra', 'ipsi', 'diff']:
                key = f'{stim}_{meas}'
                arr = subject_data[prob_cond][key][c_idx]
                cohort_data[prob_cond][c_name][key] = arr
                cohort_data[prob_cond][c_name][f'{key}_mean'] = np.nanmean(arr, axis=0) * AMPLITUDE_SCALE_FACTOR
                cohort_data[prob_cond][c_name][f'{key}_sem'] = (np.nanstd(arr, axis=0) / np.sqrt(n_c)) * AMPLITUDE_SCALE_FACTOR if n_c > 1 else np.zeros(len(times_vector))

# --- 1. Run Within-Cohort Permutation Cluster Tests (Contra vs. Ipsi within Young and Old) ---
print("\n--- Running Within-Cohort Permutation Cluster Tests (Contra vs. Ipsi) ---")
for prob_cond in conditions:
    for cond_name in ['Target', 'Distractor']:
        stim_key = f'{cond_name.lower()}_diff'
        for c_name in ['Young', 'Old']:
            diff_arr = cohort_data[prob_cond][c_name][stim_key]
            n_c = cohort_data[prob_cond][c_name]['n']
            if n_c >= 2:
                df_c = n_c - 1
                t_thresh_c = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, df_c)
                t_obs_c, clusters_c, cluster_pv_c, _ = permutation_cluster_1samp_test(
                    diff_arr, threshold=t_thresh_c, n_permutations=N_PERMUTATIONS_CLUSTER,
                    tail=CLUSTER_TAIL, n_jobs=N_JOBS, out_type="mask", seed=SEED, verbose=False
                )
                cohort_cluster_results[prob_cond][cond_name][c_name] = {
                    't_obs': t_obs_c, 'clusters': clusters_c, 'cluster_pv': cluster_pv_c,
                    'n': n_c, 'df': df_c, 'diff_arr': diff_arr
                }
                n_sig = np.sum(cluster_pv_c < ALPHA_STAT_CLUSTER)
                print(f"  {cond_name} ({prob_cond}) - {c_name} (N={n_c}, df={df_c}): {n_sig} significant cluster(s).")
            else:
                print(f"  {cond_name} ({prob_cond}) - {c_name}: Insufficient sample size (N={n_c}). Skipping.")

# --- 2. Run Between-Cohort Permutation Cluster Tests (Young vs. Old Difference Waves) ---
print("\n--- Running Between-Cohort Permutation Cluster Tests (Young vs. Old) ---")
from mne.stats import permutation_cluster_test
for prob_cond in conditions:
    for cond_name in ['Target', 'Distractor']:
        stim_key = f'{cond_name.lower()}_diff'
        y_diff = cohort_data[prob_cond]['Young'][stim_key]
        o_diff = cohort_data[prob_cond]['Old'][stim_key]
        n_y = cohort_data[prob_cond]['Young']['n']
        n_o = cohort_data[prob_cond]['Old']['n']
        if n_y >= 2 and n_o >= 2:
            F_obs, clusters, cluster_pv, _ = permutation_cluster_test(
                [y_diff, o_diff], n_permutations=N_PERMUTATIONS_CLUSTER, tail=1,
                threshold=None, out_type='indices', verbose=False, seed=SEED
            )
            age_cluster_results[prob_cond][cond_name] = {
                'F_obs': F_obs, 'clusters': clusters, 'cluster_pv': cluster_pv,
                'n_young': n_y, 'n_old': n_o, 'young_diffs': y_diff, 'old_diffs': o_diff
            }
            n_sig = np.sum(np.array(cluster_pv) < ALPHA_STAT_CLUSTER) if len(cluster_pv) > 0 else 0
            print(f"  {cond_name} ({prob_cond}) - Young vs Old (df=(1, {n_y + n_o - 2})): {n_sig} significant cluster(s).")
        else:
            print(f"  {cond_name} ({prob_cond}) - Young vs Old: Insufficient sample size. Skipping.")

# ==============================================================================
# --- FIGURE 1: 2x2 AGE-SPLIT DIFFERENCE WAVE PLOTS (N2ac / Pd) ---
# ==============================================================================
# Simplified version of the age-split figure: instead of 8 panels with three
# overlaid lines each (Contra, Ipsi, Difference), each panel now shows only
# the lateralized difference wave (Contra - Ipsi) at C3/C4, with the two
# probability conditions (HP vs. LP stimulus location) overlaid per cohort.
# The raw Contra/Ipsi waves remain available in the per-cohort standalone
# figures (erp_lateral_young.png / erp_lateral_old.png) below.
fig_age_split, axes_split = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

cohort_cols = [
    ('Young', 'Younger Adults'),
    ('Old', 'Older Adults'),
]
# Consistent visual coding of the probability factor across all panels
PROB_LINESTYLES = {'hp': '-', 'lp': '--'}
PROB_LABELS = {'hp': 'Stimulus at HP location', 'lp': 'Stimulus at LP location'}

for row_idx, cond_name in enumerate(['Target', 'Distractor']):
    stim_lower = cond_name.lower()
    for col_idx, (c_name, c_label) in enumerate(cohort_cols):
        ax = axes_split[row_idx, col_idx]
        c_color = STIMULUS_COLORS[cond_name][c_name]

        for prob_cond in conditions:
            c_data = cohort_data[prob_cond][c_name]
            n_c = c_data['n']
            diff_mean = c_data[f'{stim_lower}_diff_mean']

            # Difference wave (Contra - Ipsi): solid = HP location, dashed = LP location.
            # No ±1 SEM error bands here: with two overlaid probability
            # conditions per panel they are visually too much; the
            # significant-cluster bars already convey the reliable effects.
            ax.plot(times_vector, diff_mean, color=c_color,
                    linestyle=PROB_LINESTYLES[prob_cond], lw=2.8,
                    label=f"{PROB_LABELS[prob_cond]} (N={n_c})")

            # Significance cluster line (Contra vs. Ipsi within this
            # cohort/condition); linestyle matched to the condition's wave
            c_res = cohort_cluster_results[prob_cond][cond_name][c_name]
            sig_drawn = False
            if c_res and c_res['clusters'] is not None:
                for i_cl, cl_mask in enumerate(c_res['clusters']):
                    if c_res['cluster_pv'][i_cl] < ALPHA_STAT_CLUSTER:
                        cl_mask_arr = cl_mask[0] if isinstance(cl_mask, tuple) else cl_mask
                        cl_times = times_vector[cl_mask_arr]
                        if len(cl_times) > 0:
                            ax.hlines(y=SIG_LINE_Y_OFFSET, xmin=cl_times[0], xmax=cl_times[-1],
                                      color=COLOR_DIFF, linestyle=PROB_LINESTYLES[prob_cond],
                                      linewidth=SIG_LINE_LW, alpha=SIG_LINE_ALPHA,
                                      label=f'Sig. cluster ({prob_cond.upper()})' if not sig_drawn else '_nolegend_')
                            sig_drawn = True

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)

        wave_name = "N2ac (Target)" if cond_name == "Target" else "Pd (Distractor)"
        ax.set_title(f"{wave_name} — {c_label}", fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel("Contra − Ipsi Amplitude (μV)", fontsize=13)
        if row_idx == 1:
            ax.set_xlabel("Time (s)", fontsize=13)
        ax.legend(loc='upper right', frameon=True, fontsize=10)
        sns.despine(ax=ax)

fig_age_split.suptitle("Lateralized Difference Waves (N2ac / Pd) Split by Age Cohort", fontsize=16, fontweight='bold', y=0.98)
fig_age_split.text(0.5, 0.945,
                   "Contra − Ipsi difference waves at C3/C4 | Solid: stimulus at the high-probability (HP) location, "
                   "dashed: at a low-probability (LP) location | "
                   "Black bars: significant Contra vs. Ipsi clusters (p < 0.05)",
                   ha='center', fontsize=11, fontstyle='italic', color='#333333')
fig_age_split.tight_layout(rect=[0, 0, 1, 0.93])
fig_age_split.canvas.draw_idle()
out_age_split = os.path.join(PLOTS_DIR, "erp_lateral_age_split.png")
fig_age_split.savefig(out_age_split, dpi=300, bbox_inches='tight')
print(f"Saved age-split ERP plot to: {out_age_split}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_age_split)

# ==============================================================================
# --- FIGURE 2 & 3: STANDALONE 2x2 ERP PLOTS PER AGE COHORT ---
# ==============================================================================
# Mirrors the exact original 2x2 layout of fig_erp for each age cohort individually.
for c_name, c_info in cohort_defs.items():
    fig_c, axes_c = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

    for row_idx, cond_name in enumerate(['Target', 'Distractor']):
        c_color = STIMULUS_COLORS[cond_name][c_name]
        c_diff = COLOR_DIFF
        for col_idx, prob_cond in enumerate(conditions):
            ax = axes_c[row_idx, col_idx]
            c_data = cohort_data[prob_cond][c_name]
            n_c = c_data['n']
            stim_lower = cond_name.lower()

            contra_mean = c_data[f'{stim_lower}_contra_mean']
            contra_sem = c_data[f'{stim_lower}_contra_sem']
            ipsi_mean = c_data[f'{stim_lower}_ipsi_mean']
            ipsi_sem = c_data[f'{stim_lower}_ipsi_sem']
            diff_mean = c_data[f'{stim_lower}_diff_mean']
            diff_sem = c_data[f'{stim_lower}_diff_sem']

            ax.plot(times_vector, contra_mean, color=c_color, linestyle='-', lw=2.2, label="Contra")

            ax.plot(times_vector, ipsi_mean, color=c_color, linestyle='--', lw=2.2, label="Ipsi")

            diff_label = "N2ac (Contra-Ipsi)" if cond_name == "Target" else "Pd (Contra-Ipsi)"
            ax.plot(times_vector, diff_mean, color=c_diff, linestyle='-', lw=2.8, label=diff_label)

            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='k', linestyle=':', linewidth=0.8)

            c_res = cohort_cluster_results[prob_cond][cond_name][c_name]
            sig_drawn = False
            if c_res and c_res['clusters'] is not None:
                for i_cl, cl_mask in enumerate(c_res['clusters']):
                    if c_res['cluster_pv'][i_cl] < ALPHA_STAT_CLUSTER:
                        cl_mask_arr = cl_mask[0] if isinstance(cl_mask, tuple) else cl_mask
                        cl_times = times_vector[cl_mask_arr]
                        if len(cl_times) > 0:
                            ax.hlines(y=SIG_LINE_Y_OFFSET, xmin=cl_times[0], xmax=cl_times[-1],
                                      color=c_diff, linewidth=SIG_LINE_LW, alpha=SIG_LINE_ALPHA,
                                      label='Sig. Cluster (p < 0.05)' if not sig_drawn else '_nolegend_')
                            sig_drawn = True

            ax.set_title(f"{cond_name}: {STIM_LEVEL_DESC[cond_name][prob_cond]} ({c_info['label']}, N={n_c})", fontsize=13, fontweight='bold')
            ax.set_ylabel("Amplitude (μV)", fontsize=12)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.legend(loc='upper right', frameon=True)
            sns.despine(ax=ax)

    fig_c.suptitle(f"{c_info['label']} — Lateralized ERPs (Target & Distractor)", fontsize=16, fontweight='bold', y=0.98)
    fig_c.tight_layout(rect=[0, 0, 1, 0.96])
    fig_c.canvas.draw_idle()
    out_c = os.path.join(PLOTS_DIR, f"erp_lateral_{c_name.lower()}.png")
    fig_c.savefig(out_c, dpi=300, bbox_inches='tight')
    print(f"Saved {c_name} standalone ERP plot to: {out_c}")
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close(fig_c)

# ==============================================================================
# --- FIGURE 4: 2x2 COHORT OVERLAY ERP PLOT (YOUNG & OLD ON SAME AXIS) ---
# ==============================================================================
# Overlays Young and Old Contra/Ipsi and Difference waveforms on the same subplots.
fig_overlay, axes_overlay = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

for row_idx, cond_name in enumerate(['Target', 'Distractor']):
    color_young = STIMULUS_COLORS[cond_name]['Young']
    color_old = STIMULUS_COLORS[cond_name]['Old']

    for col_idx, prob_cond in enumerate(conditions):
        ax = axes_overlay[row_idx, col_idx]
        stim_lower = cond_name.lower()

        # Young waveforms (Light Green for Target, Light Red for Distractor)
        y_data = cohort_data[prob_cond]['Young']
        ax.plot(times_vector, y_data[f'{stim_lower}_contra_mean'], color=color_young, linestyle='-', lw=2.0, label="Young Contra")
        ax.plot(times_vector, y_data[f'{stim_lower}_ipsi_mean'], color=color_young, linestyle='--', lw=2.0, label="Young Ipsi")
        ax.plot(times_vector, y_data[f'{stim_lower}_diff_mean'], color=color_young, linestyle=':', lw=2.6, label="Young Diff")

        # Old waveforms (Dark Green for Target, Dark Red for Distractor)
        o_data = cohort_data[prob_cond]['Old']
        ax.plot(times_vector, o_data[f'{stim_lower}_contra_mean'], color=color_old, linestyle='-', lw=2.0, label="Old Contra")
        ax.plot(times_vector, o_data[f'{stim_lower}_ipsi_mean'], color=color_old, linestyle='--', lw=2.0, label="Old Ipsi")
        ax.plot(times_vector, o_data[f'{stim_lower}_diff_mean'], color=color_old, linestyle=':', lw=2.6, label="Old Diff")

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)

        # Significant difference windows between Young and Old (if any)
        res_age = age_cluster_results[prob_cond].get(cond_name)
        if res_age and res_age['clusters']:
            for cl, p_val in zip(res_age['clusters'], res_age['cluster_pv']):
                if p_val < ALPHA_STAT_CLUSTER:
                    ax.axvspan(times_vector[cl[0][0]], times_vector[cl[0][-1]],
                               color='purple', alpha=0.15, lw=0, label='Sig. Age Diff (p < 0.05)')

        ax.set_title(f"{cond_name}: {STIM_LEVEL_DESC[cond_name][prob_cond]} (Young vs Old Overlaid)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Amplitude (μV)", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.legend(loc='upper right', frameon=True, fontsize=9, ncol=2)
        sns.despine(ax=ax)

fig_overlay.suptitle("Overlaid Lateralized ERPs: Younger vs. Older Adults", fontsize=16, fontweight='bold', y=0.98)
fig_overlay.tight_layout(rect=[0, 0, 1, 0.96])
fig_overlay.canvas.draw_idle()
out_overlay = os.path.join(PLOTS_DIR, "erp_lateral_age_overlay.png")
fig_overlay.savefig(out_overlay, dpi=300, bbox_inches='tight')
print(f"Saved age overlay ERP plot to: {out_overlay}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_overlay)

# ==============================================================================
# --- FIGURE 5: 2x2 DIFFERENCE WAVE COMPARISON (YOUNG VS OLD) ---
# ==============================================================================
# Visualizes direct difference waves comparison with cluster test results.
fig_diff_comp, axes_diff_comp = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

for row_idx, cond_name in enumerate(['Target', 'Distractor']):
    color_young = STIMULUS_COLORS[cond_name]['Young']
    color_old = STIMULUS_COLORS[cond_name]['Old']

    for col_idx, prob_cond in enumerate(conditions):
        ax = axes_diff_comp[row_idx, col_idx]
        stim_lower = cond_name.lower()

        y_data = cohort_data[prob_cond]['Young']
        o_data = cohort_data[prob_cond]['Old']
        n_y = y_data['n']
        n_o = o_data['n']

        y_diff_mean = y_data[f'{stim_lower}_diff_mean']
        y_diff_sem = y_data[f'{stim_lower}_diff_sem']
        o_diff_mean = o_data[f'{stim_lower}_diff_mean']
        o_diff_sem = o_data[f'{stim_lower}_diff_sem']

        # Plot Young Difference Wave (Light Green for Target, Light Red for Distractor)
        ax.plot(times_vector, y_diff_mean, color=color_young, lw=2.5, label=f"Young (N={n_y})")

        # Plot Old Difference Wave (Dark Green for Target, Dark Red for Distractor)
        ax.plot(times_vector, o_diff_mean, color=color_old, lw=2.5, label=f"Old (N={n_o})")

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)

        # Highlight cluster test outcomes
        res_age = age_cluster_results[prob_cond].get(cond_name)
        cluster_pv = res_age['cluster_pv'] if res_age else []
        sig_diff_plotted = False
        if res_age and res_age['clusters']:
            for cl, p_val in zip(res_age['clusters'], cluster_pv):
                if p_val < ALPHA_STAT_CLUSTER:
                    ax.axvspan(times_vector[cl[0][0]], times_vector[cl[0][-1]],
                               color='purple', alpha=0.18, lw=0, label='Sig. Diff (p < 0.05)' if not sig_diff_plotted else '_nolegend_')
                    sig_diff_plotted = True

        ax.set_title(f"{cond_name} Difference Wave: {STIM_LEVEL_DESC[cond_name][prob_cond]} (Young vs Old)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Amplitude Difference (μV)", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)

        if sig_diff_plotted:
            ax.legend(loc='upper right', frameon=True)
        else:
            ax.legend(loc='upper right', frameon=True)
            min_p = np.min(cluster_pv) if len(cluster_pv) > 0 else np.nan
            p_text = f"Cluster test: n.s. (min p = {min_p:.2f})" if not np.isnan(min_p) else "Cluster test: n.s."
            ax.text(0.04, 0.06, p_text, transform=ax.transAxes, fontsize=11, fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        sns.despine(ax=ax)

fig_diff_comp.suptitle("Difference Wave Comparison (Young vs. Old)", fontsize=16, fontweight='bold', y=0.98)
fig_diff_comp.text(0.5, 0.945, "Non-parametric cluster-based permutation tests on difference waves",
                   ha='center', fontsize=12, fontstyle='italic', color='#333333')
fig_diff_comp.tight_layout(rect=[0, 0, 1, 0.94])
fig_diff_comp.canvas.draw_idle()
out_diff_comp = os.path.join(PLOTS_DIR, "erp_lateral_age_diff_comparison.png")
fig_diff_comp.savefig(out_diff_comp, dpi=300, bbox_inches='tight')
print(f"Saved difference wave comparison plot to: {out_diff_comp}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig_diff_comp)

# ==============================================================================
# --- STATISTICAL REPORTING & CONSOLE LOGGING ---
# ==============================================================================
report_lines = []
report_lines.append("="*80)
report_lines.append("SPACECUE LATERALIZED ERP STATISTICAL INFERENCE REPORT")
report_lines.append("="*80 + "\n")

print("\n" + "="*80)
print("--- SUMMARY OF CLUSTER-BASED PERMUTATION TEST RESULTS ---")
print("="*80)

# Section 1: Grand Average Lateralization (All Subjects)
print("\n[SECTION 1] GRAND AVERAGE LATERALIZATION (CONTRA VS. IPSI, ALL SUBJECTS)")
report_lines.append("[SECTION 1] GRAND AVERAGE LATERALIZATION (CONTRA VS. IPSI, ALL SUBJECTS)")
for prob_cond in conditions:
    n_subs = len(processed_subjects[prob_cond])
    deg_f = n_subs - 1
    for cond_name in ['Target', 'Distractor']:
        res = cluster_results[prob_cond].get(cond_name)
        line_header = f"{cond_name} ({prob_cond}, N={n_subs}, df={deg_f}):"
        print(f"\n  {line_header}")
        report_lines.append(f"\n  {line_header}")
        if res and res['clusters'] is not None and len(res['clusters']) > 0:
            sig_found = False
            for i_cl, cl_mask in enumerate(res['clusters']):
                p_val = res['p_values'][i_cl]
                if p_val < ALPHA_STAT_CLUSTER:
                    sig_found = True
                    cl_mask_arr = cl_mask[0] if isinstance(cl_mask, tuple) else cl_mask
                    cl_times = times_vector[cl_mask_arr]
                    t_obs_cl = res['t_obs'][cl_mask_arr]
                    mean_t = np.mean(t_obs_cl)
                    peak_t = t_obs_cl[np.argmax(np.abs(t_obs_cl))]

                    # Effect size: Cohen's dz on temporally collapsed cluster data (per guidelines)
                    subj_means = np.mean(subject_data[prob_cond][f'{cond_name.lower()}_diff'][:, cl_mask_arr], axis=1) * AMPLITUDE_SCALE_FACTOR
                    std_subj = np.std(subj_means, ddof=1)
                    cohens_dz = np.mean(subj_means) / std_subj if std_subj > 0 else 0.0

                    out_str = (f"    Cluster {i_cl+1} [SIGNIFICANT]: {cl_times[0]*1000:.0f} - {cl_times[-1]*1000:.0f} ms | "
                               f"p = {p_val:.4f} | Mean t = {mean_t:.2f}, Peak t = {peak_t:.2f} | Cohen's dz = {cohens_dz:.2f}")
                    print(out_str)
                    report_lines.append(out_str)
            if not sig_found:
                min_p = np.min(res['p_values']) if len(res['p_values']) > 0 else np.nan
                out_str = f"    No significant clusters found (min p = {min_p:.4f})."
                print(out_str)
                report_lines.append(out_str)
        else:
            out_str = "    No clusters formed at cluster-defining threshold."
            print(out_str)
            report_lines.append(out_str)

# Section 2: Within-Cohort Lateralization (Young vs Old Separately)
print("\n[SECTION 2] WITHIN-COHORT LATERALIZATION (CONTRA VS. IPSI)")
report_lines.append("\n\n[SECTION 2] WITHIN-COHORT LATERALIZATION (CONTRA VS. IPSI)")
for c_name in ['Young', 'Old']:
    print(f"\n--- Cohort: {cohort_defs[c_name]['label']} ---")
    report_lines.append(f"\n--- Cohort: {cohort_defs[c_name]['label']} ---")
    for prob_cond in conditions:
        for cond_name in ['Target', 'Distractor']:
            res_c = cohort_cluster_results[prob_cond][cond_name][c_name]
            if res_c is not None:
                n_c = res_c['n']
                df_c = res_c['df']
                line_header = f"{cond_name} ({prob_cond}, N={n_c}, df={df_c}):"
                print(f"  {line_header}")
                report_lines.append(f"  {line_header}")
                if res_c['clusters'] is not None and len(res_c['clusters']) > 0:
                    sig_found = False
                    for i_cl, cl_mask in enumerate(res_c['clusters']):
                        p_val = res_c['cluster_pv'][i_cl]
                        if p_val < ALPHA_STAT_CLUSTER:
                            sig_found = True
                            cl_mask_arr = cl_mask[0] if isinstance(cl_mask, tuple) else cl_mask
                            cl_times = times_vector[cl_mask_arr]
                            t_obs_cl = res_c['t_obs'][cl_mask_arr]
                            mean_t = np.mean(t_obs_cl)
                            peak_t = t_obs_cl[np.argmax(np.abs(t_obs_cl))]

                            # Cohen's dz on cluster-averaged data
                            subj_means = np.mean(res_c['diff_arr'][:, cl_mask_arr], axis=1) * AMPLITUDE_SCALE_FACTOR
                            std_subj = np.std(subj_means, ddof=1)
                            cohens_dz = np.mean(subj_means) / std_subj if std_subj > 0 else 0.0

                            out_str = (f"    Cluster {i_cl+1} [SIGNIFICANT]: {cl_times[0]*1000:.0f} - {cl_times[-1]*1000:.0f} ms | "
                                       f"p = {p_val:.4f} | Mean t = {mean_t:.2f}, Peak t = {peak_t:.2f} | Cohen's dz = {cohens_dz:.2f}")
                            print(out_str)
                            report_lines.append(out_str)
                    if not sig_found:
                        min_p = np.min(res_c['cluster_pv']) if len(res_c['cluster_pv']) > 0 else np.nan
                        out_str = f"    No significant clusters found (min p = {min_p:.4f})."
                        print(out_str)
                        report_lines.append(out_str)
                else:
                    out_str = "    No clusters formed at cluster-defining threshold."
                    print(out_str)
                    report_lines.append(out_str)

# Section 3: Between-Cohort Permutation Tests (Young vs Old Difference Waves)
print("\n[SECTION 3] BETWEEN-COHORT TESTS (YOUNG VS. OLD DIFFERENCE WAVES)")
report_lines.append("\n\n[SECTION 3] BETWEEN-COHORT TESTS (YOUNG VS. OLD DIFFERENCE WAVES)")
for prob_cond in conditions:
    for cond_name in ['Target', 'Distractor']:
        res_age = age_cluster_results[prob_cond].get(cond_name)
        if res_age is not None:
            n_y = res_age['n_young']
            n_o = res_age['n_old']
            df = (1, n_y + n_o - 2)
            clusters = res_age['clusters']
            cluster_pv = res_age['cluster_pv']
            F_obs = res_age['F_obs']
            y_d = res_age['young_diffs']
            o_d = res_age['old_diffs']

            line_header = f"{cond_name} ({prob_cond}, Young N={n_y}, Old N={n_o}, df={df}):"
            print(f"  {line_header}")
            report_lines.append(f"  {line_header}")

            if len(clusters) == 0:
                out_str = f"    No clusters formed at primary threshold (p < 0.05). Max F = {np.max(F_obs):.3f}."
                print(out_str)
                report_lines.append(out_str)
            else:
                for i_c, (cl, p_val) in enumerate(zip(clusters, cluster_pv)):
                    t_start_ms = times_vector[cl[0][0]] * 1000
                    t_end_ms = times_vector[cl[0][-1]] * 1000
                    mean_f = np.mean(F_obs[cl[0]])
                    peak_f = np.max(F_obs[cl[0]])

                    # Cohen's d on cluster-averaged data
                    y_means = np.mean(y_d[:, cl[0]], axis=1) * AMPLITUDE_SCALE_FACTOR
                    o_means = np.mean(o_d[:, cl[0]], axis=1) * AMPLITUDE_SCALE_FACTOR
                    s_pooled = np.sqrt(((n_y - 1) * np.var(y_means, ddof=1) + (n_o - 1) * np.var(o_means, ddof=1)) / (n_y + n_o - 2))
                    cohens_d = (np.mean(y_means) - np.mean(o_means)) / s_pooled if s_pooled > 0 else 0.0

                    sig_label = "SIGNIFICANT" if p_val < ALPHA_STAT_CLUSTER else "n.s."
                    out_str = (f"    Cluster {i_c+1} [{sig_label}]: {t_start_ms:.1f} - {t_end_ms:.1f} ms | p = {p_val:.5f} | "
                               f"Mean F = {mean_f:.3f}, Peak F = {peak_f:.3f} | Cohen's d = {cohens_d:.3f}")
                    print(out_str)
                    report_lines.append(out_str)

# Save report to text file
report_path = os.path.join(PLOTS_DIR, "erp_lateral_statistical_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"\nSaved statistical report to: {report_path}")

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

# ==============================================================================
# --- AVERAGE DIFFERENCE TOPOGRAPHIES (N2ac / Pd TIME WINDOWS) ---
# ==============================================================================
# Simple publication-style average topographies of the lateralized difference
# waves (Contra - Ipsi) in FIXED component windows (N2ac: 200-400 ms,
# Pd: 300-400 ms), meant to be placed next to the ERP time courses. The 2x2
# layout (rows: N2ac/Pd, columns: HP/LP stimulus location) mirrors the grand
# average ERP figure; a single common color scale is used across all panels
# (matching the sharey=True convention of the ERP figure). Each panel is also
# saved as a standalone PNG for flexible layouting.
AVG_TOPO_WINDOWS = {'Target': N2AC_TOPO_CLUSTER_WINDOW,    # N2ac: 200-400 ms
                    'Distractor': PD_TOPO_CLUSTER_WINDOW}  # Pd:  300-400 ms
AVG_TOPO_COMPONENT_NAMES = {'Target': 'N2ac (Target)', 'Distractor': 'Pd (Distractor)'}

avg_topo_data = {}
vmax_avg_topo = 0.0
for cond_name in ['Target', 'Distractor']:
    win = AVG_TOPO_WINDOWS[cond_name]
    t_start_idx = np.argmin(np.abs(times_vector - win[0]))
    t_end_idx = np.argmin(np.abs(times_vector - win[1]))
    for prob_cond in conditions:
        ga_topo = ga_data[prob_cond][f'{cond_name.lower()}_diff_topo']
        if np.all(np.isnan(ga_topo)):
            print(f"Skipping average topo for {cond_name} ({prob_cond}): grand average topo data is all NaN (PLOT_TOPOS=False?).")
            continue
        # Average the grand-average difference topography over the window
        data_avg = np.mean(ga_topo[:, t_start_idx:t_end_idx + 1], axis=1) * AMPLITUDE_SCALE_FACTOR
        avg_topo_data[(cond_name, prob_cond)] = data_avg
        vmax_avg_topo = max(vmax_avg_topo, np.nanmax(np.abs(data_avg)))
if vmax_avg_topo == 0:
    vmax_avg_topo = 1.0

if avg_topo_data:
    # --- Combined 2x2 figure (rows: N2ac/Pd, columns: HP/LP) ---
    fig_avg_topo, axes_avg = plt.subplots(2, len(conditions), figsize=(4.6 * len(conditions), 4.9))
    im_avg = None
    for row_idx, cond_name in enumerate(['Target', 'Distractor']):
        for col_idx, prob_cond in enumerate(conditions):
            ax = axes_avg[row_idx, col_idx]
            if (cond_name, prob_cond) not in avg_topo_data:
                ax.axis('off')
                continue
            n_subs = len(processed_subjects[prob_cond])
            im_avg, _ = mne.viz.plot_topomap(
                avg_topo_data[(cond_name, prob_cond)], epochs_info, axes=ax,
                cmap=TOPO_CMAP, vlim=(-vmax_avg_topo, vmax_avg_topo),
                show=False, sensors=False, outlines='head'
            )
            win = AVG_TOPO_WINDOWS[cond_name]
            ax.set_title(f"{AVG_TOPO_COMPONENT_NAMES[cond_name]} — {STIM_LEVEL_DESC[cond_name][prob_cond]}\n"
                         f"{win[0] * 1000:.0f}-{win[1] * 1000:.0f} ms avg (N={n_subs})", fontsize=12)

    if im_avg:
        fig_avg_topo.subplots_adjust(right=0.86, top=0.86, wspace=0.25, hspace=0.35)
        cbar_ax_avg = fig_avg_topo.add_axes([0.88, 0.14, 0.025, 0.7])
        cbar_avg = plt.colorbar(im_avg, cax=cbar_ax_avg)
        cbar_avg.set_label('Contra − Ipsi Amplitude (μV)')
    fig_avg_topo.suptitle("Average Lateralized Difference Topographies (N2ac / Pd)", fontsize=15, y=0.97)
    out_avg_topo = os.path.join(PLOTS_DIR, "erp_lateral_topo_avg_n2ac_pd.png")
    fig_avg_topo.savefig(out_avg_topo, dpi=300, bbox_inches='tight')
    print(f"Saved average N2ac/Pd difference topographies (2x2) to: {out_avg_topo}")
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close(fig_avg_topo)

    # --- Standalone per-panel figures (for placing next to ERP panels) ---
    for (cond_name, prob_cond), data_avg in avg_topo_data.items():
        fig_p, ax_p = plt.subplots(figsize=(4.6, 4.6))
        mne.viz.plot_topomap(
            data_avg, epochs_info, axes=ax_p, cmap=TOPO_CMAP,
            vlim=(-vmax_avg_topo, vmax_avg_topo),
            show=False, sensors=False, outlines='head'
        )
        n_subs = len(processed_subjects[prob_cond])
        win = AVG_TOPO_WINDOWS[cond_name]
        ax_p.set_title(f"{AVG_TOPO_COMPONENT_NAMES[cond_name]} — {STIM_LEVEL_DESC[cond_name][prob_cond]}\n"
                       f"{win[0] * 1000:.0f}-{win[1] * 1000:.0f} ms avg (N={n_subs})", fontsize=12)
        out_p = os.path.join(PLOTS_DIR, f"erp_lateral_topo_avg_{cond_name.lower()}_{prob_cond}.png")
        fig_p.savefig(out_p, dpi=300, bbox_inches='tight')
        print(f"Saved average topo panel to: {out_p}")
        if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
            plt.show()
        else:
            plt.close(fig_p)

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
        # 1. Define reshaped data, RESTRICTED TO LATERAL CHANNELS: midline
        # channels are structurally zero in the lateralization maps (zero
        # variance across subjects -> undefined t-values / NaN columns)
        lateral_ch_idx = np.where(~CH_IS_MIDLINE)[0]
        info_lateral = mne.pick_info(epochs_info, sel=lateral_ch_idx)
        X = dist_diff_topo[:, lateral_ch_idx, :].transpose([0, 2, 1])
        # 2. Define the Spatial Connectivity
        connectivity, ch_names = find_ch_adjacency(info_lateral, ch_type='eeg')

        # 3. Run the Mass Univariate Test
        t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=connectivity,
            n_permutations=1000,
            tail=0,  # Two-tailed test
            threshold=None,  # Uses a standard t-threshold based on p < 0.05
            seed=SEED,       # Reproducibility, consistent with the other tests
            n_jobs=N_JOBS
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
                mask = np.zeros(len(info_lateral.ch_names), dtype=bool)
                mask[np.unique(ch_inds)] = True

                # Average T-values over the cluster time window
                t_map = np.nanmean(t_obs[time_inds.min():time_inds.max() + 1, :], axis=0)

                # --- Calculate Effect Sizes for Spatio-Temporal Cluster ---
                t_vals_cluster = t_obs[time_inds, ch_inds]
                mean_t_cluster = np.mean(t_vals_cluster)
                peak_t_cluster = t_vals_cluster[np.argmax(np.abs(t_vals_cluster))]

                # Cohen's dz on the spatio-temporally collapsed cluster data:
                # collapse each subject's raw difference topography across the
                # exact cluster bounds, then compute a single dz on the
                # resulting subject-level averages (averaging point-wise
                # T-values would inflate the effect size by circularity)
                subj_cluster_means = np.array([
                    X[i][np.ix_(time_inds, ch_inds)].mean()
                    for i in range(X.shape[0])
                ]) * AMPLITUDE_SCALE_FACTOR
                sd_cluster = np.std(subj_cluster_means, ddof=1)
                cohens_dz_st = (np.mean(subj_cluster_means) / sd_cluster
                                if sd_cluster > 0 else 0.0)

                fig, ax = plt.subplots(figsize=(5, 5))
                im, _ = mne.viz.plot_topomap(
                    t_map,
                    info_lateral,
                    mask=mask,
                    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8),
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-v_max_abs, v_max_abs)
                )
                plt.colorbar(im, ax=ax, label='T-value')
                ax.set_title(f"Cluster {i_clust + 1} ({prob_cond}, p={cluster_pv[clust_idx]:.3f})\n{t_min * 1000:.0f} - {t_max * 1000:.0f} ms\nMean t={mean_t_cluster:.2f}, dz={cohens_dz_st:.2f}")
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
        t_evoked = mne.EvokedArray(t_obs_plot.T, info_lateral, tmin=times_vector[0], nave=n_subs, comment='T-values')

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

        # Create ROIs by checking channel labels (drop empty groups that
        # result from excluding the midline channels)
        selections = mne.channels.make_1020_channel_selections(info_lateral, midline="12z")
        selections = {k: v for k, v in selections.items() if len(v) > 0}

        figs_image = t_evoked.plot_image(
            picks='all',
            units=dict(eeg='T-value'), # Label colorbar correctly
            cmap='RdBu_r',
            show_names='all',          # Show electrode names
            mask=mask_to_plot,         # Mask non-significant values
            show=False,
            group_by=selections
        )
        # plot_image returns a LIST of figures when group_by yields multiple
        # channel groups (one figure per group)
        figs_image = figs_image if isinstance(figs_image, list) else [figs_image]
        cond_str = prob_cond.replace("-", "_")
        group_names = list(selections.keys())
        for g_name, fig_image in zip(group_names, figs_image):
            g_str = g_name.lower().replace(' ', '_')
            out_image = os.path.join(PLOTS_DIR, f"erp_lateral_spatiotemporal_image_{cond_str}_{g_str}.png")
            fig_image.savefig(out_image, dpi=200, bbox_inches='tight')
            print(f"Saved spatio-temporal image plot ({g_name}) to: {out_image}")
            if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                plt.show()
            else:
                plt.close(fig_image)
    else:
        print(f"\nSkipping mass univariate spatio-temporal analysis for {prob_cond}: not enough subjects (N < 2) or topography data missing (PLOT_TOPOS=False).")
