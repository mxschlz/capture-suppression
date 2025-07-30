import mne
import matplotlib.pyplot as plt
import SPACEPRIME  # Use the SPACEPRIME package for loading
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import t, ttest_rel, sem
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import itertools

# Assuming this is your custom outlier removal function
from stats import remove_outliers

plt.ion()

# --- Parameters ---
EPOCH_TMIN, EPOCH_TMAX = 0, 0.7
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3
AMPLITUDE_SCALE_FACTOR = 1e6

# --- Outlier Removal Configuration ---
OUTLIER_RT_THRESHOLD = 2.0
REACTION_TIME_COL = 'rt'
SUBJECT_ID_COL = 'subject_id'

# --- Pd Electrode Definition (using the same comprehensive pool as N2ac) ---
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
left_electrodes = [pair[0] for pair in PD_ELECTRODES]
right_electrodes = [pair[1] for pair in PD_ELECTRODES]

# Cluster Test Parameters
N_JOBS = 5
SEED = 42
TAIL = 0  # Two-tailed for all tests
N_PERMUTATIONS_CLUSTER = 10000
ALPHA_CLUSTER_SIGNIFICANCE = 0.05
ALPHA_CLUSTER_FORMING = 0.05

# Y-offsets for significance lines (1-sample tests vs 0)
Y_OFFSET_SIG_BASE = -0.01
Y_OFFSET_SIG_STEP = -0.005

# Parameters for Pairwise Comparison Plots
PERFORM_PAIRWISE_COMPARISONS = True
Y_OFFSET_SIG_PAIRWISE = -0.025
PAIRWISE_SIG_LINE_COLOR = 'purple'

# --- 1. Load and Preprocess Concatenated Data ---
print("--- Step 1: Loading and Preprocessing Concatenated Data ---")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")

if epochs is None:
    print("ERROR: Could not load concatenated epochs. Exiting.")
    exit()

# --- Outlier Removal Step ---
df = epochs.metadata.copy()
initial_trial_count = len(df)
df_clean = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
boolean_mask = df.index.isin(df_clean.index)
epochs = epochs[boolean_mask]
final_trial_count = len(epochs)
n_outliers = initial_trial_count - final_trial_count
print(f"Removed {n_outliers} outlier trials in total based on RT (>{OUTLIER_RT_THRESHOLD} SD).")
print(f"Total epochs after outlier removal: {final_trial_count}")

# --- Crop and get time vector ---
epochs.crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Cropped all epochs to {EPOCH_TMIN}-{EPOCH_TMAX}s.")
times_vector = epochs.times.copy()

# --- Data Storage and Setup for Subject Loop ---
subject_diff_waves = {}
processed_subject_count = 0
priming_conditions_map = {
    "no_prime": "Priming==0",
    "neg_prime": "Priming==-1",
    "pos_prime": "Priming==1"
}

# --- 2. Subject-Level Processing Loop ---
print("\n--- Step 2: Starting subject-level processing for Pd ---")
all_subject_ids = epochs.metadata[SUBJECT_ID_COL].unique()

for subject_id in all_subject_ids:
    subject_str = f"sub-{int(subject_id):02d}"
    print(f"\n--- Processing Subject: {subject_str} ---")
    subject_diff_waves[subject_str] = {}

    try:
        epochs_sub = epochs[epochs.metadata[SUBJECT_ID_COL] == subject_id]
        print(f"  Processing {len(epochs_sub)} epochs for this subject.")
        subject_contributed_to_any_diff_wave_this_subject = False

        for key, prime_event_selector in priming_conditions_map.items():
            print(f"  Processing '{key}' priming condition...")
            primed_epochs_sub = epochs_sub[prime_event_selector]

            if len(primed_epochs_sub) == 0:
                print(f"    No epochs for '{key}' priming for subject {subject_str}.")
                continue

            event_ids_in_primed_subset = list(primed_epochs_sub.event_id.keys())

            # --- MODIFIED: Event definitions for Pd (distractor-locked) ---
            left_distractor_events = [eid for eid in event_ids_in_primed_subset if
                                      "Singleton-1" in eid and "Target-2" in eid]
            right_distractor_events = [eid for eid in event_ids_in_primed_subset if
                                       "Singleton-3" in eid and "Target-2" in eid]

            epochs_left_distractor_sub = primed_epochs_sub[left_distractor_events]
            epochs_right_distractor_sub = primed_epochs_sub[right_distractor_events]

            print(
                f"    '{key}' priming: Left distractor trials: {len(epochs_left_distractor_sub)}, Right distractor trials: {len(epochs_right_distractor_sub)}")

            if len(epochs_left_distractor_sub) > 0 and len(epochs_right_distractor_sub) > 0:
                # Left distractor: contra is right hemi, ipsi is left hemi
                ev_L_D_contra_data = epochs_left_distractor_sub.copy().pick(right_electrodes).average().data.mean(
                    axis=0)
                ev_L_D_ipsi_data = epochs_left_distractor_sub.copy().pick(left_electrodes).average().data.mean(axis=0)

                # Right distractor: contra is left hemi, ipsi is right hemi
                ev_R_D_contra_data = epochs_right_distractor_sub.copy().pick(left_electrodes).average().data.mean(
                    axis=0)
                ev_R_D_ipsi_data = epochs_right_distractor_sub.copy().pick(right_electrodes).average().data.mean(axis=0)

                # Average the two contra waves and the two ipsi waves
                contra_data_sub = np.mean([ev_L_D_contra_data, ev_R_D_contra_data], axis=0)
                ipsi_data_sub = np.mean([ev_L_D_ipsi_data, ev_R_D_ipsi_data], axis=0)

                diff_wave_sub = contra_data_sub - ipsi_data_sub
                subject_diff_waves[subject_str][key] = diff_wave_sub
                subject_contributed_to_any_diff_wave_this_subject = True
                print(f"    Calculated and stored Pd diff wave for '{key}'.")
            else:
                print(
                    f"    Skipping Pd diff wave for '{key}' for subject {subject_str} due to insufficient trials.")

        if subject_contributed_to_any_diff_wave_this_subject:
            processed_subject_count += 1
        else:
            if subject_str in subject_diff_waves and not subject_diff_waves[subject_str]:
                del subject_diff_waves[subject_str]

    except Exception as e:
        print(f"  Error processing subject {subject_str}: {e}. Skipping.")
        if subject_str in subject_diff_waves:
            del subject_diff_waves[subject_str]
        continue

print(
    f"\n--- Finished subject-level processing. Processed data for {processed_subject_count} subjects. ---")

if processed_subject_count == 0:
    print("No subjects were processed successfully. Exiting.")
    exit()

# --- 3. Smooth Data and Prepare for Group-Level Analysis ---
print("\n--- Step 3: Smoothing data and preparing for group-level analysis ---")
ga_diff_waves = {}
stacked_diff_waves_for_1samp_test = {}
subject_diff_waves_smoothed = {}
condition_keys = list(priming_conditions_map.keys())

for key in condition_keys:
    smoothed_waves_list = []
    for sub_str_loop in subject_diff_waves.keys():
        if key in subject_diff_waves[sub_str_loop]:
            raw_wave = subject_diff_waves[sub_str_loop][key]
            smoothed_wave = savgol_filter(raw_wave, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)
            if sub_str_loop not in subject_diff_waves_smoothed:
                subject_diff_waves_smoothed[sub_str_loop] = {}
            subject_diff_waves_smoothed[sub_str_loop][key] = smoothed_wave
            smoothed_waves_list.append(smoothed_wave)

    if smoothed_waves_list:
        current_stacked_waves = np.array(smoothed_waves_list)
        if current_stacked_waves.shape[0] > 0:
            stacked_diff_waves_for_1samp_test[key] = current_stacked_waves
            ga_diff_waves[key] = np.mean(current_stacked_waves, axis=0)
            print(f"  GA Pd Diff Wave for '{key}': {current_stacked_waves.shape[0]} subjects.")
    else:
        ga_diff_waves[key] = None
        stacked_diff_waves_for_1samp_test[key] = None
        print(f"  No data for GA Pd Diff Wave for '{key}'.")

# --- 4. Temporal Permutation Cluster Test (1-Sample vs 0) ---
print("\n--- Step 4: Running 1-Sample Temporal Permutation Cluster Tests (vs 0) ---")
cluster_test_results_1samp = {}

for key, X_condition in stacked_diff_waves_for_1samp_test.items():
    if X_condition is None or X_condition.shape[0] < 2:
        print(f"  Condition '{key}': Not enough data for 1-sample cluster test. Skipping.")
        cluster_test_results_1samp[key] = None
        continue

    n_subjects_condition = X_condition.shape[0]
    df_condition = n_subjects_condition - 1
    t_threshold = t.ppf(1 - ALPHA_CLUSTER_FORMING / 2, df_condition)
    print(f"  Condition '{key}': {n_subjects_condition} subjects. t-threshold={t_threshold:.3f}")

    t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_condition, threshold=t_threshold, n_permutations=N_PERMUTATIONS_CLUSTER,
        tail=TAIL, n_jobs=N_JOBS, out_type='mask', seed=SEED
    )
    cluster_test_results_1samp[key] = {
        't_obs': t_obs, 'clusters': clusters, 'cluster_p_values': cluster_p_values
    }
    sig_clusters_found = np.sum(cluster_p_values < ALPHA_CLUSTER_SIGNIFICANCE)
    print(f"    Found {sig_clusters_found} significant cluster(s).")

# --- 5. Plotting Grand Averages with Error Bands and Test Results ---
print("\n--- Step 5: Plotting Grand Averages and Test Results ---")
plot_colors = {"no_prime": "grey", "neg_prime": "darkred", "pos_prime": "darkgreen"}
y_offsets_for_sig_lines = {key: Y_OFFSET_SIG_BASE + (i * Y_OFFSET_SIG_STEP) for i, key in enumerate(condition_keys)}

fig_diff, ax_diff = plt.subplots(figsize=(12, 8))
plot_successful = False

for key_plot in condition_keys:
    ga_wave_data = ga_diff_waves.get(key_plot)
    stacked_wave_data = stacked_diff_waves_for_1samp_test.get(key_plot)

    if ga_wave_data is not None and stacked_wave_data is not None and stacked_wave_data.shape[0] > 0:
        n_subs_this_cond = stacked_wave_data.shape[0]
        sem_wave = sem(stacked_wave_data, axis=0)
        df_ci = n_subs_this_cond - 1
        t_crit = t.ppf(1 - 0.05 / 2, df_ci) if df_ci > 0 else 1.96
        ci_range = sem_wave * t_crit

        ga_wave_plot = ga_wave_data * AMPLITUDE_SCALE_FACTOR
        ci_range_plot = ci_range * AMPLITUDE_SCALE_FACTOR

        ax_diff.plot(times_vector, ga_wave_plot, color=plot_colors[key_plot], lw=2.5,
                     label=f"{key_plot.replace('_', ' ').title()} (N={n_subs_this_cond})")
        ax_diff.fill_between(times_vector, ga_wave_plot - ci_range_plot, ga_wave_plot + ci_range_plot,
                             color=plot_colors[key_plot], alpha=0.1, label='_nolegend_')
        plot_successful = True

        if cluster_test_results_1samp.get(key_plot):
            stats = cluster_test_results_1samp[key_plot]
            for i, cl_mask in enumerate(stats['clusters']):
                p_val = stats['cluster_p_values'][i]
                if p_val < ALPHA_CLUSTER_SIGNIFICANCE:
                    cluster_times = times_vector[cl_mask]
                    if len(cluster_times) > 0:
                        ax_diff.hlines(y=y_offsets_for_sig_lines[key_plot], xmin=cluster_times[0],
                                       xmax=cluster_times[-1],
                                       color=plot_colors[key_plot], linewidth=6, alpha=0.9)
                        p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        ax_diff.text(cluster_times.mean(), y_offsets_for_sig_lines[key_plot], p_text,
                                     ha='center', va='center', color='white', weight='bold', fontsize=9)


# --- 6. Pairwise Temporal Permutation Cluster Tests ---
if PERFORM_PAIRWISE_COMPARISONS:
    print("\n--- Step 6: Running Pairwise Temporal Permutation Cluster Tests ---")
    condition_pairs = list(itertools.combinations(condition_keys, 2))
    y_offset_pairwise_current = Y_OFFSET_SIG_PAIRWISE

    for cond1, cond2 in condition_pairs:
        pair_key = f"{cond1}_vs_{cond2}"
        print(f"\n--- Comparing: {cond1} vs {cond2} ---")
        subs1 = {sub for sub, d in subject_diff_waves_smoothed.items() if cond1 in d}
        subs2 = {sub for sub, d in subject_diff_waves_smoothed.items() if cond2 in d}
        common_subjects = sorted(list(subs1.intersection(subs2)))

        if len(common_subjects) < 2:
            print(f"  Not enough common subjects ({len(common_subjects)}) to compare. Skipping.")
            continue

        print(f"  Found {len(common_subjects)} common subjects.")
        X1_paired = np.array([subject_diff_waves_smoothed[sub][cond1] for sub in common_subjects])
        X2_paired = np.array([subject_diff_waves_smoothed[sub][cond2] for sub in common_subjects])

        df_paired = len(common_subjects) - 1
        t_threshold_paired = t.ppf(1 - ALPHA_CLUSTER_FORMING / 2, df_paired)
        print(f"    Using t-threshold for cluster forming: {t_threshold_paired:.3f}")

        t_obs_p, clusters_p, p_values_p, _ = permutation_cluster_test(
            [X1_paired, X2_paired], stat_fun=None, threshold=t_threshold_paired,
            n_permutations=N_PERMUTATIONS_CLUSTER, tail=TAIL, n_jobs=N_JOBS, seed=SEED, out_type='mask'
        )
        print(f"    Found {np.sum(p_values_p < ALPHA_CLUSTER_SIGNIFICANCE)} significant cluster(s).")

        if plot_successful:
            for i, cl_mask in enumerate(clusters_p):
                if p_values_p[i] < ALPHA_CLUSTER_SIGNIFICANCE:
                    cluster_times = times_vector[cl_mask]
                    if len(cluster_times) > 0:
                        ax_diff.hlines(y=y_offset_pairwise_current, xmin=cluster_times[0], xmax=cluster_times[-1],
                                       color=PAIRWISE_SIG_LINE_COLOR, linewidth=6, alpha=0.9)
                        p_val = p_values_p[i]
                        p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        ax_diff.text(cluster_times.mean(), y_offset_pairwise_current,
                                     f"{cond1.split('_')[0]} vs {cond2.split('_')[0]} ({p_text})",
                                     ha='center', va='center', color='white', weight='bold', fontsize=9)
            y_offset_pairwise_current += Y_OFFSET_SIG_STEP

# --- Finalize Plot ---
if plot_successful:
    ax_diff.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_diff.axvline(0, color="black", linestyle=":", linewidth=1)
    ax_diff.legend(loc='upper right', frameon=True, fontsize=12)
    ax_diff.set_title("Grand Average Pd (Contra-Ipsi) by Priming Condition", fontsize=16, weight='bold')
    ax_diff.set_ylabel("Amplitude (µV)", fontsize=14)
    ax_diff.set_xlabel("Time (s)", fontsize=14)
    ax_diff.grid(True, linestyle=':', alpha=0.6)
    ax_diff.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
else:
    print("  Skipping Difference Wave plot as no GA data is available.")
    if 'fig_diff' in locals() and fig_diff:
        plt.close(fig_diff)

plt.show(block=True)
