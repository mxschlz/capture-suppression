import mne
import matplotlib.pyplot as plt
import os
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test  # Added permutation_cluster_test
import itertools  # For pairwise combinations

plt.ion()

# --- Parameters ---
EPOCH_TMIN, EPOCH_TMAX = 0, 0.7
TARGET_ELECTRODES_CONTRA_IPSI = ("FC5", "FC6")
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3
AMPLITUDE_SCALE_FACTOR = 1e6

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
PERFORM_PAIRWISE_COMPARISONS = False
Y_OFFSET_SIG_PAIRWISE = -0.01  # Single y-offset for significance lines on pairwise difference plots
PAIRWISE_SIG_LINE_COLOR = 'purple'  # Color for significance lines on pairwise plots

# --- Data Storage for Subject-Level Results ---
# subject_diff_waves[subject_string][condition_key] = diff_wave_numpy_array
subject_diff_waves = {}

times_vector = None
processed_subject_count = 0

# --- Subject Loop ---
print("Starting subject-level processing...")
for subject_id_int in subject_ids:
    subject_str = f"sub-{subject_id_int:02d}"
    print(f"\n--- Processing Subject: {subject_str} ---")
    subject_diff_waves[subject_str] = {}  # Initialize dict for this subject

    try:
        # ... (epoch loading and cropping remains the same) ...
        epoch_file_path_pattern = os.path.join(get_data_path(), "derivatives", "epoching", subject_str, "eeg",
                                               f"{subject_str}_task-spaceprime-epo.fif")
        epoch_files = glob.glob(epoch_file_path_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for {subject_str} using pattern: {epoch_file_path_pattern}. Skipping.")
            continue

        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub)} epochs.")
        epochs_sub.crop(EPOCH_TMIN, EPOCH_TMAX)
        print(f"  Cropped epochs to {EPOCH_TMIN}-{EPOCH_TMAX}s. {len(epochs_sub)} epochs remaining.")

        if times_vector is None:
            times_vector = epochs_sub.times.copy()

        priming_conditions_map = {
            "no_prime": "Priming==0",
            "neg_prime": "Priming==-1",
            "pos_prime": "Priming==1"
        }
        subject_contributed_to_any_diff_wave_this_subject = False

        for key, prime_event_selector in priming_conditions_map.items():
            print(f"  Processing '{key}' priming condition...")
            primed_epochs_sub = epochs_sub[prime_event_selector]

            if len(primed_epochs_sub) == 0:
                print(f"    No epochs for '{key}' priming for subject {subject_str}.")
                continue

            event_ids_in_primed_subset = list(primed_epochs_sub.event_id.keys())
            left_target_events = [eid for eid in event_ids_in_primed_subset if
                                  "Target-1" in eid and "Singleton-2" in eid]
            right_target_events = [eid for eid in event_ids_in_primed_subset if
                                   "Target-3" in eid and "Singleton-2" in eid]

            epochs_left_target_sub = primed_epochs_sub[left_target_events]
            epochs_right_target_sub = primed_epochs_sub[right_target_events]

            print(
                f"    '{key}' priming: Left target trials: {len(epochs_left_target_sub)}, Right target trials: {len(epochs_right_target_sub)}")

            if len(epochs_left_target_sub) > 0 and len(epochs_right_target_sub) > 0:
                ev_L_T_contra_data = epochs_left_target_sub.copy().average(picks=TARGET_ELECTRODES_CONTRA_IPSI[1]).data
                ev_R_T_contra_data = epochs_right_target_sub.copy().average(picks=TARGET_ELECTRODES_CONTRA_IPSI[0]).data
                contra_data_sub = np.mean([ev_L_T_contra_data, ev_R_T_contra_data], axis=0)

                ev_L_T_ipsi_data = epochs_left_target_sub.copy().average(picks=TARGET_ELECTRODES_CONTRA_IPSI[0]).data
                ev_R_T_ipsi_data = epochs_right_target_sub.copy().average(picks=TARGET_ELECTRODES_CONTRA_IPSI[1]).data
                ipsi_data_sub = np.mean([ev_L_T_ipsi_data, ev_R_T_ipsi_data], axis=0)

                diff_wave_sub = contra_data_sub - ipsi_data_sub
                subject_diff_waves[subject_str][key] = diff_wave_sub  # Store by subject_str then key
                subject_contributed_to_any_diff_wave_this_subject = True
                print(f"    Calculated and stored diff wave for '{key}'.")
            else:
                print(
                    f"    Skipping diff wave for '{key}' for subject {subject_str} due to insufficient left/right target trials.")

        if subject_contributed_to_any_diff_wave_this_subject:  # If subject contributed to any condition
            processed_subject_count += 1
        else:  # If subject contributed to no conditions, remove their entry
            if subject_str in subject_diff_waves and not subject_diff_waves[subject_str]:
                del subject_diff_waves[subject_str]


    except Exception as e:
        print(f"  Error processing subject {subject_str}: {e}. Skipping this subject.")
        if subject_str in subject_diff_waves:  # Clean up partial data for this subject on error
            del subject_diff_waves[subject_str]
        continue

print(
    f"\n--- Finished subject-level processing. Processed data for {processed_subject_count} subjects who contributed to at least one condition. ---")

if processed_subject_count == 0:
    print("No subjects were processed successfully. Exiting.")
    exit()
if times_vector is None:
    print("Times vector could not be determined. Exiting.")
    exit()

# --- Grand Average Calculation & Data Stacking for 1-Sample Tests ---
print("\nCalculating Grand Averages and Stacking Data for 1-Sample Tests...")
ga_diff_waves = {}
stacked_diff_waves_for_1samp_test = {}  # Renamed for clarity

condition_keys = list(priming_conditions_map.keys())

for key in condition_keys:
    temp_list_for_stacking = []
    for sub_str_loop in subject_diff_waves.keys():  # Iterate over subjects who have *any* data
        if key in subject_diff_waves[sub_str_loop]:
            temp_list_for_stacking.append(subject_diff_waves[sub_str_loop][key])

    if temp_list_for_stacking:
        current_stacked_waves = np.array(temp_list_for_stacking).squeeze()
        if current_stacked_waves.ndim == 1 and current_stacked_waves.size > 0:  # Single subject case
            current_stacked_waves = current_stacked_waves[np.newaxis, :]

        if current_stacked_waves.shape[0] > 0:
            stacked_diff_waves_for_1samp_test[key] = current_stacked_waves
            ga_diff_waves[key] = np.mean(current_stacked_waves, axis=0)
            print(f"  GA Diff Wave for '{key}': {current_stacked_waves.shape[0]} subjects.")
        else:  # Should not happen if temp_list_for_stacking was non-empty
            ga_diff_waves[key] = None
            print(f"  No data for GA Diff Wave for '{key}' (after stacking).")
    else:
        ga_diff_waves[key] = None
        stacked_diff_waves_for_1samp_test[key] = None  # Explicitly None
        print(f"  No data for GA Diff Wave for '{key}'.")

# --- Temporal Permutation Cluster Test (1-Sample vs 0) ---
print("\nRunning 1-Sample Temporal Permutation Cluster Tests (vs 0)...")
cluster_test_results_1samp = {}  # Renamed for clarity

for key, X_condition in stacked_diff_waves_for_1samp_test.items():
    if X_condition is None or X_condition.shape[0] == 0:
        print(f"  Condition '{key}': No data available for 1-sample cluster test.")
        cluster_test_results_1samp[key] = None
        continue
    # ... (1-sample test logic remains the same, using X_condition) ...
    n_subjects_condition = X_condition.shape[0]
    print(f"  Condition '{key}': {n_subjects_condition} subjects.")

    if n_subjects_condition < 2:
        print(
            f"    Skipping 1-sample cluster test for '{key}', not enough subjects (N={n_subjects_condition}, min: 2).")
        cluster_test_results_1samp[key] = None
        continue

    df_condition = n_subjects_condition - 1
    t_threshold = t.ppf(1 - ALPHA_CLUSTER_FORMING / 2, df_condition)
    print(f"    Using t-threshold for cluster forming: {t_threshold:.3f} (df={df_condition})")

    t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_condition, threshold=t_threshold, n_permutations=N_PERMUTATIONS_CLUSTER,
        tail=TAIL, n_jobs=N_JOBS, out_type='mask', seed=SEED
    )
    cluster_test_results_1samp[key] = {
        't_obs': t_obs, 'clusters': clusters, 'cluster_p_values': cluster_p_values,
        'H0': H0, 't_threshold_used': t_threshold
    }
    sig_clusters_found = np.sum(cluster_p_values < ALPHA_CLUSTER_SIGNIFICANCE)
    print(f"    Found {sig_clusters_found} significant cluster(s) for '{key}' (p < {ALPHA_CLUSTER_SIGNIFICANCE}).")

# --- Plotting Grand Averages (1-Sample Test Results) ---
print("\nPlotting Grand Averages (1-Sample Test Results)...")
plot_colors = {"no_prime": "grey", "neg_prime": "darkred", "pos_prime": "darkgreen"}
y_offsets_for_sig_lines = {}
for i, cond_key_plot in enumerate(condition_keys):
    y_offsets_for_sig_lines[cond_key_plot] = Y_OFFSET_SIG_BASE + (i * Y_OFFSET_SIG_STEP)

fig_diff_1samp, ax_diff_1samp = plt.subplots(figsize=(10, 6))  # Renamed fig and ax
plot_successful_1samp = False

for key_plot in condition_keys:
    if ga_diff_waves.get(key_plot) is not None:
        n_subs_this_cond = 0
        if stacked_diff_waves_for_1samp_test.get(key_plot) is not None:
            n_subs_this_cond = stacked_diff_waves_for_1samp_test[key_plot].shape[0]
        if n_subs_this_cond == 0: continue

        ax_diff_1samp.plot(times_vector,
                           savgol_filter(ga_diff_waves[key_plot] * AMPLITUDE_SCALE_FACTOR,
                                         window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER),
                           color=plot_colors[key_plot],
                           label=f"{key_plot.replace('_', ' ').title()} (N={n_subs_this_cond})")
        plot_successful_1samp = True

        if cluster_test_results_1samp.get(key_plot):
            stats = cluster_test_results_1samp[key_plot]
            current_y_offset = y_offsets_for_sig_lines[key_plot]
            for i, cl_mask in enumerate(stats['clusters']):
                if stats['cluster_p_values'][i] < ALPHA_CLUSTER_SIGNIFICANCE:
                    cluster_times = times_vector[cl_mask]
                    if len(cluster_times) > 0:
                        ax_diff_1samp.hlines(y=current_y_offset, xmin=cluster_times[0], xmax=cluster_times[-1],
                                             color=plot_colors[key_plot], linewidth=5, alpha=0.7, label='_nolegend_')
if plot_successful_1samp:
    ax_diff_1samp.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_diff_1samp.axvline(0, color="black", linestyle=":", linewidth=0.8)
    handles, labels = ax_diff_1samp.get_legend_handles_labels()
    ax_diff_1samp.legend(handles, labels, loc='upper right')
    ax_diff_1samp.set_title(
        f"Grand Average N2ac (vs 0) at {TARGET_ELECTRODES_CONTRA_IPSI[0]}/{TARGET_ELECTRODES_CONTRA_IPSI[1]}")
    ax_diff_1samp.set_ylabel("Amplitude (µV)")
    ax_diff_1samp.set_xlabel("Time (s)")
else:
    print("  Skipping 1-Sample Difference Wave plot as no GA data is available.")
    if fig_diff_1samp: plt.close(fig_diff_1samp)

# --- Pairwise Temporal Permutation Cluster Tests ---
if PERFORM_PAIRWISE_COMPARISONS:
    print("\nRunning Pairwise Temporal Permutation Cluster Tests...")
    pairwise_cluster_results = {}
    pairwise_ga_diffs = {}  # To store GA(cond1) - GA(cond2) for plotting

    # Generate pairs of conditions
    condition_pairs = list(itertools.combinations(condition_keys, 2))

    for cond1, cond2 in condition_pairs:
        pair_key = f"{cond1}_vs_{cond2}"
        print(f"  Comparing: {cond1} vs {cond2}")

        X1_paired_list = []
        X2_paired_list = []
        common_subject_ids = []

        for sub_str_loop in subject_diff_waves.keys():
            if cond1 in subject_diff_waves[sub_str_loop] and cond2 in subject_diff_waves[sub_str_loop]:
                X1_paired_list.append(subject_diff_waves[sub_str_loop][cond1])
                X2_paired_list.append(subject_diff_waves[sub_str_loop][cond2])
                common_subject_ids.append(sub_str_loop)

        if not common_subject_ids:
            print(f"    No common subjects found for {pair_key}. Skipping.")
            pairwise_cluster_results[pair_key] = None
            pairwise_ga_diffs[pair_key] = None
            continue

        X1_paired = np.array(X1_paired_list).squeeze()
        X2_paired = np.array(X2_paired_list).squeeze()

        # Ensure they are 2D if only one common subject (though test needs N>=2)
        if X1_paired.ndim == 1: X1_paired = X1_paired[np.newaxis, :]
        if X2_paired.ndim == 1: X2_paired = X2_paired[np.newaxis, :]

        n_common_subjects = len(common_subject_ids)
        print(f"    Number of common subjects for {pair_key}: {n_common_subjects}")

        if n_common_subjects < 2:
            print(
                f"    Skipping cluster test for {pair_key}, not enough common subjects (N={n_common_subjects}, min: 2).")
            pairwise_cluster_results[pair_key] = None
            pairwise_ga_diffs[pair_key] = None
            continue

        df_paired = n_common_subjects - 1
        # For paired t-test, threshold is on the differences X1-X2
        t_threshold_paired = t.ppf(1 - ALPHA_CLUSTER_FORMING / 2, df_paired)
        print(f"    Using paired t-threshold for cluster forming: {t_threshold_paired:.3f} (df={df_paired})")

        # permutation_cluster_test with two arrays performs a paired test by default
        t_obs_p, clusters_p, cluster_p_values_p, H0_p = permutation_cluster_test(
            [X1_paired, X2_paired],  # Pass as a list of two arrays
            threshold=t_threshold_paired,
            n_permutations=N_PERMUTATIONS_CLUSTER,
            tail=TAIL,  # Two-tailed comparison
            n_jobs=N_JOBS,
            out_type='mask',
            seed=SEED
        )
        pairwise_cluster_results[pair_key] = {
            't_obs': t_obs_p, 'clusters': clusters_p, 'cluster_p_values': cluster_p_values_p,
            'H0': H0_p, 't_threshold_used': t_threshold_paired, 'n_common': n_common_subjects
        }
        sig_clusters_found_p = np.sum(cluster_p_values_p < ALPHA_CLUSTER_SIGNIFICANCE)
        print(
            f"    Found {sig_clusters_found_p} significant cluster(s) for {pair_key} (p < {ALPHA_CLUSTER_SIGNIFICANCE}).")

        # Calculate GA of difference for plotting: GA(X1_paired) - GA(X2_paired)
        ga_X1_paired = np.mean(X1_paired, axis=0)
        ga_X2_paired = np.mean(X2_paired, axis=0)
        pairwise_ga_diffs[pair_key] = ga_X1_paired - ga_X2_paired

    # --- Plotting Pairwise Comparison Results ---
    if pairwise_cluster_results:  # Check if any pairwise results exist
        n_pairs = len(condition_pairs)
        if n_pairs > 0:
            fig_pairwise, axes_pairwise = plt.subplots(n_pairs, 1, figsize=(10, 4 * n_pairs), sharex=True, sharey=True)
            if n_pairs == 1:  # Ensure axes_pairwise is always an array
                axes_pairwise = [axes_pairwise]
            fig_pairwise.suptitle("Pairwise N2ac Comparisons", fontsize=16)
            plot_successful_pairwise = False

            for i_pair, (cond1, cond2) in enumerate(condition_pairs):
                pair_key = f"{cond1}_vs_{cond2}"
                ax_p = axes_pairwise[i_pair]

                ga_difference_to_plot = pairwise_ga_diffs.get(pair_key)
                stats_p = pairwise_cluster_results.get(pair_key)

                if ga_difference_to_plot is not None:
                    n_common = stats_p['n_common'] if stats_p else 'N/A'
                    ax_p.plot(times_vector,
                              savgol_filter(ga_difference_to_plot * AMPLITUDE_SCALE_FACTOR,
                                            window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER),
                              color='black',  # Plot the difference wave in black
                              label=f"GA Difference (N={n_common})")
                    plot_successful_pairwise = True

                    if stats_p:
                        for i_cl, cl_mask_p in enumerate(stats_p['clusters']):
                            if stats_p['cluster_p_values'][i_cl] < ALPHA_CLUSTER_SIGNIFICANCE:
                                cluster_times_p = times_vector[cl_mask_p]
                                if len(cluster_times_p) > 0:
                                    ax_p.hlines(y=Y_OFFSET_SIG_PAIRWISE,
                                                xmin=cluster_times_p[0], xmax=cluster_times_p[-1],
                                                color=PAIRWISE_SIG_LINE_COLOR, linewidth=5, alpha=0.7,
                                                label='_nolegend_')  # Significance line

                    ax_p.axhline(0, color="black", linestyle="--", linewidth=0.8)
                    ax_p.axvline(0, color="black", linestyle=":", linewidth=0.8)
                    ax_p.legend(loc='upper right')
                    title_cond1 = cond1.replace('_', ' ').title()
                    title_cond2 = cond2.replace('_', ' ').title()
                    ax_p.set_title(f"{title_cond1} vs. {title_cond2}")
                    ax_p.set_ylabel("Amplitude Diff. (µV)")
                else:
                    ax_p.text(0.5, 0.5, "No data for this comparison", ha='center', va='center',
                              transform=ax_p.transAxes)

            if plot_successful_pairwise:
                axes_pairwise[-1].set_xlabel("Time (s)")  # Set x-label only on the last subplot
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
            else:
                if fig_pairwise: plt.close(fig_pairwise)

plt.show(block=True)
print("\nDone.")
