import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os  # Added for path joining
from scipy.stats import ttest_rel, t, sem  # Import t for theoretical threshold
# from mne.stats import permutation_t_test # For cluster stats (not used in this part)
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import seaborn as sns  # For nicer plots

plt.ion()

# --- Analysis Parameters ---
# TFR settings
freqs = np.arange(5, 20, 1)  # Frequency band for LI calculation
n_cycles = freqs / 2  # Or a fixed number like 7
method = "morlet"  # Wavelet method
decim = 5  # Decimation factor
n_jobs = 5  # Number of jobs for parallel processing

# Channels to average over for the Left-Right difference calculation in get_lateralization_index
# This defines the channels whose power is used to compute L-R power for each stimulus side.
# The resulting LI TFR will have data for these channels.
picks_for_diff = 'eeg'


print(f"Calculating Left-Right difference using picks: {picks_for_diff}")

# Time window of interest for analysis and plotting (adjust as needed)
tmin_analysis, tmax_analysis = -1.0, 1.0  # Example time window relative to stimulus

# --- Priming Conditions ---
priming_conditions_map = {
    "no_priming": {"query": "Priming==0", "label": "No Priming (Priming==0)"},
    "neg_priming": {"query": "Priming==-1", "label": "Negative Priming (Priming==-1)"},
    "pos_priming": {"query": "Priming==1", "label": "Positive Priming (Priming==1)"}
}

# --- Data Storage ---
# Dictionaries to store subject-level results (Left - Right alpha power difference TFR objects)
subject_results = {}
for p_key in priming_conditions_map:
    subject_results[f'target_li_{p_key}'] = []
    subject_results[f'singleton_li_{p_key}'] = []

processed_subjects_list = []  # Renamed to avoid conflict
times_vector = None  # To store time vector from the first subject

# --- Subject Loop ---
for subject in subject_ids[:]:
    subject_processed_at_least_one_condition = None
    print(f"\n--- Processing Subject: {subject} ---")
    try:
        epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject}", "eeg",
                                          f"sub-{subject}_task-spaceprime-epo.fif")
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for subject {subject} using pattern: {epoch_file_pattern}. Skipping.")
            continue
        epochs_sub_full = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub_full)} epochs.")
    except Exception as e:
        print(f"  Error loading data for subject {subject}: {e}. Skipping.")
        continue

    for p_key, p_info in priming_conditions_map.items():
        p_query = p_info["query"]
        print(f"    Processing priming condition: {p_key} ({p_query})")

        try:
            epochs_sub_priming = epochs_sub_full[p_query].copy()
            if len(epochs_sub_priming) == 0:
                print(f"      No trials for priming condition {p_key}. Skipping this condition for this subject.")
                subject_results[f'target_li_{p_key}'].append(
                    None)  # Append None to keep subject counts aligned if needed
                subject_results[f'singleton_li_{p_key}'].append(None)
                continue

            all_conds_sub_priming = list(epochs_sub_priming.event_id.keys())


            # --- Function to compute TFR, average over trials/freqs/channels, and return Left-Right difference ---
            # This function calculates LI per channel specified in picks_calc
            def get_lateralization_index(left_epochs, right_epochs, picks_calc, freqs_calc, n_cycles_calc, decim_calc,
                                         method_calc,
                                         tmin_crop=None, tmax_crop=None, n_jobs_calc=-1):
                tfr_left = left_epochs.compute_tfr(
                    method=method_calc, picks=picks_calc, freqs=freqs_calc,
                    n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_calc,
                    return_itc=False, average=False, verbose=False)  # average=False gives per-trial TFRs
                if tmin_crop is not None or tmax_crop is not None: tfr_left.crop(tmin=tmin_crop, tmax=tmax_crop)
                alpha_left_avg = tfr_left.average()  # Now average over trials -> (n_channels, n_freqs, n_times)

                tfr_right = right_epochs.compute_tfr(
                    method=method_calc, picks=picks_calc, freqs=freqs_calc,
                    n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_calc,
                    return_itc=False, average=False, verbose=False)
                if tmin_crop is not None or tmax_crop is not None: tfr_right.crop(tmin=tmin_crop, tmax=tmax_crop)
                alpha_right_avg = tfr_right.average()

                # LI = (Power_Contra - Power_Ipsi) / (Power_Contra + Power_Ipsi)
                # Here, left_epochs are e.g. Left Target, right_epochs are Right Target.
                # So alpha_left_avg is power when stimulus is Left. alpha_right_avg is power when stimulus is Right.
                # The LI formula here is (Left Stim Power - Right Stim Power) / (Left Stim Power + Right Stim Power)
                # This is not directly contra-ipsi LI unless picks_calc are chosen carefully or interpreted later.
                # The original script implies this is a (Left Stimulus - Right Stimulus) difference.
                # This will be interpreted by picking left/right ROIs later.
                li_data = (alpha_left_avg.get_data() - alpha_right_avg.get_data()) / (
                            alpha_left_avg.get_data() + alpha_right_avg.get_data())

                # Create an info object for the LI TFR. It should have the same channels as alpha_left_avg.
                # Ensure channel names are preserved.
                li_info = alpha_left_avg.info.copy()

                li_power = mne.time_frequency.AverageTFRArray(data=li_data, info=li_info, freqs=freqs_calc,
                                                              times=alpha_left_avg.times)  # Use times from one of the averaged TFRs
                times_calc = alpha_left_avg.times
                return li_power, times_calc


            # Targets
            alpha_diff_target_lr_p = None
            try:
                left_target_epochs_sub_p = epochs_sub_priming[
                    [x for x in all_conds_sub_priming if "Target-1-Singleton-2" in x]].copy()
                right_target_epochs_sub_p = epochs_sub_priming[
                    [x for x in all_conds_sub_priming if "Target-3-Singleton-2" in x]].copy()
                if len(left_target_epochs_sub_p) == 0 or len(right_target_epochs_sub_p) == 0:
                    print(f"      Zero trials in one of the target conditions for {p_key}. Skipping target LI.")
                else:
                    #mne.epochs.equalize_epoch_counts([left_target_epochs_sub_p, right_target_epochs_sub_p])
                    print(
                        f"      Target trials for {p_key}: {len(left_target_epochs_sub_p)} left, {len(right_target_epochs_sub_p)} right")
                    alpha_diff_target_lr_p, times_t = get_lateralization_index(
                        left_target_epochs_sub_p, right_target_epochs_sub_p, picks_for_diff,
                        freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs
                    )
                    if times_vector is None: times_vector = times_t
                    subject_processed_at_least_one_condition = True
            except Exception as e_target:
                print(f"      Error processing targets for {p_key}, subject {subject}: {e_target}")
            subject_results[f'target_li_{p_key}'].append(alpha_diff_target_lr_p)

            # Singletons
            alpha_diff_singleton_lr_p = None
            try:
                left_singleton_epochs_sub_p = epochs_sub_priming[
                    [x for x in all_conds_sub_priming if "Target-2-Singleton-1" in x]].copy()
                right_singleton_epochs_sub_p = epochs_sub_priming[
                    [x for x in all_conds_sub_priming if "Target-2-Singleton-3" in x]].copy()
                if len(left_singleton_epochs_sub_p) == 0 or len(right_singleton_epochs_sub_p) == 0:
                    print(f"      Zero trials in one of the singleton conditions for {p_key}. Skipping singleton LI.")
                else:
                    #mne.epochs.equalize_epoch_counts([left_singleton_epochs_sub_p, right_singleton_epochs_sub_p])
                    print(
                        f"      Singleton trials for {p_key}: {len(left_singleton_epochs_sub_p)} left, {len(right_singleton_epochs_sub_p)} right")
                    alpha_diff_singleton_lr_p, times_s = get_lateralization_index(
                        left_singleton_epochs_sub_p, right_singleton_epochs_sub_p, picks_for_diff,
                        freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs
                    )
                    if times_vector is None: times_vector = times_s
                    subject_processed_at_least_one_condition = True
            except Exception as e_singleton:
                print(f"      Error processing singletons for {p_key}, subject {subject}: {e_singleton}")
            subject_results[f'singleton_li_{p_key}'].append(alpha_diff_singleton_lr_p)

        except Exception as e_prime_loop:
            print(f"    Error in priming loop for {p_key}, subject {subject}: {e_prime_loop}. Appending None.")
            subject_results[f'target_li_{p_key}'].append(None)
            subject_results[f'singleton_li_{p_key}'].append(None)
            continue

    if subject_processed_at_least_one_condition:
        processed_subjects_list.append(subject)

print(
    f"\n--- Successfully processed {len(processed_subjects_list)} subjects who had data for at least one condition: {processed_subjects_list} ---")

if not processed_subjects_list:  # Check if any subject was processed at all
    raise RuntimeError("No subjects were successfully processed for any condition. Cannot continue.")
if times_vector is None:
    raise RuntimeError("Could not determine the time vector for analysis. Ensure at least one TFR was computed.")

# --- Group Level Analysis ---

# Commented out initial TFR plots to focus on hemispheric analysis per priming condition
ga_li_target_pos_priming = mne.grand_average(subject_results['target_li_pos_priming'])
ga_li_target_neg_priming = mne.grand_average(subject_results['target_li_neg_priming'])
ga_li_target_no_priming = mne.grand_average(subject_results['target_li_no_priming'])

ga_li_singleton_pos_priming = mne.grand_average(subject_results['singleton_li_pos_priming'])
ga_li_singleton_neg_priming = mne.grand_average(subject_results["singleton_li_neg_priming"])
ga_li_singleton_no_priming = mne.grand_average(subject_results["singleton_li_no_priming"])

ga_diff_pos_priming = ga_li_target_pos_priming - ga_li_singleton_pos_priming
ga_diff_neg_priming = ga_li_target_neg_priming - ga_li_singleton_neg_priming
ga_diff_no_priming = ga_li_target_no_priming - ga_li_singleton_no_priming

# ... plotting logic ...
#ga_diff_no_priming.plot_topo()
#ga_diff_neg_priming.plot_topo()
#ga_diff_pos_priming.plot_topo()


# --- Data Storage for Consolidated Plotting ---
plot_data_all_conditions = {
    "target_li_left": {}, "target_li_right": {},
    "target_sem_left": {}, "target_sem_right": {},
    "target_t_values": {}, "target_t_thresh": {}, "target_n_subs": {},

    "singleton_li_left": {}, "singleton_li_right": {},
    "singleton_sem_left": {}, "singleton_sem_right": {},
    "singleton_t_values": {}, "singleton_t_thresh": {}, "singleton_n_subs": {}
}

# --- Hemispheric Analysis Loop (Data Extraction) ---
# Define ROI for hemispheric analysis
left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
alpha_fmin, alpha_fmax = 8, 12  # Alpha band for averaging hemispheric LI
alpha_stat = 0.05  # Significance level for t-tests

for p_key, p_info in priming_conditions_map.items():
    p_label = p_info["label"]
    print(f"\n--- Extracting Hemispheric LI Data for: {p_label} ---")

    current_target_li_list = [res for res in subject_results[f'target_li_{p_key}'] if res is not None]
    current_singleton_li_list = [res for res in subject_results[f'singleton_li_{p_key}'] if res is not None]

    n_subs_condition_target = len(current_target_li_list)
    n_subs_condition_singleton = len(current_singleton_li_list)

    print(f"  Number of subjects for Target LI ({p_label}): {n_subs_condition_target}")
    print(f"  Number of subjects for Singleton LI ({p_label}): {n_subs_condition_singleton}")

    if n_subs_condition_target < 1 and n_subs_condition_singleton < 1:  # Allow N=1 for plotting GA, T-test needs N>=2
        print(f"  Not enough subjects with data for {p_label}. Skipping data extraction for this condition.")
        continue

    target_data_tc = dict(left_hemisphere=[], right_hemisphere=[])
    singleton_data_tc = dict(left_hemisphere=[], right_hemisphere=[])

    # Extract hemispheric time courses for Targets
    if n_subs_condition_target > 0:
        for tfr_target_li in current_target_li_list:
            try:
                current_left_roi = [ch for ch in left_roi if ch in tfr_target_li.ch_names]
                current_right_roi = [ch for ch in right_roi if ch in tfr_target_li.ch_names]

                if not current_left_roi: print(
                    f"    Warning: No channels from left_roi found in target TFR for a subject in {p_label}.")
                if not current_right_roi: print(
                    f"    Warning: No channels from right_roi found in target TFR for a subject in {p_label}.")

                if current_left_roi:
                    lh_tc = tfr_target_li.copy().pick(current_left_roi).crop(fmin=alpha_fmin,
                                                                             fmax=alpha_fmax).get_data().mean(
                        axis=(0, 1))
                    target_data_tc["left_hemisphere"].append(lh_tc)
                else:
                    target_data_tc["left_hemisphere"].append(np.full_like(times_vector, np.nan))

                if current_right_roi:
                    rh_tc = tfr_target_li.copy().pick(current_right_roi).crop(fmin=alpha_fmin,
                                                                              fmax=alpha_fmax).get_data().mean(
                        axis=(0, 1))
                    target_data_tc["right_hemisphere"].append(rh_tc)
                else:
                    target_data_tc["right_hemisphere"].append(np.full_like(times_vector, np.nan))
            except Exception as e_hemi_extract:
                print(f"    Error extracting hemispheric target LI for a subject in {p_label}: {e_hemi_extract}")
                target_data_tc["left_hemisphere"].append(np.full_like(times_vector, np.nan))
                target_data_tc["right_hemisphere"].append(np.full_like(times_vector, np.nan))

    # Extract hemispheric time courses for Singletons
    if n_subs_condition_singleton > 0:
        for tfr_singleton_li in current_singleton_li_list:
            try:
                current_left_roi = [ch for ch in left_roi if ch in tfr_singleton_li.ch_names]
                current_right_roi = [ch for ch in right_roi if ch in tfr_singleton_li.ch_names]

                if not current_left_roi: print(
                    f"    Warning: No channels from left_roi found in singleton TFR for a subject in {p_label}.")
                if not current_right_roi: print(
                    f"    Warning: No channels from right_roi found in singleton TFR for a subject in {p_label}.")

                if current_left_roi:
                    lh_tc = tfr_singleton_li.copy().pick(current_left_roi).crop(fmin=alpha_fmin,
                                                                                fmax=alpha_fmax).get_data().mean(
                        axis=(0, 1))
                    singleton_data_tc["left_hemisphere"].append(lh_tc)
                else:
                    singleton_data_tc["left_hemisphere"].append(np.full_like(times_vector, np.nan))

                if current_right_roi:
                    rh_tc = tfr_singleton_li.copy().pick(current_right_roi).crop(fmin=alpha_fmin,
                                                                                 fmax=alpha_fmax).get_data().mean(
                        axis=(0, 1))
                    singleton_data_tc["right_hemisphere"].append(rh_tc)
                else:
                    singleton_data_tc["right_hemisphere"].append(np.full_like(times_vector, np.nan))
            except Exception as e_hemi_extract:
                print(f"    Error extracting hemispheric singleton LI for a subject in {p_label}: {e_hemi_extract}")
                singleton_data_tc["left_hemisphere"].append(np.full_like(times_vector, np.nan))
                singleton_data_tc["right_hemisphere"].append(np.full_like(times_vector, np.nan))

    target_left_all = np.array([arr for arr in target_data_tc["left_hemisphere"] if not np.all(np.isnan(arr))])
    target_right_all = np.array([arr for arr in target_data_tc["right_hemisphere"] if not np.all(np.isnan(arr))])
    singleton_left_all = np.array([arr for arr in singleton_data_tc["left_hemisphere"] if not np.all(np.isnan(arr))])
    singleton_right_all = np.array([arr for arr in singleton_data_tc["right_hemisphere"] if not np.all(np.isnan(arr))])

    min_subs_target = 0
    if target_left_all.ndim > 1 and target_right_all.ndim > 1:
        min_subs_target = min(target_left_all.shape[0], target_right_all.shape[0])
        if min_subs_target > 0:
            target_left_all = target_left_all[:min_subs_target, :]
            target_right_all = target_right_all[:min_subs_target, :]
        else:
            target_left_all = np.empty((0, len(times_vector)))
            target_right_all = np.empty((0, len(times_vector)))
    elif not (target_left_all.ndim > 1 and target_left_all.shape[0] > 0):  # handles empty or 1D empty
        target_left_all = np.empty((0, len(times_vector)))
    elif not (target_right_all.ndim > 1 and target_right_all.shape[0] > 0):
        target_right_all = np.empty((0, len(times_vector)))

    min_subs_singleton = 0
    if singleton_left_all.ndim > 1 and singleton_right_all.ndim > 1:
        min_subs_singleton = min(singleton_left_all.shape[0], singleton_right_all.shape[0])
        if min_subs_singleton > 0:
            singleton_left_all = singleton_left_all[:min_subs_singleton, :]
            singleton_right_all = singleton_right_all[:min_subs_singleton, :]
        else:
            singleton_left_all = np.empty((0, len(times_vector)))
            singleton_right_all = np.empty((0, len(times_vector)))
    elif not (singleton_left_all.ndim > 1 and singleton_left_all.shape[0] > 0):
        singleton_left_all = np.empty((0, len(times_vector)))
    elif not (singleton_right_all.ndim > 1 and singleton_right_all.shape[0] > 0):
        singleton_right_all = np.empty((0, len(times_vector)))

    # Store N for t-test (paired, so min_subs_target/singleton is the N for the t-test)
    plot_data_all_conditions["target_n_subs"][p_key] = min_subs_target
    plot_data_all_conditions["singleton_n_subs"][p_key] = min_subs_singleton

    # T-tests
    stat_target, t_thresh_target = np.full_like(times_vector, np.nan), np.inf
    if min_subs_target >= 2:
        stat_target, _ = ttest_rel(target_left_all, target_right_all, axis=0, nan_policy='omit')
        df_target = min_subs_target - 1
        t_thresh_target = t.ppf(1 - alpha_stat / 2, df=df_target)
    else:
        print(f"  Not enough paired data for target t-test in {p_label} (N={min_subs_target}).")
    plot_data_all_conditions["target_t_values"][p_key] = stat_target
    plot_data_all_conditions["target_t_thresh"][p_key] = t_thresh_target

    stat_singleton, t_thresh_singleton = np.full_like(times_vector, np.nan), np.inf
    if min_subs_singleton >= 2:
        stat_singleton, _ = ttest_rel(singleton_left_all, singleton_right_all, axis=0, nan_policy='omit')
        df_singleton = min_subs_singleton - 1
        t_thresh_singleton = t.ppf(1 - alpha_stat / 2, df=df_singleton)
    else:
        print(f"  Not enough paired data for singleton t-test in {p_label} (N={min_subs_singleton}).")
    plot_data_all_conditions["singleton_t_values"][p_key] = stat_singleton
    plot_data_all_conditions["singleton_t_thresh"][p_key] = t_thresh_singleton

    # Grand Averages and SEM (use original N before matching for t-test for GA/SEM if desired, or use min_subs)
    # For consistency with t-test N, let's use min_subs_target/singleton for GA/SEM as well.
    # If target_left_all or target_right_all became empty, min_subs_target would be 0.

    # N for GA/SEM should be based on how many subjects contributed to target_left_all and target_right_all *before* pairing for t-test
    # However, the current structure uses min_subs_target for GA as well. Let's stick to that for now.
    n_ga_target_left = target_left_all.shape[0] if target_left_all.ndim > 1 else 0
    n_ga_target_right = target_right_all.shape[0] if target_right_all.ndim > 1 else 0
    n_ga_singleton_left = singleton_left_all.shape[0] if singleton_left_all.ndim > 1 else 0
    n_ga_singleton_right = singleton_right_all.shape[0] if singleton_right_all.ndim > 1 else 0

    plot_data_all_conditions["target_li_left"][p_key] = np.nanmean(target_left_all,
                                                                   axis=0) if n_ga_target_left > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["target_sem_left"][p_key] = sem(target_left_all, axis=0,
                                                             nan_policy='omit') if n_ga_target_left > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["target_li_right"][p_key] = np.nanmean(target_right_all,
                                                                    axis=0) if n_ga_target_right > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["target_sem_right"][p_key] = sem(target_right_all, axis=0,
                                                              nan_policy='omit') if n_ga_target_right > 0 else np.full_like(
        times_vector, np.nan)

    plot_data_all_conditions["singleton_li_left"][p_key] = np.nanmean(singleton_left_all,
                                                                      axis=0) if n_ga_singleton_left > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["singleton_sem_left"][p_key] = sem(singleton_left_all, axis=0,
                                                                nan_policy='omit') if n_ga_singleton_left > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["singleton_li_right"][p_key] = np.nanmean(singleton_right_all,
                                                                       axis=0) if n_ga_singleton_right > 0 else np.full_like(
        times_vector, np.nan)
    plot_data_all_conditions["singleton_sem_right"][p_key] = sem(singleton_right_all, axis=0,
                                                                 nan_policy='omit') if n_ga_singleton_right > 0 else np.full_like(
        times_vector, np.nan)

# --- Consolidated Plotting ---
fig_compact, axes_compact = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
fig_compact.suptitle(f"Hemispheric Alpha LI ({alpha_fmin}-{alpha_fmax} Hz) - Across Priming Conditions", fontsize=16)

line_colors = {"no_priming": "black", "neg_priming": "red", "pos_priming": "green"}
fill_colors = {"no_priming": "grey", "neg_priming": "salmon", "pos_priming": "lightgreen"}

# Plot 1: Target LI (Left vs Right Hemisphere)
ax = axes_compact[0, 0]
for p_key, p_info in priming_conditions_map.items():
    if p_key not in plot_data_all_conditions["target_li_left"]: continue

    ga_left = plot_data_all_conditions["target_li_left"][p_key]
    sem_left = plot_data_all_conditions["target_sem_left"][p_key]
    ga_right = plot_data_all_conditions["target_li_right"][p_key]
    sem_right = plot_data_all_conditions["target_sem_right"][p_key]
    # N for LI plot can be the N that went into the GA for left/right separately, or the N for t-test.
    # Using N for t-test (min_subs_target) for consistency in labeling.
    n_s = plot_data_all_conditions["target_n_subs"].get(p_key, 0)

    if n_s > 0 and not np.all(np.isnan(ga_left)):  # Check if data exists
        ax.plot(times_vector, ga_left, color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} Left (N={n_s})")
        ax.fill_between(times_vector, ga_left - sem_left, ga_left + sem_left,
                        color=fill_colors[p_key], alpha=0.3)
    if n_s > 0 and not np.all(np.isnan(ga_right)):
        ax.plot(times_vector, ga_right, color=line_colors[p_key], linestyle='--',
                label=f"{p_info['label']} Right (N={n_s})")
        ax.fill_between(times_vector, ga_right - sem_right, ga_right + sem_right,
                        color=fill_colors[p_key], alpha=0.3, hatch='..')  # Different hatch/alpha for clarity

ax.axhline(0, color='k', linestyle=':', lw=0.8)
ax.axvline(0, color='k', linestyle=':', lw=0.8)
ax.set_ylabel("Alpha LI")
ax.set_title("Target LI (Left vs Right Hemisphere)")
ax.legend(loc="best", fontsize='small')
sns.despine(ax=ax)

# Plot 2: Singleton LI (Left vs Right Hemisphere)
ax = axes_compact[0, 1]
for p_key, p_info in priming_conditions_map.items():
    if p_key not in plot_data_all_conditions["singleton_li_left"]: continue

    ga_left = plot_data_all_conditions["singleton_li_left"][p_key]
    sem_left = plot_data_all_conditions["singleton_sem_left"][p_key]
    ga_right = plot_data_all_conditions["singleton_li_right"][p_key]
    sem_right = plot_data_all_conditions["singleton_sem_right"][p_key]
    n_s = plot_data_all_conditions["singleton_n_subs"].get(p_key, 0)

    if n_s > 0 and not np.all(np.isnan(ga_left)):
        ax.plot(times_vector, ga_left, color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} Left (N={n_s})")
        ax.fill_between(times_vector, ga_left - sem_left, ga_left + sem_left,
                        color=fill_colors[p_key], alpha=0.3)
    if n_s > 0 and not np.all(np.isnan(ga_right)):
        ax.plot(times_vector, ga_right, color=line_colors[p_key], linestyle='--',
                label=f"{p_info['label']} Right (N={n_s})")
        ax.fill_between(times_vector, ga_right - sem_right, ga_right + sem_right,
                        color=fill_colors[p_key], alpha=0.3, hatch='..')

ax.axhline(0, color='k', linestyle=':', lw=0.8)
ax.axvline(0, color='k', linestyle=':', lw=0.8)
ax.set_title("Singleton LI (Left vs Right Hemisphere)")
ax.legend(loc="best", fontsize='small')
sns.despine(ax=ax)
axes_compact[0, 1].sharey(axes_compact[0, 0])  # Share Y-axis for LI plots if desired

# Plot 3: Target T-values
ax = axes_compact[1, 0]
for p_key, p_info in priming_conditions_map.items():
    if p_key not in plot_data_all_conditions["target_t_values"]: continue

    t_vals = plot_data_all_conditions["target_t_values"][p_key]
    t_thr = plot_data_all_conditions["target_t_thresh"][p_key]
    n_s = plot_data_all_conditions["target_n_subs"].get(p_key, 0)

    if n_s >= 2 and not np.all(np.isnan(t_vals)):
        ax.plot(times_vector, t_vals, color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} (N={n_s}, |t|>{t_thr:.2f})")

        sig_mask = np.abs(t_vals) > t_thr
        significant_times = times_vector[sig_mask]
        significant_t_values = t_vals[sig_mask]
        if len(significant_times) > 0:
            ax.scatter(significant_times, significant_t_values,
                       color=line_colors[p_key], marker='o', s=15, zorder=5,
                       label=f"_nolegend_")
    else:
        ax.plot([], [], color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} (N={n_s}, No T-test)")

ax.axhline(0, color='k', linestyle=':', lw=0.8)
ax.axvline(0, color='k', linestyle=':', lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("T-value")
ax.set_title("Target: T-test (L vs R Hemi LI)")
ax.legend(loc="best", fontsize='small')
sns.despine(ax=ax)

# Plot 4: Singleton T-values
ax = axes_compact[1, 1]
for p_key, p_info in priming_conditions_map.items():
    if p_key not in plot_data_all_conditions["singleton_t_values"]: continue

    t_vals = plot_data_all_conditions["singleton_t_values"][p_key]
    t_thr = plot_data_all_conditions["singleton_t_thresh"][p_key]
    n_s = plot_data_all_conditions["singleton_n_subs"].get(p_key, 0)

    if n_s >= 2 and not np.all(np.isnan(t_vals)):
        ax.plot(times_vector, t_vals, color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} (N={n_s}, |t|>{t_thr:.2f})")

        sig_mask = np.abs(t_vals) > t_thr
        significant_times = times_vector[sig_mask]
        significant_t_values = t_vals[sig_mask]
        if len(significant_times) > 0:
            ax.scatter(significant_times, significant_t_values,
                       color=line_colors[p_key], marker='o', s=15, zorder=5,
                       label=f"_nolegend_")
    else:
        ax.plot([], [], color=line_colors[p_key], linestyle='-',
                label=f"{p_info['label']} (N={n_s}, No T-test)")

ax.axhline(0, color='k', linestyle=':', lw=0.8)
ax.axvline(0, color='k', linestyle=':', lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_title("Singleton: T-test (L vs R Hemi LI)")
ax.legend(loc="best", fontsize='small')
sns.despine(ax=ax)
ax.sharey(axes_compact[1, 0])  # Share Y-axis for T-value plots

plt.tight_layout()  # Adjust rect for suptitle
