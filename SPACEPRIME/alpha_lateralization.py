import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os  # Added for path joining
from scipy.stats import ttest_rel, t, sem  # Import t for theoretical threshold
from mne.stats import permutation_cluster_1samp_test  # For cluster stats
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import seaborn as sns  # For nicer plots

plt.ion()

# --- Analysis Parameters ---
# TFR settings
freqs = np.arange(5, 20, 1)  # Freq range for TFR
n_cycles = freqs / 2  # Or a fixed number like 7
method = "morlet"  # Wavelet method
decim = 5  # Decimation factor
n_jobs = 5  # Do not use all cores for TFR computation

# Channels to average over for the Left-Right difference calculation
picks_for_diff = 'eeg'
print(f"Calculating Left-Right difference using picks: {picks_for_diff}")

# Time window of interest for analysis and plotting (adjust as needed)
tmin_analysis, tmax_analysis = -0.5, 1.0  # Example time window relative to stimulus

# ROI and frequency band for hemispheric time course extraction
left_roi_hemi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi_hemi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
alpha_fmin_hemi, alpha_fmax_hemi = 8, 12  # Alpha band for hemispheric LI time courses

# Cluster Permutation Test Parameters
n_permutations_cluster = 10000
alpha_stat_cluster = 0.05
cluster_tail = 0  # Two-tailed test

# Significance Line Plotting Parameters (for T-value plots)
SIG_LINE_Y_OFFSET_TVAL = 0.01  # Y-offset in t-value units from y=0. Adjust based on typical t-value range.
# Can be negative to plot below y=0.
SIG_LINE_LW_TVAL = 4  # Linewidth for significance lines
SIG_LINE_ALPHA_TVAL = 0.7  # Alpha for significance lines
SIG_LINE_COLOR_TVAL = 'purple'  # Color for significance lines

# --- Data Storage ---
subject_results = {
    'target_li': [],
    'singleton_li': []
}
processed_subjects = []
times_vector = None  # To store time vector from the first subject

# --- Subject Loop ---
for subject in subject_ids[:]:
    print(f"\n--- Processing Subject: {subject} ---")
    try:
        epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject}", "eeg",
                                          f"sub-{subject}_task-spaceprime-epo.fif")
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for subject {subject} using pattern: {epoch_file_pattern}. Skipping.")
            continue
        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for subject {subject}: {e}. Skipping.")
        continue

    all_conds_sub = list(epochs_sub.event_id.keys())

    try:
        left_target_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
        right_target_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]].copy()
        print(f"  Target trials: {len(left_target_epochs_sub)} left, {len(right_target_epochs_sub)} right")
        if len(left_target_epochs_sub) == 0 or len(right_target_epochs_sub) == 0:
            raise ValueError("Zero trials in one of the target conditions.")

        left_singleton_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
        right_singleton_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]].copy()
        print(f"  Singleton trials: {len(left_singleton_epochs_sub)} left, {len(right_singleton_epochs_sub)} right")
        if len(left_singleton_epochs_sub) == 0 or len(right_singleton_epochs_sub) == 0:
            raise ValueError("Zero trials in one of the singleton conditions.")
    except Exception as e:
        print(f"  Error splitting conditions for subject {subject}: {e}. Skipping.")
        continue


    def get_lateralization_index(left_epochs, right_epochs, picks_calc, freqs_calc, n_cycles_calc, decim_calc,
                                 method_calc,
                                 tmin_crop=None, tmax_crop=None, n_jobs_tfr=-1):  # Renamed n_jobs to n_jobs_tfr

        tfr_left = left_epochs.compute_tfr(
            method=method_calc, picks=picks_calc, freqs=freqs_calc,
            n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_tfr,
            return_itc=False, average=False)  # average=False, then .average()
        if tmin_crop is not None or tmax_crop is not None:
            tfr_left.crop(tmin=tmin_crop, tmax=tmax_crop)
        alpha_left_avg = tfr_left.average()  # Average over trials

        tfr_right = right_epochs.compute_tfr(
            method=method_calc, picks=picks_calc, freqs=freqs_calc,
            n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_tfr,
            return_itc=False, average=False)  # average=False, then .average()
        if tmin_crop is not None or tmax_crop is not None:
            tfr_right.crop(tmin=tmin_crop, tmax=tmax_crop)
        alpha_right_avg = tfr_right.average()  # Average over trials

        # LI: (Contra - Ipsi) / (Contra + Ipsi)
        # Here, left_epochs are effectively "contralateral" to a right-side effect,
        # and right_epochs are "ipsilateral" to a right-side effect (or vice-versa for left-side effect).
        # The definition of LI can vary. This is (Power_Left_Cond - Power_Right_Cond) / (Power_Left_Cond + Power_Right_Cond)
        # If picks_calc = 'eeg', this LI is calculated per channel.
        li_data = (alpha_left_avg.data - alpha_right_avg.data) / (alpha_left_avg.data + alpha_right_avg.data)

        # Create an info object that matches the LI data structure
        # If picks_for_diff='eeg', then alpha_left_avg.info is fine.
        # If picks_for_diff was a specific ROI that was averaged *before* this function,
        # then info would need to reflect that (e.g., a single virtual channel).
        # Given current structure, alpha_left_avg.info should be correct.
        info_for_li = alpha_left_avg.info  # or tfr_right.info, should be same channel set

        li_power = mne.time_frequency.AverageTFRArray(
            data=li_data, info=info_for_li, times=alpha_left_avg.times,
            freqs=freqs_calc, nave=(alpha_left_avg.nave + alpha_right_avg.nave) // 2,  # Approximate nave
            comment='Lateralization Index', method=method_calc  # method here is descriptive
        )
        times_calc = alpha_left_avg.times
        return li_power, times_calc


    try:
        alpha_diff_target_lr, times_t = get_lateralization_index(
            left_target_epochs_sub, right_target_epochs_sub, picks_for_diff,
            freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs
        )
        alpha_diff_singleton_lr, times_s = get_lateralization_index(
            left_singleton_epochs_sub, right_singleton_epochs_sub, picks_for_diff,
            freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs
        )
        if times_vector is None:
            times_vector = times_t
        subject_results['target_li'].append(alpha_diff_target_lr)
        subject_results['singleton_li'].append(alpha_diff_singleton_lr)
        processed_subjects.append(subject)
    except Exception as e:
        print(f"  Error during TFR/LI calculation for subject {subject}: {e}. Skipping.")
        continue

print(f"\n--- Successfully processed {len(processed_subjects)} subjects: {processed_subjects} ---")

if not processed_subjects:
    raise RuntimeError("No subjects were successfully processed. Cannot continue.")
if times_vector is None:
    raise RuntimeError("Could not determine the time vector for analysis.")

# --- Group Level Analysis ---
ga_li_target = mne.grand_average(subject_results['target_li'])
ga_li_singleton = mne.grand_average(subject_results['singleton_li'])
# ga_diff = ga_li_target - ga_li_singleton # Difference of LIs

# Plot Grand Average LI TFRs
fig_tfr, axes_tfr = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
axes_tfr = axes_tfr.flatten()

ga_li_target.plot(axes=axes_tfr[0], show=False, colorbar=True, combine="mean")
axes_tfr[0].axhline(0, color='k', linestyle='--', linewidth=0.5)  # Freq = 0, not meaningful for LI value
axes_tfr[0].axvline(0, color='k', linestyle=':', linewidth=0.5)  # Time = 0
axes_tfr[0].set_title(f"GA Target LI ({freqs[0]}-{freqs[-1]} Hz)\n(N={len(processed_subjects)})")
axes_tfr[0].set_xlabel("Time (s)")
axes_tfr[0].set_ylabel("Frequency (Hz)")  # Y-axis is Frequency for TFR plot
sns.despine(ax=axes_tfr[0])

ga_li_singleton.plot(axes=axes_tfr[1], show=False, colorbar=True, combine="mean")
axes_tfr[1].axhline(0, color='k', linestyle='--', linewidth=0.5)  # Freq = 0
axes_tfr[1].axvline(0, color='k', linestyle=':', linewidth=0.5)  # Time = 0
axes_tfr[1].set_title(f"GA Singleton LI ({freqs[0]}-{freqs[-1]} Hz)\n(N={len(processed_subjects)})")
axes_tfr[1].set_xlabel("Time (s)")
sns.despine(ax=axes_tfr[1])
plt.tight_layout()
# fig_tfr.show() # Not needed if plt.ion() and plt.show(block=True) at end

# Prepare Data for Hemispheric Time Course Analysis
target_data_time_courses = dict(left_hemisphere=[], right_hemisphere=[])
singleton_data_time_courses = dict(left_hemisphere=[], right_hemisphere=[])

for i, sub_id in enumerate(processed_subjects):
    # Target LI TFR for this subject
    tfr_target_li_sub = subject_results["target_li"][i].copy()
    # Check if ROIs exist
    ch_names_sub = tfr_target_li_sub.ch_names
    valid_left_roi = [ch for ch in left_roi_hemi if ch in ch_names_sub]
    valid_right_roi = [ch for ch in right_roi_hemi if ch in ch_names_sub]

    if not valid_left_roi or not valid_right_roi:
        print(
            f"Warning: Subject {sub_id} missing some ROI channels for Target. Left: {len(valid_left_roi)}/{len(left_roi_hemi)}, Right: {len(valid_right_roi)}/{len(right_roi_hemi)}. Skipping this subject for hemispheric TC.")
        continue  # Skip this subject for this specific part if ROIs are not fully present

    tfr_target_left_hemisphere = tfr_target_li_sub.copy().pick(valid_left_roi).crop(fmin=alpha_fmin_hemi,
                                                                                    fmax=alpha_fmax_hemi).data.mean(
        axis=(0, 1))  # Avg over channels and freqs
    tfr_target_right_hemisphere = tfr_target_li_sub.copy().pick(valid_right_roi).crop(fmin=alpha_fmin_hemi,
                                                                                      fmax=alpha_fmax_hemi).data.mean(
        axis=(0, 1))
    target_data_time_courses["left_hemisphere"].append(tfr_target_left_hemisphere)
    target_data_time_courses["right_hemisphere"].append(tfr_target_right_hemisphere)

    # Singleton LI TFR for this subject
    tfr_singleton_li_sub = subject_results["singleton_li"][i].copy()
    ch_names_sub_s = tfr_singleton_li_sub.ch_names  # Re-check for singleton TFR if info differs
    valid_left_roi_s = [ch for ch in left_roi_hemi if ch in ch_names_sub_s]
    valid_right_roi_s = [ch for ch in right_roi_hemi if ch in ch_names_sub_s]

    if not valid_left_roi_s or not valid_right_roi_s:
        print(
            f"Warning: Subject {sub_id} missing some ROI channels for Singleton. Left: {len(valid_left_roi_s)}/{len(left_roi_hemi)}, Right: {len(valid_right_roi_s)}/{len(right_roi_hemi)}. Skipping this subject for hemispheric TC.")
        # Note: if a subject is skipped here but not for target, array lengths might mismatch later.
        # It's better to exclude subject from all hemispheric analyses if any part is missing.
        # For simplicity now, we append if valid, leading to potentially different N per condition for TCs.
        continue

    tfr_singleton_left_hemisphere = tfr_singleton_li_sub.copy().pick(valid_left_roi_s).crop(fmin=alpha_fmin_hemi,
                                                                                            fmax=alpha_fmax_hemi).data.mean(
        axis=(0, 1))
    tfr_singleton_right_hemisphere = tfr_singleton_li_sub.copy().pick(valid_right_roi_s).crop(fmin=alpha_fmin_hemi,
                                                                                              fmax=alpha_fmax_hemi).data.mean(
        axis=(0, 1))
    singleton_data_time_courses["left_hemisphere"].append(tfr_singleton_left_hemisphere)
    singleton_data_time_courses["right_hemisphere"].append(tfr_singleton_right_hemisphere)

# Convert lists of time courses to NumPy arrays
# Ensure all lists being converted have data, otherwise np.array might behave unexpectedly or error
target_left_tc_all = np.array(target_data_time_courses["left_hemisphere"]) if target_data_time_courses[
    "left_hemisphere"] else np.empty((0, len(times_vector)))
target_right_tc_all = np.array(target_data_time_courses["right_hemisphere"]) if target_data_time_courses[
    "right_hemisphere"] else np.empty((0, len(times_vector)))
singleton_left_tc_all = np.array(singleton_data_time_courses["left_hemisphere"]) if singleton_data_time_courses[
    "left_hemisphere"] else np.empty((0, len(times_vector)))
singleton_right_tc_all = np.array(singleton_data_time_courses["right_hemisphere"]) if singleton_data_time_courses[
    "right_hemisphere"] else np.empty((0, len(times_vector)))

# Check if we have enough subjects for statistical analysis of hemispheric time courses
# The number of subjects for TC analysis might be less than total processed_subjects
# if some were skipped due to missing ROI channels.
# For simplicity, we use n_subs_tc for the smallest N across conditions if they differ.
# Or, ensure subjects are consistently included/excluded.
# Here, we'll proceed if any TC data exists and let permutation test handle N.

n_subs_target_tc = target_left_tc_all.shape[0]
n_subs_singleton_tc = singleton_left_tc_all.shape[0]

# --- Statistics with Permutation Cluster Test on Hemispheric LI Time Courses ---
if n_subs_target_tc < 2 and n_subs_singleton_tc < 2:
    print(f"Not enough subjects for hemispheric LI cluster permutation test. Skipping.")
    t_obs_t, cluster_pv_t, clusters_t = None, None, None
    t_obs_d, cluster_pv_d, clusters_d = None, None, None
    t_thresh_cluster = None
else:
    # Use the smaller N for df calculation if N differs, or handle separately
    # For simplicity, assume we want to run tests if any condition has enough subjects
    # A more robust approach would be to ensure same N for L/R tc and across target/singleton
    # or run tests only on subjects present in all.

    # We test (Left Hemi LI - Right Hemi LI) against 0.
    # This difference represents the "lateralization of the lateralization index".
    diff_target_tc = target_left_tc_all - target_right_tc_all
    diff_singleton_tc = singleton_left_tc_all - singleton_right_tc_all

    # Determine a common df for threshold or calculate per test
    # For now, calculate t_thresh based on the smaller N if running combined analysis,
    # or ideally, ensure N is consistent or run tests separately.
    # Let's assume we run tests if N >= 2 for that specific condition.

    t_thresh_cluster_target, t_thresh_cluster_singleton = None, None
    t_obs_t, clusters_t, cluster_pv_t = None, None, None
    t_obs_d, clusters_d, cluster_pv_d = None, None, None

    if n_subs_target_tc >= 2:
        df_t = n_subs_target_tc - 1
        t_thresh_cluster_target = t.ppf(1 - alpha_stat_cluster / 2, df_t)
        print(f"\n--- Running Cluster Permutation Test for Target Hemispheric LI (N={n_subs_target_tc}) ---")
        print(f"Using t-threshold: {t_thresh_cluster_target:.3f} (df={df_t})")
        t_obs_t, clusters_t, cluster_pv_t, _ = permutation_cluster_1samp_test(
            diff_target_tc, threshold=t_thresh_cluster_target, n_permutations=n_permutations_cluster,
            tail=cluster_tail, n_jobs=n_jobs, out_type="mask", seed=42, verbose=False
        )
        print(
            f"  Target: Found {np.sum(cluster_pv_t < alpha_stat_cluster) if cluster_pv_t is not None else 0} significant cluster(s).")

    if n_subs_singleton_tc >= 2:
        df_d = n_subs_singleton_tc - 1
        t_thresh_cluster_singleton = t.ppf(1 - alpha_stat_cluster / 2, df_d)
        print(f"\n--- Running Cluster Permutation Test for Singleton Hemispheric LI (N={n_subs_singleton_tc}) ---")
        print(f"Using t-threshold: {t_thresh_cluster_singleton:.3f} (df={df_d})")
        t_obs_d, clusters_d, cluster_pv_d, _ = permutation_cluster_1samp_test(
            diff_singleton_tc, threshold=t_thresh_cluster_singleton, n_permutations=n_permutations_cluster,
            tail=cluster_tail, n_jobs=n_jobs, out_type="mask", seed=42, verbose=False
        )
        print(
            f"  Singleton: Found {np.sum(cluster_pv_d < alpha_stat_cluster) if cluster_pv_d is not None else 0} significant cluster(s).")

    # For plotting, we need a single t_thresh_cluster if axes are shared, or use specific ones.
    # Let's use the target one for plots if available, or singleton one.
    t_thresh_cluster = t_thresh_cluster_target if t_thresh_cluster_target is not None else t_thresh_cluster_singleton

# Grand Averages and SEM for plotting hemispheric LIs
ga_target_left = target_left_tc_all.mean(axis=0) if n_subs_target_tc > 0 else np.full_like(times_vector, np.nan)
sem_target_left = sem(target_left_tc_all, axis=0, nan_policy='omit') if n_subs_target_tc > 0 else np.full_like(
    times_vector, np.nan)
ga_target_right = target_right_tc_all.mean(axis=0) if n_subs_target_tc > 0 else np.full_like(times_vector, np.nan)
sem_target_right = sem(target_right_tc_all, axis=0, nan_policy='omit') if n_subs_target_tc > 0 else np.full_like(
    times_vector, np.nan)

ga_singleton_left = singleton_left_tc_all.mean(axis=0) if n_subs_singleton_tc > 0 else np.full_like(times_vector,
                                                                                                    np.nan)
sem_singleton_left = sem(singleton_left_tc_all, axis=0, nan_policy='omit') if n_subs_singleton_tc > 0 else np.full_like(
    times_vector, np.nan)
ga_singleton_right = singleton_right_tc_all.mean(axis=0) if n_subs_singleton_tc > 0 else np.full_like(times_vector,
                                                                                                      np.nan)
sem_singleton_right = sem(singleton_right_tc_all, axis=0,
                          nan_policy='omit') if n_subs_singleton_tc > 0 else np.full_like(times_vector, np.nan)

# --- Plotting Hemispheric Time Courses and Cluster T-tests ---
fig_hemi, axes_hemi = plt.subplots(2, 2, figsize=(16, 10), sharex=True,
                                   sharey='row')  # Share Y for LI plots, and for T-value plots
fig_hemi.suptitle("Hemispheric Alpha LI Time Courses and Cluster Permutation T-tests", fontsize=16)

# --- Plot 1: Target - Left vs Right Hemisphere LI ---
ax = axes_hemi[0, 0]
ax.plot(times_vector, ga_target_left, label="Left Hemisphere LI", color="blue")
ax.fill_between(times_vector, ga_target_left - sem_target_left, ga_target_left + sem_target_left,
                color="blue", alpha=0.2)
ax.plot(times_vector, ga_target_right, label="Right Hemisphere LI", color="red")
ax.fill_between(times_vector, ga_target_right - sem_target_right, ga_target_right + sem_target_right,
                color="red", alpha=0.2)
ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
ax.set_ylabel(f"Alpha LI ({alpha_fmin_hemi}-{alpha_fmax_hemi} Hz)")
ax.set_title(f"Lateral targets (N={n_subs_target_tc})")
ax.legend(loc="best")
sns.despine(ax=ax)

# --- Plot 2: Singleton - Left vs Right Hemisphere LI ---
ax = axes_hemi[0, 1]
ax.plot(times_vector, ga_singleton_left, label="Left Hemisphere LI", color="blue")
ax.fill_between(times_vector, ga_singleton_left - sem_singleton_left,
                ga_singleton_left + sem_singleton_left, color="blue", alpha=0.2)
ax.plot(times_vector, ga_singleton_right, label="Right Hemisphere LI", color="red")
ax.fill_between(times_vector, ga_singleton_right - sem_singleton_right,
                ga_singleton_right + sem_singleton_right, color="red", alpha=0.2)
ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
ax.set_title(f"Lateral distractors (N={n_subs_singleton_tc})")
ax.legend(loc="best")
sns.despine(ax=ax)

# --- Plot 3: Target - T-values (Left Hemi LI - Right Hemi LI) with Significant Clusters ---
ax = axes_hemi[1, 0]
if t_obs_t is not None:
    ax.plot(times_vector, t_obs_t, label="Observed T-value (L-R Hemi LI)", color="black", linewidth=1.5)
    if t_thresh_cluster_target is not None:
        ax.axhline(t_thresh_cluster_target, color='gray', linestyle='--', linewidth=1,
                   label=f'Cluster threshold (t={t_thresh_cluster_target:.2f})')
        if cluster_tail == 0:
            ax.axhline(-t_thresh_cluster_target, color='gray', linestyle='--', linewidth=1)

    significant_cluster_found_t = False
    if clusters_t is not None:
        for i, cl_mask in enumerate(clusters_t):
            if cluster_pv_t[i] < alpha_stat_cluster:
                cluster_times = times_vector[cl_mask]
                if len(cluster_times) > 0:
                    label_str_sig = None
                    if not significant_cluster_found_t:
                        label_str_sig = f'Significant Cluster (p < {alpha_stat_cluster})'
                        significant_cluster_found_t = True
                    ax.hlines(y=SIG_LINE_Y_OFFSET_TVAL,
                              xmin=cluster_times[0], xmax=cluster_times[-1],
                              color=SIG_LINE_COLOR_TVAL, linewidth=SIG_LINE_LW_TVAL,
                              alpha=SIG_LINE_ALPHA_TVAL,
                              label=label_str_sig if label_str_sig else '_nolegend_')
    if significant_cluster_found_t or (t_thresh_cluster_target is not None):
        ax.legend(loc="best")
else:
    ax.text(0.5, 0.5, "No data/stats for Target T-values", ha='center', va='center', transform=ax.transAxes)

ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("T-value")
ax.set_title("Target: Cluster Test (L-R Hemi LI)")
sns.despine(ax=ax)

# --- Plot 4: Singleton - T-values (Left Hemi LI - Right Hemi LI) with Significant Clusters ---
ax = axes_hemi[1, 1]
if t_obs_d is not None:
    ax.plot(times_vector, t_obs_d, label="Observed T-value (L-R Hemi LI)", color="black", linewidth=1.5)
    if t_thresh_cluster_singleton is not None:
        ax.axhline(t_thresh_cluster_singleton, color='gray', linestyle='--', linewidth=1,
                   label=f'Cluster threshold (t={t_thresh_cluster_singleton:.2f})')
        if cluster_tail == 0:
            ax.axhline(-t_thresh_cluster_singleton, color='gray', linestyle='--', linewidth=1)

    significant_cluster_found_d = False
    if clusters_d is not None:
        for i, cl_mask in enumerate(clusters_d):
            if cluster_pv_d[i] < alpha_stat_cluster:
                cluster_times = times_vector[cl_mask]
                if len(cluster_times) > 0:
                    label_str_sig = None
                    if not significant_cluster_found_d:
                        label_str_sig = f'Significant Cluster (p < {alpha_stat_cluster})'
                        significant_cluster_found_d = True
                    ax.hlines(y=SIG_LINE_Y_OFFSET_TVAL,
                              xmin=cluster_times[0], xmax=cluster_times[-1],
                              color=SIG_LINE_COLOR_TVAL, linewidth=SIG_LINE_LW_TVAL,
                              alpha=SIG_LINE_ALPHA_TVAL,
                              label=label_str_sig if label_str_sig else '_nolegend_')
    if significant_cluster_found_d or (t_thresh_cluster_singleton is not None):
        ax.legend(loc="best")
else:
    ax.text(0.5, 0.5, "No data/stats for Singleton T-values", ha='center', va='center', transform=ax.transAxes)

ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
ax.set_xlabel("Time (s)")
# ax.set_ylabel("T-value") # Y-axis shared with plot 3
ax.set_title("Singleton: Cluster Test (L-R Hemi LI)")
sns.despine(ax=ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
