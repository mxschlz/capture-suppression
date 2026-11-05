import sys
import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import mne
import scipy.stats
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
# Add project path to sys.path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from SPACECUE.subjects import subject_ids
import SPACECUE


def run_computation(force_recompute=False):
    cache_dir = os.path.join(SPACECUE.get_data_path(), "derivatives", "cache")
    cache_path = os.path.join(cache_dir, "pre_stim_alpha_subject_data.pkl")

    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached subject data from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                subject_data = pickle.load(f)
            print(f"Successfully loaded {len(subject_data)} subjects from cache.")
            return subject_data
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    print("Running Time-Frequency analysis for all subjects...")

    # Analysis parameters matching alpha_lateralization.py
    freqs = np.arange(5, 20, 1)
    n_cycles = freqs / 2
    method = "morlet"
    decim = 5
    
    subject_data = {}
    montage = mne.channels.make_standard_montage("easycap-M1")

    for subject in subject_ids:
        print(f"Processing Subject: {subject}...")
        epoch_file_pattern = os.path.join(SPACECUE.get_data_path(), "derivatives", "epoching", f"sci-{subject}", "eeg",
                                          f"sci-{subject}_task-spacecue-epo.fif")
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for subject {subject}. Skipping.")
            continue
            
        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        epochs_sub.set_montage(montage)
        
        # Determine cohort based on Age
        age = float(epochs_sub.metadata['Age'].iloc[0])
        cohort = 'Young' if age <= 35 else 'Old'
        
        # Mask for expected distractor location
        hp_loc_numeric = pd.to_numeric(epochs_sub.metadata['HP_Distractor_Loc'], errors='coerce')
        left_expected_epochs = epochs_sub[hp_loc_numeric == 1.0]
        right_expected_epochs = epochs_sub[hp_loc_numeric == 3.0]
        
        if len(left_expected_epochs) == 0 or len(right_expected_epochs) == 0:
            print(f"  Skipping subject {subject} due to zero trials in one condition.")
            continue
            
        # Compute TFR for Left expected distractor
        tfr_left = left_expected_epochs.compute_tfr(
            method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
            return_itc=False, average=False, n_jobs=4
        )
        alpha_left_avg = tfr_left.average()
        
        # Compute TFR for Right expected distractor
        tfr_right = right_expected_epochs.compute_tfr(
            method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
            return_itc=False, average=False, n_jobs=4
        )
        alpha_right_avg = tfr_right.average()
        
        # Calculate LI: (Left - Right) / (Left + Right)
        li_data = (alpha_left_avg.data - alpha_right_avg.data) / (alpha_left_avg.data + alpha_right_avg.data)
        
        # Calculate mean power: (Left + Right) / 2
        mean_power_data = (alpha_left_avg.data + alpha_right_avg.data) / 2
        
        # Store essential data
        subject_data[subject] = {
            'li_data': li_data,
            'mean_power_data': mean_power_data,
            'times': alpha_left_avg.times,
            'freqs': alpha_left_avg.freqs,
            'ch_names': alpha_left_avg.ch_names,
            'cohort': cohort,
            'age': age
        }

    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(subject_data, f)
        print(f"Saved calculated LI and Power TFRs to cache at {cache_path}.")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return subject_data


def compute_collapsed_cohens_d(group1_data, group2_data, indices):
    """
    Computes Cohen's d by first collapsing (averaging) each subject's signal
    across the spatial or temporal cluster indices, strictly adhering to
    scientific guidelines against point-wise effect size circularity.
    """
    g1_means = np.mean(group1_data[:, indices], axis=1)
    g2_means = np.mean(group2_data[:, indices], axis=1)
    n1, n2 = len(g1_means), len(g2_means)
    var1, var2 = np.var(g1_means, ddof=1), np.var(g2_means, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(g1_means) - np.mean(g2_means)) / s_pooled if s_pooled > 0 else 0.0


def compute_one_sample_cohens_dz(data, indices):
    """
    Computes Cohen's d_z for a one-sample (against zero) test by first
    collapsing (averaging) each subject's signal across the cluster indices.
    This strictly adheres to scientific guidelines against circular point-wise
    effect size estimation: the cluster is treated as a single, indivisible
    response before computing d_z = mean(collapsed) / SD(collapsed).
    """
    collapsed = np.mean(data[:, indices], axis=1)
    sd = np.std(collapsed, ddof=1)
    return np.mean(collapsed) / sd if sd > 0 else 0.0


def main():
    subject_data = run_computation(force_recompute=False)

    # Left and Right ROI definitions
    left_roi_hemi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
    right_roi_hemi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]

    subject_ids_list = list(subject_data.keys())
    epoch_file_pattern = os.path.join(SPACECUE.get_data_path(), "derivatives", "epoching", f"sci-{subject_ids_list[0]}", "eeg",
                                      f"sci-{subject_ids_list[0]}_task-spacecue-epo.fif")
    first_subject_epochs = mne.read_epochs(glob.glob(epoch_file_pattern)[0], preload=False)
    montage = mne.channels.make_standard_montage("easycap-M1")
    first_subject_epochs.set_montage(montage)
    info = first_subject_epochs.info

    cohorts = ['Young', 'Old']
    times_vector = subject_data[subject_ids_list[0]]['times']
    freqs = subject_data[subject_ids_list[0]]['freqs']
    
    # 8-12 Hz Alpha band index
    alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
    
    # Pre-stimulus time window [-0.75, 0.0] index
    pre_stim_idx = np.where((times_vector >= -0.75) & (times_vector <= 0.0))[0]

    # Data structures for Time Courses (tc) and Topographies (topo)
    li_tcs = {c: [] for c in cohorts}
    li_topos = {c: [] for c in cohorts}
    
    power_tcs = {c: [] for c in cohorts}
    power_topos = {c: [] for c in cohorts}

    for sub_id, sub_info in subject_data.items():
        li_data = sub_info['li_data']
        power_raw = sub_info['mean_power_data']
        ch_names = sub_info['ch_names']
        cohort = sub_info['cohort']
        
        # Total alpha power is analysed in log space: raw Morlet power is
        # strongly right-skewed (approximately log-normal; subject-level
        # skew ~2.5, Shapiro-Wilk p < .001). The former subject-wise
        # normalization (division by the subject's overall mean power) is
        # applied in the log domain as a subtraction of the subject's mean
        # log power, so power_scaled is the natural-log power relative to
        # the subject's GEOMETRIC mean power (0 = at the subject's own
        # geometric mean). Non-positive values (numerically impossible for
        # Morlet power) are clipped to the smallest positive float to keep
        # np.log finite.
        power_log = np.log(np.maximum(power_raw, np.finfo(float).tiny))
        power_scaled = power_log - power_log.mean()
        
        # 1. Topography: Mean pre-stimulus alpha value per channel
        li_topo = li_data[:, alpha_idx, :][:, :, pre_stim_idx].mean(axis=(1, 2))
        power_topo = power_scaled[:, alpha_idx, :][:, :, pre_stim_idx].mean(axis=(1, 2))
        
        li_topos[cohort].append(li_topo)
        power_topos[cohort].append(power_topo)
        
        # 2. Time Courses: Average over ROI channels & alpha band
        valid_left_chans = [ch_names.index(ch) for ch in left_roi_hemi if ch in ch_names]
        valid_right_chans = [ch_names.index(ch) for ch in right_roi_hemi if ch in ch_names]
        
        # For LI, hemispheric difference: Left Hemi - Right Hemi
        li_left_tc = li_data[valid_left_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        li_right_tc = li_data[valid_right_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        li_tcs[cohort].append(li_left_tc - li_right_tc)
        
        # For Power, mean over all ROI channels
        roi_power_left = power_scaled[valid_left_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        roi_power_right = power_scaled[valid_right_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        power_tcs[cohort].append((roi_power_left + roi_power_right) / 2)

    # Convert to numpy arrays
    y_li_tc = np.array(li_tcs['Young'])
    o_li_tc = np.array(li_tcs['Old'])
    y_li_topo = np.array(li_topos['Young'])
    o_li_topo = np.array(li_topos['Old'])

    y_pwr_tc = np.array(power_tcs['Young'])
    o_pwr_tc = np.array(power_tcs['Old'])
    y_pwr_topo = np.array(power_topos['Young'])
    o_pwr_topo = np.array(power_topos['Old'])

    n_young = len(y_li_tc)
    n_old = len(o_li_tc)
    df_between = (1, n_young + n_old - 2)

    print("\n=================================================================")
    print(f"  PERMUTATION CLUSTER TESTS: PRE-STIMULUS ALPHA (Young vs. Old)  ")
    print(f"  Sample sizes: Young N = {n_young}, Old N = {n_old}, df = {df_between}")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # 1. TIME COURSE CLUSTER PERMUTATION TESTS (Alpha LI: Young vs Old)
    # -------------------------------------------------------------
    print("--- 1. Alpha LI Time Course Permutation Cluster Tests (Pre-defined ROI) ---")
    # Pre-stimulus focus (-0.75 to 0.0 s)
    y_li_pre = y_li_tc[:, pre_stim_idx]
    o_li_pre = o_li_tc[:, pre_stim_idx]
    F_li_pre, cl_li_pre, pv_li_pre, _ = permutation_cluster_test(
        [y_li_pre, o_li_pre], n_permutations=10000, tail=1, threshold=None,
        out_type='indices', verbose=False, seed=42
    )

    # Full time-course (-1.0 to 1.0 s)
    F_li_full, cl_li_full, pv_li_full, _ = permutation_cluster_test(
        [y_li_tc, o_li_tc], n_permutations=10000, tail=1, threshold=None,
        out_type='indices', verbose=False, seed=42
    )

    sig_li_windows = []
    is_li_sig = np.zeros(len(times_vector), dtype=bool)
    if len(cl_li_full) > 0:
        for cl, p_val in zip(cl_li_full, pv_li_full):
            d_val = compute_collapsed_cohens_d(y_li_tc, o_li_tc, cl[0])
            t_start, t_end = times_vector[cl[0][0]], times_vector[cl[0][-1]]
            if p_val < 0.05:
                status = "SIGNIFICANT"
            elif p_val < 0.10:
                status = "TREND"
            else:
                status = "n.s."
            print(f"  Full TC Cluster [{status}]: {t_start*1000:.0f} to {t_end*1000:.0f} ms | p = {p_val:.4f} | Peak F = {np.max(F_li_full[cl[0]]):.3f} | Cohen's d = {d_val:.3f}")
            if p_val < 0.05:
                is_li_sig[cl[0]] = True
                sig_li_windows.append(f"{t_start*1000:.0f}-{t_end*1000:.0f} ms (p={p_val:.4f})")
    else:
        print(f"  No clusters formed on full LI time course. Max F = {np.max(F_li_full):.3f}")

    if len(cl_li_pre) > 0:
        for cl, p_val in zip(cl_li_pre, pv_li_pre):
            d_val = compute_collapsed_cohens_d(y_li_pre, o_li_pre, cl[0])
            t_start, t_end = times_vector[pre_stim_idx[cl[0][0]]], times_vector[pre_stim_idx[cl[0][-1]]]
            status = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            print(f"  Pre-stim Window Cluster [{status}]: {t_start*1000:.0f} to {t_end*1000:.0f} ms | p = {p_val:.4f} | Peak F = {np.max(F_li_pre[cl[0]]):.3f} | Cohen's d = {d_val:.3f}")
    else:
        print(f"  No clusters formed in pre-stimulus window. Max F = {np.max(F_li_pre):.3f}")

    # Standard pre-stimulus mean comparison across [-0.75, 0.0] s (temporal collapse)
    y_li_pre_mean = y_li_pre.mean(axis=1)
    o_li_pre_mean = o_li_pre.mean(axis=1)
    tt_li_pre = scipy.stats.ttest_ind(y_li_pre_mean, o_li_pre_mean)
    s_pool_li = np.sqrt(((n_young - 1)*np.var(y_li_pre_mean, ddof=1) + (n_old - 1)*np.var(o_li_pre_mean, ddof=1)) / (n_young + n_old - 2))
    d_li_pre = (np.mean(y_li_pre_mean) - np.mean(o_li_pre_mean)) / s_pool_li
    print(f"  Collapsed Pre-stim Window [-750 to 0 ms] Mean LI:")
    print(f"    Young: {y_li_pre_mean.mean():.4f} ± {y_li_pre_mean.std():.4f} (SEM: {sem(y_li_pre_mean):.4f})")
    print(f"    Old:   {o_li_pre_mean.mean():.4f} ± {o_li_pre_mean.std():.4f} (SEM: {sem(o_li_pre_mean):.4f})")
    print(f"    Independent t-test: t({n_young + n_old - 2}) = {tt_li_pre.statistic:.4f}, p = {tt_li_pre.pvalue:.4f}, Cohen's d = {d_li_pre:.4f}")

    # -------------------------------------------------------------
    # 1b. ONE-SAMPLE CLUSTER PERMUTATION TESTS AGAINST ZERO (per cohort)
    #     Two-tailed tests of whether the alpha LI time course deviates
    #     from zero within each cohort, with circularity-safe collapsed
    #     Cohen's d_z effect sizes.
    # -------------------------------------------------------------
    print("\n--- 1b. One-Sample Tests Against Zero (Alpha LI Time Course) ---")
    is_li_sig_1samp = {c: np.zeros(len(times_vector), dtype=bool) for c in cohorts}
    one_sample_p_pre = {c: [] for c in cohorts}
    one_sample_report = {c: [] for c in cohorts}
    # Collapsed pre-stimulus mean LI vs. 0 (one-sample t-test) statistics,
    # stored per cohort for visualizing trends towards significance in the
    # Alpha LI time course plot.
    pre_mean_stats = {}

    for c, li_tc_arr in [('Young', y_li_tc), ('Old', o_li_tc)]:
        n_c = len(li_tc_arr)
        df_one = n_c - 1
        print(f"  {c} cohort (N = {n_c}, df = {df_one}):")

        # Full time course (-1.0 to 1.0 s), two-tailed against zero
        T_full_1s, cl_full_1s, pv_full_1s, _ = permutation_cluster_1samp_test(
            li_tc_arr, n_permutations=10000, tail=0, threshold=None,
            out_type='indices', verbose=False, seed=42
        )
        if len(cl_full_1s) > 0:
            for i_cl, (cl, p_val) in enumerate(zip(cl_full_1s, pv_full_1s)):
                dz_val = compute_one_sample_cohens_dz(li_tc_arr, cl[0])
                t_start, t_end = times_vector[cl[0][0]], times_vector[cl[0][-1]]
                if p_val < 0.05:
                    status = "SIGNIFICANT"
                elif p_val < 0.10:
                    status = "TREND"
                else:
                    status = "n.s."
                mean_dir = "positive LI" if np.mean(li_tc_arr[:, cl[0]]) > 0 else "negative LI"
                print(f"    Full TC Cluster [{status}]: {t_start*1000:.0f} to {t_end*1000:.0f} ms | p = {p_val:.4f} | Peak |t| = {np.max(np.abs(T_full_1s[cl[0]])):.3f} | Cohen's d_z = {dz_val:.3f} ({mean_dir})")
                one_sample_report[c].append(
                    f"       Full TC Cluster {i_cl + 1} [{status}]: {t_start*1000:.0f}-{t_end*1000:.0f} ms | "
                    f"p = {p_val:.5f} | Peak |t| = {np.max(np.abs(T_full_1s[cl[0]])):.3f} | Cohen's d_z = {dz_val:.3f}\n")
                if p_val < 0.05:
                    is_li_sig_1samp[c][cl[0]] = True
        else:
            print(f"    No clusters formed on full LI time course. Max |t| = {np.max(np.abs(T_full_1s)):.3f}")
            one_sample_report[c].append(
                f"       No clusters formed on full time course. Max |t| = {np.max(np.abs(T_full_1s)):.3f}\n")

        # Pre-stimulus window [-0.75, 0.0] s, two-tailed against zero
        li_pre_c = li_tc_arr[:, pre_stim_idx]
        T_pre_1s, cl_pre_1s, pv_pre_1s, _ = permutation_cluster_1samp_test(
            li_pre_c, n_permutations=10000, tail=0, threshold=None,
            out_type='indices', verbose=False, seed=42
        )
        one_sample_p_pre[c] = pv_pre_1s
        if len(cl_pre_1s) > 0:
            for i_cl, (cl, p_val) in enumerate(zip(cl_pre_1s, pv_pre_1s)):
                dz_val = compute_one_sample_cohens_dz(li_pre_c, cl[0])
                t_start, t_end = times_vector[pre_stim_idx[cl[0][0]]], times_vector[pre_stim_idx[cl[0][-1]]]
                if p_val < 0.05:
                    status = "SIGNIFICANT"
                elif p_val < 0.10:
                    status = "TREND"
                else:
                    status = "n.s."
                mean_dir = "positive LI" if np.mean(li_pre_c[:, cl[0]]) > 0 else "negative LI"
                print(f"    Pre-stim Window Cluster [{status}]: {t_start*1000:.0f} to {t_end*1000:.0f} ms | p = {p_val:.4f} | Peak |t| = {np.max(np.abs(T_pre_1s[cl[0]])):.3f} | Cohen's d_z = {dz_val:.3f} ({mean_dir})")
                one_sample_report[c].append(
                    f"       Pre-stim Cluster {i_cl + 1} [{status}]: {t_start*1000:.0f}-{t_end*1000:.0f} ms | "
                    f"p = {p_val:.5f} | Peak |t| = {np.max(np.abs(T_pre_1s[cl[0]])):.3f} | Cohen's d_z = {dz_val:.3f}\n")
        else:
            print(f"    No clusters formed in pre-stimulus window. Max |t| = {np.max(np.abs(T_pre_1s)):.3f}")
            one_sample_report[c].append(
                f"       No clusters formed in pre-stimulus window. Max |t| = {np.max(np.abs(T_pre_1s)):.3f}\n")

        # Collapsed pre-stimulus mean one-sample t-test (with Shapiro-Wilk normality check)
        pre_mean_c = li_pre_c.mean(axis=1)
        sh_w, sh_p = scipy.stats.shapiro(pre_mean_c)
        tt_1s = scipy.stats.ttest_1samp(pre_mean_c, 0.0)
        dz_pre = np.mean(pre_mean_c) / np.std(pre_mean_c, ddof=1) if np.std(pre_mean_c, ddof=1) > 0 else 0.0
        print(f"    Collapsed Pre-stim Mean LI vs. 0: t({df_one}) = {tt_1s.statistic:.4f}, p = {tt_1s.pvalue:.4f}, Cohen's d_z = {dz_pre:.4f}")
        print(f"      Shapiro-Wilk normality (collapsed means): W = {sh_w:.3f}, p = {sh_p:.4f}")
        one_sample_report[c].append(
            f"       Collapsed Pre-stim Mean vs. 0: t({df_one}) = {tt_1s.statistic:.4f}, p = {tt_1s.pvalue:.4f}, Cohen's d_z = {dz_pre:.4f}\n"
            f"       Shapiro-Wilk normality (collapsed means): W = {sh_w:.3f}, p = {sh_p:.4f}\n")
        # Store collapsed pre-stimulus mean statistics for plot annotation
        pre_mean_stats[c] = {'t': tt_1s.statistic, 'p': tt_1s.pvalue,
                             'dz': dz_pre, 'df': df_one,
                             'mean': float(np.mean(pre_mean_c))}

    # -------------------------------------------------------------
    # 2. PRE-DEFINED ROI SPATIAL CLUSTER PERMUTATION TESTS (-0.75 to 0.0 s)
    # -------------------------------------------------------------
    roi_channels = left_roi_hemi + right_roi_hemi
    roi_indices = [info['ch_names'].index(ch) for ch in roi_channels if ch in info['ch_names']]
    roi_mask = np.isin(info['ch_names'], roi_channels)
    info_roi = mne.pick_info(info, roi_indices)
    adj_roi, _ = mne.channels.find_ch_adjacency(info_roi, 'eeg')

    print("\n--- 2. Pre-defined ROI Spatial Topography Permutation Cluster Tests (-0.75 to 0.0 s) ---")
    y_li_roi = y_li_topo[:, roi_indices]
    o_li_roi = o_li_topo[:, roi_indices]
    F_li_roi, cl_li_roi, pv_li_roi, _ = permutation_cluster_test(
        [y_li_roi, o_li_roi], adjacency=adj_roi, n_permutations=10000,
        tail=1, threshold=None, verbose=False, seed=42
    )

    if len(cl_li_roi) > 0:
        for cl, p_val in zip(cl_li_roi, pv_li_roi):
            d_val = compute_collapsed_cohens_d(y_li_roi, o_li_roi, cl[0])
            status = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            ch_names_sig = [info_roi['ch_names'][i] for i in cl[0]]
            print(f"  ROI LI Spatial Cluster [{status}]: {len(cl[0])} channels ({', '.join(ch_names_sig[:6])}) | p = {p_val:.4f} | Peak F = {np.max(F_li_roi[cl[0]]):.3f} | Cohen's d = {d_val:.3f}")
    else:
        print(f"  No spatial clusters formed for Alpha LI in pre-defined ROI. Max F = {np.max(F_li_roi):.3f}")

    # -------------------------------------------------------------
    # 3. TOTAL ALPHA POWER CLUSTER PERMUTATION TESTS (Young vs Old)
    # -------------------------------------------------------------
    print("\n--- 3. Total Alpha Power Permutation Cluster Tests (Pre-defined ROI) ---")
    y_pwr_pre = y_pwr_tc[:, pre_stim_idx]
    o_pwr_pre = o_pwr_tc[:, pre_stim_idx]
    F_pwr_pre, cl_pwr_pre, pv_pwr_pre, _ = permutation_cluster_test(
        [y_pwr_pre, o_pwr_pre], n_permutations=10000, tail=1, threshold=None,
        out_type='indices', verbose=False, seed=42
    )
    F_pwr_full, cl_pwr_full, pv_pwr_full, _ = permutation_cluster_test(
        [y_pwr_tc, o_pwr_tc], n_permutations=10000, tail=1, threshold=None,
        out_type='indices', verbose=False, seed=42
    )

    is_pwr_sig = np.zeros(len(times_vector), dtype=bool)
    if len(cl_pwr_full) > 0:
        for cl, p_val in zip(cl_pwr_full, pv_pwr_full):
            d_val = compute_collapsed_cohens_d(y_pwr_tc, o_pwr_tc, cl[0])
            t_start, t_end = times_vector[cl[0][0]], times_vector[cl[0][-1]]
            status = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            print(f"  Power Full TC Cluster [{status}]: {t_start*1000:.0f} to {t_end*1000:.0f} ms | p = {p_val:.4f} | Peak F = {np.max(F_pwr_full[cl[0]]):.3f} | Cohen's d = {d_val:.3f}")
            if p_val < 0.05:
                is_pwr_sig[cl[0]] = True
    else:
        print(f"  No clusters formed on full Power time course. Max F = {np.max(F_pwr_full):.3f}")

    # Spatial cluster test on Relative Alpha Power within pre-defined ROI
    y_pwr_roi = y_pwr_topo[:, roi_indices]
    o_pwr_roi = o_pwr_topo[:, roi_indices]
    F_pwr_roi, cl_pwr_roi, pv_pwr_roi, _ = permutation_cluster_test(
        [y_pwr_roi, o_pwr_roi], adjacency=adj_roi, n_permutations=10000,
        tail=1, threshold=None, verbose=False, seed=42
    )
    if len(cl_pwr_roi) > 0:
        for cl, p_val in zip(cl_pwr_roi, pv_pwr_roi):
            d_val = compute_collapsed_cohens_d(y_pwr_roi, o_pwr_roi, cl[0])
            status = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            ch_names_sig = [info_roi['ch_names'][i] for i in cl[0]]
            print(f"  Power ROI Spatial Cluster [{status}]: {len(cl[0])} channels ({', '.join(ch_names_sig[:6])}) | p = {p_val:.4f} | Peak F = {np.max(F_pwr_roi[cl[0]]):.3f} | Cohen's d = {d_val:.3f}")
    else:
        print(f"  No spatial clusters formed for Power in pre-defined ROI. Max F = {np.max(F_pwr_roi):.3f}")

    # --- Calculations of Means and SEMs for Plotting ---
    avg_li_tc = {}; sem_li_tc = {}; avg_li_topo = {}
    avg_power_tc = {}; sem_power_tc = {}; avg_power_topo = {}
    
    for c in cohorts:
        avg_li_tc[c] = np.mean(li_tcs[c], axis=0)
        sem_li_tc[c] = sem(li_tcs[c], axis=0)
        avg_li_topo[c] = np.mean(li_topos[c], axis=0)
        
        avg_power_tc[c] = np.mean(power_tcs[c], axis=0)
        sem_power_tc[c] = sem(power_tcs[c], axis=0)
        avg_power_topo[c] = np.mean(power_topos[c], axis=0)

    # --- PLOTTING (2 Rows, 3 Columns) ---
    # Grid layout focused on the Alpha LI time course: the time-course
    # column is given a much larger share of the canvas width than the
    # topography columns, and the LI row (row 1) is taller than the power
    # row (row 2). The topographies are correspondingly rendered smaller.
    sns.set_theme(style="ticks", context="talk")
    fig = plt.figure(figsize=(16.5, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0],
                          height_ratios=[1.25, 1.0])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                     for r in range(2)])
    
    from SPACECUE.colors import COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD
    colors = {'Young': COLOR_DISTRACTOR_YOUNG, 'Old': COLOR_DISTRACTOR_OLD}
    # Highlight the 12 pre-defined parieto-occipital electrode pairs
    roi_marker_params = dict(marker='o', markerfacecolor='black', markeredgecolor='black',
                             linewidth=0, markersize=3.5, markeredgewidth=0)

    # ------------------ ROW 1: ALPHA LI ------------------
    # Time course (-1.0 to 1.0 s)
    ax_li_tc = axes[0, 0]
    for c in cohorts:
        n_c = n_young if c == 'Young' else n_old
        ax_li_tc.plot(times_vector, avg_li_tc[c], color=colors[c], lw=2.8, label=f"{c} (N={n_c})")
        ax_li_tc.fill_between(times_vector, avg_li_tc[c] - sem_li_tc[c], avg_li_tc[c] + sem_li_tc[c], color=colors[c], alpha=0.18)
    
    # Highlight pre-stimulus window [-0.75, 0.0 s]
    ax_li_tc.axvspan(-0.75, 0.0, color='gray', alpha=0.10, label='−750–0 ms')
    ax_li_tc.axhline(0, color='gray', linestyle='--', alpha=0.6, lw=1)
    ax_li_tc.axvline(0, color='black', linestyle=':', alpha=0.6, lw=1)
    ax_li_tc.set_xlim(-1.0, 1.0)
    ax_li_tc.set_ylabel("Alpha LI (Left − Right Hemi)", fontsize=13)
    ax_li_tc.set_title("Alpha LI Time Course", fontsize=14, fontweight='semibold')

    # Shade significant cluster intervals: one-sample clusters against zero
    # (in each cohort's color, Old additionally hatched) and between-group
    # Young vs. Old clusters (purple).
    dt = times_vector[1] - times_vector[0]

    # 1) One-sample clusters (LI differs from 0 within cohort)
    for c in cohorts:
        if np.any(is_li_sig_1samp[c]):
            first_idx_1s = np.where(is_li_sig_1samp[c])[0][0]
            for i in range(len(times_vector)):
                if is_li_sig_1samp[c][i]:
                    ax_li_tc.axvspan(times_vector[i], times_vector[i] + dt,
                                     facecolor=colors[c], alpha=0.15, lw=0,
                                     hatch='///' if c == 'Old' else None,
                                     edgecolor=colors[c] if c == 'Old' else 'none',
                                     label=f"{c} differs from 0 (p < 0.05)" if i == first_idx_1s else '_nolegend_')

    # 2) Between-group clusters (Young vs. Old)
    if np.any(is_li_sig):
        first_idx_grp = np.where(is_li_sig)[0][0]
        for i in range(len(times_vector)):
            if is_li_sig[i]:
                ax_li_tc.axvspan(times_vector[i], times_vector[i] + dt, color='purple', alpha=0.20, lw=0,
                                label='Young ≠ Old (p < 0.05)' if i == first_idx_grp else '_nolegend_')

    # 3) Trend towards significance in the collapsed pre-stimulus mean LI:
    #    one-sample t-test of the window-averaged LI [-0.75, 0] s against 0.
    #    Instead of enumerating all non-significant cluster results, this
    #    trend is visualized directly: each trending cohort's window-
    #    collapsed mean LI is drawn as a thick horizontal segment across
    #    the pre-stimulus window, annotated with the full statistics
    #    (t, df, exact unrounded p, Cohen's d_z).
    trend_txt_lines = []
    for c in cohorts:
        st = pre_mean_stats.get(c)
        if st is not None and 0.05 <= st['p'] < 0.10:
            trend_txt_lines.append(
                f"{c}: pre-stim mean LI vs. 0\n"
                f"t({st['df']}) = {st['t']:.2f}, p = {st['p']:.4f}, d = {st['dz']:.2f}"
            )
    if trend_txt_lines:
        ax_li_tc.text(0.04, 0.08, "\n".join(trend_txt_lines), transform=ax_li_tc.transAxes,
                      fontsize=10.5, fontstyle='italic', ha='left', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none'))

    ax_li_tc.legend(frameon=True, fontsize=10.5, loc='upper right')
    sns.despine(ax=ax_li_tc)

    # Dynamic LI topo scaling based strictly on actual data max (no artificial minimum floor)
    li_vmax = max(np.abs(avg_li_topo['Young']).max(), np.abs(avg_li_topo['Old']).max())
    li_vmin = -li_vmax
    
    # Younger LI Topomap
    ax_li_topo_y = axes[0, 1]
    im_li_y, _ = mne.viz.plot_topomap(
        avg_li_topo['Young'], info, axes=ax_li_topo_y, cmap='RdBu_r',
        vlim=(li_vmin, li_vmax), show=False, outlines='head', sensors=False,
        mask=roi_mask, mask_params=roi_marker_params
    )
    topo_y_title = f"Younger Cohort (N={n_young})\n−750 to 0 ms"
    ax_li_topo_y.set_title(topo_y_title, fontsize=13, fontweight='semibold')
    
    # Older LI Topomap
    ax_li_topo_o = axes[0, 2]
    im_li_o, _ = mne.viz.plot_topomap(
        avg_li_topo['Old'], info, axes=ax_li_topo_o, cmap='RdBu_r',
        vlim=(li_vmin, li_vmax), show=False, outlines='head', sensors=False,
        mask=roi_mask, mask_params=roi_marker_params
    )
    topo_o_title = f"Older Cohort (N={n_old})\n−750 to 0 ms"
    ax_li_topo_o.set_title(topo_o_title, fontsize=13, fontweight='semibold')
    
    # Shared LI colorbar
    cbar_li_ax = fig.add_axes([0.92, 0.58, 0.015, 0.3])
    cbar_li = plt.colorbar(im_li_y, cax=cbar_li_ax)
    cbar_li.set_label('Alpha LI', fontsize=11)
    cbar_li.ax.tick_params(labelsize=10)

    # ------------------ ROW 2: TOTAL ALPHA POWER ------------------
    # Time course (-1.0 to 1.0 s)
    ax_pwr_tc = axes[1, 0]
    for c in cohorts:
        n_c = n_young if c == 'Young' else n_old
        ax_pwr_tc.plot(times_vector, avg_power_tc[c], color=colors[c], lw=2.8, label=f"{c} (N={n_c})")
        ax_pwr_tc.fill_between(times_vector, avg_power_tc[c] - sem_power_tc[c], avg_power_tc[c] + sem_power_tc[c], color=colors[c], alpha=0.18)
        
    ax_pwr_tc.axvspan(-0.75, 0.0, color='gray', alpha=0.10)
    ax_pwr_tc.axhline(0.0, color='gray', linestyle='--', alpha=0.6, lw=1)
    ax_pwr_tc.axvline(0, color='black', linestyle=':', alpha=0.6, lw=1)
    ax_pwr_tc.set_xlim(-1.0, 1.0)
    ax_pwr_tc.set_xlabel("Time (s)", fontsize=13)
    ax_pwr_tc.set_ylabel("Log Relative Alpha Power (ln)", fontsize=13)
    ax_pwr_tc.set_title("Total Alpha Power Time Course\n", fontsize=14, fontweight='semibold')

    if np.any(is_pwr_sig):
        dt = times_vector[1] - times_vector[0]
        for i in range(len(times_vector)):
            if is_pwr_sig[i]:
                ax_pwr_tc.axvspan(times_vector[i], times_vector[i] + dt, color='purple', alpha=0.20, lw=0,
                                 label='Sig. Cluster (p < 0.05)' if i == np.where(is_pwr_sig)[0][0] else '_nolegend_')
        ax_pwr_tc.legend(frameon=True, fontsize=10.5, loc='upper right')
    else:
        min_p_pwr = np.min(pv_pwr_pre) if len(pv_pwr_pre) > 0 else np.nan
        ax_pwr_tc.legend(frameon=True, fontsize=10.5, loc='upper right')
    sns.despine(ax=ax_pwr_tc)

    # Consistent Power topo scaling: log-relative power is signed around 0
    # (0 = the subject's geometric mean power), so a symmetric diverging
    # scale centered on zero is used
    pwr_absmax = max(np.abs(avg_power_topo['Young']).max(),
                     np.abs(avg_power_topo['Old']).max())
    pwr_vmax, pwr_vmin = pwr_absmax, -pwr_absmax
    
    # Younger Power Topomap
    ax_pwr_topo_y = axes[1, 1]
    im_pwr_y, _ = mne.viz.plot_topomap(
        avg_power_topo['Young'], info, axes=ax_pwr_topo_y, cmap='RdBu_r',
        vlim=(pwr_vmin, pwr_vmax), show=False, outlines='head', sensors=False,
        mask=roi_mask, mask_params=roi_marker_params
    )
    ax_pwr_topo_y.set_title(f"Younger Cohort (N={n_young})\n−750 to 0 ms", fontsize=13, fontweight='semibold')
    
    # Older Power Topomap
    ax_pwr_topo_o = axes[1, 2]
    im_pwr_o, _ = mne.viz.plot_topomap(
        avg_power_topo['Old'], info, axes=ax_pwr_topo_o, cmap='RdBu_r',
        vlim=(pwr_vmin, pwr_vmax), show=False, outlines='head', sensors=False,
        mask=roi_mask, mask_params=roi_marker_params
    )
    pwr_o_title = f"Older Cohort (N={n_old})\n−750 to 0 ms"
    ax_pwr_topo_o.set_title(pwr_o_title, fontsize=13, fontweight='semibold')
    
    # Shared Power colorbar
    cbar_pwr_ax = fig.add_axes([0.92, 0.12, 0.015, 0.3])
    cbar_pwr = plt.colorbar(im_pwr_y, cax=cbar_pwr_ax)
    cbar_pwr.set_label('Log Rel. Alpha Power (ln)', fontsize=11)
    cbar_pwr.ax.tick_params(labelsize=10)

    # Labels and layout adjustments
    plt.tight_layout(rect=[0, 0, 0.9, 0.94])
    
    # Save the output plot
    plots_dir = os.path.normpath(os.path.join(project_root, "spacecue", "plots"))
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "pre_stim_alpha_lateralization.png")
    
    for attempt in range(3):
        try:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"\nSaved updated figure to {output_path}")
            break
        except Exception as e_save:
            if attempt < 2:
                time.sleep(1.0)
            else:
                print(f"Warning: Could not save to {output_path} ({e_save}). Attempting fallback filename...")
                fallback_path = os.path.join(plots_dir, "pre_stim_alpha_lateralization_updated.png")
                fig.savefig(fallback_path, bbox_inches='tight', dpi=300)
                print(f"Saved figure to fallback: {fallback_path}")

    # Save detailed statistical report
    report_path = os.path.join(plots_dir, "pre_stim_alpha_lateralization_report.txt")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("=================================================================\n")
        f_rep.write("  PRE-STIMULUS ALPHA LATERALIZATION: YOUNG VS. OLD COHORTS REPORT\n")
        f_rep.write("=================================================================\n\n")
        f_rep.write(f"Sample sizes: Young Cohort N = {n_young}, Old Cohort N = {n_old}\n")
        f_rep.write(f"Degrees of freedom (between-subjects): df = {df_between}\n\n")
        f_rep.write("1. ALPHA LATERALIZATION INDEX (LI) TIME-COURSE CLUSTER TESTS (12 ROI PAIRS):\n")
        f_rep.write(f"   Pre-stimulus window [-750 to 0 ms]:\n")
        if len(cl_li_pre) > 0:
            for i, (cl, p) in enumerate(zip(cl_li_pre, pv_li_pre)):
                d_val = compute_collapsed_cohens_d(y_li_pre, o_li_pre, cl[0])
                f_rep.write(f"     Cluster {i+1}: {times_vector[pre_stim_idx[cl[0][0]]]*1000:.0f}-{times_vector[pre_stim_idx[cl[0][-1]]]*1000:.0f} ms | p = {p:.5f} | Peak F = {np.max(F_li_pre[cl[0]]):.3f} | Cohen's d = {d_val:.3f}\n")
        else:
            f_rep.write(f"     No clusters formed at primary threshold. Max F = {np.max(F_li_pre):.3f}\n")
        f_rep.write(f"\n   Collapsed Pre-stimulus Mean LI [-750 to 0 ms]:\n")
        f_rep.write(f"     Young: {y_li_pre_mean.mean():.4f} ± {y_li_pre_mean.std():.4f} (SEM: {sem(y_li_pre_mean):.4f})\n")
        f_rep.write(f"     Old:   {o_li_pre_mean.mean():.4f} ± {o_li_pre_mean.std():.4f} (SEM: {sem(o_li_pre_mean):.4f})\n")
        f_rep.write(f"     t({n_young + n_old - 2}) = {tt_li_pre.statistic:.4f}, p = {tt_li_pre.pvalue:.4f}, Cohen's d = {d_li_pre:.4f}\n\n")
        f_rep.write("1b. ONE-SAMPLE TESTS AGAINST ZERO (ALPHA LI TIME COURSE, TWO-TAILED):\n")
        for c in cohorts:
            n_c = n_young if c == 'Young' else n_old
            f_rep.write(f"   {c} Cohort (N = {n_c}, df = {n_c - 1}):\n")
            if one_sample_report[c]:
                for line in one_sample_report[c]:
                    f_rep.write(line)
            else:
                f_rep.write("       No results available.\n")
        f_rep.write("\n")
        f_rep.write("2. PRE-DEFINED ROI SPATIAL CLUSTER TESTS (24 PARIETO-OCCIPITAL CHANNELS):\n")
        f_rep.write(f"   Alpha LI Topography in Pre-defined ROI:\n")
        if len(cl_li_roi) > 0:
            for i, (cl, p) in enumerate(zip(cl_li_roi, pv_li_roi)):
                d_val = compute_collapsed_cohens_d(y_li_roi, o_li_roi, cl[0])
                f_rep.write(f"     Spatial Cluster {i+1}: {len(cl[0])} channels | p = {p:.5f} | Peak F = {np.max(F_li_roi[cl[0]]):.3f} | Cohen's d = {d_val:.3f}\n")
        else:
            f_rep.write(f"     No spatial clusters formed in pre-defined ROI. Max F = {np.max(F_li_roi):.3f}\n")
        f_rep.write(f"   Total Alpha Power Topography in Pre-defined ROI:\n")
        if len(cl_pwr_roi) > 0:
            for i, (cl, p) in enumerate(zip(cl_pwr_roi, pv_pwr_roi)):
                d_val = compute_collapsed_cohens_d(y_pwr_roi, o_pwr_roi, cl[0])
                f_rep.write(f"     Spatial Cluster {i+1}: {len(cl[0])} channels | p = {p:.5f} | Peak F = {np.max(F_pwr_roi[cl[0]]):.3f} | Cohen's d = {d_val:.3f}\n")
        else:
            f_rep.write(f"     No spatial clusters formed in pre-defined ROI. Max F = {np.max(F_pwr_roi):.3f}\n")
        f_rep.write("\n3. TOTAL ALPHA POWER TIME COURSE TESTS (12 ROI PAIRS):\n")
        f_rep.write(f"     Pre-stimulus time course Max F = {np.max(F_pwr_pre):.3f}\n")
        f_rep.write(f"     Full time course Max F = {np.max(F_pwr_full):.3f}\n")
    print(f"Saved statistical report to {report_path}")

    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()

