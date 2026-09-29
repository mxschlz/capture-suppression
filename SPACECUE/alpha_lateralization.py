import sys
import mne
import numpy as np
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import t, sem
from mne.stats import permutation_cluster_1samp_test
import pandas as pd
import seaborn as sns
import SPACECUE
from SPACECUE.subjects import subject_ids

sns.set_theme(context="talk", style="ticks")

# --- Analysis Parameters ---
freqs = np.arange(5, 20, 1)
n_cycles = freqs / 2
method = "morlet"
decim = 5
n_jobs = 5

picks_for_diff = 'eeg'
tmin_analysis, tmax_analysis = -1.0, 1.0

left_roi_hemi  = ["TP9",  "TP7",  "CP5", "CP3", "CP1", "P7",  "P5",  "P3",  "P1",  "PO7", "PO3", "O1"]
right_roi_hemi = ["TP10", "TP8",  "CP6", "CP4", "CP2", "P8",  "P6",  "P4",  "P2",  "PO8", "PO4", "O2"]
alpha_fmin_hemi, alpha_fmax_hemi = 8, 12

n_permutations_cluster = 10000
alpha_stat_cluster = 0.05
cluster_tail = 0

SIG_LINE_Y_OFFSET_TVAL = 0.01
SIG_LINE_LW_TVAL       = 4
SIG_LINE_ALPHA_TVAL    = 0.7
SIG_LINE_COLOR_TVAL    = 'purple'

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
PLOTS_DIR = os.path.join(project_root, "spacecue", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data Storage ---
subject_results = {
    'target_li':      [],
    'singleton_li':   [],
    'expected_li':    [],
    'expected_li_08': [],
    'expected_li_06': []
}
subject_cohorts = []   # 'Young' or 'Old' per processed subject
processed_subjects = []
times_vector = None

# --- Subject Loop ---
for subject in subject_ids[:]:
    print(f"\n--- Processing Subject: {subject} ---")
    try:
        epoch_file_pattern = os.path.join(
            SPACECUE.get_data_path(), "derivatives", "epoching",
            f"sci-{subject}", "eeg", f"sci-{subject}_task-spacecue-epo.fif"
        )
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for subject {subject}. Skipping.")
            continue
        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for subject {subject}: {e}. Skipping.")
        continue

    # Determine cohort from metadata Age
    cohort = 'Unknown'
    if epochs_sub.metadata is not None and 'Age' in epochs_sub.metadata.columns:
        age = float(epochs_sub.metadata['Age'].iloc[0])
        cohort = 'Young' if age <= 35 else 'Old'

    all_conds_sub = list(epochs_sub.event_id.keys())

    try:
        left_target_epochs_sub  = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
        right_target_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]].copy()
        if len(left_target_epochs_sub) == 0 or len(right_target_epochs_sub) == 0:
            raise ValueError("Zero trials in one of the target conditions.")

        left_singleton_epochs_sub  = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
        right_singleton_epochs_sub = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]].copy()
        if len(left_singleton_epochs_sub) == 0 or len(right_singleton_epochs_sub) == 0:
            raise ValueError("Zero trials in one of the singleton conditions.")

        # Expected distractor
        if epochs_sub.metadata is not None and 'HP_Distractor_Loc' in epochs_sub.metadata.columns:
            hp_loc_numeric = pd.to_numeric(epochs_sub.metadata['HP_Distractor_Loc'], errors='coerce')
            left_expected_epochs_sub  = epochs_sub[hp_loc_numeric == 1.0].copy()
            right_expected_epochs_sub = epochs_sub[hp_loc_numeric == 3.0].copy()

            if 'HP_Distractor_Prob' in epochs_sub.metadata.columns:
                hp_prob_numeric = pd.to_numeric(epochs_sub.metadata['HP_Distractor_Prob'], errors='coerce')
                left_expected_08_epochs_sub  = epochs_sub[(hp_loc_numeric == 1.0) & (hp_prob_numeric == 0.8)].copy()
                right_expected_08_epochs_sub = epochs_sub[(hp_loc_numeric == 3.0) & (hp_prob_numeric == 0.8)].copy()
                left_expected_06_epochs_sub  = epochs_sub[(hp_loc_numeric == 1.0) & (hp_prob_numeric == 0.6)].copy()
                right_expected_06_epochs_sub = epochs_sub[(hp_loc_numeric == 3.0) & (hp_prob_numeric == 0.6)].copy()
            else:
                left_expected_08_epochs_sub = right_expected_08_epochs_sub = None
                left_expected_06_epochs_sub = right_expected_06_epochs_sub = None
        else:
            left_expected_epochs_sub  = right_expected_epochs_sub  = None
            left_expected_08_epochs_sub = right_expected_08_epochs_sub = None
            left_expected_06_epochs_sub = right_expected_06_epochs_sub = None
    except Exception as e:
        print(f"  Error splitting conditions for subject {subject}: {e}. Skipping.")
        continue

    def get_lateralization_index(left_epochs, right_epochs, picks_calc, freqs_calc, n_cycles_calc,
                                 decim_calc, method_calc, tmin_crop=None, tmax_crop=None, n_jobs_tfr=-1):
        tfr_left = left_epochs.compute_tfr(
            method=method_calc, picks=picks_calc, freqs=freqs_calc,
            n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_tfr,
            return_itc=False, average=False)
        if tmin_crop is not None or tmax_crop is not None:
            tfr_left.crop(tmin=tmin_crop, tmax=tmax_crop)
        alpha_left_avg = tfr_left.average()

        tfr_right = right_epochs.compute_tfr(
            method=method_calc, picks=picks_calc, freqs=freqs_calc,
            n_cycles=n_cycles_calc, decim=decim_calc, n_jobs=n_jobs_tfr,
            return_itc=False, average=False)
        if tmin_crop is not None or tmax_crop is not None:
            tfr_right.crop(tmin=tmin_crop, tmax=tmax_crop)
        alpha_right_avg = tfr_right.average()

        li_data = (alpha_left_avg.data - alpha_right_avg.data) / (alpha_left_avg.data + alpha_right_avg.data)
        li_power = mne.time_frequency.AverageTFRArray(
            data=li_data, info=alpha_left_avg.info, times=alpha_left_avg.times,
            freqs=freqs_calc, nave=(alpha_left_avg.nave + alpha_right_avg.nave) // 2,
            comment='Lateralization Index', method=method_calc
        )
        return li_power, alpha_left_avg.times

    try:
        alpha_diff_target_lr,    times_t = get_lateralization_index(
            left_target_epochs_sub, right_target_epochs_sub, picks_for_diff,
            freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs)
        alpha_diff_singleton_lr, _ = get_lateralization_index(
            left_singleton_epochs_sub, right_singleton_epochs_sub, picks_for_diff,
            freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs)

        if left_expected_epochs_sub is not None and len(left_expected_epochs_sub) > 0 and len(right_expected_epochs_sub) > 0:
            alpha_diff_expected_lr, _ = get_lateralization_index(
                left_expected_epochs_sub, right_expected_epochs_sub, picks_for_diff,
                freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs)
            subject_results['expected_li'].append(alpha_diff_expected_lr)
        if left_expected_08_epochs_sub is not None and len(left_expected_08_epochs_sub) > 0 and len(right_expected_08_epochs_sub) > 0:
            alpha_diff_expected_08_lr, _ = get_lateralization_index(
                left_expected_08_epochs_sub, right_expected_08_epochs_sub, picks_for_diff,
                freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs)
            subject_results['expected_li_08'].append(alpha_diff_expected_08_lr)
        if left_expected_06_epochs_sub is not None and len(left_expected_06_epochs_sub) > 0 and len(right_expected_06_epochs_sub) > 0:
            alpha_diff_expected_06_lr, _ = get_lateralization_index(
                left_expected_06_epochs_sub, right_expected_06_epochs_sub, picks_for_diff,
                freqs, n_cycles, decim, method, tmin_analysis, tmax_analysis, n_jobs_tfr=n_jobs)
            subject_results['expected_li_06'].append(alpha_diff_expected_06_lr)

        if times_vector is None:
            times_vector = times_t
        subject_results['target_li'].append(alpha_diff_target_lr)
        subject_results['singleton_li'].append(alpha_diff_singleton_lr)
        subject_cohorts.append(cohort)
        processed_subjects.append(subject)
    except Exception as e:
        print(f"  Error during TFR/LI calculation for subject {subject}: {e}. Skipping.")
        continue

print(f"\n--- Successfully processed {len(processed_subjects)} subjects: {processed_subjects} ---")

if not processed_subjects:
    raise RuntimeError("No subjects were successfully processed.")
if times_vector is None:
    raise RuntimeError("Could not determine the time vector.")

# --- Align time vectors ---
all_tfrs = (subject_results['target_li'] + subject_results['singleton_li'] +
            subject_results['expected_li'] + subject_results['expected_li_08'] + subject_results['expected_li_06'])
time_lengths = [len(tfr.times) for tfr in all_tfrs]

if len(set(time_lengths)) > 1:
    common_tmin = max(tfr.times[0] for tfr in all_tfrs)
    common_tmax = min(tfr.times[-1] for tfr in all_tfrs)
    for tfr in all_tfrs:
        tfr.crop(tmin=common_tmin, tmax=common_tmax)
    shortest_tfr = min(all_tfrs, key=lambda x: len(x.times))
    times_vector = shortest_tfr.times.copy()
    from scipy.interpolate import interp1d
    from mne.time_frequency import AverageTFRArray
    for i, tfr in enumerate(all_tfrs):
        if len(tfr.times) != len(times_vector) or not np.allclose(tfr.times, times_vector):
            f_interp = interp1d(tfr.times, tfr.data, axis=-1, bounds_error=False, fill_value="extrapolate")
            new_data = f_interp(times_vector)
            all_tfrs[i] = AverageTFRArray(
                info=tfr.info, data=new_data, times=times_vector, freqs=tfr.freqs,
                nave=tfr.nave, comment=tfr.comment, method=tfr.method)
    len_t   = len(subject_results['target_li'])
    len_s   = len(subject_results['singleton_li'])
    len_e   = len(subject_results['expected_li'])
    len_e08 = len(subject_results['expected_li_08'])
    len_e06 = len(subject_results['expected_li_06'])
    idx = 0
    subject_results['target_li']      = all_tfrs[idx:idx+len_t];   idx += len_t
    subject_results['singleton_li']   = all_tfrs[idx:idx+len_s];   idx += len_s
    subject_results['expected_li']    = all_tfrs[idx:idx+len_e];   idx += len_e
    subject_results['expected_li_08'] = all_tfrs[idx:idx+len_e08]; idx += len_e08
    subject_results['expected_li_06'] = all_tfrs[idx:idx+len_e06]
else:
    times_vector = all_tfrs[0].times.copy()

# --- Helper functions ---
def extract_hemi_tc(li_sub):
    """Return (left_tc, right_tc) time courses averaged over posterior ROI + alpha band."""
    ch = li_sub.ch_names
    vl = [c for c in left_roi_hemi  if c in ch]
    vr = [c for c in right_roi_hemi if c in ch]
    if not vl or not vr:
        return None, None
    l_tc = li_sub.copy().pick(vl).crop(fmin=alpha_fmin_hemi, fmax=alpha_fmax_hemi).data.mean(axis=(0, 1))
    r_tc = li_sub.copy().pick(vr).crop(fmin=alpha_fmin_hemi, fmax=alpha_fmax_hemi).data.mean(axis=(0, 1))
    return l_tc, r_tc

def run_cluster_test(diff_tc_all):
    """1-sample cluster permutation test on (N_subjects, times) difference array."""
    n = diff_tc_all.shape[0]
    if n < 2:
        return None, None, None, None
    df = n - 1
    thresh = t.ppf(1 - alpha_stat_cluster / 2, df)
    t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        diff_tc_all, threshold=thresh, n_permutations=n_permutations_cluster,
        tail=cluster_tail, n_jobs=n_jobs, out_type="mask", seed=42, verbose=False)
    return t_obs, clusters, cluster_pv, thresh

def plot_tc(ax, ga, se, color, label):
    ax.plot(times_vector, ga, color=color, lw=2.5, label=label)
    ax.fill_between(times_vector, ga - se, ga + se, color=color, alpha=0.18)

def plot_sig_clusters(ax, t_obs, thresh, clusters, cluster_pv):
    if t_obs is None:
        ax.text(0.5, 0.5, "No stats", ha='center', va='center', transform=ax.transAxes)
        return
    ax.plot(times_vector, t_obs, color='black', lw=1.5, label="Observed T")
    if thresh is not None:
        ax.axhline( thresh, color='gray', ls='--', lw=1, label=f'Threshold (t={thresh:.2f})')
        ax.axhline(-thresh, color='gray', ls='--', lw=1)
    sig_found = False
    if clusters is not None:
        for i, cl_mask in enumerate(clusters):
            if cluster_pv[i] < alpha_stat_cluster:
                c_times = times_vector[cl_mask]
                if len(c_times) > 0:
                    lbl = f'Sig. cluster (p<{alpha_stat_cluster})' if not sig_found else '_nolegend_'
                    sig_found = True
                    ax.hlines(y=SIG_LINE_Y_OFFSET_TVAL, xmin=c_times[0], xmax=c_times[-1],
                              color=SIG_LINE_COLOR_TVAL, lw=SIG_LINE_LW_TVAL,
                              alpha=SIG_LINE_ALPHA_TVAL, label=lbl)
    ax.legend(loc='best', fontsize=10)

def ax_decor(ax, xlabel=None, ylabel=None):
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.axvline(0, color='k', ls=':', lw=0.8)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    sns.despine(ax=ax)

# Shortcuts
has_expected    = len(subject_results['expected_li'])    > 0
has_expected_08 = len(subject_results['expected_li_08']) > 0
has_expected_06 = len(subject_results['expected_li_06']) > 0

# ============================================================
# FIGURE 1 — GA TFR Spectrograms Overview
#   Row 0: Target LI | Singleton LI | Target-Singleton diff
#   Row 1: Expected (all) | Expected (0.8) | Expected (0.6)
# ============================================================
print("\n--- Plotting Figure 1: TFR Overview ---")
ga_li_target    = mne.grand_average(subject_results['target_li'])
ga_li_singleton = mne.grand_average(subject_results['singleton_li'])
ga_diff_ts      = ga_li_target - ga_li_singleton

fig1, axes1 = plt.subplots(2, 3, figsize=(21, 11), sharey=True)
fig1.suptitle("Grand Average Alpha Lateralization Index — TFR Spectrograms", fontsize=17, fontweight='semibold')

row0_specs = [
    (axes1[0, 0], ga_li_target,    f"Target LI\n(N={len(subject_results['target_li'])})"),
    (axes1[0, 1], ga_li_singleton, f"Singleton LI\n(N={len(subject_results['singleton_li'])})"),
    (axes1[0, 2], ga_diff_ts,       "Target − Singleton"),
]
for ax, ga_obj, title in row0_specs:
    ga_obj.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="coolwarm")
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    sns.despine(ax=ax)

ga_li_expected = ga_li_exp08 = ga_li_exp06 = None
row1_specs = [
    (axes1[1, 0], 'expected_li',    "Expected Distractor LI"),
    (axes1[1, 1], 'expected_li_08', "Expected Distractor — Prob 0.8"),
    (axes1[1, 2], 'expected_li_06', "Expected Distractor — Prob 0.6"),
]
for ax, key, title in row1_specs:
    li_list = subject_results[key]
    if li_list:
        ga_obj = mne.grand_average(li_list)
        if key == 'expected_li':    ga_li_expected = ga_obj
        elif key == 'expected_li_08': ga_li_exp08 = ga_obj
        elif key == 'expected_li_06': ga_li_exp06 = ga_obj
        ga_obj.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="coolwarm")
        ax.axvline(0, color='k', ls=':', lw=0.8)
        ax.set_title(f"{title}\n(N={len(li_list)})", fontsize=13)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        sns.despine(ax=ax)
    else:
        ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(PLOTS_DIR, "alpha_lat_tfr_overview.png")
fig1.savefig(out1, bbox_inches='tight', dpi=200)
print(f"  Saved: {out1}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig1)

# ============================================================
# FIGURE 2 — Hemispheric Time Courses + Cluster Stats
#   3 columns: Target | Singleton | Expected Distractor
#   Row 0: Left vs Right Hemi LI time courses
#   Row 1: Cluster T-stats (L-R difference)
# ============================================================
print("\n--- Plotting Figure 2: Hemispheric Time Courses ---")

conditions_hemi = [
    ('Target',              subject_results['target_li']),
    ('Singleton',           subject_results['singleton_li']),
    ('Expected Distractor', subject_results['expected_li'] if has_expected else []),
]
col_colors = {'Left': '#1565C0', 'Right': '#C62828'}

fig2, axes2 = plt.subplots(2, 3, figsize=(21, 11), sharex=True)
fig2.suptitle("Hemispheric Alpha LI Time Courses + Cluster Permutation Tests", fontsize=17, fontweight='semibold')

for col, (cond_name, li_list) in enumerate(conditions_hemi):
    left_tcs, right_tcs = [], []
    for li_sub in li_list:
        l, r = extract_hemi_tc(li_sub)
        if l is not None:
            left_tcs.append(l); right_tcs.append(r)
    n = len(left_tcs)
    ax_tc   = axes2[0, col]
    ax_stat = axes2[1, col]

    if n == 0:
        ax_tc.set_visible(False); ax_stat.set_visible(False); continue

    left_arr  = np.array(left_tcs);  right_arr = np.array(right_tcs)
    ga_l, ga_r = left_arr.mean(0), right_arr.mean(0)
    se_l, se_r = sem(left_arr, 0),  sem(right_arr, 0)

    plot_tc(ax_tc, ga_l, se_l, col_colors['Left'],  'Left Hemi')
    plot_tc(ax_tc, ga_r, se_r, col_colors['Right'], 'Right Hemi')
    ax_tc.set_title(f"{cond_name} (N={n})", fontsize=13, fontweight='semibold')
    ax_tc.set_ylabel(f"Alpha LI ({alpha_fmin_hemi}–{alpha_fmax_hemi} Hz)" if col == 0 else "")
    ax_tc.legend(loc='best', fontsize=10, frameon=False)
    ax_decor(ax_tc)

    diff_arr = left_arr - right_arr
    t_obs, clusters, cluster_pv, thresh = run_cluster_test(diff_arr)
    plot_sig_clusters(ax_stat, t_obs, thresh, clusters, cluster_pv)
    ax_stat.set_ylabel("T-value" if col == 0 else "")
    ax_stat.set_title(f"{cond_name}: Cluster Test (L−R)", fontsize=12)
    ax_decor(ax_stat, xlabel="Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = os.path.join(PLOTS_DIR, "alpha_lat_hemispheric_timecourses.png")
fig2.savefig(out2, bbox_inches='tight', dpi=200)
print(f"  Saved: {out2}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig2)

# ============================================================
# FIGURE 3 — Probability Comparison: 0.6 vs 0.8
#   Col 0: Overlaid L-R LI time courses
#   Col 1: Paired cluster test (0.8 − 0.6)
#   Col 2: TFR difference spectrogram (0.8 − 0.6)
# ============================================================
print("\n--- Plotting Figure 3: Probability Comparison ---")

if has_expected_08 and has_expected_06:
    exp08_ltcs, exp08_rtcs, exp06_ltcs, exp06_rtcs = [], [], [], []
    for li_sub in subject_results['expected_li_08']:
        l, r = extract_hemi_tc(li_sub)
        if l is not None:
            exp08_ltcs.append(l); exp08_rtcs.append(r)
    for li_sub in subject_results['expected_li_06']:
        l, r = extract_hemi_tc(li_sub)
        if l is not None:
            exp06_ltcs.append(l); exp06_rtcs.append(r)

    n08, n06 = len(exp08_ltcs), len(exp06_ltcs)
    exp08_l, exp08_r = np.array(exp08_ltcs), np.array(exp08_rtcs)
    exp06_l, exp06_r = np.array(exp06_ltcs), np.array(exp06_rtcs)
    diff08 = exp08_l - exp08_r
    diff06 = exp06_l - exp06_r
    ga08 = diff08.mean(0); se08 = sem(diff08, 0) if n08 > 1 else np.zeros_like(ga08)
    ga06 = diff06.mean(0); se06 = sem(diff06, 0) if n06 > 1 else np.zeros_like(ga06)

    fig3, axes3 = plt.subplots(1, 3, figsize=(21, 6))
    fig3.suptitle("Expected Distractor: Probability 0.6 vs 0.8 — Alpha Lateralisation",
                  fontsize=17, fontweight='semibold')

    # Col 0: Overlaid time courses
    ax = axes3[0]
    plot_tc(ax, ga08, se08, '#6A0DAD', f'Prob 0.8 (N={n08})')
    plot_tc(ax, ga06, se06, '#E87722', f'Prob 0.6 (N={n06})')
    ax.axvspan(-0.75, 0.0, color='gray', alpha=0.08, label='Pre-stim window')
    ax.set_ylabel(f"L−R Alpha LI ({alpha_fmin_hemi}–{alpha_fmax_hemi} Hz)")
    ax.set_title("Lateralisation Strength over Time", fontsize=13, fontweight='semibold')
    ax.legend(loc='best', fontsize=11, frameon=False)
    ax_decor(ax, xlabel="Time (s)")

    # Col 1: Paired cluster test
    ax = axes3[1]
    if n08 == n06 and n08 >= 2:
        paired_diff = diff08 - diff06
        t_obs_p, cl_p, pv_p, thresh_p = run_cluster_test(paired_diff)
        plot_sig_clusters(ax, t_obs_p, thresh_p, cl_p, pv_p)
        ax.set_title("Paired Cluster Test (0.8 − 0.6)", fontsize=13, fontweight='semibold')
        ax.set_ylabel("T-value")
    else:
        ax.text(0.5, 0.5, f"Unequal N\n(N08={n08}, N06={n06})\nPaired test unavailable",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax_decor(ax, xlabel="Time (s)")

    # Col 2: TFR difference spectrogram
    ax = axes3[2]
    if ga_li_exp08 is not None and ga_li_exp06 is not None:
        ga_diff_prob = ga_li_exp08 - ga_li_exp06
        ga_diff_prob.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="coolwarm")
        ax.axvline(0, color='k', ls=':', lw=0.8)
        ax.set_title("TFR Difference Spectrogram (0.8 − 0.6)", fontsize=13, fontweight='semibold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out3 = os.path.join(PLOTS_DIR, "alpha_lat_probability_comparison.png")
    fig3.savefig(out3, bbox_inches='tight', dpi=200)
    print(f"  Saved: {out3}")
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close(fig3)
else:
    print("  Skipping Figure 3: insufficient probability data.")

# ============================================================
# FIGURE 4 — Age Split: Pre- and Post-Stimulus Alpha LI
#   Rows: Young | Old
#   Cols: Target | Singleton | Expected Distractor
#   Each panel: L-R hemi LI time course with window shading
# ============================================================
print("\n--- Plotting Figure 4: Age Split ---")

cohorts = ['Young', 'Old']
cohort_colors = {'Young': '#1F77B4', 'Old': '#FF7F0E'}

cond_keys   = ['target_li', 'singleton_li', 'expected_li']
cond_labels = ['Target',    'Singleton',    'Expected Distractor']

# Build per-cohort L-R diff time courses
# target_li and singleton_li are aligned 1:1 with subject_cohorts
cond_cohort_tcs = {}
for key in cond_keys[:2]:
    data = {c: [] for c in cohorts}
    for i, li_sub in enumerate(subject_results[key]):
        coh = subject_cohorts[i]
        l, r = extract_hemi_tc(li_sub)
        if l is not None:
            data[coh].append(l - r)
    cond_cohort_tcs[key] = data

# expected_li: appended in the same order as processed_subjects
# Use processed_subjects index to recover cohort
proc_set = list(processed_subjects)
data_exp = {c: [] for c in cohorts}
exp_idx  = 0
for i, sub in enumerate(proc_set):
    if exp_idx >= len(subject_results['expected_li']):
        break
    # expected_li entries correspond to subjects who had expected epochs;
    # they were appended in subject order, so we track with exp_idx.
    coh = subject_cohorts[i]
    l, r = extract_hemi_tc(subject_results['expected_li'][exp_idx])
    if l is not None:
        data_exp[coh].append(l - r)
    exp_idx += 1
cond_cohort_tcs['expected_li'] = data_exp

fig4, axes4 = plt.subplots(2, 3, figsize=(21, 11), sharex=True)
fig4.suptitle("Alpha Lateralisation: Young vs Old — Pre- and Post-Stimulus",
              fontsize=17, fontweight='semibold')

for col, (key, label) in enumerate(zip(cond_keys, cond_labels)):
    for row, cohort in enumerate(cohorts):
        ax = axes4[row, col]
        tcs = cond_cohort_tcs[key][cohort]
        n   = len(tcs)

        if row == 0:
            ax.set_title(label, fontsize=13, fontweight='semibold')
        if col == 0:
            ax.set_ylabel(f"{cohort}\nL−R Alpha LI ({alpha_fmin_hemi}–{alpha_fmax_hemi} Hz)", fontsize=11)
        if row == 1:
            ax.set_xlabel("Time (s)", fontsize=11)

        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax_decor(ax)
            continue

        arr = np.array(tcs)
        ga  = arr.mean(0)
        se  = sem(arr, 0) if n > 1 else np.zeros_like(ga)

        plot_tc(ax, ga, se, cohort_colors[cohort], f'{cohort} (N={n})')
        ax.axvspan(-0.75, 0.0, color='steelblue', alpha=0.07, label='Pre-stim (−0.75–0 s)')
        ax.axvspan( 0.0,  0.5, color='coral',     alpha=0.07, label='Post-stim (0–0.5 s)')
        ax.legend(loc='upper left', fontsize=10, frameon=False)
        ax_decor(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out4 = os.path.join(PLOTS_DIR, "alpha_lat_age_split.png")
fig4.savefig(out4, bbox_inches='tight', dpi=200)
print(f"  Saved: {out4}")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close(fig4)

print("\n=== All figures saved to:", PLOTS_DIR, "===")
