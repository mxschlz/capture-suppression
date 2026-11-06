"""
SPACECUE — Alpha Lateralization Pipeline & Cueing Strength (0.8 vs 0.6) Analysis
================================================================================
Interactive script structured into sequential cells (# %%) for PyCharm / IPython.

Uses the shared incremental pickle cache (pre_stim_alpha_subject_data.pkl)
so existing subjects are loaded in < 1 second and new subjects compute only once.
"""

# %% [Cell 1: Setup, Imports & Configuration]
import sys
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import t, sem
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test

# Add project root to sys.path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

import SPACECUE
from SPACECUE.subjects import subject_ids
from SPACECUE.pre_stimulus_alpha_lateralization import run_computation

# Publication styling adhering strictly to AGENTS.md
sns.set_theme(context="talk", style="ticks")
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.ion()

PLOTS_DIR = os.path.normpath(os.path.join(project_root, "SPACECUE", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

# ROI definitions
left_roi_hemi  = ["TP9",  "TP7",  "CP5", "CP3", "CP1", "P7",  "P5",  "P3",  "P1",  "PO7", "PO3", "O1"]
right_roi_hemi = ["TP10", "TP8",  "CP6", "CP4", "CP2", "P8",  "P6",  "P4",  "P2",  "PO8", "PO4", "O2"]
alpha_fmin_hemi, alpha_fmax_hemi = 8, 12

print("=" * 80)
print("SPACECUE ALPHA LATERALIZATION & CUEING STRENGTH PIPELINE")
print(f"Output directory: {PLOTS_DIR}")
print("=" * 80)


# %% [Cell 2: Load Multi-Condition TFR Data from Shared Incremental Cache]
# Loads all condition splits: expected (all), expected_08, expected_06, target, singleton
cached_subject_data = run_computation(force_recompute=False, include_all_conditions=True)

# Reconstruct MNE AverageTFRArray objects from cached dictionaries
subject_ids_available = sorted(list(cached_subject_data.keys()))
print(f"\nReconstructing TFR objects for {len(subject_ids_available)} subjects...")

# Load template info from first subject epoch file
first_sub = subject_ids_available[0]
first_epo_pattern = os.path.join(SPACECUE.get_data_path(), "derivatives", "epoching",
                                 f"sci-{first_sub}", "eeg", f"sci-{first_sub}_task-spacecue-epo.fif")
template_info = mne.read_epochs(glob.glob(first_epo_pattern)[0], preload=False, verbose=False).info

times_vector = cached_subject_data[first_sub]['times']
freqs_vector = cached_subject_data[first_sub]['freqs']

subject_results = {
    'target_li': [],
    'singleton_li': [],
    'expected_li': [],
    'expected_li_08': [],
    'expected_li_06': []
}
subject_cohorts = []
processed_subjects = []

# Mappings from subject_results keys to cached condition keys
cond_map = {
    'target_li': 'target',
    'singleton_li': 'singleton',
    'expected_li': 'expected_all',
    'expected_li_08': 'expected_08',
    'expected_li_06': 'expected_06'
}

for sub in subject_ids_available:
    s_info = cached_subject_data[sub]
    conds = s_info.get('conditions', {})
    
    # Check if this subject has expected condition data
    exp_all = conds.get('expected_all', None)
    if exp_all is None:
        continue
        
    for res_key, c_key in cond_map.items():
        c_data = conds.get(c_key, None)
        if c_data is not None:
            tfr_obj = mne.time_frequency.AverageTFRArray(
                info=template_info,
                data=c_data['li_data'],
                times=times_vector,
                freqs=freqs_vector,
                nave=c_data.get('nave', 100),
                comment=f"sci-{sub}_{c_key}",
                method="morlet"
            )
            subject_results[res_key].append(tfr_obj)
            
    subject_cohorts.append(s_info['cohort'])
    processed_subjects.append(sub)

print(f"Successfully loaded {len(processed_subjects)} subjects with full condition splits.")


# %% [Cell 3: Helper Functions for Feature Extraction & Statistics]
def extract_hemi_tc(li_sub):
    """Return (left_tc, right_tc) time courses averaged over posterior ROI + alpha band."""
    ch = li_sub.ch_names
    vl = [c for c in left_roi_hemi if c in ch]
    vr = [c for c in right_roi_hemi if c in ch]
    if not vl or not vr:
        return None, None
    l_tc = li_sub.copy().pick(vl).crop(fmin=alpha_fmin_hemi, fmax=alpha_fmax_hemi).data.mean(axis=(0, 1))
    r_tc = li_sub.copy().pick(vr).crop(fmin=alpha_fmin_hemi, fmax=alpha_fmax_hemi).data.mean(axis=(0, 1))
    return l_tc, r_tc

def compute_collapsed_cohens_d(group1_data, group2_data, indices):
    """Computes circularity-safe Cohen's d collapsed across significant cluster indices."""
    g1 = np.mean(group1_data[:, indices], axis=1)
    g2 = np.mean(group2_data[:, indices], axis=1)
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / sp if sp > 0 else 0.0

def compute_one_sample_cohens_dz(data, indices):
    """Computes circularity-safe Cohen's d_z for 1-sample test collapsed across cluster."""
    collapsed = np.mean(data[:, indices], axis=1)
    sd = np.std(collapsed, ddof=1)
    return np.mean(collapsed) / sd if sd > 0 else 0.0

def run_cluster_test_1samp(diff_tc_all, n_permutations=10000, seed=42):
    """1-sample cluster permutation test on (N_subjects, times) array against 0."""
    n = diff_tc_all.shape[0]
    if n < 2:
        return None, [], [], None
    df = n - 1
    thresh = t.ppf(1 - 0.05 / 2, df)
    t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        diff_tc_all, threshold=thresh, n_permutations=n_permutations,
        tail=0, n_jobs=4, out_type="mask", seed=seed, verbose=False
    )
    return t_obs, clusters, cluster_pv, thresh

def ax_decor(ax, xlabel=None, ylabel=None):
    ax.axhline(0, color='k', ls=':', lw=0.8, alpha=0.7)
    ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.7)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    sns.despine(ax=ax)


# %% [Cell 4: Pre-Stimulus Alpha Lateralization by Cueing Strength (0.8 vs 0.6)]
# Extract L-R hemispheric difference time courses for 0.8 and 0.6
exp08_diffs, exp06_diffs = [], []
exp08_cohorts, exp06_cohorts = [], []
exp08_subs, exp06_subs = [], []

for idx, li_sub in enumerate(subject_results['expected_li_08']):
    l, r = extract_hemi_tc(li_sub)
    if l is not None:
        exp08_diffs.append(l - r)
        exp08_cohorts.append(subject_cohorts[idx])
        exp08_subs.append(processed_subjects[idx])

for idx, li_sub in enumerate(subject_results['expected_li_06']):
    l, r = extract_hemi_tc(li_sub)
    if l is not None:
        exp06_diffs.append(l - r)
        exp06_cohorts.append(subject_cohorts[idx])
        exp06_subs.append(processed_subjects[idx])

# Align paired subjects who have both 0.8 and 0.6
common_subs = [s for s in exp08_subs if s in exp06_subs]
idx08 = [exp08_subs.index(s) for s in common_subs]
idx06 = [exp06_subs.index(s) for s in common_subs]

arr08 = np.array([exp08_diffs[i] for i in idx08])
arr06 = np.array([exp06_diffs[i] for i in idx06])
paired_cohorts = [exp08_cohorts[i] for i in idx08]

n_total = len(common_subs)
pre_stim_mask = (times_vector >= -0.75) & (times_vector <= 0.0)

# Pre-stimulus window collapsed means per subject
pre08_means = arr08[:, pre_stim_mask].mean(axis=1)
pre06_means = arr06[:, pre_stim_mask].mean(axis=1)
pre_diff = pre08_means - pre06_means

# 1. Overall Statistics (All Subjects N=41)
print("\n" + "=" * 80)
print(f"CUEING STRENGTH STATISTICS: PROB 0.8 vs 0.6 (N = {n_total})")
print("=" * 80)

sh_08_w, sh_08_p = scipy.stats.shapiro(pre08_means)
sh_06_w, sh_06_p = scipy.stats.shapiro(pre06_means)
sh_df_w, sh_df_p = scipy.stats.shapiro(pre_diff)

tt_08 = scipy.stats.ttest_1samp(pre08_means, 0.0)
tt_06 = scipy.stats.ttest_1samp(pre06_means, 0.0)
tt_paired = scipy.stats.ttest_rel(pre08_means, pre06_means)

dz_08 = np.mean(pre08_means) / np.std(pre08_means, ddof=1)
dz_06 = np.mean(pre06_means) / np.std(pre06_means, ddof=1)
dz_paired = np.mean(pre_diff) / np.std(pre_diff, ddof=1)

print(f"Pre-stimulus Mean Alpha LI [-750 to 0 ms]:")
print(f"  Prob 0.8: M = {pre08_means.mean():.4f} ± {pre08_means.std(ddof=1):.4f} (SEM = {sem(pre08_means):.4f}) | t({n_total-1}) = {tt_08.statistic:.4f}, p = {tt_08.pvalue:.4f}, d_z = {dz_08:.4f}")
print(f"  Prob 0.6: M = {pre06_means.mean():.4f} ± {pre06_means.std(ddof=1):.4f} (SEM = {sem(pre06_means):.4f}) | t({n_total-1}) = {tt_06.statistic:.4f}, p = {tt_06.pvalue:.4f}, d_z = {dz_06:.4f}")
print(f"  Paired (0.8 − 0.6): Diff M = {pre_diff.mean():.4f} | t({n_total-1}) = {tt_paired.statistic:.4f}, p = {tt_paired.pvalue:.4f}, d_z = {dz_paired:.4f}")

# 2. Cohort Breakdown (Young vs Old × 0.8 vs 0.6)
young_mask = np.array([c == 'Young' for c in paired_cohorts])
old_mask = np.array([c == 'Old' for c in paired_cohorts])

n_young = np.sum(young_mask)
n_old = np.sum(old_mask)

y_pre08 = pre08_means[young_mask]
y_pre06 = pre06_means[young_mask]
y_diff = y_pre08 - y_pre06

o_pre08 = pre08_means[old_mask]
o_pre06 = pre06_means[old_mask]
o_diff = o_pre08 - o_pre06

y_tt_08 = scipy.stats.ttest_1samp(y_pre08, 0.0)
y_tt_06 = scipy.stats.ttest_1samp(y_pre06, 0.0)
y_tt_rel = scipy.stats.ttest_rel(y_pre08, y_pre06)

o_tt_08 = scipy.stats.ttest_1samp(o_pre08, 0.0)
o_tt_06 = scipy.stats.ttest_1samp(o_pre06, 0.0)
o_tt_rel = scipy.stats.ttest_rel(o_pre08, o_pre06)

print(f"\nCohort Breakdown:")
print(f"  Younger Cohort (N={n_young}):")
print(f"    0.8: M = {y_pre08.mean():.4f} (SEM = {sem(y_pre08):.4f}) | t({n_young-1}) = {y_tt_08.statistic:.4f}, p = {y_tt_08.pvalue:.4f}, d_z = {y_pre08.mean()/np.std(y_pre08, ddof=1):.4f}")
print(f"    0.6: M = {y_pre06.mean():.4f} (SEM = {sem(y_pre06):.4f}) | t({n_young-1}) = {y_tt_06.statistic:.4f}, p = {y_tt_06.pvalue:.4f}, d_z = {y_pre06.mean()/np.std(y_pre06, ddof=1):.4f}")
print(f"    0.8 vs 0.6: t({n_young-1}) = {y_tt_rel.statistic:.4f}, p = {y_tt_rel.pvalue:.4f}, d_z = {y_diff.mean()/np.std(y_diff, ddof=1):.4f}")
print(f"  Older Cohort (N={n_old}):")
print(f"    0.8: M = {o_pre08.mean():.4f} (SEM = {sem(o_pre08):.4f}) | t({n_old-1}) = {o_tt_08.statistic:.4f}, p = {o_tt_08.pvalue:.4f}, d_z = {o_pre08.mean()/np.std(o_pre08, ddof=1):.4f}")
print(f"    0.6: M = {o_pre06.mean():.4f} (SEM = {sem(o_pre06):.4f}) | t({n_old-1}) = {o_tt_06.statistic:.4f}, p = {o_tt_06.pvalue:.4f}, d_z = {o_pre06.mean()/np.std(o_pre06, ddof=1):.4f}")
print(f"    0.8 vs 0.6: t({n_old-1}) = {o_tt_rel.statistic:.4f}, p = {o_tt_rel.pvalue:.4f}, d_z = {o_diff.mean()/np.std(o_diff, ddof=1):.4f}")

# 2x2 Mixed ANOVA using pingouin if available
anova_report_str = ""
try:
    import pingouin as pg
    df_anova = pd.DataFrame({
        'Subject': np.concatenate([common_subs, common_subs]),
        'Cohort': np.concatenate([paired_cohorts, paired_cohorts]),
        'Probability': np.concatenate([['Prob_08']*n_total, ['Prob_06']*n_total]),
        'Alpha_LI': np.concatenate([pre08_means, pre06_means])
    })
    aov = pg.mixed_anova(data=df_anova, dv='Alpha_LI', within='Probability', between='Cohort', subject='Subject')
    anova_report_str = aov.to_string()
    print("\n2x2 Mixed ANOVA (Cohort × Probability):")
    print(aov[['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']])
except Exception as e_aov:
    print(f"\nCould not run mixed ANOVA: {e_aov}")

# Cluster permutation tests on time courses (Paired 0.8 - 0.6)
t_obs_pair, cl_pair, pv_pair, thresh_pair = run_cluster_test_1samp(arr08 - arr06)

# Save full text report
report_path = os.path.join(PLOTS_DIR, "alpha_lateralization_cueing_strength_report.txt")
with open(report_path, "w", encoding="utf-8") as f_rep:
    f_rep.write("=" * 80 + "\n")
    f_rep.write("  PRE-STIMULUS ALPHA LATERALIZATION: CUEING STRENGTH (0.8 vs 0.6) REPORT\n")
    f_rep.write("=" * 80 + "\n\n")
    f_rep.write(f"Total Paired Subjects N = {n_total} (Young N = {n_young}, Old N = {n_old})\n\n")
    f_rep.write("1. COLLAPSED PRE-STIMULUS WINDOW [-750 to 0 ms] (12 Posterior ROI Pairs, 8-12 Hz):\n")
    f_rep.write(f"   Overall (N={n_total}):\n")
    f_rep.write(f"     Prob 0.8: M = {pre08_means.mean():.4f} ± {pre08_means.std(ddof=1):.4f} (SEM = {sem(pre08_means):.4f}) | t({n_total-1}) = {tt_08.statistic:.4f}, p = {tt_08.pvalue:.4f}, d_z = {dz_08:.4f}\n")
    f_rep.write(f"     Prob 0.6: M = {pre06_means.mean():.4f} ± {pre06_means.std(ddof=1):.4f} (SEM = {sem(pre06_means):.4f}) | t({n_total-1}) = {tt_06.statistic:.4f}, p = {tt_06.pvalue:.4f}, d_z = {dz_06:.4f}\n")
    f_rep.write(f"     Paired (0.8 - 0.6): t({n_total-1}) = {tt_paired.statistic:.4f}, p = {tt_paired.pvalue:.4f}, d_z = {dz_paired:.4f}\n\n")
    f_rep.write(f"   Younger Cohort (N={n_young}):\n")
    f_rep.write(f"     Prob 0.8: M = {y_pre08.mean():.4f} (SEM = {sem(y_pre08):.4f}) | t({n_young-1}) = {y_tt_08.statistic:.4f}, p = {y_tt_08.pvalue:.4f}, d_z = {y_pre08.mean()/np.std(y_pre08, ddof=1):.4f}\n")
    f_rep.write(f"     Prob 0.6: M = {y_pre06.mean():.4f} (SEM = {sem(y_pre06):.4f}) | t({n_young-1}) = {y_tt_06.statistic:.4f}, p = {y_tt_06.pvalue:.4f}, d_z = {y_pre06.mean()/np.std(y_pre06, ddof=1):.4f}\n")
    f_rep.write(f"     Paired (0.8 - 0.6): t({n_young-1}) = {y_tt_rel.statistic:.4f}, p = {y_tt_rel.pvalue:.4f}, d_z = {y_diff.mean()/np.std(y_diff, ddof=1):.4f}\n\n")
    f_rep.write(f"   Older Cohort (N={n_old}):\n")
    f_rep.write(f"     Prob 0.8: M = {o_pre08.mean():.4f} (SEM = {sem(o_pre08):.4f}) | t({n_old-1}) = {o_tt_08.statistic:.4f}, p = {o_tt_08.pvalue:.4f}, d_z = {o_pre08.mean()/np.std(o_pre08, ddof=1):.4f}\n")
    f_rep.write(f"     Prob 0.6: M = {o_pre06.mean():.4f} (SEM = {sem(o_pre06):.4f}) | t({n_old-1}) = {o_tt_06.statistic:.4f}, p = {o_tt_06.pvalue:.4f}, d_z = {o_pre06.mean()/np.std(o_pre06, ddof=1):.4f}\n")
    f_rep.write(f"     Paired (0.8 - 0.6): t({n_old-1}) = {o_tt_rel.statistic:.4f}, p = {o_tt_rel.pvalue:.4f}, d_z = {o_diff.mean()/np.std(o_diff, ddof=1):.4f}\n\n")
    if anova_report_str:
        f_rep.write("2. 2x2 MIXED ANOVA (Cohort × Probability):\n")
        f_rep.write(anova_report_str + "\n\n")
print(f"Saved full statistical report to {report_path}")


# %% [Cell 5: Generate Publication Figures (Strictly AGENTS.md)]
plt.close('all')

# ------------------------------------------------------------------------------
# FIGURE 5: Pre-Stimulus Cueing Strength (0.8 vs 0.6) × Age Cohort Focus
# ------------------------------------------------------------------------------
print("\n--- Generating Figure 5: Cueing Strength Focus ---")
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))

color_08 = "#7B1FA2"  # Deep Purple for 0.8
color_06 = "#F57C00"  # Orange for 0.6

# Panel A: Younger Cohort Time Course (0.8 vs 0.6)
ax_y = axes5[0]
y_08_tc, y_06_tc = arr08[young_mask], arr06[young_mask]
ga_y08, se_y08 = y_08_tc.mean(0), sem(y_08_tc, 0)
ga_y06, se_y06 = y_06_tc.mean(0), sem(y_06_tc, 0)

ax_y.plot(times_vector, ga_y08, color=color_08, lw=2.5, label=f"Prob 0.8 (N={n_young})")
ax_y.fill_between(times_vector, ga_y08 - se_y08, ga_y08 + se_y08, color=color_08, alpha=0.3)
ax_y.plot(times_vector, ga_y06, color=color_06, lw=2.5, label=f"Prob 0.6 (N={n_young})")
ax_y.fill_between(times_vector, ga_y06 - se_y06, ga_y06 + se_y06, color=color_06, alpha=0.3)
ax_y.axvspan(-0.75, 0.0, color='gray', alpha=0.10, label='Pre-stim (−750–0 ms)')
ax_decor(ax_y, xlabel="Time (s)", ylabel="Alpha LI (Left − Right Hemi)")
ax_y.set_title(f"Younger Cohort: Alpha LI Time Course\n0.8 vs 0.6 p = {y_tt_rel.pvalue:.3f}", fontsize=13, fontweight='semibold')
ax_y.set_xlim(-1.0, 1.0)
ax_y.legend(loc="upper right", fontsize=10, frameon=True)

# Panel B: Older Cohort Time Course (0.8 vs 0.6)
ax_o = axes5[1]
o_08_tc, o_06_tc = arr08[old_mask], arr06[old_mask]
ga_o08, se_o08 = o_08_tc.mean(0), sem(o_08_tc, 0)
ga_o06, se_o06 = o_06_tc.mean(0), sem(o_06_tc, 0)

ax_o.plot(times_vector, ga_o08, color=color_08, lw=2.5, label=f"Prob 0.8 (N={n_old})")
ax_o.fill_between(times_vector, ga_o08 - se_o08, ga_o08 + se_o08, color=color_08, alpha=0.3)
ax_o.plot(times_vector, ga_o06, color=color_06, lw=2.5, label=f"Prob 0.6 (N={n_old})")
ax_o.fill_between(times_vector, ga_o06 - se_o06, ga_o06 + se_o06, color=color_06, alpha=0.3)
ax_o.axvspan(-0.75, 0.0, color='gray', alpha=0.10, label='Pre-stim (−750–0 ms)')
ax_decor(ax_o, xlabel="Time (s)")
ax_o.set_title(f"Older Cohort: Alpha LI Time Course\n0.8 vs 0.6 p = {o_tt_rel.pvalue:.3f}", fontsize=13, fontweight='semibold')
ax_o.set_xlim(-1.0, 1.0)
ax_o.legend(loc="upper right", fontsize=10, frameon=True)

# Panel C: Paired Subject-Level Slope Plot across Cueing Strengths
ax_slope = axes5[2]
x_pos = np.array([0, 1])

# Younger lines
for i in range(n_young):
    ax_slope.plot([0.8, 1.2], [y_pre06[i], y_pre08[i]], color="#1976D2", alpha=0.25, lw=1.2)
# Older lines
for i in range(n_old):
    ax_slope.plot([1.8, 2.2], [o_pre06[i], o_pre08[i]], color="#E64A19", alpha=0.25, lw=1.2)

# Group means + error bars
ax_slope.errorbar([0.8, 1.2], [y_pre06.mean(), y_pre08.mean()], yerr=[sem(y_pre06), sem(y_pre08)],
                  color="#0D47A1", lw=3.0, capsize=4, marker='o', markersize=8, label=f"Young (N={n_young})")
ax_slope.errorbar([1.8, 2.2], [o_pre06.mean(), o_pre08.mean()], yerr=[sem(o_pre06), sem(o_pre08)],
                  color="#BF360C", lw=3.0, capsize=4, marker='s', markersize=8, label=f"Old (N={n_old})")

ax_slope.axhline(0, color='black', linestyle=':', lw=0.8, alpha=0.7)
ax_slope.set_xticks([0.8, 1.2, 1.8, 2.2])
ax_slope.set_xticklabels(["0.6", "0.8", "0.6", "0.8"])
ax_slope.set_xlabel("Cueing Strength (Prob)", fontsize=13)
ax_slope.set_ylabel("Pre-Stim Mean Alpha LI", fontsize=13)
ax_slope.set_title("Individual & Group Alpha LI\n(−750 to 0 ms)", fontsize=13, fontweight='semibold')
ax_slope.legend(loc="upper left", frameon=True, fontsize=10)
sns.despine(ax=ax_slope)

fig5.tight_layout()
out5 = os.path.join(PLOTS_DIR, "alpha_lat_prestim_cueing_strength_cohort.png")
fig5.savefig(out5, dpi=300, bbox_inches="tight")
print(f"Saved: {out5}")

# ------------------------------------------------------------------------------
# FIGURE 1: GA TFR Spectrograms Overview
# ------------------------------------------------------------------------------
print("\n--- Generating Figure 1: TFR Overview ---")
ga_li_target    = mne.grand_average(subject_results['target_li'])
ga_li_singleton = mne.grand_average(subject_results['singleton_li'])
ga_diff_ts      = ga_li_target - ga_li_singleton

fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9.5), sharey=True)
fig1.suptitle("Grand Average Alpha Lateralization Index — TFR Spectrograms", fontsize=16, fontweight='semibold')

row0_specs = [
    (axes1[0, 0], ga_li_target,    f"Target LI (N={len(subject_results['target_li'])})"),
    (axes1[0, 1], ga_li_singleton, f"Singleton LI (N={len(subject_results['singleton_li'])})"),
    (axes1[0, 2], ga_diff_ts,       "Target − Singleton"),
]
for ax, ga_obj, title in row0_specs:
    ga_obj.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="RdBu_r")
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    sns.despine(ax=ax)

row1_specs = [
    (axes1[1, 0], 'expected_li',    f"Expected Distractor (All, N={len(subject_results['expected_li'])})"),
    (axes1[1, 1], 'expected_li_08', f"Expected Distractor — Prob 0.8 (N={len(subject_results['expected_li_08'])})"),
    (axes1[1, 2], 'expected_li_06', f"Expected Distractor — Prob 0.6 (N={len(subject_results['expected_li_06'])})"),
]
for ax, key, title in row1_specs:
    li_list = subject_results[key]
    if li_list:
        ga_obj = mne.grand_average(li_list)
        ga_obj.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="RdBu_r")
        ax.axvline(0, color='k', ls=':', lw=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        sns.despine(ax=ax)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(PLOTS_DIR, "alpha_lat_tfr_overview.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
print(f"Saved: {out1}")

# ------------------------------------------------------------------------------
# FIGURE 3: Probability Comparison (0.8 vs 0.6)
# ------------------------------------------------------------------------------
print("\n--- Generating Figure 3: Probability Comparison ---")
ga08 = arr08.mean(0)
se08 = sem(arr08, 0)
ga06 = arr06.mean(0)
se06 = sem(arr06, 0)

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
fig3.suptitle("Expected Distractor: Probability 0.8 vs 0.6 — Alpha Lateralization", fontsize=16, fontweight='semibold')

# Col 0: Time courses
ax = axes3[0]
ax.plot(times_vector, ga08, color=color_08, lw=2.5, label=f"Prob 0.8 (N={n_total})")
ax.fill_between(times_vector, ga08 - se08, ga08 + se08, color=color_08, alpha=0.3)
ax.plot(times_vector, ga06, color=color_06, lw=2.5, label=f"Prob 0.6 (N={n_total})")
ax.fill_between(times_vector, ga06 - se06, ga06 + se06, color=color_06, alpha=0.3)
ax.axvspan(-0.75, 0.0, color='gray', alpha=0.10, label='Pre-stim window')
ax.set_ylabel("L−R Alpha LI (8–12 Hz)", fontsize=12)
ax.set_title("Lateralization Strength over Time", fontsize=13, fontweight='semibold')
ax.legend(loc='best', fontsize=10, frameon=True)
ax_decor(ax, xlabel="Time (s)")

# Col 1: Paired Cluster Test
ax = axes3[1]
if t_obs_pair is not None:
    ax.plot(times_vector, t_obs_pair, color='black', lw=1.5, label="Observed T")
    ax.axhline(thresh_pair, color='gray', ls='--', lw=1, label=f"Threshold (t={thresh_pair:.2f})")
    ax.axhline(-thresh_pair, color='gray', ls='--', lw=1)
    sig_found = False
    for i, cl_mask in enumerate(cl_pair):
        if pv_pair[i] < 0.05:
            c_times = times_vector[cl_mask]
            if len(c_times) > 0:
                lbl = 'Sig. cluster (p < 0.05)' if not sig_found else '_nolegend_'
                sig_found = True
                ax.hlines(y=0.01, xmin=c_times[0], xmax=c_times[-1], color='purple', lw=4, alpha=0.7, label=lbl)
    ax.legend(loc='best', fontsize=10)
else:
    ax.text(0.5, 0.5, "No stats available", ha='center', va='center', transform=ax.transAxes)
ax.set_title("Paired Cluster Test (0.8 − 0.6)", fontsize=13, fontweight='semibold')
ax.set_ylabel("T-value", fontsize=12)
ax_decor(ax, xlabel="Time (s)")

# Col 2: TFR Difference Spectrogram
ax = axes3[2]
ga_exp08_all = mne.grand_average(subject_results['expected_li_08'])
ga_exp06_all = mne.grand_average(subject_results['expected_li_06'])
ga_diff_prob = ga_exp08_all - ga_exp06_all
ga_diff_prob.plot(axes=ax, show=False, colorbar=True, combine="mean", cmap="RdBu_r")
ax.axvline(0, color='k', ls=':', lw=0.8)
ax.set_title("TFR Difference Spectrogram (0.8 − 0.6)", fontsize=13, fontweight='semibold')
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Frequency (Hz)", fontsize=11)
sns.despine(ax=ax)

fig3.tight_layout()
out3 = os.path.join(PLOTS_DIR, "alpha_lat_probability_comparison.png")
fig3.savefig(out3, dpi=300, bbox_inches="tight")
print(f"Saved: {out3}")

print("\n" + "=" * 80)
print("ALPHA LATERALIZATION ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"All figures saved to: {PLOTS_DIR}")
print("=" * 80)
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    plt.show()
