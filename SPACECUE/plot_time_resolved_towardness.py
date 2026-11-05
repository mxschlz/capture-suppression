"""
SPACECUE: Time-Resolved Towardness Dynamics Across Reaching Trajectories

This script calculates and plots continuous, time-resolved towardness waveforms
TT(t), DT(t), and CT(t) across normalized trajectory progression (0 to 100%):
  - Panel A: Tri-stimulus Time-Resolved Towardness (Target, Distractor, Control)
             computed on target response trials (Grand Average ± 1 SEM, N = 36).
  - Panel B: Target Towardness Accumulation: Younger vs. Older Adults
             comparing evidence accumulation rates and fractional peak latencies.
  - Panel C: Net Distractor Suppression: Delta_Suppression(t) = CT(t) - DT(t)
             isolating active distractor inhibition from neutral baseline, with
             cluster-based permutation tests for both age cohorts.
  - Panel D: Towardness Velocity (d/dt TT(t)): Continuous rate of decision commitment.
  - Panel E: Chosen-Option Time-Resolved Towardness Waveforms by Selected Outcome
             (Target vs. Control vs. Distractor) within Younger and Older Adults,
             with grand-average t50 markers and within-cohort cluster permutation tests.
  - Panel F: Subject-Level 50% Fractional Peak Latency (t50) by Selected Outcome
             (Target vs. Control vs. Distractor) within Younger and Older Adults,
             with paired statistical comparisons (Holm-Bonferroni corrected + Cohen's dz).

Adheres strictly to project AGENTS.md conventions:
  - Muted, perceptually uniform colors from SPACECUE/colors.py
  - Shaded SEM error bands on all trajectories
  - Cluster-level Cohen's dz / d computed on temporally collapsed cluster bounds per subject
  - Full statistical reporting (test statistics, df, exact p-values, effect sizes, assumption checks)
  - Units on all axes; top/right spines despined
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
import mne
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
from SPACECUE.colors import (
    COLOR_TARGET, COLOR_TARGET_YOUNG, COLOR_TARGET_OLD,
    COLOR_DISTRACTOR, COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD,
    COLOR_CONTROL, COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD,
)

# -------------------------------------------------------------------
# 0. AESTHETICS & PLOT CONFIGURATION
# -------------------------------------------------------------------
sns.set_theme(context="talk", style="ticks")
plt.ion()
plt.close('all')

NORM_STEPS = 101
time_pct = np.linspace(0, 100, NORM_STEPS)
COHORTS = ['Younger Adults', 'Older Adults']
OUTCOMES = ['Target', 'Control', 'Distractor']
PAIRS_OUTCOME = [('Target', 'Control'), ('Control', 'Distractor'), ('Target', 'Distractor')]
X_OF = {'Target': 0, 'Control': 1, 'Distractor': 2}


def stars_from_p(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def paired_comparison(vals_a, vals_b, label_a, label_b):
    """Paired comparison A vs B on subject-level values (via pingouin).

    Assumption check: Shapiro-Wilk on paired differences (pg.normality).
    If normality holds (p >= .05), runs a paired t-test (pg.ttest); otherwise
    runs a Wilcoxon signed-rank test (pg.wilcoxon). Effect size is Cohen's
    dz = mean(diff) / sd(diff) on the raw subject-level differences.
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    diff = vals_a - vals_b
    n = diff.size
    dz = diff.mean() / diff.std(ddof=1)
    norm_df = pg.normality(diff)
    sh_w = float(norm_df['W'].iloc[0])
    sh_p = float(norm_df['pval'].iloc[0])
    if sh_p >= 0.05:
        t_df = pg.ttest(vals_a, vals_b, paired=True)
        res = {'test': 't', 'stat': float(t_df['T'].iloc[0]),
               'df': int(t_df['dof'].iloc[0]), 'p': float(t_df['p-val'].iloc[0])}
    else:
        w_df = pg.wilcoxon(vals_a, vals_b)
        res = {'test': 'W', 'stat': float(w_df['W-val'].iloc[0]),
               'df': np.nan, 'p': float(w_df['p-val'].iloc[0])}
    res.update({
        'label_a': label_a, 'label_b': label_b, 'n': n,
        'mean_diff': float(diff.mean()),
        'shapiro_W': sh_w, 'shapiro_p': sh_p,
        'dz': float(dz),
    })
    return res


def draw_sig_bracket(ax, x1, x2, y, h, text):
    """Draw a significance bracket between x1 and x2 at height y with text on top."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='#212121', lw=1.2, zorder=7)
    ax.text((x1 + x2) / 2.0, y + h * 1.3, text, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#212121', zorder=8, linespacing=1.1)


def compute_t50_interpolated(curve):
    """Return the interpolated trajectory progression (%) where curve reaches 50% of its final value."""
    half_val = 0.5 * curve[-1]
    idx_above = np.where(curve >= half_val)[0]
    if len(idx_above) == 0:
        return 100.0
    if idx_above[0] == 0:
        return 0.0
    k = int(idx_above[0])
    y0, y1 = curve[k - 1], curve[k]
    frac = (half_val - y0) / (y1 - y0) if (y1 != y0) else 0.0
    return float((k - 1) + frac)


# -------------------------------------------------------------------
# 1. LOAD DATA & COMPUTE TIME-RESOLVED TOWARDNESS
# -------------------------------------------------------------------
print("[INFO] Loading continuous trajectory cache and behavioral metadata...")
data_path = SPACECUE.get_data_path()
cache_path = os.path.join(data_path, 'concatenated', 'aligned_mouse_resampled_101.npz')
archive = np.load(cache_path)
df_traj = pd.DataFrame(archive['data'], columns=archive['columns'].tolist())

# Restrict trajectory table to spatial/temporal columns before inner-joining
# the outlier-cleaned behavioral table (prevents stale embedded columns from leaking)
traj_keep = ['subject_id', 'block', 'trial_nr', 'step', 'time_norm', 'x_aligned', 'y_aligned']
df_traj = df_traj[[c for c in traj_keep if c in df_traj.columns]]

beh_path = os.path.join(data_path, 'concatenated', 'concatenated_behavioral.csv')
df_beh = pd.read_csv(beh_path)
beh_cols = [
    'subject_id', 'block', 'trial_nr', 'IsCorrect',
    'SelectControl', 'SelectDistractor', 'Age', 'SingletonPresent'
]
df_beh_sub = df_beh[beh_cols].drop_duplicates(['subject_id', 'block', 'trial_nr'])
df = pd.merge(df_traj, df_beh_sub, on=['subject_id', 'block', 'trial_nr'], how='inner')

# Exclude misses, categorize age cohort and click outcome
df = df[df['IsCorrect'].notna()].copy()
df['age_cohort'] = np.where(df['Age'] <= 35, 'Younger Adults', 'Older Adults')

df['outcome'] = 'Miss'
df.loc[df['IsCorrect'] == 1, 'outcome'] = 'Target'
df.loc[df['SelectControl'] == 1, 'outcome'] = 'Control'
df.loc[df['SelectDistractor'] == 1, 'outcome'] = 'Distractor'

# Mathematical Projection onto Stimulus Axes (Radius = 1.0 dva in aligned space):
#   Target:     (0, 1)              -> Proj_T = y * 100%
#   Distractor: (-sqrt(3)/2, -1/2)  -> Proj_D = (-x * sqrt(3) - y) / 2.0 * 100%
#   Control:    (+sqrt(3)/2, -1/2)  -> Proj_C = (+x * sqrt(3) - y) / 2.0 * 100%
sqrt3 = np.sqrt(3.0)
df['target_tt'] = df['y_aligned'] * 100.0
df['distractor_tt'] = (-df['x_aligned'] * sqrt3 - df['y_aligned']) / 2.0 * 100.0
df['control_tt'] = (df['x_aligned'] * sqrt3 - df['y_aligned']) / 2.0 * 100.0
df['suppression_tt'] = df['control_tt'] - df['distractor_tt']  # = (x * sqrt(3)) * 100%

# Chosen-option time-resolved towardness: projection onto whichever stimulus was selected
df['chosen_tt'] = np.where(
    df['outcome'] == 'Target', df['target_tt'],
    np.where(df['outcome'] == 'Control', df['control_tt'], df['distractor_tt'])
)

# -------------------------------------------------------------------
# 2. PARTICIPANT-LEVEL AGGREGATION
# -------------------------------------------------------------------
# 2A. Target-response trials (Panels A–D)
df_target = df[df['IsCorrect'] == 1].copy()

sub_tt = df_target.groupby(['subject_id', 'age_cohort', 'step'])['target_tt'].mean().unstack('step')
sub_dt = df_target.groupby(['subject_id', 'age_cohort', 'step'])['distractor_tt'].mean().unstack('step')
sub_ct = df_target.groupby(['subject_id', 'age_cohort', 'step'])['control_tt'].mean().unstack('step')
sub_st = df_target.groupby(['subject_id', 'age_cohort', 'step'])['suppression_tt'].mean().unstack('step')

X_tt_young = sub_tt.xs('Younger Adults', level='age_cohort').values  # (18, 101)
X_tt_old = sub_tt.xs('Older Adults', level='age_cohort').values      # (18, 101)

X_dt_young = sub_dt.xs('Younger Adults', level='age_cohort').values
X_dt_old = sub_dt.xs('Older Adults', level='age_cohort').values

X_ct_young = sub_ct.xs('Younger Adults', level='age_cohort').values
X_ct_old = sub_ct.xs('Older Adults', level='age_cohort').values

X_st_young = sub_st.xs('Younger Adults', level='age_cohort').values
X_st_old = sub_st.xs('Older Adults', level='age_cohort').values

ga_tt, sem_tt = np.mean(sub_tt.values, axis=0), stats.sem(sub_tt.values, axis=0)
ga_dt, sem_dt = np.mean(sub_dt.values, axis=0), stats.sem(sub_dt.values, axis=0)
ga_ct, sem_ct = np.mean(sub_ct.values, axis=0), stats.sem(sub_ct.values, axis=0)

# 2B. Chosen-option towardness by selected outcome (Panels E–F)
df_valid = df[df['outcome'].isin(OUTCOMES)].copy()
sub_chosen_curves = {}  # [cohort][outcome] -> DataFrame (n_subjects, 101)
sub_t50_df = {}         # [cohort] -> DataFrame (n_subjects, 3 outcomes)
sub_auc_df = {}         # [cohort] -> DataFrame (n_subjects, 3 outcomes)

for cohort in COHORTS:
    sub_chosen_curves[cohort] = {}
    for outcome in OUTCOMES:
        mat = df_valid[(df_valid['age_cohort'] == cohort) & (df_valid['outcome'] == outcome)].groupby(
            ['subject_id', 'step']
        )['chosen_tt'].mean().unstack('step')
        sub_chosen_curves[cohort][outcome] = mat

    subjects = sub_chosen_curves[cohort]['Target'].index
    t50_mat = pd.DataFrame(index=subjects, columns=OUTCOMES, dtype=float)
    auc_mat = pd.DataFrame(index=subjects, columns=OUTCOMES, dtype=float)
    for outcome in OUTCOMES:
        X_out = sub_chosen_curves[cohort][outcome].loc[subjects].values
        t50_mat[outcome] = [compute_t50_interpolated(X_out[i, :]) for i in range(X_out.shape[0])]
        auc_mat[outcome] = X_out.mean(axis=1)
    sub_t50_df[cohort] = t50_mat
    sub_auc_df[cohort] = auc_mat

# -------------------------------------------------------------------
# 3. STATISTICAL INFERENCE: CLUSTER-BASED PERMUTATION & PAIRED TESTS
# -------------------------------------------------------------------
print("\n--- STATISTICAL REPORT: TIME-RESOLVED CLUSTER-BASED PERMUTATION TESTS ---")

# Cluster Test 1: Net Suppression Index vs. 0 (Younger Adults)
t_obs_y, cl_y, p_y, _ = mne.stats.permutation_cluster_1samp_test(
    X_st_young, n_permutations=5000, seed=42, verbose=False
)
young_suppr_clusters = []
for cl, p in zip(cl_y, p_y):
    if p < 0.05:
        steps = cl[0]
        cl_m = X_st_young[:, steps].mean(axis=1)
        dz = cl_m.mean() / cl_m.std(ddof=1)
        young_suppr_clusters.append({'start': steps[0], 'end': steps[-1], 'p': p, 'dz': dz})
        print(f"Younger Adults Net Suppression Cluster: Steps {steps[0]}–{steps[-1]}% "
              f"({time_pct[steps[0]]:.1f}–{time_pct[steps[-1]]:.1f}%), p = {p:.6f}, Cohen's dz = {dz:.2f}")

# Cluster Test 2: Net Suppression Index vs. 0 (Older Adults)
t_obs_o, cl_o, p_o, _ = mne.stats.permutation_cluster_1samp_test(
    X_st_old, n_permutations=5000, seed=42, verbose=False
)
old_suppr_clusters = []
for cl, p in zip(cl_o, p_o):
    if p < 0.05:
        steps = cl[0]
        cl_m = X_st_old[:, steps].mean(axis=1)
        dz = cl_m.mean() / cl_m.std(ddof=1)
        old_suppr_clusters.append({'start': steps[0], 'end': steps[-1], 'p': p, 'dz': dz})
        print(f"Older Adults Net Suppression Cluster: Steps {steps[0]}–{steps[-1]}% "
              f"({time_pct[steps[0]]:.1f}–{time_pct[steps[-1]]:.1f}%), p = {p:.6f}, Cohen's dz = {dz:.2f}")

# Cluster Test 3: Target Towardness Comparison (Young vs. Old)
t_obs_diff, cl_diff, p_diff, _ = mne.stats.permutation_cluster_test(
    [X_tt_young, X_tt_old], n_permutations=5000, seed=42, verbose=False
)
tt_diff_clusters = []
for cl, p in zip(cl_diff, p_diff):
    if p < 0.05:
        steps = cl[0]
        diff_young = X_tt_young[:, steps].mean(axis=1)
        diff_old = X_tt_old[:, steps].mean(axis=1)
        pooled_sd = np.sqrt((np.var(diff_young, ddof=1) + np.var(diff_old, ddof=1)) / 2.0)
        d = (diff_young.mean() - diff_old.mean()) / pooled_sd
        tt_diff_clusters.append({'start': steps[0], 'end': steps[-1], 'p': p, 'd': d})
        print(f"Target Towardness Cohort Difference Cluster: Steps {steps[0]}–{steps[-1]}%, "
              f"p = {p:.6f}, Cohen's d = {d:.2f}")

# Cluster Test 4: Within-Cohort Chosen-Option Towardness (Target vs Control vs Distractor, Panel E)
print("\n--- PANEL E: WITHIN-COHORT CHOSEN-OPTION TOWARDNESS CLUSTER PERMUTATION TESTS ---")
outcome_cluster_info = {}  # [cohort][(a, b)] -> list of significant cluster dicts
for cohort in COHORTS:
    outcome_cluster_info[cohort] = {}
    print(f"[{cohort}]")
    for a, b in PAIRS_OUTCOME:
        Xa = sub_chosen_curves[cohort][a].values
        Xb = sub_chosen_curves[cohort][b].values
        diff_ab = Xa - Xb
        _, cl_ab, p_ab, _ = mne.stats.permutation_cluster_1samp_test(
            diff_ab, n_permutations=5000, seed=42, verbose=False
        )
        sig_ab = []
        for cl, p in zip(cl_ab, p_ab):
            if p < 0.05:
                steps = cl[0]
                cl_sub = diff_ab[:, steps].mean(axis=1)
                dz = cl_sub.mean() / cl_sub.std(ddof=1)
                sig_ab.append({
                    'start': int(steps[0]), 'end': int(steps[-1]),
                    'p': float(p), 'dz': float(dz),
                    'mean_diff': float(cl_sub.mean()), 'sd_diff': float(cl_sub.std(ddof=1))
                })
        sig_ab.sort(key=lambda c: c['p'])
        outcome_cluster_info[cohort][(a, b)] = sig_ab
        if sig_ab:
            for c_info in sig_ab:
                print(f"  {a} vs {b}: Steps {c_info['start']}–{c_info['end']}%, "
                      f"p = {c_info['p']:.6f}, d_z = {c_info['dz']:.2f}, "
                      f"mean_diff = {c_info['mean_diff']:+.2f}%, SD = {c_info['sd_diff']:.2f}%")
        else:
            print(f"  {a} vs {b}: no significant cluster (all p >= 0.05)")

# Panel F: Within-Cohort 50% Fractional Peak Latency (t50) RM-ANOVA & Pairwise Tests
print("\n--- PANEL F: WITHIN-COHORT 50% FRACTIONAL PEAK LATENCY (t50) STATISTICS ---")
panel_f_stats = {}
for cohort in COHORTS:
    t50_sub = sub_t50_df[cohort]
    long_t50 = t50_sub.reset_index().melt(id_vars='subject_id', var_name='outcome', value_name='t50')
    rm_res = pg.rm_anova(data=long_t50, dv='t50', within='outcome', subject='subject_id',
                         detailed=True, effsize='np2')
    f_val = float(rm_res['F'].iloc[0])
    df1, df2 = int(rm_res['DF'].iloc[0]), int(rm_res['DF'].iloc[1])
    p_rm = float(rm_res['p-unc'].iloc[0])
    np2 = float(rm_res['np2'].iloc[0])
    print(f"[{cohort}] One-way RM-ANOVA on t50: F({df1}, {df2}) = {f_val:.3f}, "
          f"p = {p_rm:.6g}, np2 = {np2:.4f}")
    for out in OUTCOMES:
        vals = t50_sub[out].values
        print(f"  {out:10s}: Mean = {vals.mean():.2f}%, SD = {vals.std(ddof=1):.2f}%, SEM = {stats.sem(vals):.2f}%")

    res_f = [paired_comparison(t50_sub[a].values, t50_sub[b].values, a, b) for a, b in PAIRS_OUTCOME]
    _, p_corr_f = pg.multicomp([r['p'] for r in res_f], alpha=0.05, method='holm')
    for r, ph in zip(res_f, p_corr_f):
        r['p_holm'] = float(ph)
        stat_s = f"t({r['df']}) = {r['stat']:.3f}" if r['test'] == 't' else f"W = {r['stat']:.1f}"
        print(f"  {r['label_a']} vs {r['label_b']}: {stat_s}, p = {r['p']:.6g}, "
              f"p_holm = {r['p_holm']:.6g}, d_z = {r['dz']:.2f}, "
              f"mean_diff = {r['mean_diff']:+.2f}%, Shapiro-W p = {r['shapiro_p']:.4g}")
    panel_f_stats[cohort] = res_f

# -------------------------------------------------------------------
# 4. FIGURE RENDERING (3x2 PUBLICATION LAYOUT)
# -------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(18, 20), gridspec_kw={'hspace': 0.36, 'wspace': 0.25})

# ===================================================================
# PANEL A: Tri-Stimulus Time-Resolved Towardness (N = 36)
# ===================================================================
ax_a = axes[0, 0]
ax_a.set_title("A — Time-Resolved Towardness Waveforms (All N = 36)", loc='left', pad=12, fontweight='bold', fontsize=14)
ax_a.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.6)

ax_a.plot(time_pct, ga_tt, color=COLOR_TARGET, lw=3.6, label='Target Towardness', zorder=6)
ax_a.fill_between(time_pct, ga_tt - sem_tt, ga_tt + sem_tt, color=COLOR_TARGET, alpha=0.25, zorder=5)

ax_a.plot(time_pct, ga_ct, color=COLOR_CONTROL, lw=3.0, label='Control Towardness', zorder=6)
ax_a.fill_between(time_pct, ga_ct - sem_ct, ga_ct + sem_ct, color=COLOR_CONTROL, alpha=0.22, zorder=5)

ax_a.plot(time_pct, ga_dt, color=COLOR_DISTRACTOR, lw=3.0, label='Distractor Towardness', zorder=6)
ax_a.fill_between(time_pct, ga_dt - sem_dt, ga_dt + sem_dt, color=COLOR_DISTRACTOR, alpha=0.22, zorder=5)

ax_a.annotate("Distractor Suppression\n(DT < CT)", xy=(50, -32.5), xytext=(65, -18),
              arrowprops=dict(arrowstyle="->", color=COLOR_DISTRACTOR, lw=1.8),
              fontsize=10.5, fontweight='bold', color=COLOR_DISTRACTOR, ha='left')

ax_a.set_xlabel("Trajectory Progression (%)")
ax_a.set_ylabel("Towardness Score (%)")
ax_a.set_ylim(-55, 95)
ax_a.legend(loc='upper left', frameon=True, fontsize=10.5, framealpha=0.92)
sns.despine(ax=ax_a)

# ===================================================================
# PANEL B: Target Towardness Accumulation (Young vs. Old)
# ===================================================================
ax_b = axes[0, 1]
ax_b.set_title("B — Target Towardness Accumulation by Cohort", loc='left', pad=12, fontweight='bold', fontsize=14)
ax_b.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.6)

m_tt_y, s_tt_y = np.mean(X_tt_young, axis=0), stats.sem(X_tt_young, axis=0)
m_tt_o, s_tt_o = np.mean(X_tt_old, axis=0), stats.sem(X_tt_old, axis=0)

ax_b.plot(time_pct, m_tt_y, color=COLOR_TARGET_YOUNG, lw=3.5, label='Younger Adults (N = 18)', zorder=6)
ax_b.fill_between(time_pct, m_tt_y - s_tt_y, m_tt_y + s_tt_y, color=COLOR_TARGET_YOUNG, alpha=0.28, zorder=5)

ax_b.plot(time_pct, m_tt_o, color=COLOR_TARGET_OLD, lw=3.5, label='Older Adults (N = 18)', zorder=6)
ax_b.fill_between(time_pct, m_tt_o - s_tt_o, m_tt_o + s_tt_o, color=COLOR_TARGET_OLD, alpha=0.28, zorder=5)

t50_idx_y = np.where(m_tt_y >= (m_tt_y[-1] * 0.5))[0][0]
t50_idx_o = np.where(m_tt_o >= (m_tt_o[-1] * 0.5))[0][0]
ax_b.plot([time_pct[t50_idx_y], time_pct[t50_idx_y]], [0, m_tt_y[t50_idx_y]], color=COLOR_TARGET_YOUNG, linestyle=':', lw=2.0)
ax_b.plot([time_pct[t50_idx_o], time_pct[t50_idx_o]], [0, m_tt_o[t50_idx_o]], color=COLOR_TARGET_OLD, linestyle=':', lw=2.0)

ax_b.text(time_pct[t50_idx_o] - 2, 40, f"t50 = {time_pct[t50_idx_o]:.0f}%",
          ha='right', va='center', fontsize=9.5, fontweight='bold', color=COLOR_TARGET_OLD)
ax_b.text(time_pct[t50_idx_y] + 2, 40, f"t50 = {time_pct[t50_idx_y]:.0f}%",
          ha='left', va='center', fontsize=9.5, fontweight='bold', color=COLOR_TARGET_YOUNG)

if tt_diff_clusters:
    c_df = tt_diff_clusters[0]
    ax_b.plot([c_df['start'], c_df['end']], [92, 92], color='#212121', lw=3.2)
    ax_b.text((c_df['start'] + c_df['end']) / 2.0, 95,
              f"** Cohort Difference Cluster (p = {c_df['p']:.4f}, d = {abs(c_df['d']):.2f})",
              ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#212121')

ax_b.set_xlabel("Trajectory Progression (%)")
ax_b.set_ylabel("Target Towardness (%)")
ax_b.set_ylim(-10, 105)
ax_b.legend(loc='lower right', frameon=True, fontsize=10.5, framealpha=0.92)
sns.despine(ax=ax_b)

# ===================================================================
# PANEL C: Net Distractor Suppression Dynamics (CT - DT)
# ===================================================================
ax_c = axes[1, 0]
ax_c.set_title("C — Net Distractor Suppression Dynamics (CT - DT)", loc='left', pad=12, fontweight='bold', fontsize=14)
ax_c.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.6)

m_st_y, s_st_y = np.mean(X_st_young, axis=0), stats.sem(X_st_young, axis=0)
m_st_o, s_st_o = np.mean(X_st_old, axis=0), stats.sem(X_st_old, axis=0)

ax_c.plot(time_pct, m_st_y, color=COLOR_TARGET_YOUNG, lw=3.5, label='Younger Adults (N = 18)', zorder=6)
ax_c.fill_between(time_pct, m_st_y - s_st_y, m_st_y + s_st_y, color=COLOR_TARGET_YOUNG, alpha=0.28, zorder=5)

ax_c.plot(time_pct, m_st_o, color=COLOR_TARGET_OLD, lw=3.5, label='Older Adults (N = 18)', zorder=6)
ax_c.fill_between(time_pct, m_st_o - s_st_o, m_st_o + s_st_o, color=COLOR_TARGET_OLD, alpha=0.28, zorder=5)

if young_suppr_clusters:
    c_y = young_suppr_clusters[0]
    ax_c.plot([c_y['start'], c_y['end']], [-1.8, -1.8], color=COLOR_TARGET_YOUNG, lw=3.5, zorder=8)
    ax_c.text((c_y['start'] + c_y['end']) / 2.0, -2.8, f"*** Young (p = {c_y['p']:.4f}, dz = {c_y['dz']:.2f})",
              ha='center', va='top', fontsize=9.0, fontweight='bold', color=COLOR_TARGET_YOUNG)

if old_suppr_clusters:
    c_o = old_suppr_clusters[0]
    ax_c.plot([c_o['start'], c_o['end']], [-4.2, -4.2], color=COLOR_TARGET_OLD, lw=3.5, zorder=8)
    ax_c.text((c_o['start'] + c_o['end']) / 2.0, -5.2, f"*** Old (p = {c_o['p']:.4f}, dz = {c_o['dz']:.2f})",
              ha='center', va='top', fontsize=9.0, fontweight='bold', color=COLOR_TARGET_OLD)

ax_c.set_xlabel("Trajectory Progression (%)")
ax_c.set_ylabel("Net Suppression Index (%)\n[Control - Distractor Towardness]")
ax_c.set_ylim(-7.0, 7.5)
ax_c.legend(loc='upper left', frameon=True, fontsize=10.5, framealpha=0.92)
sns.despine(ax=ax_c)

# ===================================================================
# PANEL D: Towardness Velocity / Accumulation Rate (d/dt TT)
# ===================================================================
ax_d = axes[1, 1]
ax_d.set_title("D — Towardness Velocity: Rate of Decision Commitment", loc='left', pad=12, fontweight='bold', fontsize=14)
ax_d.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.6)

dt_step = time_pct[1] - time_pct[0]  # = 1.0% progression
vel_tt_young = np.gradient(X_tt_young, dt_step, axis=1)
vel_tt_old = np.gradient(X_tt_old, dt_step, axis=1)

m_vel_y, s_vel_y = np.mean(vel_tt_young, axis=0), stats.sem(vel_tt_young, axis=0)
m_vel_o, s_vel_o = np.mean(vel_tt_old, axis=0), stats.sem(vel_tt_old, axis=0)

ax_d.plot(time_pct, m_vel_y, color=COLOR_TARGET_YOUNG, lw=3.5, label='Younger Adults (N = 18)', zorder=6)
ax_d.fill_between(time_pct, m_vel_y - s_vel_y, m_vel_y + s_vel_y, color=COLOR_TARGET_YOUNG, alpha=0.28, zorder=5)

ax_d.plot(time_pct, m_vel_o, color=COLOR_TARGET_OLD, lw=3.5, label='Older Adults (N = 18)', zorder=6)
ax_d.fill_between(time_pct, m_vel_o - s_vel_o, m_vel_o + s_vel_o, color=COLOR_TARGET_OLD, alpha=0.28, zorder=5)

peak_idx_y = np.argmax(m_vel_y)
peak_idx_o = np.argmax(m_vel_o)

ax_d.plot(time_pct[peak_idx_y], m_vel_y[peak_idx_y], 'o', color=COLOR_TARGET_YOUNG, markersize=8, zorder=8)
ax_d.text(time_pct[peak_idx_y] + 2, m_vel_y[peak_idx_y] + 0.08,
          f"Peak: {m_vel_y[peak_idx_y]:.2f}%/step\nat {time_pct[peak_idx_y]:.0f}%",
          ha='left', va='bottom', fontsize=9.0, fontweight='bold', color=COLOR_TARGET_YOUNG)

ax_d.plot(time_pct[peak_idx_o], m_vel_o[peak_idx_o], 'o', color=COLOR_TARGET_OLD, markersize=8, zorder=8)
ax_d.text(time_pct[peak_idx_o] - 2, m_vel_o[peak_idx_o] + 0.08,
          f"Peak: {m_vel_o[peak_idx_o]:.2f}%/step\nat {time_pct[peak_idx_o]:.0f}%",
          ha='right', va='bottom', fontsize=9.0, fontweight='bold', color=COLOR_TARGET_OLD)

ax_d.set_xlabel("Trajectory Progression (%)")
ax_d.set_ylabel("Towardness Velocity (% / step)\n[d/dt Target Towardness]")
ax_d.set_ylim(-0.4, 3.2)
ax_d.legend(loc='upper right', frameon=True, fontsize=10.5, framealpha=0.92)
sns.despine(ax=ax_d)

# ===================================================================
# PANEL E: Chosen-Option Time-Resolved Towardness by Selected Outcome
#          (Split side-by-side: Younger Adults vs. Older Adults)
# ===================================================================
gs_e = axes[2, 0].get_subplotspec().subgridspec(1, 2, wspace=0.22)
axes[2, 0].remove()
ax_e1 = fig.add_subplot(gs_e[0])
ax_e2 = fig.add_subplot(gs_e[1])

cohort_e_cfg = [
    (ax_e1, 'Younger Adults',
     {'Target': COLOR_TARGET_YOUNG, 'Control': COLOR_CONTROL_YOUNG, 'Distractor': COLOR_DISTRACTOR_YOUNG},
     "E — Younger Adults (N = 18)"),
    (ax_e2, 'Older Adults',
     {'Target': COLOR_TARGET_OLD, 'Control': COLOR_CONTROL_OLD, 'Distractor': COLOR_DISTRACTOR_OLD},
     "Older Adults (N = 18)"),
]

for ax_e, cohort, col_map, title_str in cohort_e_cfg:
    ax_e.set_title(title_str, loc='left', pad=12, fontweight='bold', fontsize=13)
    ax_e.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)

    for out in OUTCOMES:
        X_out = sub_chosen_curves[cohort][out].values
        m_out, s_out = np.mean(X_out, axis=0), stats.sem(X_out, axis=0)
        c_out = col_map[out]
        ax_e.plot(time_pct, m_out, color=c_out, lw=3.0, label=out, zorder=6)
        ax_e.fill_between(time_pct, m_out - s_out, m_out + s_out, color=c_out, alpha=0.24, zorder=5)

        # Grand-average t50 marker for each selected outcome
        t50_ga = compute_t50_interpolated(m_out)
        y_half = 0.5 * m_out[-1]
        ax_e.plot([t50_ga, t50_ga], [0, y_half], color=c_out, linestyle=':', lw=1.8, zorder=7)
        ax_e.plot(t50_ga, y_half, 'o', color=c_out, markersize=5.5, zorder=8)

    # Plot within-cohort cluster bars at the bottom of the panel
    bar_levels = [
        (('Target', 'Distractor'), -11.0, col_map['Target']),
        (('Control', 'Distractor'), -19.5, col_map['Control']),
        (('Target', 'Control'), -28.0, '#455A64'),
    ]
    for (a, b), y_bar, b_col in bar_levels:
        sigs = outcome_cluster_info[cohort][(a, b)]
        if sigs:
            cl = sigs[0]
            ax_e.plot([cl['start'], cl['end']], [y_bar, y_bar], color=b_col, lw=3.0, zorder=8)
            mid_x = min(max((cl['start'] + cl['end']) / 2.0, 18.0), 82.0)
            ax_e.text(mid_x, y_bar + 1.5,
                      f"{a[0]} vs {b[0]}: {stars_from_p(cl['p'])} dz = {cl['dz']:.2f}",
                      ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#212121', zorder=9)
        else:
            ax_e.text(50.0, y_bar + 1.0,
                      f"{a[0]} vs {b[0]}: n.s. (no cluster)",
                      ha='center', va='bottom', fontsize=8.0, style='italic', color='#757575', zorder=9)

    ax_e.set_xlabel("Trajectory Progression (%)")
    ax_e.set_xticks([0, 50, 100])
    ax_e.set_ylim(-32, 95)
    ax_e.legend(loc='upper left', frameon=True, fontsize=9.0, framealpha=0.92)
    sns.despine(ax=ax_e)

ax_e1.set_ylabel("Chosen-Option Towardness (%)")
ax_e2.set_ylabel("")
ax_e2.tick_params(labelleft=False)

# ===================================================================
# PANEL F: Subject-Level 50% Fractional Peak Latency (t50) by Outcome
#          (Split side-by-side: Younger Adults vs. Older Adults)
# ===================================================================
gs_f = axes[2, 1].get_subplotspec().subgridspec(1, 2, wspace=0.25)
axes[2, 1].remove()
ax_f1 = fig.add_subplot(gs_f[0])
ax_f2 = fig.add_subplot(gs_f[1])

x_pos_f = np.arange(len(OUTCOMES))
cohort_f_cfg = [
    (ax_f1, 'Younger Adults',
     [COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG],
     "F — Younger Adults (N = 18)"),
    (ax_f2, 'Older Adults',
     [COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD],
     "Older Adults (N = 18)"),
]

np.random.seed(42)
# Shared y-range across F1 and F2 for direct visual comparison
all_t50_max = max(sub_t50_df[c][o].max() for c in COHORTS for o in OUTCOMES)
all_t50_min = min(sub_t50_df[c][o].min() for c in COHORTS for o in OUTCOMES)
yr_f = (all_t50_max + 28) - (all_t50_min - 6)
lvl_base_f = all_t50_max + 0.04 * yr_f
lvl_step_f = 0.095 * yr_f
h_f = 0.015 * yr_f

for ax_f, cohort, cols_f, title_str in cohort_f_cfg:
    t50_sub = sub_t50_df[cohort]
    ax_f.set_title(title_str, loc='left', pad=12, fontweight='bold', fontsize=13)
    ax_f.axhline(50.0, color='#9E9E9E', linestyle='--', lw=1.1, alpha=0.7, label='50% Time')

    jitter_f = np.random.uniform(-0.06, 0.06, size=len(t50_sub))
    for i in range(len(t50_sub)):
        s_row = t50_sub.iloc[i]
        y_vals = [s_row['Target'], s_row['Control'], s_row['Distractor']]
        ax_f.plot(x_pos_f + jitter_f[i], y_vals, color='gray', alpha=0.30, lw=0.8, zorder=2)

    for idx, out in enumerate(OUTCOMES):
        vals = t50_sub[out].values
        ax_f.scatter(np.full_like(vals, x_pos_f[idx]) + jitter_f, vals,
                     color=cols_f[idx], alpha=0.65, s=36, zorder=3)
        m, sem = vals.mean(), stats.sem(vals)
        ax_f.errorbar(x_pos_f[idx], m, yerr=sem, fmt='o', color='black', ecolor='black',
                      elinewidth=2.2, capsize=5, capthick=1.8, markersize=7, zorder=6)

    stat_f = {(r['label_a'], r['label_b']): r for r in panel_f_stats[cohort]}
    lvl_f = {
        'Target_Control': lvl_base_f,
        'Control_Distractor': lvl_base_f + lvl_step_f,
        'Target_Distractor': lvl_base_f + 2 * lvl_step_f,
    }
    for a, b in PAIRS_OUTCOME:
        r = stat_f[(a, b)]
        draw_sig_bracket(ax_f, X_OF[a], X_OF[b], lvl_f[f'{a}_{b}'], h_f,
                         f"{stars_from_p(r['p_holm'])} $d_z$ = {r['dz']:.2f}")

    ax_f.set_xticks(x_pos_f)
    ax_f.set_xticklabels(OUTCOMES, fontsize=11)
    ax_f.set_xlim(-0.45, 2.45)
    ax_f.set_ylim(all_t50_min - 6, all_t50_max + 0.34 * yr_f)
    sns.despine(ax=ax_f)

ax_f1.set_ylabel("50% Fractional Peak Latency t50 (%)")
ax_f2.set_ylabel("")
ax_f2.tick_params(labelleft=False)

# -------------------------------------------------------------------
# 5. SAVE FIGURES
# -------------------------------------------------------------------
plots_dir = os.path.join(project_root, 'SPACECUE', 'plots')
os.makedirs(plots_dir, exist_ok=True)
png_path = os.path.join(plots_dir, 'time_resolved_towardness.png')
svg_path = os.path.join(plots_dir, 'time_resolved_towardness.svg')

for attempt in range(5):
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        break
    except OSError:
        time.sleep(1.0)

print(f"[INFO] Successfully saved time-resolved towardness figure to:\n       {png_path}\n       {svg_path}")

plt.show()
