"""
SPACECUE: Dissecting Metacognitive Confidence vs. Ballistic Motor Control
and Comparing Choice Confidence Across Target, Control, and Distractor Outcomes.

This script investigates two critical methodological and theoretical questions
regarding Panel A of `SPACECUE/plot_mean_cursor_trajectories.py`, where older
adults exhibit straighter, less variable cursor trajectories and stronger
directional commitment despite reduced objective accuracy:

1. Question 1 (Motor Bias vs. Metacognitive Confidence):
   When cursor trajectories are decomposed into **Initiation Time** (stimulus
   onset to movement onset) and **Movement Time** (movement onset to response
   click), do older adults delay movement onset ("wait-and-move" ballistic
   strategy)? We also test continuous radial velocity profiles and covariate-
   adjusted regressions / Linear Mixed-Effects Models (LMMs) to confirm that
   older adults' elevated trajectory commitment holds after controlling for
   trial-by-trial motor timing.

2. Question 2 & Outcome Comparison (Target vs. Control vs. Distractor):
   Does trajectory confidence/straightness differ between Target, non-salient
   Control, and salient pitch-deviant Distractor selections? Specifically, on
   error trials, do older adults show equal straightness/confidence when
   clicking the salient Distractor vs. the non-salient Control, or does the
   non-salient Control drive target-like "false hearing" while the salient
   Distractor triggers decisional conflict and trajectory curvature?

Panels:
    A - Motor timing decomposition: Initiation Time (s) and Movement Time (s)
        across cohorts and outcomes (Target / Control / Distractor)
    B - Continuous radial velocity profiles (dva/s) & chosen-axis commitment
        waveforms (%) over the response interval (Younger vs. Older adults,
        with cluster-based permutation tests)
    C - Chosen-option towardness (%) per outcome (Target / Control / Distractor):
        within-cohort pairwise comparisons (Panel C1) and between-cohort contrasts (Panel C2)
    D - Active perpendicular deviation (|Perp. Deviation|, dva) from the chosen
        button path per outcome: within-cohort pairwise comparisons (Panel D1)
        and between-cohort contrasts (Panel D2)
    E - Time-resolved perpendicular deviation (dva) from the chosen axis across
        Target, Control, and Distractor selections for Younger and Older adults
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mne.stats import permutation_cluster_test
import pingouin as pg
import statsmodels.formula.api as smf

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
    COLOR_CONTROL, COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD,
    COLOR_DISTRACTOR, COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD,
)

# ---------------------------------------------------------------------
# 0. Plotting & Analysis Configuration
# ---------------------------------------------------------------------
sns.set_theme(context="talk", style="ticks")
plt.ion()
plt.close('all')

NORM_STEPS = 101
COORD_SCALE = 1.0
STIM_DURATION_S = 0.25
RESP_WINDOW_S = 1.75
MOVEMENT_ONSET_THRESH_DVA = 0.05  # 5% of the 1.0 dva center-to-button distance

COHORTS = ['Younger adults', 'Older adults']
OUTCOMES = ['Target', 'Control', 'Distractor']
PAIRS = [('Target', 'Control'), ('Control', 'Distractor'), ('Target', 'Distractor')]
X_OF = {'Target': 0, 'Control': 1, 'Distractor': 2}

# Canonical button unit vectors in scaled arena coordinates (radius = 1.0 dva)
CANONICAL_LOCS = {
    'Target': np.array([0.0, 1.0]),
    'Distractor': np.array([-np.sqrt(3.0) / 2.0, -0.5]),
    'Control': np.array([np.sqrt(3.0) / 2.0, -0.5]),
}

OUTCOME_COLS = {
    'Target': (COLOR_TARGET_YOUNG, COLOR_TARGET_OLD),
    'Control': (COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD),
    'Distractor': (COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD),
}

COHORT_PALETTE = {
    'Younger adults': COLOR_TARGET_YOUNG,
    'Older adults': COLOR_TARGET_OLD,
}


# ---------------------------------------------------------------------
# 1. Statistical Helper Functions (Strict AGENTS.md Compliance)
# ---------------------------------------------------------------------
def stars_from_p(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def paired_comparison(vals_a, vals_b, label_a, label_b):
    """Paired comparison A vs B on subject-level averages.

    Checks normality of paired differences via Shapiro-Wilk (pg.normality).
    Uses paired t-test if normal (p >= 0.05), otherwise Wilcoxon signed-rank.
    Computes Cohen's dz = mean(diff) / SD(diff) on subject-level differences.
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
    vals_a, vals_b = vals_a[valid], vals_b[valid]
    diff = vals_a - vals_b
    n = diff.size
    dz = float(diff.mean() / diff.std(ddof=1))
    norm_df = pg.normality(diff)
    sh_w = float(norm_df['W'].iloc[0])
    sh_p = float(norm_df['pval'].iloc[0])
    if sh_p >= 0.05:
        t_df = pg.ttest(vals_a, vals_b, paired=True)
        res = {
            'test': 't',
            'stat': float(t_df['T'].iloc[0]),
            'df': float(t_df['dof'].iloc[0]),
            'p': float(t_df['p-val'].iloc[0]),
        }
    else:
        w_df = pg.wilcoxon(vals_a, vals_b)
        res = {
            'test': 'W',
            'stat': float(w_df['W-val'].iloc[0]),
            'df': float(n - 1),
            'p': float(w_df['p-val'].iloc[0]),
        }
    res.update({
        'label_a': label_a, 'label_b': label_b, 'n': int(n),
        'mean_a': float(vals_a.mean()), 'sd_a': float(vals_a.std(ddof=1)),
        'mean_b': float(vals_b.mean()), 'sd_b': float(vals_b.std(ddof=1)),
        'mean_diff': float(diff.mean()), 'sd_diff': float(diff.std(ddof=1)),
        'shapiro_W': sh_w, 'shapiro_p': sh_p, 'dz': dz,
    })
    return res


def independent_comparison(vals_a, vals_b, label_a, label_b):
    """Independent-samples comparison A vs B on subject-level averages.

    Checks Shapiro-Wilk normality in both groups and Levene's test for
    homoscedasticity. Runs Welch/Student t-test if both groups are normal,
    otherwise Mann-Whitney U test. Computes pooled Cohen's d.
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    vals_a = vals_a[~np.isnan(vals_a)]
    vals_b = vals_b[~np.isnan(vals_b)]
    na, nb = vals_a.size, vals_b.size
    sp = np.sqrt(((na - 1) * vals_a.var(ddof=1) + (nb - 1) * vals_b.var(ddof=1))
                 / (na + nb - 2))
    d_val = float((vals_a.mean() - vals_b.mean()) / sp)
    sh_a_df = pg.normality(vals_a)
    sh_b_df = pg.normality(vals_b)
    sh_w_a, sh_p_a = float(sh_a_df['W'].iloc[0]), float(sh_a_df['pval'].iloc[0])
    sh_w_b, sh_p_b = float(sh_b_df['W'].iloc[0]), float(sh_b_df['pval'].iloc[0])
    lev_stat, lev_p = stats.levene(vals_a, vals_b)
    if sh_p_a >= 0.05 and sh_p_b >= 0.05:
        t_df = pg.ttest(vals_a, vals_b, correction='auto')
        res = {
            'test': 't',
            'stat': float(t_df['T'].iloc[0]),
            'df': float(t_df['dof'].iloc[0]),
            'p': float(t_df['p-val'].iloc[0]),
        }
    else:
        mw = pg.mwu(vals_a, vals_b)
        res = {
            'test': 'U',
            'stat': float(mw['U-val'].iloc[0]),
            'df': float(na + nb - 2),
            'p': float(mw['p-val'].iloc[0]),
        }
    res.update({
        'label_a': label_a, 'label_b': label_b, 'n_a': int(na), 'n_b': int(nb),
        'mean_a': float(vals_a.mean()), 'sd_a': float(vals_a.std(ddof=1)),
        'mean_b': float(vals_b.mean()), 'sd_b': float(vals_b.std(ddof=1)),
        'd': d_val,
        'shapiro_W_a': sh_w_a, 'shapiro_p_a': sh_p_a,
        'shapiro_W_b': sh_w_b, 'shapiro_p_b': sh_p_b,
        'levene_W': float(lev_stat), 'levene_p': float(lev_p),
    })
    return res


def run_2group_cluster_test(X_young, X_old, n_permutations=5000, seed=42):
    """Two-sample cluster-based permutation test (Younger vs. Older).

    Follows AGENTS.md critical methodological rule: Cohen's d for each
    significant cluster is computed by collapsing (averaging) the raw
    continuous waveform across the cluster's temporal bounds for each
    subject first, and then computing pooled Cohen's d on the 1D subject
    averages.
    """
    t_obs, clusters, cluster_pv, _ = permutation_cluster_test(
        [X_young, X_old], n_permutations=n_permutations, seed=seed, verbose=False
    )
    sig_clusters = []
    for cl, p in zip(clusters, cluster_pv):
        if p < 0.05:
            arr = np.asarray(cl[0] if isinstance(cl, tuple) else cl)
            if arr.ndim > 1:
                steps = np.where(arr.any(axis=0))[0]
            elif arr.dtype == bool:
                steps = np.where(arr)[0]
            else:
                steps = arr.astype(int)
            sub_y = X_young[:, steps].mean(axis=1)
            sub_o = X_old[:, steps].mean(axis=1)
            na, nb = sub_y.size, sub_o.size
            sp = np.sqrt(((na - 1) * sub_y.var(ddof=1) + (nb - 1) * sub_o.var(ddof=1))
                         / (na + nb - 2))
            d_cluster = float((sub_y.mean() - sub_o.mean()) / sp)
            t_cl = pg.ttest(sub_y, sub_o, correction='auto')
            sig_clusters.append({
                'start_step': int(steps[0]),
                'end_step': int(steps[-1]),
                'p': float(p),
                'd': d_cluster,
                't_collapsed': float(t_cl['T'].iloc[0]),
                'df_collapsed': float(t_cl['dof'].iloc[0]),
                'mean_young': float(sub_y.mean()),
                'sd_young': float(sub_y.std(ddof=1)),
                'mean_old': float(sub_o.mean()),
                'sd_old': float(sub_o.std(ddof=1)),
            })
    sig_clusters.sort(key=lambda c: c['p'])
    return sig_clusters


def draw_sig_bracket(ax, x1, x2, y, h, text, color='#212121'):
    """Draw a clean significance bracket between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.2, zorder=7, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h * 1.25, text, ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color=color, zorder=8, linespacing=1.1, clip_on=False)


def add_panel_letter(ax, letter):
    """Place a bold panel letter left of the subplot title."""
    if letter:
        ax.annotate(letter, xy=(0, 1.0), xycoords='axes fraction',
                    xytext=(-8, 12), textcoords='offset points',
                    ha='right', va='bottom', fontsize=19, fontweight='bold',
                    color='#000000', zorder=12)


# ---------------------------------------------------------------------
# 2. Load & Synchronize Trajectory and Behavioral Data
# ---------------------------------------------------------------------
# Mirrors the exact trial-cleaning and inner-join logic of
# SPACECUE/plot_mean_cursor_trajectories.py so that the analyzed trial
# set is 100% identical to Panel A.
print("[INFO] Loading resampled mouse trajectories and behavioral table...")
data_path = SPACECUE.get_data_path()

archive = np.load(os.path.join(data_path, 'concatenated', 'aligned_mouse_resampled_101.npz'))
df_traj = pd.DataFrame(archive['data'], columns=archive['columns'].tolist())
traj_keep = ['subject_id', 'block', 'trial_nr', 'step', 'time_norm', 'x_aligned', 'y_aligned']
df_traj = df_traj[[c for c in traj_keep if c in df_traj.columns]]

df_beh = pd.read_csv(os.path.join(data_path, 'concatenated', 'concatenated_behavioral.csv'))
beh_cols = [
    'subject_id', 'block', 'trial_nr', 'rt', 'SelectDistractor', 'SelectControl',
    'SelectOther', 'IsCorrect', 'target_towardness', 'distractor_towardness',
    'control_towardness', 'Age'
]
df_beh = df_beh[beh_cols].drop_duplicates(['subject_id', 'block', 'trial_nr'])

# Inner join restricts trajectories to the outlier-cleaned behavioral trial set
df_traj = pd.merge(df_traj, df_beh, on=['subject_id', 'block', 'trial_nr'], how='inner')

df_traj['outcome'] = 'Miss / No Click'
df_traj.loc[df_traj['IsCorrect'] == 1, 'outcome'] = 'Target'
df_traj.loc[df_traj['SelectControl'] == 1, 'outcome'] = 'Control'
df_traj.loc[df_traj['SelectDistractor'] == 1, 'outcome'] = 'Distractor'

# Filter for valid response selections (Target, Control, Distractor)
df_traj = df_traj[df_traj['outcome'].isin(OUTCOMES)].copy()
df_traj['age_cohort'] = np.where(df_traj['Age'] <= 35, 'Younger adults', 'Older adults')

# Scale aligned coordinates into canonical unit-radius arena (r = 1.0 dva)
df_traj['x_scaled'] = df_traj['x_aligned'] * COORD_SCALE
df_traj['y_scaled'] = df_traj['y_aligned'] * COORD_SCALE

# Ensure strict sorting by trial and normalized step (0..100)
df_traj = df_traj.sort_values(['subject_id', 'block', 'trial_nr', 'step']).reset_index(drop=True)
n_trials = len(df_traj) // NORM_STEPS
assert len(df_traj) == n_trials * NORM_STEPS, "Incomplete 101-step trajectory detected."

# Reshape continuous trajectories to (n_trials, 101)
X_mat = df_traj['x_scaled'].values.reshape(n_trials, NORM_STEPS)
Y_mat = df_traj['y_scaled'].values.reshape(n_trials, NORM_STEPS)
trials_df = df_traj.iloc[::NORM_STEPS][
    ['subject_id', 'block', 'trial_nr', 'age_cohort', 'outcome', 'rt', 'Age']
].reset_index(drop=True)

print(f"[INFO] Loaded {n_trials:,} valid response trials across "
      f"{trials_df['subject_id'].nunique()} subjects.")


# ---------------------------------------------------------------------
# 3. Single-Trial Motor Timing & Confidence Metric Extraction
# ---------------------------------------------------------------------
# Temporal alignment documentation:
#   - Stimulus presentation runs from t = 0.00 s to t = 0.25 s.
#   - Response interval (Phase 1 in raw mouse logs) runs for 1.75 s from
#     t = 0.25 s to t = 2.00 s post-stimulus onset.
#   - Each step in `aligned_mouse_resampled_101.npz` (time_norm = 0.00..1.00)
#     corresponds to dt = 1.75 / 100 = 0.0175 s (17.5 ms) of physical time.
#   - Behavioral `rt` is logged in seconds relative to stimulus onset.
dt_step_s = RESP_WINDOW_S / (NORM_STEPS - 1)  # 0.0175 s per step
phys_time_stim_s = STIM_DURATION_S + np.linspace(0.0, RESP_WINDOW_S, NORM_STEPS)
time_pct = np.linspace(0.0, 100.0, NORM_STEPS)

# Radial displacement from the trial's initial position (step 0)
dx0 = X_mat - X_mat[:, [0]]
dy0 = Y_mat - Y_mat[:, [0]]
dist_from_start = np.hypot(dx0, dy0)

# Radial velocity (dva/s) along the physical response clock
vx_mat = np.gradient(X_mat, dt_step_s, axis=1)
vy_mat = np.gradient(Y_mat, dt_step_s, axis=1)
speed_mat = np.hypot(vx_mat, vy_mat)
radial_vel_mat = np.gradient(dist_from_start, dt_step_s, axis=1)

# 1. Movement Initiation Time (t_init): stimulus onset to first crossing of
#    MOVEMENT_ONSET_THRESH_DVA (0.05 dva = 5% of button distance)
moved_mask = dist_from_start >= MOVEMENT_ONSET_THRESH_DVA
has_moved = moved_mask.any(axis=1)
init_step_float = np.where(has_moved, np.argmax(moved_mask, axis=1), np.nan)
init_step_int = np.where(has_moved, np.argmax(moved_mask, axis=1), 0).astype(int)

trials_df['t_init_s'] = STIM_DURATION_S + init_step_float * dt_step_s
trials_df['t_init_resp_s'] = init_step_float * dt_step_s

# 2. Movement Execution Duration (t_move): movement onset to click (rt - t_init)
trials_df['t_move_s'] = trials_df['rt'] - trials_df['t_init_s']

# Active transit duration (t_exec_active): movement onset to 90% of peak displacement
max_disp = dist_from_start.max(axis=1, keepdims=True)
arrive_mask = (dist_from_start >= 0.90 * max_disp) & (max_disp >= MOVEMENT_ONSET_THRESH_DVA)
arrive_step_float = np.where(arrive_mask.any(axis=1), np.argmax(arrive_mask, axis=1), np.nan)
trials_df['t_arrive_s'] = STIM_DURATION_S + arrive_step_float * dt_step_s
trials_df['t_exec_active_s'] = trials_df['t_arrive_s'] - trials_df['t_init_s']

# 3. Rotate each trial onto the CHOSEN button's canonical axis:
#    u_chosen = unit vector pointing from (0, 0) to the clicked button
#    - Parallel projection (proj_par): progress toward the chosen button (% of 1.0 dva)
#    - Perpendicular deviation (proj_perp): orthogonal distance from ideal straight path (dva)
u_x = trials_df['outcome'].map(lambda o: CANONICAL_LOCS[o][0]).values[:, None]
u_y = trials_df['outcome'].map(lambda o: CANONICAL_LOCS[o][1]).values[:, None]

proj_par_pct = (X_mat * u_x + Y_mat * u_y) * 100.0  # % of target radius (1.0 dva)
proj_perp_dva = X_mat * u_y - Y_mat * u_x           # signed perpendicular deviation (dva)
abs_perp_dva = np.abs(proj_perp_dva)                # unsigned perpendicular deviation (dva)

# Pre-click mask (samples up to the response click `t_cut = rt - 0.25 s`)
click_step_int = np.clip(
    np.floor((trials_df['rt'].values - STIM_DURATION_S) / dt_step_s).astype(int),
    1, NORM_STEPS - 1
)
step_grid = np.arange(NORM_STEPS)[None, :]
pre_click_mask = step_grid <= click_step_int[:, None]
active_move_mask = (step_grid >= init_step_int[:, None]) & pre_click_mask

# Confidence Metric 1: Chosen-Option Towardness (%, pre-click average projection)
trials_df['chosen_towardness_pct'] = np.nanmean(
    np.where(pre_click_mask, proj_par_pct, np.nan), axis=1
)
trials_df['chosen_towardness_full_pct'] = proj_par_pct.mean(axis=1)

# Confidence Metric 2: Active Perpendicular Deviation (dva, from initiation to click)
trials_df['active_perp_dev_dva'] = np.nanmean(
    np.where(active_move_mask, abs_perp_dva, np.nan), axis=1
)
trials_df['full_perp_dev_dva'] = abs_perp_dva.mean(axis=1)

# Confidence Metric 3: Path Straightness Index (net displacement / total path length)
seg_lengths = np.hypot(np.diff(X_mat, axis=1), np.diff(Y_mat, axis=1))
total_path_len = seg_lengths.sum(axis=1)
net_disp = np.hypot(X_mat[:, -1] - X_mat[:, 0], Y_mat[:, -1] - Y_mat[:, 0])
trials_df['straightness_index'] = np.divide(
    net_disp, total_path_len,
    out=np.full_like(net_disp, np.nan, dtype=float),
    where=total_path_len >= MOVEMENT_ONSET_THRESH_DVA
)

# Attach step-wise chosen-axis projections back to df_traj for time-resolved analyses
df_traj['proj_par_pct'] = proj_par_pct.ravel()
df_traj['proj_perp_dva'] = proj_perp_dva.ravel()
df_traj['abs_perp_dva'] = abs_perp_dva.ravel()
df_traj['radial_vel_dva_s'] = radial_vel_mat.ravel()


# ---------------------------------------------------------------------
# 4. Subject-Level Aggregation & Comprehensive Statistical Inference
# ---------------------------------------------------------------------
report_lines = []


def log_line(msg=""):
    print(msg)
    report_lines.append(msg)


log_line("=" * 88)
log_line("SPACECUE STATISTICAL REPORT: CONFIDENCE VS. BALLISTIC MOTOR CONTROL")
log_line("=" * 88)

# 4A. Overall Subject-Level Motor Timing & Confidence (Collapsed Across Outcomes)
METRICS_OVERALL = [
    ('t_init_s', 'Initiation Time from Stimulus Onset (s)'),
    ('t_init_resp_s', 'Initiation Time from Response Window Onset (s)'),
    ('t_move_s', 'Movement Duration: Onset to Click (s)'),
    ('t_exec_active_s', 'Active Transit Time: Onset to 90% Displacement (s)'),
    ('rt', 'Total Reaction Time from Stimulus Onset (s)'),
    ('chosen_towardness_pct', 'Pre-Click Chosen-Option Towardness (%)'),
    ('active_perp_dev_dva', 'Active Perpendicular Deviation |d_perp| (dva)'),
    ('straightness_index', 'Path Straightness Index (Net / Total Path)'),
]

sub_overall = trials_df.groupby(['age_cohort', 'subject_id'])[
    [m[0] for m in METRICS_OVERALL]
].mean()

log_line("\n--- SECTION 1: OVERALL COHORT COMPARISONS (YOUNGER VS. OLDER ADULTS, N = 18 / GROUP) ---")
overall_cohort_stats = {}
for col_key, col_label in METRICS_OVERALL:
    vy = sub_overall.loc['Younger adults', col_key].values
    vo = sub_overall.loc['Older adults', col_key].values
    res = independent_comparison(vy, vo, 'Younger adults', 'Older adults')
    overall_cohort_stats[col_key] = res
    stat_str = f"t({res['df']:.1f}) = {res['stat']:.4f}" if res['test'] == 't' else f"U = {res['stat']:.1f}"
    log_line(
        f"\n{col_label}:\n"
        f"  Younger adults (n={res['n_a']}): Mean = {res['mean_a']:.4f}, SD = {res['sd_a']:.4f}\n"
        f"  Older adults   (n={res['n_b']}): Mean = {res['mean_b']:.4f}, SD = {res['sd_b']:.4f}\n"
        f"  Normality (Shapiro-Wilk): Young W = {res['shapiro_W_a']:.4f} (p = {res['shapiro_p_a']:.4g}), "
        f"Old W = {res['shapiro_W_b']:.4f} (p = {res['shapiro_p_b']:.4g})\n"
        f"  Homoscedasticity (Levene): W = {res['levene_W']:.4f}, p = {res['levene_p']:.4g}\n"
        f"  Contrast: {stat_str}, p = {res['p']:.6g} ({stars_from_p(res['p'])}), Cohen's d = {res['d']:.3f}"
    )

# Sensitivity check across movement onset thresholds (0.02, 0.05, 0.10 dva)
log_line("\n--- SECTION 1B: SENSITIVITY CHECK ACROSS MOVEMENT INITIATION THRESHOLDS ---")
for thresh_check in [0.02, 0.05, 0.10]:
    m_mask = dist_from_start >= thresh_check
    h_mov = m_mask.any(axis=1)
    i_step = np.where(h_mov, np.argmax(m_mask, axis=1), np.nan)
    trials_df[f'_t_init_{thresh_check}'] = STIM_DURATION_S + i_step * dt_step_s
    sub_th = trials_df.groupby(['age_cohort', 'subject_id'])[f'_t_init_{thresh_check}'].mean()
    r_th = independent_comparison(
        sub_th.loc['Younger adults'].values, sub_th.loc['Older adults'].values,
        'Younger adults', 'Older adults'
    )
    st_s = f"t({r_th['df']:.1f}) = {r_th['stat']:.3f}" if r_th['test'] == 't' else f"U = {r_th['stat']:.1f}"
    log_line(
        f"  Threshold = {thresh_check:.2f} dva: Young = {r_th['mean_a']:.4f} ± {r_th['sd_a']:.4f} s, "
        f"Old = {r_th['mean_b']:.4f} ± {r_th['sd_b']:.4f} s | {st_s}, p = {r_th['p']:.4g}, d = {r_th['d']:.2f}"
    )

# 4B. Subject-Level Breakdown by Outcome (Target, Control, Distractor)
KEY_OUTCOME_METRICS = [
    ('t_init_s', 'Initiation Time (s)'),
    ('t_move_s', 'Movement Duration (s)'),
    ('rt', 'Reaction Time (s)'),
    ('chosen_towardness_pct', 'Chosen Towardness (%)'),
    ('active_perp_dev_dva', 'Active |Perp. Deviation| (dva)'),
]

sub_by_outcome = trials_df.groupby(['age_cohort', 'outcome', 'subject_id'])[
    [m[0] for m in KEY_OUTCOME_METRICS]
].mean()

# Also compute Panel A Subject-Mean Trajectory Curvature (|perp_dev| of the subject's mean trace)
# and Within-Subject Trial-to-Trial SD of Perpendicular Deviation around the chosen path
sub_mean_traces = df_traj.groupby(['age_cohort', 'outcome', 'subject_id', 'step'])[
    ['proj_par_pct', 'proj_perp_dva', 'abs_perp_dva']
].mean().reset_index()
sub_mean_traces['mean_trace_abs_perp'] = sub_mean_traces['proj_perp_dva'].abs()
sub_trace_curv = sub_mean_traces.groupby(['age_cohort', 'outcome', 'subject_id'])[
    'mean_trace_abs_perp'
].mean()
sub_by_outcome['panel_a_trace_curv_dva'] = sub_trace_curv

sub_sd_perp = df_traj.groupby(['age_cohort', 'outcome', 'subject_id', 'step'])[
    'proj_perp_dva'
].std().groupby(['age_cohort', 'outcome', 'subject_id']).mean()
sub_by_outcome['trial_sd_perp_dva'] = sub_sd_perp

ALL_OUTCOME_METRICS = KEY_OUTCOME_METRICS + [
    ('panel_a_trace_curv_dva', 'Panel A Mean-Trajectory Curvature |d_perp| (dva)'),
    ('trial_sd_perp_dva', 'Within-Subject Trial-to-Trial SD of Perp. Deviation (dva)'),
]

log_line("\n--- SECTION 2: MIXED ANOVAs (AGE COHORT x OUTCOME: TARGET / CONTROL / DISTRACTOR) ---")
mixed_anova_tables = {}
for col_key, col_label in ALL_OUTCOME_METRICS:
    df_aov = sub_by_outcome[col_key].reset_index().dropna()
    aov = pg.mixed_anova(
        data=df_aov, dv=col_key, between='age_cohort', within='outcome', subject='subject_id'
    )
    spher = pg.sphericity(data=df_aov, dv=col_key, within='outcome', subject='subject_id')
    mixed_anova_tables[col_key] = aov
    log_line(f"\n[{col_label}] Mixed ANOVA (Mauchly's W = {spher.W:.4f}, p = {spher.pval:.4g}):")
    for _, row in aov.iterrows():
        p_use = row['p-GG-corr'] if ('p-GG-corr' in row and pd.notna(row['p-GG-corr']) and spher.pval < 0.05) else row['p-unc']
        log_line(
            f"  {row['Source']:12s}: F({int(row['DF1'])}, {int(row['DF2'])}) = {row['F']:.4f}, "
            f"p = {p_use:.6g}, np2 = {row['np2']:.4f}"
        )

# 4C. Within-Cohort Pairwise Outcome Comparisons (Target vs Control vs Distractor)
#     and Between-Cohort Contrasts per Outcome
within_cohort_stats = {m[0]: {} for m in ALL_OUTCOME_METRICS}
between_cohort_stats = {m[0]: {} for m in ALL_OUTCOME_METRICS}

log_line("\n--- SECTION 3: WITHIN-COHORT OUTCOME COMPARISONS (TARGET VS. CONTROL VS. DISTRACTOR) ---")
for col_key, col_label in ALL_OUTCOME_METRICS:
    log_line(f"\n=== {col_label} ===")
    tbl = sub_by_outcome[col_key].unstack('outcome')
    for cohort in COHORTS:
        c_df = tbl.loc[cohort].dropna()
        rm = pg.rm_anova(c_df, effsize='np2')
        log_line(
            f"  [{cohort}] One-Way RM-ANOVA across Outcomes: "
            f"F({int(rm['ddof1'].iloc[0])}, {int(rm['ddof2'].iloc[0])}) = {rm['F'].iloc[0]:.4f}, "
            f"p = {rm['p-unc'].iloc[0]:.6g}, np2 = {rm['np2'].iloc[0]:.4f}"
        )
        res_list = [paired_comparison(c_df[a].values, c_df[b].values, a, b) for a, b in PAIRS]
        _, p_holm_arr = pg.multicomp([r['p'] for r in res_list], alpha=0.05, method='holm')
        for r, ph in zip(res_list, p_holm_arr):
            r['p_holm'] = float(ph)
            st_s = f"t({int(r['df'])}) = {r['stat']:.3f}" if r['test'] == 't' else f"W = {r['stat']:.1f}"
            log_line(
                f"    {r['label_a']:10s} ({r['mean_a']:.4f} ± {r['sd_a']:.4f}) vs. "
                f"{r['label_b']:10s} ({r['mean_b']:.4f} ± {r['sd_b']:.4f}): "
                f"{st_s}, p = {r['p']:.5g}, p_holm = {r['p_holm']:.5g} ({stars_from_p(r['p_holm'])}), "
                f"dz = {r['dz']:.3f}, Shapiro p = {r['shapiro_p']:.4g}"
            )
        within_cohort_stats[col_key][cohort] = res_list

log_line("\n--- SECTION 4: BETWEEN-COHORT COMPARISONS PER OUTCOME (YOUNGER VS. OLDER ADULTS) ---")
for col_key, col_label in ALL_OUTCOME_METRICS:
    log_line(f"\n=== {col_label} ===")
    tbl = sub_by_outcome[col_key].unstack('outcome')
    res_out = {}
    for outcome in OUTCOMES:
        vy = tbl.loc['Younger adults', outcome].dropna().values
        vo = tbl.loc['Older adults', outcome].dropna().values
        res_out[outcome] = independent_comparison(vy, vo, 'Younger adults', 'Older adults')
    _, p_holm_out = pg.multicomp([res_out[o]['p'] for o in OUTCOMES], alpha=0.05, method='holm')
    for outcome, ph in zip(OUTCOMES, p_holm_out):
        r = res_out[outcome]
        r['p_holm'] = float(ph)
        st_s = f"t({r['df']:.1f}) = {r['stat']:.3f}" if r['test'] == 't' else f"U = {r['stat']:.1f}"
        log_line(
            f"  [{outcome:10s}] Young = {r['mean_a']:.4f} ± {r['sd_a']:.4f} vs. "
            f"Old = {r['mean_b']:.4f} ± {r['sd_b']:.4f}: "
            f"{st_s}, p = {r['p']:.5g}, p_holm = {r['p_holm']:.5g} ({stars_from_p(r['p_holm'])}), "
            f"d = {r['d']:.3f}, Levene p = {r['levene_p']:.3g}"
        )
    between_cohort_stats[col_key] = res_out


# ---------------------------------------------------------------------
# 5. Controlling for Motor Timing: ANCOVA & Single-Trial LMMs
# ---------------------------------------------------------------------
log_line("\n--- SECTION 5: CONFIRMING CONFIDENCE EFFECT AFTER CONTROLLING FOR MOTOR TIMING ---")
sub_reg_df = sub_overall.reset_index().copy()
sub_reg_df['is_older'] = (sub_reg_df['age_cohort'] == 'Older adults').astype(int)

ols_tow = smf.ols(
    'chosen_towardness_pct ~ is_older + t_init_s + t_move_s', data=sub_reg_df
).fit()
log_line("\n1) Subject-Level Multiple Regression: Chosen Towardness (%) ~ Age_Cohort + t_init + t_move")
log_line(f"   Model R^2 = {ols_tow.rsquared:.4f}, F({int(ols_tow.df_model)}, {int(ols_tow.df_resid)}) = {ols_tow.fvalue:.3f}, p = {ols_tow.f_pvalue:.6g}")
for term in ['Intercept', 'is_older', 't_init_s', 't_move_s']:
    log_line(
        f"   {term:12s}: beta = {ols_tow.params[term]:+8.4f}, SE = {ols_tow.bse[term]:.4f}, "
        f"t({int(ols_tow.df_resid)}) = {ols_tow.tvalues[term]:+7.3f}, p = {ols_tow.pvalues[term]:.6g}"
    )

# Single-trial Linear Mixed-Effects Model (LMM) with random subject intercepts
lmm_df = trials_df.dropna(subset=['chosen_towardness_pct', 'active_perp_dev_dva', 't_init_s', 't_move_s']).copy()
lmm_df['is_older'] = (lmm_df['age_cohort'] == 'Older adults').astype(int)

lmm_tow = smf.mixedlm(
    "chosen_towardness_pct ~ is_older + C(outcome, Treatment(reference='Target')) + t_init_s + t_move_s",
    data=lmm_df, groups=lmm_df['subject_id']
).fit(reml=True)

log_line(f"\n2) Single-Trial Linear Mixed-Effects Model (N = {len(lmm_df):,} trials, 36 subjects):")
log_line("   Formula: chosen_towardness_pct ~ is_older + C(outcome, ref='Target') + t_init_s + t_move_s + (1 | subject_id)")
for param_name in lmm_tow.fe_params.index:
    b = lmm_tow.fe_params[param_name]
    se = lmm_tow.bse_fe[param_name]
    z = lmm_tow.tvalues[param_name]
    p = lmm_tow.pvalues[param_name]
    log_line(f"   {param_name:52s}: beta = {b:+8.4f}, SE = {se:.4f}, z = {z:+7.3f}, p = {p:.6g}")


# ---------------------------------------------------------------------
# 6. Time-Resolved Cluster-Based Permutation Tests
# ---------------------------------------------------------------------
log_line("\n--- SECTION 6: TIME-RESOLVED CLUSTER PERMUTATION TESTS (5,000 PERMUTATIONS) ---")

# 6A. Radial Velocity Profile (Younger vs. Older, collapsed across valid trials per subject)
sub_vel_curves = df_traj.groupby(['age_cohort', 'subject_id', 'step'])[
    'radial_vel_dva_s'
].mean().unstack('step')
X_vel_young = sub_vel_curves.loc['Younger adults'].values
X_vel_old = sub_vel_curves.loc['Older adults'].values
vel_clusters = run_2group_cluster_test(X_vel_young, X_vel_old)

if vel_clusters:
    for cl in vel_clusters:
        t0, t1 = phys_time_stim_s[cl['start_step']], phys_time_stim_s[cl['end_step']]
        log_line(
            f"  [Radial Velocity: Young vs. Old] Cluster {cl['start_step']}–{cl['end_step']}% "
            f"({t0:.2f}–{t1:.2f} s post-stim): t({cl['df_collapsed']:.1f}) = {cl['t_collapsed']:.3f}, "
            f"p_cluster = {cl['p']:.4g}, Cohen's d = {cl['d']:.2f}"
        )
else:
    log_line("  [Radial Velocity: Young vs. Old] No significant cluster (p >= 0.05).")

# 6B. Chosen-Option Towardness Waveform (Younger vs. Older, collapsed across outcomes)
sub_par_curves = df_traj.groupby(['age_cohort', 'subject_id', 'step'])[
    'proj_par_pct'
].mean().unstack('step')
X_par_young = sub_par_curves.loc['Younger adults'].values
X_par_old = sub_par_curves.loc['Older adults'].values
par_clusters = run_2group_cluster_test(X_par_young, X_par_old)

for cl in par_clusters:
    t0, t1 = phys_time_stim_s[cl['start_step']], phys_time_stim_s[cl['end_step']]
    log_line(
        f"  [Chosen Towardness Waveform: Young vs. Old] Cluster {cl['start_step']}–{cl['end_step']}% "
        f"({t0:.2f}–{t1:.2f} s post-stim): t({cl['df_collapsed']:.1f}) = {cl['t_collapsed']:.3f}, "
        f"p_cluster = {cl['p']:.4g}, Cohen's d = {cl['d']:.2f}"
    )

# 6C. Time-Resolved Unsigned Perpendicular Deviation |d_perp(t)| per Outcome (Younger vs. Older)
perp_curves_by_outcome = {}
perp_clusters_by_outcome = {}
for outcome in OUTCOMES:
    d_out = df_traj[df_traj['outcome'] == outcome]
    c_sub = d_out.groupby(['age_cohort', 'subject_id', 'step'])['abs_perp_dva'].mean().unstack('step')
    Xy = c_sub.loc['Younger adults'].dropna().values
    Xo = c_sub.loc['Older adults'].dropna().values
    perp_curves_by_outcome[outcome] = {'Younger adults': Xy, 'Older adults': Xo}
    cls_out = run_2group_cluster_test(Xy, Xo)
    perp_clusters_by_outcome[outcome] = cls_out
    if cls_out:
        for cl in cls_out:
            log_line(
                f"  [|Perp. Deviation| {outcome}: Young vs. Old] Cluster {cl['start_step']}–{cl['end_step']}%: "
                f"t({cl['df_collapsed']:.1f}) = {cl['t_collapsed']:.3f}, p_cluster = {cl['p']:.4g}, Cohen's d = {cl['d']:.2f}"
            )
    else:
        log_line(f"  [|Perp. Deviation| {outcome}: Young vs. Old] No significant cluster (p >= 0.05).")


# ---------------------------------------------------------------------
# 7. Publication Figure Rendering (3 Rows x 2 Columns, matching SPACECUE style)
# ---------------------------------------------------------------------
print("\n[INFO] Rendering publication-grade multi-panel figure...")
fig, axes = plt.subplots(3, 2, figsize=(16, 16), gridspec_kw={'hspace': 0.44, 'wspace': 0.28})
np.random.seed(42)

# =====================================================================
# PANEL A: Motor Timing Decomposition (Initiation Time vs. Movement Time)
# =====================================================================
gs_a = axes[0, 0].get_subplotspec().subgridspec(1, 2, wspace=0.32)
axes[0, 0].remove()
ax_a1 = fig.add_subplot(gs_a[0])
ax_a2 = fig.add_subplot(gs_a[1])

timing_panels = [
    (ax_a1, 't_init_s', 'Initiation time', 'Time (s)', 'A'),
    (ax_a2, 't_move_s', 'Movement duration', 'Time (s)', ''),
]

x_pos_out = np.arange(len(OUTCOMES))
dodge_w = 0.18

for ax_t, metric_key, title_str, ylabel_str, panel_lbl in timing_panels:
    ax_t.set_title(title_str, loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_t, panel_lbl)

    tbl_t = sub_by_outcome[metric_key].unstack('outcome')
    y_max_t = float(tbl_t.max().max())
    y_min_t = float(tbl_t.min().min())
    yr_t = y_max_t - y_min_t

    for gi, outcome in enumerate(OUTCOMES):
        for ci, cohort in enumerate(COHORTS):
            vals = tbl_t.loc[cohort, outcome].dropna().values
            col = OUTCOME_COLS[outcome][ci]
            xg = x_pos_out[gi] + (-dodge_w if ci == 0 else dodge_w)
            jit = np.random.uniform(-0.05, 0.05, size=vals.size)
            ax_t.scatter(
                xg + jit, vals, color=col, alpha=0.60, s=34, zorder=3,
                label=cohort if (gi == 0 and metric_key == 't_init_s') else None
            )
            ax_t.errorbar(
                xg, vals.mean(), yerr=stats.sem(vals), fmt='o', color='black',
                ecolor='black', elinewidth=2.0, capsize=4, capthick=1.6,
                markersize=6.5, zorder=6
            )

        # Cohort contrast bracket above each outcome
        r_coh = between_cohort_stats[metric_key][outcome]
        lvl_t = y_max_t + (0.08 + 0.14 * gi) * yr_t
        draw_sig_bracket(
            ax_t, x_pos_out[gi] - dodge_w, x_pos_out[gi] + dodge_w,
            lvl_t, 0.025 * yr_t,
            f"{stars_from_p(r_coh['p_holm'])} $d$ = {r_coh['d']:.2f}"
        )

    ax_t.set_xticks(x_pos_out)
    ax_t.set_xticklabels(OUTCOMES, fontsize=9.5)
    ax_t.set_xlim(-0.55, 2.55)
    ax_t.set_ylim(y_min_t - 0.12 * yr_t, y_max_t + 0.56 * yr_t)
    ax_t.set_ylabel(ylabel_str)
    sns.despine(ax=ax_t)

ax_a2.set_ylabel('')
ax_a1.legend(loc='upper left', frameon=True, fontsize=8.0, framealpha=0.92)


# =====================================================================
# PANEL B: Continuous Radial Velocity (dva/s) & Chosen Commitment (%)
# =====================================================================
gs_b = axes[0, 1].get_subplotspec().subgridspec(1, 2, wspace=0.38)
axes[0, 1].remove()
ax_b1 = fig.add_subplot(gs_b[0])
ax_b2 = fig.add_subplot(gs_b[1])

# B1: Radial Velocity over physical time from stimulus onset (s)
ax_b1.set_title('Radial velocity', loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
add_panel_letter(ax_b1, 'B')
ax_b1.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)

for cohort, col in [('Younger adults', COLOR_TARGET_YOUNG), ('Older adults', COLOR_TARGET_OLD)]:
    X_v = sub_vel_curves.loc[cohort].values
    m_v = X_v.mean(axis=0)
    s_v = stats.sem(X_v, axis=0)
    ax_b1.plot(phys_time_stim_s, m_v, color=col, lw=2.8, label=cohort, zorder=6)
    ax_b1.fill_between(phys_time_stim_s, m_v - s_v, m_v + s_v, color=col, alpha=0.28, zorder=5)

# Annotate mean movement initiation times (vertical dotted lines + non-overlapping callouts)
mean_init_y = overall_cohort_stats['t_init_s']['mean_a']
mean_init_o = overall_cohort_stats['t_init_s']['mean_b']
ax_b1.axvline(mean_init_y, color=COLOR_TARGET_YOUNG, linestyle=':', lw=1.8, zorder=4)
ax_b1.axvline(mean_init_o, color=COLOR_TARGET_OLD, linestyle=':', lw=1.8, zorder=4)
ax_b1.annotate(
    f"Old $t_{{init}}$\n{mean_init_o:.2f} s",
    xy=(mean_init_o, 0.35), xytext=(0.30, 0.58),
    arrowprops=dict(arrowstyle="->", color=COLOR_TARGET_OLD, lw=1.2),
    color=COLOR_TARGET_OLD, fontsize=8.0, fontweight='bold', ha='left', va='center', zorder=9
)
ax_b1.annotate(
    f"Young $t_{{init}}$\n{mean_init_y:.2f} s",
    xy=(mean_init_y, 0.22), xytext=(1.12, 0.28),
    arrowprops=dict(arrowstyle="->", color=COLOR_TARGET_YOUNG, lw=1.2),
    color=COLOR_TARGET_YOUNG, fontsize=8.0, fontweight='bold', ha='left', va='center', zorder=9
)

if vel_clusters:
    cl_v = vel_clusters[0]
    t0_v, t1_v = phys_time_stim_s[cl_v['start_step']], phys_time_stim_s[cl_v['end_step']]
    ax_b1.plot([t0_v, t1_v], [-0.18, -0.18], color='#212121', lw=2.8, zorder=8)
    ax_b1.text((t0_v + t1_v) / 2.0, -0.14, f"{stars_from_p(cl_v['p'])} $d$ = {cl_v['d']:.2f}",
               ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#212121')

ax_b1.set_xlabel('Time from stimulus (s)')
ax_b1.set_ylabel('Radial velocity (dva/s)')
ax_b1.set_xlim(0.25, 2.00)
ax_b1.set_ylim(-0.28, 1.68)
ax_b1.legend(loc='upper right', frameon=True, fontsize=8.0, framealpha=0.92)
sns.despine(ax=ax_b1)

# B2: Continuous Chosen-Option Commitment Waveform (%) over trajectory progression
ax_b2.set_title('Chosen-axis projection', loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
ax_b2.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)

for cohort, col in [('Younger adults', COLOR_TARGET_YOUNG), ('Older adults', COLOR_TARGET_OLD)]:
    X_p = sub_par_curves.loc[cohort].values
    m_p = X_p.mean(axis=0)
    s_p = stats.sem(X_p, axis=0)
    ax_b2.plot(time_pct, m_p, color=col, lw=2.8, label=cohort, zorder=6)
    ax_b2.fill_between(time_pct, m_p - s_p, m_p + s_p, color=col, alpha=0.28, zorder=5)

if par_clusters:
    cl_p = par_clusters[0]
    ax_b2.plot([cl_p['start_step'], cl_p['end_step']], [-12.0, -12.0], color='#212121', lw=2.8, zorder=8)
    ax_b2.text((cl_p['start_step'] + cl_p['end_step']) / 2.0, -9.5,
               f"{stars_from_p(cl_p['p'])} $d$ = {cl_p['d']:.2f}",
               ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#212121')

ax_b2.set_xlabel('Trajectory (%)')
ax_b2.set_ylabel('Chosen towardness (%)')
ax_b2.set_xticks([0, 50, 100])
ax_b2.set_ylim(-18, 92)
sns.despine(ax=ax_b2)


# =====================================================================
# PANEL C: Chosen-Option Towardness (%) Across Target, Control, Distractor
# =====================================================================
gs_c = axes[1, 0].get_subplotspec().subgridspec(1, 2, wspace=0.28)
axes[1, 0].remove()
ax_c1 = fig.add_subplot(gs_c[0])
ax_c2 = fig.add_subplot(gs_c[1])

cohort_c_cfg = [
    (ax_c1, 'Younger adults', [COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG], 'C'),
    (ax_c2, 'Older adults', [COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD], ''),
]

tbl_tow = sub_by_outcome['chosen_towardness_pct'].unstack('outcome')
data_top_c = float(tbl_tow.max().max())
data_bot_c = float(tbl_tow.min().min())
yr_c = (data_top_c + 22) - (data_bot_c - 5)

for ax_c, cohort, cols_c, panel_lbl in cohort_c_cfg:
    c_df = tbl_tow.loc[cohort]
    ax_c.set_title(cohort, loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_c, panel_lbl)
    ax_c.axhline(0, color='#9E9E9E', linestyle='--', lw=1.0, alpha=0.7)

    jitter_c = np.random.uniform(-0.06, 0.06, size=len(c_df))
    for i in range(len(c_df)):
        s_row = c_df.iloc[i]
        ax_c.plot(x_pos_out + jitter_c[i], [s_row['Target'], s_row['Control'], s_row['Distractor']],
                  color='gray', alpha=0.30, lw=0.8, zorder=2)

    for idx, cat in enumerate(OUTCOMES):
        vals = c_df[cat].dropna().values
        ax_c.scatter(np.full_like(vals, x_pos_out[idx]) + jitter_c[:vals.size], vals,
                     color=cols_c[idx], alpha=0.65, s=36, zorder=3)
        ax_c.errorbar(x_pos_out[idx], vals.mean(), yerr=stats.sem(vals), fmt='o',
                      color='black', ecolor='black', elinewidth=2.2, capsize=5,
                      capthick=1.8, markersize=7, zorder=6)

    stat_map_c = {(r['label_a'], r['label_b']): r for r in within_cohort_stats['chosen_towardness_pct'][cohort]}
    lvl_base_c = data_top_c + 0.05 * yr_c
    lvl_step_c = 0.11 * yr_c
    h_c = 0.015 * yr_c
    lvls_c = {
        'Target_Control': lvl_base_c,
        'Control_Distractor': lvl_base_c + lvl_step_c,
        'Target_Distractor': lvl_base_c + 2 * lvl_step_c,
    }
    for a, b in PAIRS:
        r = stat_map_c[(a, b)]
        draw_sig_bracket(
            ax_c, X_OF[a], X_OF[b], lvls_c[f'{a}_{b}'], h_c,
            f"{stars_from_p(r['p_holm'])} $d_z$ = {r['dz']:.2f}"
        )

    ax_c.set_xticks(x_pos_out)
    ax_c.set_xticklabels(OUTCOMES, fontsize=9.5)
    ax_c.set_xlim(-0.45, 2.45)
    ax_c.set_ylim(min(0, data_bot_c) - 5, data_top_c + 0.40 * yr_c)
    sns.despine(ax=ax_c)

ax_c1.set_ylabel('Chosen towardness (%)')
ax_c2.set_ylabel('')
ax_c2.tick_params(labelleft=False)


# =====================================================================
# PANEL D: Active Perpendicular Deviation |d_perp| (dva) by Outcome
# =====================================================================
gs_d = axes[1, 1].get_subplotspec().subgridspec(1, 2, wspace=0.28)
axes[1, 1].remove()
ax_d1 = fig.add_subplot(gs_d[0])
ax_d2 = fig.add_subplot(gs_d[1])

cohort_d_cfg = [
    (ax_d1, 'Younger adults', [COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG], 'D'),
    (ax_d2, 'Older adults', [COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD], ''),
]

tbl_perp = sub_by_outcome['active_perp_dev_dva'].unstack('outcome')
data_top_d = float(tbl_perp.max().max())
data_bot_d = float(tbl_perp.min().min())
yr_d = data_top_d - data_bot_d

for ax_d, cohort, cols_d, panel_lbl in cohort_d_cfg:
    d_df = tbl_perp.loc[cohort]
    ax_d.set_title(cohort, loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_d, panel_lbl)

    jitter_d = np.random.uniform(-0.06, 0.06, size=len(d_df))
    for i in range(len(d_df)):
        s_row = d_df.iloc[i]
        ax_d.plot(x_pos_out + jitter_d[i], [s_row['Target'], s_row['Control'], s_row['Distractor']],
                  color='gray', alpha=0.30, lw=0.8, zorder=2)

    for idx, cat in enumerate(OUTCOMES):
        vals = d_df[cat].dropna().values
        ax_d.scatter(np.full_like(vals, x_pos_out[idx]) + jitter_d[:vals.size], vals,
                     color=cols_d[idx], alpha=0.65, s=36, zorder=3)
        ax_d.errorbar(x_pos_out[idx], vals.mean(), yerr=stats.sem(vals), fmt='o',
                      color='black', ecolor='black', elinewidth=2.2, capsize=5,
                      capthick=1.8, markersize=7, zorder=6)

    stat_map_d = {(r['label_a'], r['label_b']): r for r in within_cohort_stats['active_perp_dev_dva'][cohort]}
    lvl_base_d = data_top_d + 0.06 * yr_d
    lvl_step_d = 0.14 * yr_d
    h_d = 0.02 * yr_d
    lvls_d = {
        'Target_Control': lvl_base_d,
        'Control_Distractor': lvl_base_d + lvl_step_d,
        'Target_Distractor': lvl_base_d + 2 * lvl_step_d,
    }
    for a, b in PAIRS:
        r = stat_map_d[(a, b)]
        draw_sig_bracket(
            ax_d, X_OF[a], X_OF[b], lvls_d[f'{a}_{b}'], h_d,
            f"{stars_from_p(r['p_holm'])} $d_z$ = {r['dz']:.2f}"
        )

    ax_d.set_xticks(x_pos_out)
    ax_d.set_xticklabels(OUTCOMES, fontsize=9.5)
    ax_d.set_xlim(-0.45, 2.45)
    ax_d.set_ylim(max(0.0, data_bot_d - 0.10 * yr_d), data_top_d + 0.50 * yr_d)
    sns.despine(ax=ax_d)

ax_d1.set_ylabel('Active |perp. deviation| (dva)')
ax_d2.set_ylabel('')
ax_d2.tick_params(labelleft=False)


# =====================================================================
# PANEL E: Time-Resolved Perpendicular Deviation per Outcome & Cohort
#          + Direct Cohort Contrasts of Chosen Towardness (E2)
# =====================================================================
spec_e = axes[2, 0].get_subplotspec().get_gridspec()
gs_e = spec_e[2, :].subgridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.30)
axes[2, 0].remove()
axes[2, 1].remove()

gs_e1 = gs_e[0].subgridspec(1, 3, wspace=0.42)
axs_e1 = [fig.add_subplot(gs_e1[i]) for i in range(3)]

e_ymin, e_ymax = np.inf, -np.inf
for outcome in OUTCOMES:
    for cohort in COHORTS:
        curves = perp_curves_by_outcome[outcome][cohort]
        m_c = np.nanmean(curves, axis=0)
        s_c = stats.sem(curves, axis=0, nan_policy='omit')
        e_ymin = min(e_ymin, float(np.nanmin(m_c - s_c)))
        e_ymax = max(e_ymax, float(np.nanmax(m_c + s_c)))
rng_e = e_ymax - e_ymin
e_ylim = (e_ymin - 0.28 * rng_e, e_ymax + 0.08 * rng_e)
bar_y_e = e_ymin - 0.14 * rng_e

for k, (ax_e1, outcome) in enumerate(zip(axs_e1, OUTCOMES)):
    for col, cohort in zip(OUTCOME_COLS[outcome], COHORTS):
        curves = perp_curves_by_outcome[outcome][cohort]
        m_c = np.nanmean(curves, axis=0)
        s_c = stats.sem(curves, axis=0, nan_policy='omit')
        ax_e1.plot(time_pct, m_c, color=col, lw=2.8, zorder=6,
                   label=cohort if k == 0 else None)
        ax_e1.fill_between(time_pct, m_c - s_c, m_c + s_c, color=col, alpha=0.28, zorder=5)

    if perp_clusters_by_outcome[outcome]:
        cl = perp_clusters_by_outcome[outcome][0]
        ax_e1.plot([cl['start_step'], cl['end_step']], [bar_y_e, bar_y_e],
                   color='#212121', lw=2.8, zorder=8)
        ax_e1.text((cl['start_step'] + cl['end_step']) / 2.0, bar_y_e + 0.02 * rng_e,
                   f"{stars_from_p(cl['p'])} $d$ = {cl['d']:.2f}",
                   ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#212121')

    ax_e1.set_title(outcome, loc='left', pad=10, fontweight='bold', fontsize=13, color='#212121')
    ax_e1.set_xlabel('Trajectory (%)')
    ax_e1.set_xticks([0, 50, 100])
    ax_e1.set_ylim(*e_ylim)
    sns.despine(ax=ax_e1)

add_panel_letter(axs_e1[0], 'E')
axs_e1[0].set_ylabel('|Perp. deviation| (dva)')
axs_e1[0].legend(loc='upper left', frameon=True, fontsize=8.5, framealpha=0.92)

# E2: Direct Between-Cohort Comparison of Pre-Click Chosen Towardness (%) per Outcome
ax_e2 = fig.add_subplot(gs_e[1])
ax_e2.set_title('Cohort contrast: confidence', loc='left', pad=12, fontweight='bold', fontsize=13, color='#212121')
ax_e2.axhline(0, color='#9E9E9E', linestyle='--', lw=1.0, alpha=0.7)

for gi, outcome in enumerate(OUTCOMES):
    for ci, cohort in enumerate(COHORTS):
        vals = tbl_tow.loc[cohort, outcome].dropna().values
        col = OUTCOME_COLS[outcome][ci]
        xg = x_pos_out[gi] + (-dodge_w if ci == 0 else dodge_w)
        jit = np.random.uniform(-0.05, 0.05, size=vals.size)
        ax_e2.scatter(xg + jit, vals, color=col, alpha=0.60, s=36, zorder=3)
        ax_e2.errorbar(xg, vals.mean(), yerr=stats.sem(vals), fmt='o', color='black',
                       ecolor='black', elinewidth=2.2, capsize=5, capthick=1.8,
                       markersize=7, zorder=6)

    r_tow = between_cohort_stats['chosen_towardness_pct'][outcome]
    lvl_e2 = data_top_c + (0.08 + 0.14 * gi) * (data_top_c - data_bot_c)
    draw_sig_bracket(
        ax_e2, x_pos_out[gi] - dodge_w, x_pos_out[gi] + dodge_w,
        lvl_e2, 0.03 * (data_top_c - data_bot_c),
        f"{stars_from_p(r_tow['p_holm'])} $d$ = {r_tow['d']:.2f}"
    )

ax_e2.set_xticks(x_pos_out)
ax_e2.set_xticklabels(OUTCOMES, fontsize=9.5)
ax_e2.set_ylabel('Chosen towardness (%)')
ax_e2.set_xlim(-0.55, 2.55)
ax_e2.set_ylim(min(0, data_bot_c) - 5, data_top_c + 0.55 * (data_top_c - data_bot_c))
sns.despine(ax=ax_e2)


# ---------------------------------------------------------------------
# 8. Save Figure & Statistical Report
# ---------------------------------------------------------------------
plots_dir = os.path.join(project_root, 'SPACECUE', 'plots')
os.makedirs(plots_dir, exist_ok=True)
png_path = os.path.join(plots_dir, 'confidence_vs_motor_bias_analysis.png')
svg_path = os.path.join(plots_dir, 'confidence_vs_motor_bias_analysis.svg')
report_path = os.path.join(plots_dir, 'confidence_vs_motor_bias_report.txt')

for attempt in range(5):
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        break
    except OSError:
        time.sleep(1.0)

with open(report_path, 'w', encoding='utf-8') as f_rep:
    f_rep.write("\n".join(report_lines) + "\n")

print(f"\n[INFO] Saved publication figure to:\n       {png_path}\n       {svg_path}")
print(f"[INFO] Saved statistical report to:\n       {report_path}")
plt.show()
