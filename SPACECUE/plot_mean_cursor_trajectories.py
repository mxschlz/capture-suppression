"""
Mean cursor trajectories per age cohort (young / old), with lateral deviation
over time, button selection percentages and towardness scores.

Panels:
    A - spatial cursor trajectories per cohort
    B - lateral deviation over time per click outcome (Target / Control /
        Distractor) + one-sample cluster permutation tests vs 0 (per cohort)
    C - selection percentages per button + pairwise tests (stars, Cohen's d)
    D - towardness scores per stimulus + pairwise tests (stars, Cohen's d)
    E - trial-to-trial variability of the lateral deviation per click outcome
        (Target / Control / Distractor) + cohort comparisons
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import stats
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import pingouin as pg

# project root
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
from SPACECUE.colors import (
    COLOR_TARGET_YOUNG, COLOR_TARGET_OLD,
    COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD,
    COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD,
)

sns.set_theme(context="talk", style="ticks")
plt.ion()
plt.close('all')

# stimulus locations on the unit circle (r = 1 dva)
TARGET_LOC = np.array([0.0, 1.0])
DISTRACTOR_LOC = np.array([-np.sqrt(3) / 2, -0.5])
CONTROL_LOC = np.array([np.sqrt(3) / 2, -0.5])

NORM_STEPS = 101
COORD_SCALE = 0.5
COHORTS = ['Younger adults', 'Older adults']
OUTCOMES = ['Target', 'Control', 'Distractor']

# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
data_path = SPACECUE.get_data_path()

archive = np.load(os.path.join(data_path, 'concatenated', 'aligned_mouse_resampled_101.npz'))
df_traj = pd.DataFrame(archive['data'], columns=archive['columns'].tolist())

# The NPZ archive embeds behavioral columns (age, is_correct, is_hp,
# singleton_present) from whichever behavioral table existed when it was
# created. Drop them so the freshly loaded CSV below is the single source of
# truth; otherwise stale labels silently leak into the analysis.
traj_keep = ['subject_id', 'block', 'trial_nr', 'step', 'time_norm', 'x_aligned', 'y_aligned']
df_traj = df_traj[[c for c in traj_keep if c in df_traj.columns]]

df_beh = pd.read_csv(os.path.join(data_path, 'concatenated', 'concatenated_behavioral.csv'))
beh_cols = [
    'subject_id', 'block', 'trial_nr', 'SelectDistractor', 'SelectControl', 'SelectOther',
    'IsCorrect', 'target_towardness', 'distractor_towardness', 'control_towardness', 'Age'
]
df_beh = df_beh[beh_cols].drop_duplicates(['subject_id', 'block', 'trial_nr'])

# Inner join: the CSV is outlier-cleaned (target towardness > 2 SD within
# subject removed), but the trajectory archive is NOT regenerated and still
# contains those trials. A left join would keep them with NaN behavioral
# columns, which misclassifies their cohort (NaN Age -> 'Older adults') and
# rescues removed outliers back into the Target curves via the archive's stale
# is_correct column. The inner join restricts trajectories to the cleaned
# trial set.
df_traj = pd.merge(df_traj, df_beh, on=['subject_id', 'block', 'trial_nr'], how='inner')

# which button was clicked (trials without a click stay 'Miss / No Click')
df_traj['outcome'] = 'Miss / No Click'
df_traj.loc[df_traj['IsCorrect'] == 1, 'outcome'] = 'Target'
df_traj.loc[df_traj['SelectControl'] == 1, 'outcome'] = 'Control'
df_traj.loc[df_traj['SelectDistractor'] == 1, 'outcome'] = 'Distractor'

df_traj['age_cohort'] = np.where(df_traj['Age'] <= 35, 'Younger adults', 'Older adults')
df_beh['age_cohort'] = np.where(df_beh['Age'] <= 35, 'Younger adults', 'Older adults')

df_traj['x_scaled'] = df_traj['x_aligned'] * COORD_SCALE
df_traj['y_scaled'] = df_traj['y_aligned'] * COORD_SCALE

# ---------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------
clicked = df_beh[df_beh['IsCorrect'].notna()].copy()
clicked['outcome'] = 'Miss / No Click'
clicked.loc[clicked['IsCorrect'] == 1, 'outcome'] = 'Target'
clicked.loc[clicked['SelectControl'] == 1, 'outcome'] = 'Control'
clicked.loc[clicked['SelectDistractor'] == 1, 'outcome'] = 'Distractor'
valid_clicks = clicked[clicked['outcome'] != 'Miss / No Click']

sub_pct = valid_clicks.groupby(['age_cohort', 'subject_id', 'outcome']).size() \
                      .unstack(fill_value=0)
sub_pct = sub_pct.div(sub_pct.sum(axis=1), axis=0) * 100

# subject-wise mean trajectories per outcome
sub_traj = df_traj[df_traj['outcome'].isin(['Target', 'Control', 'Distractor'])].groupby(
    ['age_cohort', 'outcome', 'subject_id', 'step']
)[['x_scaled', 'y_scaled']].mean().reset_index()

# one-sample cluster permutation tests on the lateral deviation vs 0,
# separately per click outcome and cohort
# (dz computed on cluster-averaged data per subject, sign preserved)
dev_cluster_info = {}  # [outcome][cohort] -> {'X': (n_subjects, n_steps), 'clusters': [...]}
for outcome in OUTCOMES:
    dev_cluster_info[outcome] = {}
    for cohort in COHORTS:
        sub_tx = df_traj[(df_traj['outcome'] == outcome) & (df_traj['age_cohort'] == cohort)].groupby(
            ['subject_id', 'step']
        )['x_scaled'].mean().unstack('step')
        X = sub_tx.values  # (n_subjects, n_steps)
        t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
            X, n_permutations=5000, seed=42, verbose=False
        )
        sig_clusters = []
        for cl, p in zip(clusters, cluster_pv):
            if p < 0.05:
                steps = cl[0]
                cl_mean_sub = X[:, steps].mean(axis=1)
                dz = cl_mean_sub.mean() / cl_mean_sub.std(ddof=1)
                sig_clusters.append({
                    'start_pct': steps[0], 'end_pct': steps[-1], 'p': p, 'dz': dz,
                    'mean': cl_mean_sub.mean(), 'sd': cl_mean_sub.std(ddof=1)
                })
        sig_clusters.sort(key=lambda c: c['p'])
        dev_cluster_info[outcome][cohort] = {'X': X, 'clusters': sig_clusters}

print('\n=== Lateral deviation from target path vs 0 (Panel B, one-sample cluster tests) ===')
for outcome in OUTCOMES:
    for cohort in COHORTS:
        sig = dev_cluster_info[outcome][cohort]['clusters']
        if not sig:
            print(f"[{outcome} | {cohort}] no significant cluster (p >= 0.05)")
            continue
        for cl in sig:
            print(f"[{outcome} | {cohort}] cluster {cl['start_pct']}-{cl['end_pct']} %: "
                  f"p = {cl['p']:.4g}, d_z = {cl['dz']:.2f}, "
                  f"mean = {cl['mean']:.4f} dva, SD = {cl['sd']:.4f} dva")

sub_toward = df_beh.groupby(['age_cohort', 'subject_id'])[
    ['target_towardness', 'distractor_towardness', 'control_towardness']
].mean()

# ---------------------------------------------------------------------
# pairwise statistics for Panels C & D
# (Shapiro-Wilk assumption check -> paired t-test or Wilcoxon fallback,
#  Cohen's dz on paired differences, Holm-Bonferroni within each family)
# ---------------------------------------------------------------------
PAIRS_C_D = [('Target', 'Control'), ('Control', 'Distractor'), ('Target', 'Distractor')]
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

    Assumption check: Shapiro-Wilk on the paired differences (pg.normality).
    If normality holds (p >= .05) a paired t-test is run (pg.ttest), otherwise
    a Wilcoxon signed-rank test (pg.wilcoxon). Effect size is Cohen's
    dz = mean(diff) / sd(diff) on the raw subject-level differences
    (positive dz -> A > B).
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


def independent_comparison(vals_a, vals_b, label_a, label_b):
    """Independent-samples comparison A vs B (via pingouin).

    Assumption check: Shapiro-Wilk per group. If both groups are normal
    (p >= .05) a Welch-corrected t-test is run, otherwise a Mann-Whitney U
    test. Effect size is Cohen's d using the pooled SD
    (positive d -> A > B).
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    na, nb = vals_a.size, vals_b.size
    sp = np.sqrt(((na - 1) * vals_a.var(ddof=1) + (nb - 1) * vals_b.var(ddof=1))
                 / (na + nb - 2))
    d_val = (vals_a.mean() - vals_b.mean()) / sp
    sh_a = float(pg.normality(vals_a)['pval'].iloc[0])
    sh_b = float(pg.normality(vals_b)['pval'].iloc[0])
    if sh_a >= 0.05 and sh_b >= 0.05:
        t_df = pg.ttest(vals_a, vals_b, correction='auto')
        res = {'test': 't', 'stat': float(t_df['T'].iloc[0]),
               'df': float(t_df['dof'].iloc[0]), 'p': float(t_df['p-val'].iloc[0])}
    else:
        mw = pg.mwu(vals_a, vals_b)
        res = {'test': 'U', 'stat': float(mw['U-val'].iloc[0]),
               'df': np.nan, 'p': float(mw['p-val'].iloc[0])}
    res.update({'label_a': label_a, 'label_b': label_b, 'n_a': na, 'n_b': nb,
                'd': float(d_val), 'shapiro_p_a': sh_a, 'shapiro_p_b': sh_b})
    return res


def draw_sig_bracket(ax, x1, x2, y, h, text):
    """Draw a significance bracket between x1 and x2 at height y with text on top."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='#212121', lw=1.2, zorder=7)
    ax.text((x1 + x2) / 2.0, y + h * 1.3, text, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#212121', zorder=8, linespacing=1.1)


def add_panel_letter(ax, letter):
    """Place a large bold panel letter (A/B/C/D) left of the axes title."""
    if letter:
        ax.annotate(letter, xy=(0, 1.0), xycoords='axes fraction',
                    xytext=(-6, 12), textcoords='offset points',
                    ha='right', va='bottom', fontsize=20, fontweight='bold',
                    color='#000000', zorder=12)


panel_c_stats = {}
panel_d_stats = {}
for cohort in COHORTS:
    # Panel C: subject-wise selection percentages
    c_sub = sub_pct.loc[cohort]
    res_c = [paired_comparison(c_sub[a].values, c_sub[b].values, a, b) for a, b in PAIRS_C_D]
    _, p_corr = pg.multicomp([r['p'] for r in res_c], alpha=0.05, method='holm')
    for r, p_holm in zip(res_c, p_corr):
        r['p_holm'] = float(p_holm)
    panel_c_stats[cohort] = res_c

    # Panel D: subject-wise towardness scores
    d_sub = sub_toward.loc[cohort]
    D_COL = {'Target': 'target_towardness', 'Control': 'control_towardness',
             'Distractor': 'distractor_towardness'}
    res_d = [paired_comparison(d_sub[D_COL[a]].values, d_sub[D_COL[b]].values, a, b)
             for a, b in PAIRS_C_D]
    _, p_corr = pg.multicomp([r['p'] for r in res_d], alpha=0.05, method='holm')
    for r, p_holm in zip(res_d, p_corr):
        r['p_holm'] = float(p_holm)
    panel_d_stats[cohort] = res_d

print('\n=== Pairwise statistics for Panels C & D (stars based on Holm-corrected p) ===')
for panel_name, stat_dict in [('C: Selection (%)', panel_c_stats),
                              ('D: Towardness (%)', panel_d_stats)]:
    for cohort in COHORTS:
        for r in stat_dict[cohort]:
            if r['test'] == 't':
                stat_str = f"t({r['df']}) = {r['stat']:.3f}"
            else:
                stat_str = f"W = {r['stat']:.1f}"
            print(
                f"[Panel {panel_name}] {cohort} | {r['label_a']} vs {r['label_b']}: "
                f"{stat_str}, p = {r['p']:.4g}, p_holm = {r['p_holm']:.4g}, "
                f"d = {r['dz']:.2f}, mean_diff = {r['mean_diff']:.2f}, "
                f"Shapiro-W p = {r['shapiro_p']:.4g}"
            )

# ---------------------------------------------------------------------
# trial-to-trial variability of the lateral deviation (Panel E)
# Replicated for all three click outcomes (Target / Control / Distractor).
# Within-subject SD across same-outcome trials at each normalized step,
# then per-outcome cohort comparison: cluster permutation test on the SD
# time courses and a Welch t / Mann-Whitney test on the subject-mean SD
# (Cohen's d, pooled; Holm-corrected across the three outcomes).
# ---------------------------------------------------------------------
sub_var_curves = {}    # [outcome][cohort] -> (n_subjects, n_steps) SD curves
sub_var_mean = {}      # [outcome][cohort] -> per-subject mean SD (scalar)
var_cluster_info = {}  # [outcome] -> significant clusters (sorted by p)
for outcome in OUTCOMES:
    d_out = df_traj[df_traj['outcome'] == outcome]
    sub_var_curves[outcome] = {}
    sub_var_mean[outcome] = {}
    for cohort in COHORTS:
        sd_curve = d_out[d_out['age_cohort'] == cohort].groupby(
            ['subject_id', 'step'])['x_scaled'].std().unstack('step')
        # a subject needs >= 2 trials of this outcome for a definable SD;
        # trials are complete 101-step trajectories, so each subject's SD
        # row is either fully defined or all-NaN (0 or 1 trials)
        sd_curve = sd_curve.dropna(how='all').dropna()
        sub_var_curves[outcome][cohort] = sd_curve
        sub_var_mean[outcome][cohort] = sd_curve.mean(axis=1).dropna()

    X_var_young = sub_var_curves[outcome][COHORTS[0]].values
    X_var_old = sub_var_curves[outcome][COHORTS[1]].values
    _, clusters_v, cluster_pv_v, _ = permutation_cluster_test(
        [X_var_young, X_var_old], n_permutations=5000, seed=42, verbose=False
    )
    sig_clusters_v = []
    for cl, p in zip(clusters_v, cluster_pv_v):
        if p < 0.05:
            # mne returns each cluster as a tuple; for these 1-D time series the
            # element is an int array of time-step indices (or a boolean mask)
            arr = np.asarray(cl[0] if isinstance(cl, tuple) else cl)
            if arr.ndim > 1:
                steps = np.where(arr.any(axis=0))[0]
            elif arr.dtype == bool:
                steps = np.where(arr)[0]
            else:
                steps = arr.astype(int)
            v_y = X_var_young[:, steps].mean(axis=1)
            v_o = X_var_old[:, steps].mean(axis=1)
            sp = np.sqrt(((v_y.size - 1) * v_y.var(ddof=1) + (v_o.size - 1) * v_o.var(ddof=1))
                         / (v_y.size + v_o.size - 2))
            sig_clusters_v.append({
                'start_pct': steps[0], 'end_pct': steps[-1], 'p': p,
                'd': (v_y.mean() - v_o.mean()) / sp,
            })
    sig_clusters_v.sort(key=lambda c: c['p'])
    var_cluster_info[outcome] = sig_clusters_v

# subject-level scalar (mean trial-to-trial SD), cohort contrast per outcome
var_scalar_stats = {}
for outcome in OUTCOMES:
    var_scalar_stats[outcome] = independent_comparison(
        sub_var_mean[outcome][COHORTS[0]].values, sub_var_mean[outcome][COHORTS[1]].values,
        COHORTS[0], COHORTS[1]
    )
_, p_corr_var = pg.multicomp([var_scalar_stats[o]['p'] for o in OUTCOMES],
                             alpha=0.05, method='holm')
for o, p_holm in zip(OUTCOMES, p_corr_var):
    var_scalar_stats[o]['p_holm'] = float(p_holm)

print('\n=== Trial-to-trial variability of lateral deviation per click outcome (Panel E) ===')
for outcome in OUTCOMES:
    n_y = sub_var_curves[outcome][COHORTS[0]].shape[0]
    n_o = sub_var_curves[outcome][COHORTS[1]].shape[0]
    print(f"[{outcome}] subjects with >= 2 trials: {COHORTS[0]} n = {n_y}, "
          f"{COHORTS[1]} n = {n_o}")
    for cohort in COHORTS:
        d_coh = df_traj[(df_traj['outcome'] == outcome) &
                        (df_traj['age_cohort'] == cohort)]
        nt = (d_coh[['subject_id', 'block', 'trial_nr']].drop_duplicates()
              .groupby('subject_id').size())
        print(f"    trials per subject ({cohort}): median = {nt.median():.0f}, "
              f"range = {nt.min()}-{nt.max()}")
    for cl in var_cluster_info[outcome]:
        print(f"    Cluster {cl['start_pct']}-{cl['end_pct']} %: "
              f"p = {cl['p']:.4g}, d = {cl['d']:.2f}")
    r = var_scalar_stats[outcome]
    if r['test'] == 't':
        stat_str = f"t({r['df']:.1f}) = {r['stat']:.3f}"
    else:
        stat_str = f"U = {r['stat']:.1f}"
    print(f"    {COHORTS[0]} vs {COHORTS[1]} (mean trial-to-trial SD): {stat_str}, "
          f"p = {r['p']:.4g}, p_holm = {r['p_holm']:.4g}, d = {r['d']:.2f}, "
          f"Shapiro p = {r['shapiro_p_a']:.3g} / {r['shapiro_p_b']:.3g}")

# ---------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 15), gridspec_kw={'hspace': 0.4, 'wspace': 0.25})

# ---- Panel A: spatial trajectories -----------------------------------
gs_a = axes[0, 0].get_subplotspec().subgridspec(1, 2, wspace=0.28)
axes[0, 0].remove()
ax_a1 = fig.add_subplot(gs_a[0])
ax_a2 = fig.add_subplot(gs_a[1])

cohort_a_cfg = [
    (ax_a1, 'Younger adults', COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG, 'A'),
    (ax_a2, 'Older adults', COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD, ''),
]

for ax_a, cohort, col_tgt, col_ctl, col_dis, panel_lbl in cohort_a_cfg:
    ax_a.set_aspect('equal', anchor='N')
    ax_a.set_title(f"{cohort} (n = 18)", loc='left', pad=12,
                   fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_a, panel_lbl)

    # arena and direct paths
    for loc in [TARGET_LOC, DISTRACTOR_LOC, CONTROL_LOC]:
        ax_a.plot([0, loc[0]], [0, loc[1]], color='#CFD8DC', linestyle=':', lw=1.2, zorder=3)

    c_traj = sub_traj[sub_traj['age_cohort'] == cohort]
    for out_name, out_color, out_lw in [('Target', col_tgt, 3.2),
                                        ('Control', col_ctl, 2.6),
                                        ('Distractor', col_dis, 2.6)]:
        grp_out = c_traj[c_traj['outcome'] == out_name]

        for s_id in grp_out['subject_id'].unique():
            sub_data = grp_out[grp_out['subject_id'] == s_id]
            ax_a.plot(sub_data['x_scaled'], sub_data['y_scaled'],
                      color=out_color, lw=0.60, alpha=0.18, zorder=4)

        mean_path = grp_out.groupby('step')[['x_scaled', 'y_scaled']].mean().values
        sem_path = grp_out.groupby('step')[['x_scaled', 'y_scaled']].sem().values
        ax_a.plot(mean_path[:, 0], mean_path[:, 1], color=out_color, lw=out_lw, zorder=7)

        # SEM envelope; for the target the main deviation is horizontal
        if out_name == 'Target':
            ax_a.fill_betweenx(mean_path[:, 1], mean_path[:, 0] - sem_path[:, 0],
                               mean_path[:, 0] + sem_path[:, 0],
                               color=out_color, alpha=0.25, zorder=6)
        else:
            ax_a.fill_between(mean_path[:, 0], mean_path[:, 1] - sem_path[:, 1],
                              mean_path[:, 1] + sem_path[:, 1],
                              color=out_color, alpha=0.25, zorder=6)

        # direction arrow around the middle of the trajectory
        arrow_idx = 48
        ax_a.annotate('', xy=(mean_path[arrow_idx + 4, 0], mean_path[arrow_idx + 4, 1]),
                      xytext=(mean_path[arrow_idx, 0], mean_path[arrow_idx, 1]),
                      arrowprops=dict(arrowstyle="->", color=out_color, lw=1.6,
                                      mutation_scale=11), zorder=8)

    # buttons with cohort-mean selection percentages
    # two-line labels keep the horizontal footprint small enough to stay
    # clear of the axes spines at the bottom corners of the arena
    c_pct = sub_pct.loc[cohort].mean()
    for loc, label, pct, col, y_offset in [
        (TARGET_LOC, 'Target', c_pct['Target'], col_tgt, 0.08),
        (DISTRACTOR_LOC, 'Distractor', c_pct['Distractor'], col_dis, -0.08),
        (CONTROL_LOC, 'Control', c_pct['Control'], col_ctl, -0.08),
    ]:
        ax_a.text(loc[0], loc[1] + y_offset, f"{label}\n({pct:.1f}%)",
                  ha='center', va='bottom' if y_offset > 0 else 'top',
                  fontsize=8.5, fontweight='bold', color=col, zorder=11,
                  linespacing=1.15)

    ax_a.add_patch(Circle((0, 0), 0.045, facecolor='#ECEFF1', edgecolor='#455A64', lw=1.8, zorder=10))


    ax_a.set_xlabel('X position (dva)')
    ax_a.set_xlim(-1.30, 1.30)
    ax_a.set_ylim(-1.15, 1.45)
    sns.despine(ax=ax_a)

ax_a1.set_ylabel('Y position (dva)')
ax_a2.set_ylabel('')
ax_a2.tick_params(labelleft=False)

# ---- Panel B: lateral deviation over time -----------------------------
# ---- Panel B: lateral deviation over time, per click outcome -----------
# rows = cohorts, columns = click outcomes; hue = outcome, lightness = cohort
gs_b = axes[0, 1].get_subplotspec().subgridspec(2, 3, wspace=0.55, hspace=0.50)
axes[0, 1].remove()
axs_b = {(ci, oi): fig.add_subplot(gs_b[ci, oi])
         for ci in range(len(COHORTS)) for oi in range(len(OUTCOMES))}

time_pct = np.linspace(0, 100, NORM_STEPS)

# hue = outcome, lightness = cohort (SPACECUE color scheme)
OUTCOME_COLS = {
    'Target': (COLOR_TARGET_YOUNG, COLOR_TARGET_OLD),
    'Control': (COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD),
    'Distractor': (COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD),
}

# per-outcome y-limits shared across the two cohort rows, derived from the
# subject-level traces, with a reserved bottom band for the cluster bars
dev_ylim, dev_bar_y, dev_rng = {}, {}, {}
for outcome in OUTCOMES:
    y_lo, y_hi = np.inf, -np.inf
    for cohort in COHORTS:
        X = dev_cluster_info[outcome][cohort]['X']
        y_lo = min(y_lo, X.min())
        y_hi = max(y_hi, X.max())
    dev_rng[outcome] = y_hi - y_lo
    dev_ylim[outcome] = (y_lo - 0.30 * dev_rng[outcome], y_hi + 0.06 * dev_rng[outcome])
    dev_bar_y[outcome] = y_lo - 0.24 * dev_rng[outcome]

for ci, cohort in enumerate(COHORTS):
    for oi, outcome in enumerate(OUTCOMES):
        ax_b = axs_b[(ci, oi)]
        trace_col = OUTCOME_COLS[outcome][ci]
        X = dev_cluster_info[outcome][cohort]['X']
        sig_clusters = dev_cluster_info[outcome][cohort]['clusters']

        ax_b.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)

        for s_idx in range(X.shape[0]):
            ax_b.plot(time_pct, X[s_idx, :], color=trace_col, lw=0.6, alpha=0.18, zorder=3)

        mean_xc = np.mean(X, axis=0)
        sem_xc = stats.sem(X, axis=0)
        ax_b.plot(time_pct, mean_xc, color=trace_col, lw=2.8, zorder=6)
        ax_b.fill_between(time_pct, mean_xc - sem_xc, mean_xc + sem_xc,
                          color=trace_col, alpha=0.28, zorder=5)

        # strongest significant cluster vs 0 (single-line label, as in C/D)
        if sig_clusters:
            cl = sig_clusters[0]
            rng_b = dev_rng[outcome]
            bar_y = dev_bar_y[outcome]
            ax_b.plot([cl['start_pct'], cl['end_pct']], [bar_y, bar_y],
                      color='#212121', lw=2.6, zorder=8)
            mid_lbl = min(max((cl['start_pct'] + cl['end_pct']) / 2.0, 14.0), 86.0)
            ax_b.text(mid_lbl, bar_y + 0.02 * rng_b,
                      f"{stars_from_p(cl['p'])} $d$ = {cl['dz']:.2f}",
                      ha='center', va='bottom',
                      fontsize=9, color='#212121', zorder=9)

        if ci == 0:
            ax_b.set_title(outcome, loc='left', pad=10,
                           fontweight='bold', fontsize=12, color='#212121')
            ax_b.tick_params(labelbottom=False)
        else:
            ax_b.set_xlabel('Trajectory (%)')
        if oi > 0:
            ax_b.tick_params(labelleft=False)
        ax_b.set_xticks([0, 50, 100])
        ax_b.set_ylim(*dev_ylim[outcome])
        sns.despine(ax=ax_b)

# horizontal clamp: keep the single-line cluster labels inside the mini
# panels even when the strongest cluster center sits near the panel edge
fig.canvas.draw()
_rend_b = fig.canvas.get_renderer()
for _ax_b in axs_b.values():
    for _t in _ax_b.texts:
        if '$d_z$' not in _t.get_text():
            continue
        _bb = _t.get_window_extent(_rend_b)
        _ax_bb = _ax_b.get_window_extent(_rend_b)
        if _bb.x1 > _ax_bb.x1 - 2:
            _dx_pix = _bb.x1 - (_ax_bb.x1 - 2)
        elif _bb.x0 < _ax_bb.x0 + 2:
            _dx_pix = _bb.x0 - (_ax_bb.x0 + 2)
        else:
            continue
        _x0, _x1 = _ax_b.get_xlim()
        _dx_data = _dx_pix * (_x1 - _x0) / _ax_bb.width
        _t.set_position((_t.get_position()[0] - _dx_data, _t.get_position()[1]))

add_panel_letter(axs_b[(0, 0)], 'B')
axs_b[(0, 0)].set_ylabel(f"{COHORTS[0]}\nX deviation from\ntarget path (dva)", fontsize=10)
axs_b[(1, 0)].set_ylabel(f"{COHORTS[1]}\nX deviation from\ntarget path (dva)", fontsize=10)

# ---- Panel C: selection percentages -----------------------------------
gs_c = axes[1, 0].get_subplotspec().subgridspec(1, 2, wspace=0.28)
axes[1, 0].remove()
ax_c1 = fig.add_subplot(gs_c[0])
ax_c2 = fig.add_subplot(gs_c[1])

categories_c = ['Target', 'Control', 'Distractor']
x_pos_c = np.arange(len(categories_c))

cohort_c_cfg = [
    (ax_c1, 'Younger adults', [COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG], 'C'),
    (ax_c2, 'Older adults', [COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD], ''),
]

np.random.seed(42)

for ax_c, cohort, cols_c, panel_lbl in cohort_c_cfg:
    c_sub_pct = sub_pct.loc[cohort]
    ax_c.set_title(f"{cohort}", loc='left', pad=12,
                   fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_c, panel_lbl)
    ax_c.axhline(100.0 / 3.0, color='#9E9E9E', linestyle='--', lw=1.2, label='Chance')

    jitter_c = np.random.uniform(-0.06, 0.06, size=len(c_sub_pct))
    for i in range(len(c_sub_pct)):
        s_row = c_sub_pct.iloc[i]
        y_vals = [s_row['Target'], s_row['Control'], s_row['Distractor']]
        ax_c.plot(x_pos_c + jitter_c[i], y_vals, color='gray', alpha=0.30, lw=0.8, zorder=2)

    for idx, cat in enumerate(categories_c):
        vals = c_sub_pct[cat].values
        ax_c.scatter(np.full_like(vals, x_pos_c[idx]) + jitter_c, vals,
                     color=cols_c[idx], alpha=0.60, s=36, zorder=3)
        m, sem = vals.mean(), stats.sem(vals)
        ax_c.errorbar(x_pos_c[idx], m, yerr=sem, fmt='o', color='black', ecolor='black',
                      elinewidth=2.2, capsize=5, capthick=1.8, markersize=7, zorder=6)

    # significance brackets: paired tests (Holm-corrected) + Cohen's dz.
    # Levels are stacked proportionally to the full y-range so that each
    # single-line label clears the horizontal bar of the bracket above it
    # and the topmost label stays inside the axes.
    stat_c = {(r['label_a'], r['label_b']): r for r in panel_c_stats[cohort]}
    data_top = max(c_sub_pct[cat].max() for cat in categories_c)
    yr_c = data_top + 43  # full y-range once bottom (-3) and top margin are included
    lvl_base = data_top + 0.045 * yr_c
    lvl_step = 0.100 * yr_c
    h_c = 0.013 * yr_c
    lvl = {'Target_Control': lvl_base,
           'Control_Distractor': lvl_base + lvl_step,
           'Target_Distractor': lvl_base + 2 * lvl_step}
    for a, b in PAIRS_C_D:
        r = stat_c[(a, b)]
        draw_sig_bracket(ax_c, X_OF[a], X_OF[b], lvl[f'{a}_{b}'], h_c,
                         f"{stars_from_p(r['p_holm'])} $d$ = {r['dz']:.2f}")

    ax_c.set_xticks(x_pos_c)
    ax_c.set_xticklabels(['Target', 'Control', 'Distractor'], fontsize=11)
    ax_c.set_ylim(-3, data_top + 0.36 * yr_c)
    sns.despine(ax=ax_c)

ax_c1.set_ylabel('Selected proportion (%)')
ax_c2.set_ylabel('')
ax_c2.tick_params(labelleft=False)
ax_c1.legend(loc='lower left', frameon=True, fontsize=9.5, framealpha=0.9)

# ---- Panel D: towardness scores ---------------------------------------
gs_d = axes[1, 1].get_subplotspec().subgridspec(1, 2, wspace=0.28)
axes[1, 1].remove()
ax_d1 = fig.add_subplot(gs_d[0])
ax_d2 = fig.add_subplot(gs_d[1])

categories_d = ['target_towardness', 'control_towardness', 'distractor_towardness']
x_pos_d = np.arange(len(categories_d))

cohort_d_cfg = [
    (ax_d1, 'Younger adults', [COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG], 'D'),
    (ax_d2, 'Older adults', [COLOR_TARGET_OLD, COLOR_CONTROL_OLD, COLOR_DISTRACTOR_OLD], ''),
]

for ax_d, cohort, cols_d, panel_lbl in cohort_d_cfg:
    c_st = sub_toward.loc[cohort]
    ax_d.set_title(f"{cohort}", loc='left', pad=12,
                   fontweight='bold', fontsize=13, color='#212121')
    add_panel_letter(ax_d, panel_lbl)
    ax_d.axhline(0, color='#9E9E9E', linestyle='--', lw=1.0, alpha=0.7)

    jitter_d = np.random.uniform(-0.06, 0.06, size=len(c_st))
    for i in range(len(c_st)):
        s_row = c_st.iloc[i]
        y_vals = [s_row['target_towardness'], s_row['control_towardness'], s_row['distractor_towardness']]
        ax_d.plot(x_pos_d + jitter_d[i], y_vals, color='gray', alpha=0.30, lw=0.8, zorder=2)

    for idx, cat in enumerate(categories_d):
        vals = c_st[cat].values
        ax_d.scatter(np.full_like(vals, x_pos_d[idx]) + jitter_d, vals,
                     color=cols_d[idx], alpha=0.60, s=36, zorder=3)
        m, sem = vals.mean(), stats.sem(vals)
        ax_d.errorbar(x_pos_d[idx], m, yerr=sem, fmt='o', color='black', ecolor='black',
                      elinewidth=2.2, capsize=5, capthick=1.8, markersize=7, zorder=6)

    # significance brackets: paired tests (Holm-corrected) + Cohen's dz.
    # Levels are stacked proportionally to the full y-range (see Panel C).
    stat_d = {(r['label_a'], r['label_b']): r for r in panel_d_stats[cohort]}
    data_top = max(c_st[cat].max() for cat in categories_d)
    data_bot = min(c_st[cat].min() for cat in categories_d)
    yr_d = (data_top + 38) - (data_bot - 10)  # full y-range incl. margins
    lvl_base = data_top + 0.05 * yr_d
    lvl_step = 0.105 * yr_d
    h_d = 0.015 * yr_d
    lvl = {'Target_Control': lvl_base,
           'Control_Distractor': lvl_base + lvl_step,
           'Target_Distractor': lvl_base + 2 * lvl_step}
    for a, b in PAIRS_C_D:
        r = stat_d[(a, b)]
        draw_sig_bracket(ax_d, X_OF[a], X_OF[b], lvl[f'{a}_{b}'], h_d,
                         f"{stars_from_p(r['p_holm'])} $d$ = {r['dz']:.2f}")

    ax_d.set_xticks(x_pos_d)
    ax_d.set_xticklabels(['Target', 'Control', 'Distractor'], fontsize=11)
    ax_d.set_ylim(data_bot - 10, data_top + 0.38 * yr_d)
    sns.despine(ax=ax_d)

ax_d1.set_ylabel('Towardness (%)')
ax_d2.set_ylabel('')
ax_d2.tick_params(labelleft=False)

# ---- Panel E: trial-to-trial variability per click outcome ------------
spec_e = axes[2, 0].get_subplotspec().get_gridspec()
gs_e = spec_e[2, :].subgridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.30)
axes[2, 0].remove()
axes[2, 1].remove()

# E1: one mini panel per click outcome, each contrasting the two cohorts
# (hue = outcome, lightness = cohort, following the SPACECUE color scheme)
gs_e1 = gs_e[0].subgridspec(1, 3, wspace=0.45)
axs_e1 = [fig.add_subplot(gs_e1[i]) for i in range(3)]

# OUTCOME_COLS (hue = outcome, lightness = cohort) is defined in Panel B above

# shared y-limits across mini panels for comparability, with a reserved
# band at the bottom for the cluster bars
e_ymin, e_ymax = np.inf, -np.inf
for outcome in OUTCOMES:
    for cohort in COHORTS:
        curves = sub_var_curves[outcome][cohort].values
        m_v = np.nanmean(curves, axis=0)
        s_v = stats.sem(curves, axis=0, nan_policy='omit')
        e_ymin = min(e_ymin, np.nanmin(m_v - s_v))
        e_ymax = max(e_ymax, np.nanmax(m_v + s_v))
rng_e = e_ymax - e_ymin
e_ylim = (e_ymin - 0.32 * rng_e, e_ymax + 0.06 * rng_e)
bar_y_e = e_ymin - 0.17 * rng_e

for k, (ax_e1, outcome) in enumerate(zip(axs_e1, OUTCOMES)):
    for col, cohort in zip(OUTCOME_COLS[outcome], COHORTS):
        curves = sub_var_curves[outcome][cohort].values
        m_v = np.nanmean(curves, axis=0)
        s_v = stats.sem(curves, axis=0, nan_policy='omit')
        ax_e1.plot(time_pct, m_v, color=col, lw=2.8, zorder=6,
                   label=cohort if k == 0 else None)
        ax_e1.fill_between(time_pct, m_v - s_v, m_v + s_v, color=col,
                           alpha=0.28, zorder=5)

    # strongest significant cluster of the younger-vs-older contrast
    if var_cluster_info[outcome]:
        cl = var_cluster_info[outcome][0]
        ax_e1.plot([cl['start_pct'], cl['end_pct']], [bar_y_e, bar_y_e],
                   color='#212121', lw=3.0, zorder=8)
        ax_e1.text((cl['start_pct'] + cl['end_pct']) / 2.0, bar_y_e + 0.02 * rng_e,
                   stars_from_p(cl['p']), ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color='#212121', zorder=9)
        ax_e1.text((cl['start_pct'] + cl['end_pct']) / 2.0, bar_y_e - 0.02 * rng_e,
                   f"$d$ = {cl['d']:.2f}", ha='center', va='top',
                   fontsize=9.0, color='#212121')

    ax_e1.set_title(outcome, loc='left', pad=10,
                    fontweight='bold', fontsize=13, color='#212121')
    ax_e1.set_xlabel('Trajectory (%)')
    ax_e1.set_xticks([0, 50, 100])
    ax_e1.set_ylim(*e_ylim)
    sns.despine(ax=ax_e1)

add_panel_letter(axs_e1[0], 'E')
axs_e1[0].set_ylabel('SD of lateral deviation (dva)')
axs_e1[0].legend(loc='upper left', frameon=True, fontsize=9, framealpha=0.9)

# E2: subject-level scalar (mean trial-to-trial SD) per outcome and cohort
ax_e2 = fig.add_subplot(gs_e[1])
x_pos_e2 = np.arange(len(OUTCOMES))
dodge_e2 = 0.18

data_top_e2, data_bot_e2 = -np.inf, np.inf
for outcome in OUTCOMES:
    for cohort in COHORTS:
        vv = sub_var_mean[outcome][cohort].values
        data_top_e2 = max(data_top_e2, vv.max())
        data_bot_e2 = min(data_bot_e2, vv.min())
rng2 = data_top_e2 - data_bot_e2

for gi, outcome in enumerate(OUTCOMES):
    for ci, cohort in enumerate(COHORTS):
        vals = sub_var_mean[outcome][cohort].values
        col = OUTCOME_COLS[outcome][ci]
        jit = np.random.uniform(-0.05, 0.05, size=vals.size)
        xg = x_pos_e2[gi] + (-dodge_e2 if ci == 0 else dodge_e2)
        ax_e2.scatter(xg + jit, vals, color=col, alpha=0.60, s=36, zorder=3)
        ax_e2.errorbar(xg, vals.mean(), yerr=stats.sem(vals), fmt='o', color='black',
                       ecolor='black', elinewidth=2.2, capsize=5, capthick=1.8,
                       markersize=7, zorder=6)

# cohort contrasts per outcome; brackets are staggered so the single-line
# labels of neighbouring outcomes cannot collide
for gi, outcome in enumerate(OUTCOMES):
    r = var_scalar_stats[outcome]
    lvl_e2 = data_top_e2 + (0.10 + 0.14 * gi) * rng2
    draw_sig_bracket(ax_e2, x_pos_e2[gi] - dodge_e2, x_pos_e2[gi] + dodge_e2,
                     lvl_e2, 0.03 * rng2,
                     f"{stars_from_p(r['p_holm'])} $d$ = {r['d']:.2f}")

ax_e2.set_title('Mean variability per subject', loc='left', pad=12,
                fontweight='bold', fontsize=13, color='#212121')
ax_e2.set_xticks(x_pos_e2)
ax_e2.set_xticklabels(
    [f"{o}" for o in OUTCOMES],
    fontsize=10)
ax_e2.set_xlim(-0.55, 2.55)
ax_e2.set_ylim(data_bot_e2 - 0.18 * rng2, data_top_e2 + 0.55 * rng2)
sns.despine(ax=ax_e2)

# save
plots_dir = os.path.join(project_root, 'SPACECUE', 'plots')
os.makedirs(plots_dir, exist_ok=True)
png_path = os.path.join(plots_dir, 'average_cursor_trajectories_publication.png')
svg_path = os.path.join(plots_dir, 'average_cursor_trajectories_publication.svg')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Saved figure to {png_path} and {svg_path}")

plt.show()
