"""
SPACECUE: Statistical Rule Violation Analysis
=============================================
Investigates behavioral costs, cursor towardness dynamics, and neural electrophysiological
dynamics (pre-stimulus alpha lateralization and post-stimulus N2ac/Pd ERPs) on trials where
the statistical regularity rule is violated (Target appears at High-Probability Distractor Location:
`TargetLoc == HP_Distractor_Loc`).

Evaluated across both cueing strengths:
  - 80% HP Distractor Probability
  - 60% HP Distractor Probability

Analysis Scope & Constraints:
  - All trials included (both correct and error trials), evaluating accuracy as an explicit dependent variable.
  - Conventional behavioral metrics: Accuracy (IsCorrect, %), Response Time (rt, in seconds).
  - Continuous mouse-tracking towardness metrics: target_towardness, distractor_towardness, control_towardness (%).
  - Time-resolved cursor trajectories across normalized trajectory progression (0-100% Movement Time).
  - Neural pre-stimulus alpha lateralization index (LI, 8-12 Hz, -750 to 0 ms).
  - Neural post-stimulus lateralized ERPs (C3/C4 contra - ipsi difference waveforms, N2ac).

Strict adherence to AGENTS.md neuroscientific standards:
  - Units on all axes: Time (s), Progression (% MT), Accuracy (%), RT (s), Towardness (%), Amplitude (μV).
  - Publication-ready styling: sns.set_theme(context="talk", style="ticks"), sns.despine().
  - Shaded SEM error bands with alpha=0.3 on group waveforms.
  - Non-parametric cluster-based permutation tests on continuous trajectories and ERPs.
  - Circularity-free cluster effect sizes computed on temporally averaged cluster bounds per subject.
  - Full reporting of test statistics (t, F), degrees of freedom (df), exact unrounded p-values, and effect sizes.
  - Console-first interactive execution layout (# %% cells) with plt.close('all').
"""

# %% [Cell 1: Setup, Imports, Paths & Aesthetics]
import os
import sys
import glob
import re
import time
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import mne
from mne.stats import permutation_cluster_1samp_test

# Use 'Agg' backend if non-interactive
import matplotlib
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in python path
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
    COLOR_CONTROL, COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD
)

# Publication theme
sns.set_theme(context="talk", style="ticks")
plt.close('all')

# Directory setup
data_path = SPACECUE.get_data_path()
output_dir = os.path.join(project_root, 'SPACECUE', 'plots', 'statistical_rule_violation')
os.makedirs(output_dir, exist_ok=True)
cache_dir = os.path.join(data_path, 'derivatives', 'cache')
os.makedirs(cache_dir, exist_ok=True)

# Aesthetic palette for violation conditions
# 0.8: Solid dark / light pair; 0.6: Solid dark / light pair
COLOR_08_HP = '#b71c1c'  # Deep crimson (violation under strong cue)
COLOR_08_LP = '#e57373'  # Soft salmon (standard under strong cue)
COLOR_06_HP = '#0d47a1'  # Deep navy (violation under weak cue)
COLOR_06_LP = '#64b5f6'  # Soft azure (standard under weak cue)

PALETTE = {
    '0.8 HP (Violation)': COLOR_08_HP,
    '0.8 LP (Standard)': COLOR_08_LP,
    '0.6 HP (Violation)': COLOR_06_HP,
    '0.6 LP (Standard)': COLOR_06_LP,
}


def stars_from_p(p):
    """Return significance stars string for a given p-value."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def paired_comparison(vals_a, vals_b, label_a='Condition A', label_b='Condition B'):
    """
    Run assumption check (Shapiro-Wilk) and paired test (t-test or Wilcoxon) with Cohen's dz.
    Returns dictionary with exact unrounded statistics and effect size.
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    valid_mask = (~np.isnan(vals_a)) & (~np.isnan(vals_b))
    va = vals_a[valid_mask]
    vb = vals_b[valid_mask]
    diff = va - vb
    n = len(diff)

    if n < 3:
        return {'test': 'N/A', 'stat': np.nan, 'df': np.nan, 'p': np.nan, 'dz': np.nan,
                'mean_diff': np.nan, 'shapiro_W': np.nan, 'shapiro_p': np.nan, 'n': n}

    norm_df = pg.normality(diff)
    sh_w = float(norm_df['W'].iloc[0])
    sh_p = float(norm_df['pval'].iloc[0])
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0

    if sh_p >= 0.05:
        t_df = pg.ttest(va, vb, paired=True)
        res = {
            'test': 'paired-t',
            'stat': float(t_df['T'].iloc[0]),
            'df': int(t_df['dof'].iloc[0]),
            'p': float(t_df['p-val'].iloc[0]),
            'dz': dz,
            'mean_diff': float(diff.mean()),
            'shapiro_W': sh_w,
            'shapiro_p': sh_p,
            'n': n,
            'label_a': label_a,
            'label_b': label_b
        }
    else:
        w_df = pg.wilcoxon(va, vb)
        res = {
            'test': 'Wilcoxon',
            'stat': float(w_df['W-val'].iloc[0]),
            'df': np.nan,
            'p': float(w_df['p-val'].iloc[0]),
            'dz': dz,
            'mean_diff': float(diff.mean()),
            'shapiro_W': sh_w,
            'shapiro_p': sh_p,
            'n': n,
            'label_a': label_a,
            'label_b': label_b
        }
    return res


def onesamp_comparison(vals, test_val=0.0, label='Condition'):
    """One-sample test against test_val with Shapiro-Wilk check and Cohen's dz."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    diff = vals - test_val
    n = len(diff)
    if n < 3:
        return {'test': 'N/A', 'stat': np.nan, 'df': np.nan, 'p': np.nan, 'dz': np.nan,
                'mean': np.nan, 'shapiro_W': np.nan, 'shapiro_p': np.nan, 'n': n}

    norm_df = pg.normality(diff)
    sh_w = float(norm_df['W'].iloc[0])
    sh_p = float(norm_df['pval'].iloc[0])
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0

    if sh_p >= 0.05:
        t_df = pg.ttest(diff, 0.0)
        res = {
            'test': '1samp-t',
            'stat': float(t_df['T'].iloc[0]),
            'df': int(t_df['dof'].iloc[0]),
            'p': float(t_df['p-val'].iloc[0]),
            'dz': dz,
            'mean': float(diff.mean()),
            'shapiro_W': sh_w,
            'shapiro_p': sh_p,
            'n': n,
            'label': label
        }
    else:
        w_df = pg.wilcoxon(diff)
        res = {
            'test': 'Wilcoxon-1samp',
            'stat': float(w_df['W-val'].iloc[0]),
            'df': np.nan,
            'p': float(w_df['p-val'].iloc[0]),
            'dz': dz,
            'mean': float(diff.mean()),
            'shapiro_W': sh_w,
            'shapiro_p': sh_p,
            'n': n,
            'label': label
        }
    return res


print("[INFO] Cell 1 complete: Libraries loaded, aesthetic styling configured.")

# %% [Cell 2: Load Behavioral Dataset & Define Violation Conditions]
print("\n" + "="*80)
print("CELL 2: LOADING BEHAVIORAL DATASET & DEFINING VIOLATION CONDITIONS")
print("="*80)

beh_csv_path = os.path.join(data_path, 'concatenated', 'concatenated_behavioral.csv')
if not os.path.exists(beh_csv_path):
    raise FileNotFoundError(f"Behavioral dataset not found at {beh_csv_path}")

df_raw = pd.read_csv(beh_csv_path)
print(f"[INFO] Loaded concatenated_behavioral.csv: {df_raw.shape[0]} trials across {df_raw['subject_id'].nunique()} subjects.")

# Harmonize HP_Distractor_Loc and define statistical rule violation
# Distractor predominantly appears at HP_Distractor_Loc ('Left' or 'Right').
# Rule violation: Target appears at the HP distractor location!
df_raw['hp_loc_clean'] = df_raw['HP_Distractor_Loc'].replace({
    1.0: 'Left', 3.0: 'Right', 1: 'Left', 3: 'Right',
    '1': 'Left', '3': 'Right', '1.0': 'Left', '3.0': 'Right'
}).astype(str)

df_raw['target_loc_clean'] = df_raw['TargetLoc'].astype(str)
df_raw['prob_numeric'] = pd.to_numeric(df_raw['HP_Distractor_Prob'], errors='coerce')

# Filter lateral target trials (where lateralized selection occurs: 'Left' or 'Right')
df_lat = df_raw[df_raw['target_loc_clean'].isin(['Left', 'Right', 'left', 'right'])].copy()
df_lat['is_violation'] = (df_lat['target_loc_clean'].str.lower() == df_lat['hp_loc_clean'].str.lower()).astype(int)
df_lat['placement'] = np.where(df_lat['is_violation'] == 1, 'Target at HP (Violation)', 'Target at LP (Standard)')

# Display type classification
df_lat['display_type'] = np.where(
    df_lat['SingletonLoc'].astype(str).str.lower() == 'front',
    'Pure Target (Front Dist)',
    'Lateral Distractor'
)

print("\n--- Empirical Trial Counts (All Trials Included) ---")
crosstab_counts = pd.crosstab(
    [df_lat['prob_numeric'], df_lat['display_type']],
    df_lat['placement'],
    margins=True
)
print(crosstab_counts)

# %% [Cell 3: Behavioral Statistical Models (All Trials)]
print("\n" + "="*80)
print("CELL 3: BEHAVIORAL STATISTICAL MODELS ACROSS ALL TRIALS")
print("="*80)

# Five behavioral metrics:
# 1. IsCorrect (Accuracy, binary 0/1 converted to %)
# 2. rt (Reaction time, s; raw valid responses across all trials)
# 3. target_towardness (%)
# 4. distractor_towardness (%)
# 5. control_towardness (%)

BEH_METRICS = {
    'Accuracy': {'col': 'IsCorrect', 'unit': '%', 'scale': 100.0, 'title': 'Accuracy'},
    'RT': {'col': 'rt', 'unit': 's', 'scale': 1.0, 'title': 'Response Time'},
    'Target Towardness': {'col': 'target_towardness', 'unit': '%', 'scale': 1.0, 'title': 'Target Towardness'},
    'Distractor Towardness': {'col': 'distractor_towardness', 'unit': '%', 'scale': 1.0, 'title': 'Distractor Towardness'},
    'Control Towardness': {'col': 'control_towardness', 'unit': '%', 'scale': 1.0, 'title': 'Control Towardness'}
}

# Aggregate at subject level for each condition cell
subject_beh_results = {}
report_lines = []
report_lines.append("SPACECUE: STATISTICAL RULE VIOLATION ANALYSIS REPORT")
report_lines.append("="*80)
report_lines.append(f"Cohorts: All available subjects (N = {df_lat['subject_id'].nunique()})")
report_lines.append("Trial Inclusion: All trials (both correct and error trials included)")
report_lines.append(f"Total lateral target trials analyzed: {len(df_lat)}")
report_lines.append("="*80 + "\n")

beh_summary_table = []

for m_key, m_info in BEH_METRICS.items():
    col = m_info['col']
    scale = m_info['scale']
    unit = m_info['unit']
    
    # Compute subject averages for each of the 4 cells
    # (prob: 0.8 vs 0.6) x (violation: 1 vs 0)
    sub_agg = df_lat.groupby(['subject_id', 'prob_numeric', 'is_violation'])[col].mean().reset_index()
    sub_agg['val'] = sub_agg[col] * scale
    
    # Pivot to wide format: subject x condition
    piv = sub_agg.pivot(index='subject_id', columns=['prob_numeric', 'is_violation'], values='val')
    
    # Condition series across subjects
    s_08_hp = piv[(0.8, 1)].dropna()
    s_08_lp = piv[(0.8, 0)].dropna()
    s_06_hp = piv[(0.6, 1)].dropna()
    s_06_lp = piv[(0.6, 0)].dropna()
    
    # Synchronize subjects present in all 4 conditions
    common_subs = sorted(list(set(s_08_hp.index) & set(s_08_lp.index) & set(s_06_hp.index) & set(s_06_lp.index)))
    s_08_hp = s_08_hp.loc[common_subs].values
    s_08_lp = s_08_lp.loc[common_subs].values
    s_06_hp = s_06_hp.loc[common_subs].values
    s_06_lp = s_06_lp.loc[common_subs].values
    
    # Violation Cost = Target at HP - Target at LP
    cost_08 = s_08_hp - s_08_lp
    cost_06 = s_06_hp - s_06_lp
    
    # Statistical tests:
    # 1. 0.8 Violation Cost vs 0
    t_08_cost = onesamp_comparison(cost_08, test_val=0.0, label=f'{m_key} Cost (0.8)')
    # 2. 0.6 Violation Cost vs 0
    t_06_cost = onesamp_comparison(cost_06, test_val=0.0, label=f'{m_key} Cost (0.6)')
    # 3. Cost 0.8 vs Cost 0.6
    t_cost_diff = paired_comparison(cost_08, cost_06, label_a='0.8 Cost', label_b='0.6 Cost')
    
    # 4. 2x2 Repeated Measures ANOVA (Probability: 0.8 vs 0.6 x Placement: HP vs LP)
    rm_df_list = []
    for s_idx, sid in enumerate(common_subs):
        rm_df_list.append({'subject': sid, 'prob': '0.8', 'placement': 'Target at HP', 'val': s_08_hp[s_idx]})
        rm_df_list.append({'subject': sid, 'prob': '0.8', 'placement': 'Target at LP', 'val': s_08_lp[s_idx]})
        rm_df_list.append({'subject': sid, 'prob': '0.6', 'placement': 'Target at HP', 'val': s_06_hp[s_idx]})
        rm_df_list.append({'subject': sid, 'prob': '0.6', 'placement': 'Target at LP', 'val': s_06_lp[s_idx]})
    rm_df = pd.DataFrame(rm_df_list)
    aov = pg.rm_anova(data=rm_df, dv='val', within=['prob', 'placement'], subject='subject', detailed=True)
    
    # 5. Single-trial Linear Mixed Model (pingouin mixed_lmm or statsmodels)
    try:
        sub_trials = df_lat[['subject_id', 'prob_numeric', 'is_violation', col]].dropna().copy()
        sub_trials['dv'] = sub_trials[col] * scale
        sub_trials['prob_c'] = np.where(sub_trials['prob_numeric'] == 0.8, 1, 0)
        lmm_fit = pg.mixed_lmm(data=sub_trials, dv='dv', group='subject_id', fixed=['prob_c', 'is_violation'], time=None)
    except Exception as e:
        lmm_fit = None

    subject_beh_results[m_key] = {
        'common_subs': common_subs,
        's_08_hp': s_08_hp, 's_08_lp': s_08_lp,
        's_06_hp': s_06_hp, 's_06_lp': s_06_lp,
        'cost_08': cost_08, 'cost_06': cost_06,
        't_08_cost': t_08_cost, 't_06_cost': t_06_cost,
        't_cost_diff': t_cost_diff, 'anova': aov,
        'lmm': lmm_fit
    }
    
    # Summary Table Entry
    beh_summary_table.append({
        'Metric': f"{m_key} ({unit})",
        '0.8 HP Mean ± SEM': f"{np.mean(s_08_hp):.2f} ± {np.std(s_08_hp, ddof=1)/np.sqrt(len(s_08_hp)):.2f}",
        '0.8 LP Mean ± SEM': f"{np.mean(s_08_lp):.2f} ± {np.std(s_08_lp, ddof=1)/np.sqrt(len(s_08_lp)):.2f}",
        '0.8 Cost (HP-LP)': f"{np.mean(cost_08):+.2f} (t={t_08_cost['stat']:.2f}, p={t_08_cost['p']:.4e}, dz={t_08_cost['dz']:.2f})",
        '0.6 HP Mean ± SEM': f"{np.mean(s_06_hp):.2f} ± {np.std(s_06_hp, ddof=1)/np.sqrt(len(s_06_hp)):.2f}",
        '0.6 LP Mean ± SEM': f"{np.mean(s_06_lp):.2f} ± {np.std(s_06_lp, ddof=1)/np.sqrt(len(s_06_lp)):.2f}",
        '0.6 Cost (HP-LP)': f"{np.mean(cost_06):+.2f} (t={t_06_cost['stat']:.2f}, p={t_06_cost['p']:.4e}, dz={t_06_cost['dz']:.2f})",
        'Cost Diff (0.8 vs 0.6)': f"t={t_cost_diff['stat']:.2f}, p={t_cost_diff['p']:.4e}, dz={t_cost_diff['dz']:.2f}"
    })
    
    # Detailed log
    rep_sec = [
        f"--- METRIC: {m_key.upper()} ({unit}) (N = {len(common_subs)}) ---",
        f"Condition 0.8: HP = {np.mean(s_08_hp):.3f} ± {np.std(s_08_hp, ddof=1):.3f}, LP = {np.mean(s_08_lp):.3f} ± {np.std(s_08_lp, ddof=1):.3f}",
        f"  -> Violation Cost (0.8): {np.mean(cost_08):+.3f} {unit} | {t_08_cost['test']} stat = {t_08_cost['stat']:.4f}, df = {t_08_cost['df']}, p = {t_08_cost['p']:.6e}, dz = {t_08_cost['dz']:.4f} ({stars_from_p(t_08_cost['p'])})",
        f"Condition 0.6: HP = {np.mean(s_06_hp):.3f} ± {np.std(s_06_hp, ddof=1):.3f}, LP = {np.mean(s_06_lp):.3f} ± {np.std(s_06_lp, ddof=1):.3f}",
        f"  -> Violation Cost (0.6): {np.mean(cost_06):+.3f} {unit} | {t_06_cost['test']} stat = {t_06_cost['stat']:.4f}, df = {t_06_cost['df']}, p = {t_06_cost['p']:.6e}, dz = {t_06_cost['dz']:.4f} ({stars_from_p(t_06_cost['p'])})",
        f"Cost Comparison (0.8 Cost vs 0.6 Cost):",
        f"  -> Difference = {np.mean(cost_08 - cost_06):+.3f} {unit} | {t_cost_diff['test']} stat = {t_cost_diff['stat']:.4f}, df = {t_cost_diff['df']}, p = {t_cost_diff['p']:.6e}, dz = {t_cost_diff['dz']:.4f} ({stars_from_p(t_cost_diff['p'])})",
        f"Repeated Measures ANOVA (Probability x Placement):",
    ]
    for _, row in aov.iterrows():
        df1_val = row.get('ddof1', np.nan)
        df2_val = row.get('ddof2', np.nan)
        f_val = row.get('F', np.nan)
        p_val = row.get('p-unc', np.nan)
        # Calculate partial eta squared: np2 = (F * df1) / (F * df1 + df2)
        if pd.notna(f_val) and pd.notna(df1_val) and pd.notna(df2_val) and (f_val * df1_val + df2_val) > 0:
            np2_val = (f_val * df1_val) / (f_val * df1_val + df2_val)
        else:
            np2_val = np.nan
        rep_sec.append(f"  Source: {row['Source']:<15} | DF1={df1_val}, DF2={df2_val} | F={f_val:.4f} | p={p_val:.6e} | np2={np2_val:.4f}")
    rep_sec.append("\n")
    print("\n".join(rep_sec))
    report_lines.extend(rep_sec)

# %% [Cell 4: Continuous Mouse-Tracking Towardness Dynamics (0–100% MT)]
print("\n" + "="*80)
print("CELL 4: CONTINUOUS MOUSE-TRACKING TOWARDNESS DYNAMICS (0–100% MT)")
print("="*80)

traj_cache_path = os.path.join(data_path, 'concatenated', 'aligned_mouse_resampled_101.npz')
print(f"[INFO] Loading continuous trajectory data from {traj_cache_path}...")
archive = np.load(traj_cache_path)
df_traj = pd.DataFrame(archive['data'], columns=archive['columns'].tolist())

# Merge trajectory with trial-level behavioral metadata
traj_cols_keep = ['subject_id', 'block', 'trial_nr', 'step', 'time_norm', 'x_aligned', 'y_aligned']
df_traj = df_traj[[c for c in traj_cols_keep if c in df_traj.columns]]

beh_sub_meta = df_lat[['subject_id', 'block', 'trial_nr', 'prob_numeric', 'is_violation']].drop_duplicates()
df_traj_merged = pd.merge(df_traj, beh_sub_meta, on=['subject_id', 'block', 'trial_nr'], how='inner')
print(f"[INFO] Merged trajectory table: {len(df_traj_merged)} points across {df_traj_merged['subject_id'].nunique()} subjects.")

# Compute time-resolved towardness projections (Target, Distractor, Control):
# Aligned geometry: Target is at (0, 1), Distractor is at (-sqrt(3)/2, -0.5), Control is at (+sqrt(3)/2, -0.5)
sqrt3 = np.sqrt(3.0)
df_traj_merged['target_tt'] = df_traj_merged['y_aligned'] * 100.0
df_traj_merged['distractor_tt'] = (-df_traj_merged['x_aligned'] * sqrt3 - df_traj_merged['y_aligned']) / 2.0 * 100.0
df_traj_merged['control_tt'] = (df_traj_merged['x_aligned'] * sqrt3 - df_traj_merged['y_aligned']) / 2.0 * 100.0

time_steps = np.sort(df_traj_merged['step'].unique())
n_steps = len(time_steps)
all_traj_subs = sorted(df_traj_merged['subject_id'].unique())

# Structure subject-level trajectories: shape (n_subs, 101) for each condition
traj_conditions = {
    '08_HP': (0.8, 1),
    '08_LP': (0.8, 0),
    '06_HP': (0.6, 1),
    '06_LP': (0.6, 0),
}

subj_traj_tt = {c: [] for c in traj_conditions}
subj_traj_dt = {c: [] for c in traj_conditions}
subj_traj_ct = {c: [] for c in traj_conditions}
valid_traj_subs = []

for sid in all_traj_subs:
    sub_df = df_traj_merged[df_traj_merged['subject_id'] == sid]
    has_all = True
    temp_tt, temp_dt, temp_ct = {}, {}, {}
    for c_lbl, (p_val, v_val) in traj_conditions.items():
        cell_df = sub_df[(sub_df['prob_numeric'] == p_val) & (sub_df['is_violation'] == v_val)]
        if len(cell_df) == 0:
            has_all = False
            break
        step_means = cell_df.groupby('step')[['target_tt', 'distractor_tt', 'control_tt']].mean()
        if len(step_means) != n_steps:
            has_all = False
            break
        temp_tt[c_lbl] = step_means['target_tt'].values
        temp_dt[c_lbl] = step_means['distractor_tt'].values
        temp_ct[c_lbl] = step_means['control_tt'].values
    
    if has_all:
        valid_traj_subs.append(sid)
        for c_lbl in traj_conditions:
            subj_traj_tt[c_lbl].append(temp_tt[c_lbl])
            subj_traj_dt[c_lbl].append(temp_dt[c_lbl])
            subj_traj_ct[c_lbl].append(temp_ct[c_lbl])

for c_lbl in traj_conditions:
    subj_traj_tt[c_lbl] = np.array(subj_traj_tt[c_lbl])
    subj_traj_dt[c_lbl] = np.array(subj_traj_dt[c_lbl])
    subj_traj_ct[c_lbl] = np.array(subj_traj_ct[c_lbl])

print(f"[INFO] Computed subject continuous trajectories for N = {len(valid_traj_subs)} subjects.")

# Cluster-based permutation test on continuous Target Towardness difference (HP vs LP)
# Condition 0.8: TT(HP) - TT(LP)
diff_tt_08 = subj_traj_tt['08_HP'] - subj_traj_tt['08_LP']
t_obs_tt_08, clusters_tt_08, p_vals_tt_08, _ = permutation_cluster_1samp_test(
    diff_tt_08, n_permutations=2000, tail=0, verbose=False
)

# Condition 0.6: TT(HP) - TT(LP)
diff_tt_06 = subj_traj_tt['06_HP'] - subj_traj_tt['06_LP']
t_obs_tt_06, clusters_tt_06, p_vals_tt_06, _ = permutation_cluster_1samp_test(
    diff_tt_06, n_permutations=2000, tail=0, verbose=False
)

report_lines.append("--- CONTINUOUS TRAJECTORY CLUSTER TESTS (TARGET TOWARDNESS) ---")
report_lines.append(f"Tested difference: TT(HP) - TT(LP) over 0–100% Movement Time (N = {len(valid_traj_subs)})")

# Methodological constraint from AGENTS.md:
# NEVER compute cluster Cohen's dz by point-wise averaging.
# Collapse continuous signal across temporal bounds of cluster per subject, then compute single Cohen's dz.
sig_clusters_tt_08 = []
for c_idx, (cl, p_val) in enumerate(zip(clusters_tt_08, p_vals_tt_08)):
    if p_val < 0.05:
        cl_steps = np.where(cl[0])[0]
        step_start, step_end = cl_steps[0], cl_steps[-1]
        # Collapse per subject across cluster bounds
        sub_collapsed = np.mean(diff_tt_08[:, step_start:step_end+1], axis=1)
        dz_cluster = float(sub_collapsed.mean() / sub_collapsed.std(ddof=1))
        sig_clusters_tt_08.append((step_start, step_end, p_val, dz_cluster))
        report_lines.append(f"  0.8 Significant Cluster #{c_idx+1}: Steps {step_start}–{step_end}% MT | p = {p_val:.6e} | Cluster dz = {dz_cluster:.4f}")

sig_clusters_tt_06 = []
for c_idx, (cl, p_val) in enumerate(zip(clusters_tt_06, p_vals_tt_06)):
    if p_val < 0.05:
        cl_steps = np.where(cl[0])[0]
        step_start, step_end = cl_steps[0], cl_steps[-1]
        sub_collapsed = np.mean(diff_tt_06[:, step_start:step_end+1], axis=1)
        dz_cluster = float(sub_collapsed.mean() / sub_collapsed.std(ddof=1))
        sig_clusters_tt_06.append((step_start, step_end, p_val, dz_cluster))
        report_lines.append(f"  0.6 Significant Cluster #{c_idx+1}: Steps {step_start}–{step_end}% MT | p = {p_val:.6e} | Cluster dz = {dz_cluster:.4f}")

report_lines.append("\n")

# %% [Cell 5: Neural Dynamics I — Pre-Stimulus Alpha Lateralization]
print("\n" + "="*80)
print("CELL 5: NEURAL DYNAMICS I — PRE-STIMULUS ALPHA LATERALIZATION")
print("="*80)

# Load cached alpha dataset (multi-condition cache from pre_stim_alpha_lateralization.py)
alpha_cache_path = os.path.join(data_path, 'derivatives', 'cache', 'pre_stim_alpha_subject_data.pkl')
if os.path.exists(alpha_cache_path):
    with open(alpha_cache_path, 'rb') as fp:
        alpha_cache = pickle.load(fp)
    alpha_subs = sorted(list(alpha_cache.keys()))
    print(f"[INFO] Loaded pre-stimulus alpha cache from {alpha_cache_path} ({len(alpha_subs)} subjects).")
    
    # Extract subject-level LI for 0.8 and 0.6
    alpha_li_08 = np.array([float(np.mean(alpha_cache[s]['conditions']['expected_08']['li_data'])) for s in alpha_subs])
    alpha_li_06 = np.array([float(np.mean(alpha_cache[s]['conditions']['expected_06']['li_data'])) for s in alpha_subs])
    
    # Correlate subject-level pre-stimulus alpha LI with behavioral suppression cost
    # Cross-match subjects with behavioral cost
    beh_subs_08 = subject_beh_results['RT']['common_subs']
    shared_alpha_subs = sorted(list(set(alpha_subs) & set(beh_subs_08)))
    
    idx_a = [alpha_subs.index(s) for s in shared_alpha_subs]
    idx_b = [beh_subs_08.index(s) for s in shared_alpha_subs]
    
    # 0.8 correlation: Pre-stim alpha LI vs RT Violation Cost
    li_08_shared = alpha_li_08[idx_a]
    rt_cost_08_shared = subject_beh_results['RT']['cost_08'][idx_b]
    acc_cost_08_shared = subject_beh_results['Accuracy']['cost_08'][idx_b]
    tt_cost_08_shared = subject_beh_results['Target Towardness']['cost_08'][idx_b]
    
    # 0.6 correlation: Pre-stim alpha LI vs RT Violation Cost
    li_06_shared = alpha_li_06[idx_a]
    rt_cost_06_shared = subject_beh_results['RT']['cost_06'][idx_b]
    acc_cost_06_shared = subject_beh_results['Accuracy']['cost_06'][idx_b]
    tt_cost_06_shared = subject_beh_results['Target Towardness']['cost_06'][idx_b]
    
    r_rt_08 = pg.corr(li_08_shared, rt_cost_08_shared)
    r_acc_08 = pg.corr(li_08_shared, acc_cost_08_shared)
    r_tt_08 = pg.corr(li_08_shared, tt_cost_08_shared)
    
    r_rt_06 = pg.corr(li_06_shared, rt_cost_06_shared)
    r_acc_06 = pg.corr(li_06_shared, acc_cost_06_shared)
    r_tt_06 = pg.corr(li_06_shared, tt_cost_06_shared)
    
    rep_alpha = [
        "--- NEURAL DYNAMICS I: PRE-STIMULUS ALPHA LATERALIZATION ---",
        f"Subjects with alpha and behavioral data: N = {len(shared_alpha_subs)}",
        f"0.8 Pre-stim Alpha LI vs RT Cost: r = {r_rt_08['r'].iloc[0]:.4f}, p = {r_rt_08['p-val'].iloc[0]:.4f} ({stars_from_p(r_rt_08['p-val'].iloc[0])})",
        f"0.8 Pre-stim Alpha LI vs Accuracy Cost: r = {r_acc_08['r'].iloc[0]:.4f}, p = {r_acc_08['p-val'].iloc[0]:.4f} ({stars_from_p(r_acc_08['p-val'].iloc[0])})",
        f"0.8 Pre-stim Alpha LI vs Target Towardness Cost: r = {r_tt_08['r'].iloc[0]:.4f}, p = {r_tt_08['p-val'].iloc[0]:.4f} ({stars_from_p(r_tt_08['p-val'].iloc[0])})",
        f"0.6 Pre-stim Alpha LI vs RT Cost: r = {r_rt_06['r'].iloc[0]:.4f}, p = {r_rt_06['p-val'].iloc[0]:.4f} ({stars_from_p(r_rt_06['p-val'].iloc[0])})",
        f"0.6 Pre-stim Alpha LI vs Accuracy Cost: r = {r_acc_06['r'].iloc[0]:.4f}, p = {r_acc_06['p-val'].iloc[0]:.4f} ({stars_from_p(r_acc_06['p-val'].iloc[0])})",
        f"0.6 Pre-stim Alpha LI vs Target Towardness Cost: r = {r_tt_06['r'].iloc[0]:.4f}, p = {r_tt_06['p-val'].iloc[0]:.4f} ({stars_from_p(r_tt_06['p-val'].iloc[0])})\n",
    ]
    print("\n".join(rep_alpha))
    report_lines.extend(rep_alpha)
else:
    print(f"[WARN] Alpha cache not found at {alpha_cache_path}. Skipping alpha correlations.")

# %% [Cell 6: Neural Dynamics II — Post-Stimulus Lateralized ERPs (N2ac at C3/C4)]
print("\n" + "="*80)
print("CELL 6: NEURAL DYNAMICS II — POST-STIMULUS LATERALIZED ERPS (N2AC AT C3/C4)")
print("="*80)

erp_cache_file = os.path.join(cache_dir, 'rule_violation_erp_cache.pkl')
if not os.path.exists(erp_cache_file):
    print(f"[INFO] ERP cache not found at {erp_cache_file}. Generating cache now...")
    from SPACECUE.cache_rule_violation_erps import compute_and_save_cache
    compute_and_save_cache(erp_cache_file)

with open(erp_cache_file, 'rb') as fp:
    erp_data = pickle.load(fp)

erp_times = erp_data['times']
erp_subs = erp_data['subjects']
print(f"[INFO] Loaded ERP cache: {len(erp_subs)} subjects, times = [{erp_times[0]:.2f}, {erp_times[-1]:.2f}] s.")

# Restrict ERP analysis to window [-0.2, 0.6] s for publication plots
time_mask = (erp_times >= -0.2) & (erp_times <= 0.6)
plot_times = erp_times[time_mask]

# Extract Pure Target displays (where N2ac is isolated from distractor interference)
# Shapes: (N_subs, N_times)
erp_cond_diff = {}
erp_cond_contra = {}
erp_cond_ipsi = {}

valid_erp_subs = []
for sid in erp_subs:
    sub_pure = erp_data['pure'][sid]
    # Ensure minimum 5 trials per cell
    if min(sub_pure['08_HP_n'], sub_pure['08_LP_n'], sub_pure['06_HP_n'], sub_pure['06_LP_n']) >= 5:
        valid_erp_subs.append(sid)

print(f"[INFO] Valid ERP subjects with >= 5 trials per cell: N = {len(valid_erp_subs)}.")

for c_key in ['08_HP', '08_LP', '06_HP', '06_LP']:
    erp_cond_diff[c_key] = np.array([erp_data['pure'][sid][f'{c_key}_diff'][time_mask] * 1e6 for sid in valid_erp_subs])  # in µV
    erp_cond_contra[c_key] = np.array([erp_data['pure'][sid][f'{c_key}_contra'][time_mask] * 1e6 for sid in valid_erp_subs])
    erp_cond_ipsi[c_key] = np.array([erp_data['pure'][sid][f'{c_key}_ipsi'][time_mask] * 1e6 for sid in valid_erp_subs])

# Non-parametric cluster-based permutation test on N2ac Difference Waves (HP vs LP)
# Post-stimulus window [0.1, 0.5] s
post_stim_mask = (plot_times >= 0.1) & (plot_times <= 0.5)
post_times = plot_times[post_stim_mask]

diff_n2ac_08 = erp_cond_diff['08_HP'][:, post_stim_mask] - erp_cond_diff['08_LP'][:, post_stim_mask]
diff_n2ac_06 = erp_cond_diff['06_HP'][:, post_stim_mask] - erp_cond_diff['06_LP'][:, post_stim_mask]

t_obs_n2ac_08, clusters_n2ac_08, p_vals_n2ac_08, _ = permutation_cluster_1samp_test(
    diff_n2ac_08, n_permutations=2000, tail=0, verbose=False
)
t_obs_n2ac_06, clusters_n2ac_06, p_vals_n2ac_06, _ = permutation_cluster_1samp_test(
    diff_n2ac_06, n_permutations=2000, tail=0, verbose=False
)

# N2ac Time Window Quantification (200-350 ms post-stimulus)
n2ac_win_mask = (plot_times >= 0.20) & (plot_times <= 0.35)
n2ac_amp_08_hp = np.mean(erp_cond_diff['08_HP'][:, n2ac_win_mask], axis=1)
n2ac_amp_08_lp = np.mean(erp_cond_diff['08_LP'][:, n2ac_win_mask], axis=1)
n2ac_amp_06_hp = np.mean(erp_cond_diff['06_HP'][:, n2ac_win_mask], axis=1)
n2ac_amp_06_lp = np.mean(erp_cond_diff['06_LP'][:, n2ac_win_mask], axis=1)

t_n2ac_08 = paired_comparison(n2ac_amp_08_hp, n2ac_amp_08_lp, '0.8 N2ac (HP)', '0.8 N2ac (LP)')
t_n2ac_06 = paired_comparison(n2ac_amp_06_hp, n2ac_amp_06_lp, '0.6 N2ac (HP)', '0.6 N2ac (LP)')

# 2x2 RM ANOVA on N2ac amplitude
rm_n2ac_list = []
for s_idx, sid in enumerate(valid_erp_subs):
    rm_n2ac_list.append({'subject': sid, 'prob': '0.8', 'placement': 'Target at HP', 'val': n2ac_amp_08_hp[s_idx]})
    rm_n2ac_list.append({'subject': sid, 'prob': '0.8', 'placement': 'Target at LP', 'val': n2ac_amp_08_lp[s_idx]})
    rm_n2ac_list.append({'subject': sid, 'prob': '0.6', 'placement': 'Target at HP', 'val': n2ac_amp_06_hp[s_idx]})
    rm_n2ac_list.append({'subject': sid, 'prob': '0.6', 'placement': 'Target at LP', 'val': n2ac_amp_06_lp[s_idx]})
aov_n2ac = pg.rm_anova(data=pd.DataFrame(rm_n2ac_list), dv='val', within=['prob', 'placement'], subject='subject', detailed=True)

rep_erp = [
    "--- NEURAL DYNAMICS II: POST-STIMULUS LATERALIZED ERPS (N2AC) ---",
    f"Subjects evaluated: N = {len(valid_erp_subs)}",
    f"N2ac Time Window: 200–350 ms post-stimulus (Difference wave = Contra - Ipsi)",
    f"0.8 N2ac Mean Amplitude: HP = {np.mean(n2ac_amp_08_hp):.3f} µV, LP = {np.mean(n2ac_amp_08_lp):.3f} µV",
    f"  -> Comparison: {t_n2ac_08['test']} stat = {t_n2ac_08['stat']:.4f}, df = {t_n2ac_08['df']}, p = {t_n2ac_08['p']:.6e}, dz = {t_n2ac_08['dz']:.4f} ({stars_from_p(t_n2ac_08['p'])})",
    f"0.6 N2ac Mean Amplitude: HP = {np.mean(n2ac_amp_06_hp):.3f} µV, LP = {np.mean(n2ac_amp_06_lp):.3f} µV",
    f"  -> Comparison: {t_n2ac_06['test']} stat = {t_n2ac_06['stat']:.4f}, df = {t_n2ac_06['df']}, p = {t_n2ac_06['p']:.6e}, dz = {t_n2ac_06['dz']:.4f} ({stars_from_p(t_n2ac_06['p'])})",
    "Repeated Measures ANOVA on N2ac Amplitude:",
]
for _, row in aov_n2ac.iterrows():
    df1_val = row.get('ddof1', np.nan)
    df2_val = row.get('ddof2', np.nan)
    f_val = row.get('F', np.nan)
    p_val = row.get('p-unc', np.nan)
    if pd.notna(f_val) and pd.notna(df1_val) and pd.notna(df2_val) and (f_val * df1_val + df2_val) > 0:
        np2_val = (f_val * df1_val) / (f_val * df1_val + df2_val)
    else:
        np2_val = np.nan
    rep_erp.append(f"  Source: {row['Source']:<15} | DF1={df1_val}, DF2={df2_val} | F={f_val:.4f} | p={p_val:.6e} | np2={np2_val:.4f}")
rep_erp.append("\n")
print("\n".join(rep_erp))
report_lines.extend(rep_erp)

# Save text report to disk
report_path = os.path.join(output_dir, 'statistical_rule_violation_report.txt')
with open(report_path, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(report_lines))
print(f"[SUCCESS] Exported statistical report to {report_path}")

# %% [Cell 7: Publication-Grade Multi-Panel Visualizations]

# %% [Cell 7: Publication-Grade Multi-Panel Visualizations]
print("\n" + "="*80)
print("CELL 7: GENERATING PUBLICATION-GRADE VISUALIZATIONS")
print("="*80)

# Helper function to plot paired comparison slopes + mean ± SEM
def plot_paired_metric(ax, vals_hp_08, vals_lp_08, vals_hp_06, vals_lp_06, title, ylabel, is_cost=False):
    """Plot paired condition comparisons with individual subject slopes and group error bars."""
    x_pos = [0, 1, 2.2, 3.2]
    n_sub = len(vals_hp_08)
    
    # 0.8 data (Crimson)
    for i in range(n_sub):
        ax.plot([x_pos[0], x_pos[1]], [vals_lp_08[i], vals_hp_08[i]], color=COLOR_08_LP, alpha=0.25, lw=1.2, zorder=1)
    # 0.6 data (Navy)
    for i in range(n_sub):
        ax.plot([x_pos[2], x_pos[3]], [vals_lp_06[i], vals_hp_06[i]], color=COLOR_06_LP, alpha=0.25, lw=1.2, zorder=1)
        
    # Group means ± SEM
    m_lp_08, sem_lp_08 = np.mean(vals_lp_08), np.std(vals_lp_08, ddof=1) / np.sqrt(n_sub)
    m_hp_08, sem_hp_08 = np.mean(vals_hp_08), np.std(vals_hp_08, ddof=1) / np.sqrt(n_sub)
    m_lp_06, sem_lp_06 = np.mean(vals_lp_06), np.std(vals_lp_06, ddof=1) / np.sqrt(n_sub)
    m_hp_06, sem_hp_06 = np.mean(vals_hp_06), np.std(vals_hp_06, ddof=1) / np.sqrt(n_sub)
    
    ax.errorbar([x_pos[0]], [m_lp_08], yerr=[sem_lp_08], fmt='o', color=COLOR_08_LP, ms=9, capsize=4, elinewidth=2.5, zorder=5, label='80% Standard (LP)')
    ax.errorbar([x_pos[1]], [m_hp_08], yerr=[sem_hp_08], fmt='s', color=COLOR_08_HP, ms=9, capsize=4, elinewidth=2.5, zorder=5, label='80% Violation (HP)')
    ax.plot([x_pos[0], x_pos[1]], [m_lp_08, m_hp_08], color=COLOR_08_HP, lw=3.0, zorder=4)
    
    ax.errorbar([x_pos[2]], [m_lp_06], yerr=[sem_lp_06], fmt='o', color=COLOR_06_LP, ms=9, capsize=4, elinewidth=2.5, zorder=5, label='60% Standard (LP)')
    ax.errorbar([x_pos[3]], [m_hp_06], yerr=[sem_hp_06], fmt='s', color=COLOR_06_HP, ms=9, capsize=4, elinewidth=2.5, zorder=5, label='60% Violation (HP)')
    ax.plot([x_pos[2], x_pos[3]], [m_lp_06, m_hp_06], color=COLOR_06_HP, lw=3.0, zorder=4)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['80% Cue\nStandard (LP)', '80% Cue\nViolation (HP)', 
                        '60% Cue\nStandard (LP)', '60% Cue\nViolation (HP)'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    sns.despine(ax=ax)


# -------------------------------------------------------------------
# FIGURE 1: Comprehensive Behavioral Violation Costs
# -------------------------------------------------------------------
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))

# Panel A: Accuracy
res_acc = subject_beh_results['Accuracy']
plot_paired_metric(
    axes1[0, 0], res_acc['s_08_hp'], res_acc['s_08_lp'], res_acc['s_06_hp'], res_acc['s_06_lp'],
    'Target Selection Accuracy', 'Accuracy (%)'
)

# Panel B: Response Time
res_rt = subject_beh_results['RT']
plot_paired_metric(
    axes1[0, 1], res_rt['s_08_hp'], res_rt['s_08_lp'], res_rt['s_06_hp'], res_rt['s_06_lp'],
    'Response Latency', 'RT (s)'
)

# Panel C: Target Towardness
res_tt = subject_beh_results['Target Towardness']
plot_paired_metric(
    axes1[0, 2], res_tt['s_08_hp'], res_tt['s_08_lp'], res_tt['s_06_hp'], res_tt['s_06_lp'],
    'Target Towardness', 'Towardness (%)'
)

# Panel D: Distractor Towardness
res_dt = subject_beh_results['Distractor Towardness']
plot_paired_metric(
    axes1[1, 0], res_dt['s_08_hp'], res_dt['s_08_lp'], res_dt['s_06_hp'], res_dt['s_06_lp'],
    'Distractor Towardness', 'Towardness (%)'
)

# Panel E: Control Towardness
res_ct = subject_beh_results['Control Towardness']
plot_paired_metric(
    axes1[1, 1], res_ct['s_08_hp'], res_ct['s_08_lp'], res_ct['s_06_hp'], res_ct['s_06_lp'],
    'Control Towardness', 'Towardness (%)'
)

# Panel F: Direct Violation Cost Comparison across Cueing Strengths
ax_cost = axes1[1, 2]
cost_labels = ['Accuracy\n(%)', 'RT\n(x10 ms)', 'Target\nToward (%)', 'Distractor\nToward (%)', 'Control\nToward (%)']
# Scale RT cost to 10 ms units for visual comparability
costs_08_means = [
    np.mean(res_acc['cost_08']),
    np.mean(res_rt['cost_08']) * 100.0,
    np.mean(res_tt['cost_08']),
    np.mean(res_dt['cost_08']),
    np.mean(res_ct['cost_08']),
]
costs_08_sems = [
    np.std(res_acc['cost_08'], ddof=1) / np.sqrt(len(res_acc['cost_08'])),
    np.std(res_rt['cost_08'], ddof=1) * 100.0 / np.sqrt(len(res_rt['cost_08'])),
    np.std(res_tt['cost_08'], ddof=1) / np.sqrt(len(res_tt['cost_08'])),
    np.std(res_dt['cost_08'], ddof=1) / np.sqrt(len(res_dt['cost_08'])),
    np.std(res_ct['cost_08'], ddof=1) / np.sqrt(len(res_ct['cost_08'])),
]

costs_06_means = [
    np.mean(res_acc['cost_06']),
    np.mean(res_rt['cost_06']) * 100.0,
    np.mean(res_tt['cost_06']),
    np.mean(res_dt['cost_06']),
    np.mean(res_ct['cost_06']),
]
costs_06_sems = [
    np.std(res_acc['cost_06'], ddof=1) / np.sqrt(len(res_acc['cost_06'])),
    np.std(res_rt['cost_06'], ddof=1) * 100.0 / np.sqrt(len(res_rt['cost_06'])),
    np.std(res_tt['cost_06'], ddof=1) / np.sqrt(len(res_tt['cost_06'])),
    np.std(res_dt['cost_06'], ddof=1) / np.sqrt(len(res_dt['cost_06'])),
    np.std(res_ct['cost_06'], ddof=1) / np.sqrt(len(res_ct['cost_06'])),
]

bar_x = np.arange(len(cost_labels))
bar_width = 0.35
ax_cost.axhline(0, color='gray', lw=1.0, linestyle='--', zorder=1)
ax_cost.bar(bar_x - bar_width/2, costs_08_means, yerr=costs_08_sems, width=bar_width, color=COLOR_08_HP, capsize=4, label='80% Cue Cost')
ax_cost.bar(bar_x + bar_width/2, costs_06_means, yerr=costs_06_sems, width=bar_width, color=COLOR_06_HP, capsize=4, label='60% Cue Cost')

ax_cost.set_xticks(bar_x)
ax_cost.set_xticklabels(cost_labels, fontsize=11)
ax_cost.set_ylabel('Violation Cost (HP - LP)', fontsize=14)
ax_cost.set_title('Violation Cost Comparison', fontsize=15, fontweight='bold', pad=10)
ax_cost.legend(frameon=False, fontsize=12)
sns.despine(ax=ax_cost)

fig1.tight_layout()
fig1_path = os.path.join(output_dir, 'statistical_rule_violation_behavior.png')
fig1.savefig(fig1_path, dpi=300)
print(f"[SUCCESS] Saved Figure 1 to {fig1_path}")


# -------------------------------------------------------------------
# FIGURE 2: Continuous Mouse-Tracking Towardness Dynamics
# -------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

# Common styling for trajectory traces
tt_pct = np.linspace(0, 100, n_steps)

# Panel A: Target Towardness TT(t)
ax_tt = axes2[0]
for cond_key, col_val, lbl_val in [
    ('08_HP', COLOR_08_HP, '80% Violation (HP)'),
    ('08_LP', COLOR_08_LP, '80% Standard (LP)'),
    ('06_HP', COLOR_06_HP, '60% Violation (HP)'),
    ('06_LP', COLOR_06_LP, '60% Standard (LP)'),
]:
    mean_trace = np.mean(subj_traj_tt[cond_key], axis=0)
    sem_trace = np.std(subj_traj_tt[cond_key], axis=0, ddof=1) / np.sqrt(len(valid_traj_subs))
    ax_tt.plot(tt_pct, mean_trace, color=col_val, lw=2.5, label=lbl_val)
    ax_tt.fill_between(tt_pct, mean_trace - sem_trace, mean_trace + sem_trace, color=col_val, alpha=0.3)

# Highlight significant cluster for 0.8
for c_start, c_end, p_val, dz in sig_clusters_tt_08:
    ax_tt.axvspan(tt_pct[c_start], tt_pct[c_end], color='#ffebee', alpha=0.5, zorder=0)

ax_tt.set_title('Target Towardness TT(t)', fontsize=15, fontweight='bold')
ax_tt.set_xlabel('Trajectory Progression (% MT)', fontsize=14)
ax_tt.set_ylabel('Towardness (%)', fontsize=14)
ax_tt.legend(frameon=False, fontsize=11)
sns.despine(ax=ax_tt)

# Panel B: Distractor Towardness DT(t)
ax_dt = axes2[1]
for cond_key, col_val, lbl_val in [
    ('08_HP', COLOR_08_HP, '80% Violation (HP)'),
    ('08_LP', COLOR_08_LP, '80% Standard (LP)'),
    ('06_HP', COLOR_06_HP, '60% Violation (HP)'),
    ('06_LP', COLOR_06_LP, '60% Standard (LP)'),
]:
    mean_trace = np.mean(subj_traj_dt[cond_key], axis=0)
    sem_trace = np.std(subj_traj_dt[cond_key], axis=0, ddof=1) / np.sqrt(len(valid_traj_subs))
    ax_dt.plot(tt_pct, mean_trace, color=col_val, lw=2.5, label=lbl_val)
    ax_dt.fill_between(tt_pct, mean_trace - sem_trace, mean_trace + sem_trace, color=col_val, alpha=0.3)

ax_dt.set_title('Distractor Towardness DT(t)', fontsize=15, fontweight='bold')
ax_dt.set_xlabel('Trajectory Progression (% MT)', fontsize=14)
ax_dt.set_ylabel('Towardness (%)', fontsize=14)
ax_dt.legend(frameon=False, fontsize=11)
sns.despine(ax=ax_dt)

# Panel C: Control Towardness CT(t)
ax_ct = axes2[2]
for cond_key, col_val, lbl_val in [
    ('08_HP', COLOR_08_HP, '80% Violation (HP)'),
    ('08_LP', COLOR_08_LP, '80% Standard (LP)'),
    ('06_HP', COLOR_06_HP, '60% Violation (HP)'),
    ('06_LP', COLOR_06_LP, '60% Standard (LP)'),
]:
    mean_trace = np.mean(subj_traj_ct[cond_key], axis=0)
    sem_trace = np.std(subj_traj_ct[cond_key], axis=0, ddof=1) / np.sqrt(len(valid_traj_subs))
    ax_ct.plot(tt_pct, mean_trace, color=col_val, lw=2.5, label=lbl_val)
    ax_ct.fill_between(tt_pct, mean_trace - sem_trace, mean_trace + sem_trace, color=col_val, alpha=0.3)

ax_ct.set_title('Control Towardness CT(t)', fontsize=15, fontweight='bold')
ax_ct.set_xlabel('Trajectory Progression (% MT)', fontsize=14)
ax_ct.set_ylabel('Towardness (%)', fontsize=14)
ax_ct.legend(frameon=False, fontsize=11)
sns.despine(ax=ax_ct)

fig2.tight_layout()
fig2_path = os.path.join(output_dir, 'statistical_rule_violation_cursor_dynamics.png')
fig2.savefig(fig2_path, dpi=300)
print(f"[SUCCESS] Saved Figure 2 to {fig2_path}")


# -------------------------------------------------------------------
# FIGURE 3: Neural ERP Dynamics (N2ac at C3/C4)
# -------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: 0.8 Contralateral vs Ipsilateral ERPs
ax_erp_08 = axes3[0]
for trace_data, col_val, lbl_val in [
    (erp_cond_contra['08_HP'], COLOR_08_HP, 'HP Contra'),
    (erp_cond_ipsi['08_HP'], COLOR_08_HP, 'HP Ipsi (dashed)'),
    (erp_cond_contra['08_LP'], COLOR_08_LP, 'LP Contra'),
    (erp_cond_ipsi['08_LP'], COLOR_08_LP, 'LP Ipsi (dashed)'),
]:
    m_tr = np.mean(trace_data, axis=0)
    s_tr = np.std(trace_data, axis=0, ddof=1) / np.sqrt(len(valid_erp_subs))
    l_style = '--' if 'Ipsi' in lbl_val else '-'
    ax_erp_08.plot(plot_times, m_tr, color=col_val, lw=2.2, linestyle=l_style, label=lbl_val)
    ax_erp_08.fill_between(plot_times, m_tr - s_tr, m_tr + s_tr, color=col_val, alpha=0.15)

ax_erp_08.axvline(0, color='gray', lw=1.0, linestyle='--')
ax_erp_08.axhline(0, color='gray', lw=1.0, linestyle=':')
ax_erp_08.set_title('80% Cue: Contra vs Ipsi (C3/C4)', fontsize=15, fontweight='bold')
ax_erp_08.set_xlabel('Time (s)', fontsize=14)
ax_erp_08.set_ylabel('Amplitude (μV)', fontsize=14)
ax_erp_08.legend(frameon=False, fontsize=10)
sns.despine(ax=ax_erp_08)

# Panel B: 0.6 Contralateral vs Ipsilateral ERPs
ax_erp_06 = axes3[1]
for trace_data, col_val, lbl_val in [
    (erp_cond_contra['06_HP'], COLOR_06_HP, 'HP Contra'),
    (erp_cond_ipsi['06_HP'], COLOR_06_HP, 'HP Ipsi (dashed)'),
    (erp_cond_contra['06_LP'], COLOR_06_LP, 'LP Contra'),
    (erp_cond_ipsi['06_LP'], COLOR_06_LP, 'LP Ipsi (dashed)'),
]:
    m_tr = np.mean(trace_data, axis=0)
    s_tr = np.std(trace_data, axis=0, ddof=1) / np.sqrt(len(valid_erp_subs))
    l_style = '--' if 'Ipsi' in lbl_val else '-'
    ax_erp_06.plot(plot_times, m_tr, color=col_val, lw=2.2, linestyle=l_style, label=lbl_val)
    ax_erp_06.fill_between(plot_times, m_tr - s_tr, m_tr + s_tr, color=col_val, alpha=0.15)

ax_erp_06.axvline(0, color='gray', lw=1.0, linestyle='--')
ax_erp_06.axhline(0, color='gray', lw=1.0, linestyle=':')
ax_erp_06.set_title('60% Cue: Contra vs Ipsi (C3/C4)', fontsize=15, fontweight='bold')
ax_erp_06.set_xlabel('Time (s)', fontsize=14)
ax_erp_06.set_ylabel('Amplitude (μV)', fontsize=14)
ax_erp_06.legend(frameon=False, fontsize=10)
sns.despine(ax=ax_erp_06)

# Panel C: N2ac Difference Waves (Contra - Ipsi)
ax_diff = axes3[2]
for c_key, col_val, lbl_val in [
    ('08_HP', COLOR_08_HP, '80% Violation (HP)'),
    ('08_LP', COLOR_08_LP, '80% Standard (LP)'),
    ('06_HP', COLOR_06_HP, '60% Violation (HP)'),
    ('06_LP', COLOR_06_LP, '60% Standard (LP)'),
]:
    m_diff = np.mean(erp_cond_diff[c_key], axis=0)
    s_diff = np.std(erp_cond_diff[c_key], axis=0, ddof=1) / np.sqrt(len(valid_erp_subs))
    ax_diff.plot(plot_times, m_diff, color=col_val, lw=2.5, label=lbl_val)
    ax_diff.fill_between(plot_times, m_diff - s_diff, m_diff + s_diff, color=col_val, alpha=0.25)

# Highlight N2ac quantification window [200, 350] ms
ax_diff.axvspan(0.20, 0.35, color='#eeeeee', alpha=0.6, zorder=0, label='N2ac Window')
ax_diff.axvline(0, color='gray', lw=1.0, linestyle='--')
ax_diff.axhline(0, color='gray', lw=1.0, linestyle=':')
ax_diff.set_title('N2ac Difference Waves (Contra - Ipsi)', fontsize=15, fontweight='bold')
ax_diff.set_xlabel('Time (s)', fontsize=14)
ax_diff.set_ylabel('Amplitude (μV)', fontsize=14)
ax_diff.legend(frameon=False, fontsize=10)
sns.despine(ax=ax_diff)

fig3.tight_layout()
fig3_path = os.path.join(output_dir, 'statistical_rule_violation_erps.png')
fig3.savefig(fig3_path, dpi=300)
print(f"[SUCCESS] Saved Figure 3 to {fig3_path}")


# -------------------------------------------------------------------
# FIGURE 4: Pre- & Post-Stimulus Alpha Lateralization Dynamics
# -------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5.5))

# Extract time-course of Alpha LI across 12 ROI pairs
left_roi_hemi = ['TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO3', 'O1']
right_roi_hemi = ['TP10', 'TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 'PO8', 'PO4', 'O2']

alpha_subs_all = sorted(list(alpha_cache.keys()))
ch_names = alpha_cache[alpha_subs_all[0]]['ch_names']
freqs = alpha_cache[alpha_subs_all[0]]['freqs']
alpha_times = alpha_cache[alpha_subs_all[0]]['times']
alpha_idx = (freqs >= 8) & (freqs <= 12)
valid_left = [ch_names.index(ch) for ch in left_roi_hemi if ch in ch_names]
valid_right = [ch_names.index(ch) for ch in right_roi_hemi if ch in ch_names]

# Compute subject time courses for 80% and 60% expected distractor condition
tcs_08 = []
tcs_06 = []
for s in alpha_subs_all:
    li_08_sub = alpha_cache[s]['conditions']['expected_08']['li_data']
    li_06_sub = alpha_cache[s]['conditions']['expected_06']['li_data']
    tc_08 = (li_08_sub[valid_left, :, :][:, alpha_idx, :].mean(axis=(0, 1)) - 
             li_08_sub[valid_right, :, :][:, alpha_idx, :].mean(axis=(0, 1)))
    tc_06 = (li_06_sub[valid_left, :, :][:, alpha_idx, :].mean(axis=(0, 1)) - 
             li_06_sub[valid_right, :, :][:, alpha_idx, :].mean(axis=(0, 1)))
    tcs_08.append(tc_08)
    tcs_06.append(tc_06)
tcs_08 = np.array(tcs_08)
tcs_06 = np.array(tcs_06)

# Panel A: Alpha LI Time Courses (-0.75s to +0.8s) across cueing strengths
ax_a = axes4[0]
m_08_tc, sem_08_tc = np.mean(tcs_08, axis=0), np.std(tcs_08, axis=0, ddof=1) / np.sqrt(len(alpha_subs_all))
m_06_tc, sem_06_tc = np.mean(tcs_06, axis=0), np.std(tcs_06, axis=0, ddof=1) / np.sqrt(len(alpha_subs_all))

ax_a.plot(alpha_times, m_08_tc, color=COLOR_08_HP, lw=2.5, label='80% Cue')
ax_a.fill_between(alpha_times, m_08_tc - sem_08_tc, m_08_tc + sem_08_tc, color=COLOR_08_HP, alpha=0.25)
ax_a.plot(alpha_times, m_06_tc, color=COLOR_06_HP, lw=2.5, label='60% Cue')
ax_a.fill_between(alpha_times, m_06_tc - sem_06_tc, m_06_tc + sem_06_tc, color=COLOR_06_HP, alpha=0.25)

ax_a.axvspan(-0.75, 0.0, color='#f5f5f5', alpha=0.8, zorder=0, label='Pre-Stim Window')
ax_a.axvline(0, color='gray', lw=1.2, linestyle='--')
ax_a.axhline(0, color='gray', lw=1.0, linestyle=':')
ax_a.set_xlim([-0.75, 0.8])
ax_a.set_xlabel('Time (s)', fontsize=14)
ax_a.set_ylabel('Alpha Lateralization Index (a.u.)', fontsize=14)
ax_a.set_title('Alpha Lateralization Dynamics', fontsize=15, fontweight='bold', pad=10)
ax_a.legend(frameon=False, fontsize=11, loc='upper right')
sns.despine(ax=ax_a)

# Panel B: Pre-stimulus Alpha LI leading into Violation vs Standard trials
ax_b = axes4[1]
lmm_data_path = os.path.join(data_path, 'concatenated', 'spacecue_single_trial_lmm_data.csv')
if os.path.exists(lmm_data_path):
    df_lmm_trials = pd.read_csv(lmm_data_path)
    df_lmm_lat = df_lmm_trials[df_lmm_trials['TargetLoc'].astype(str).isin(['Left', 'Right', 'left', 'right'])].copy()
    hp_lbl = df_lmm_lat['HP_Distractor_Loc'].replace({
        1.0: 'Left', 3.0: 'Right', 1: 'Left', 3: 'Right', '1': 'Left', '3': 'Right', '1.0': 'Left', '3.0': 'Right'
    }).astype(str)
    df_lmm_lat['is_violation'] = (df_lmm_lat['TargetLoc'].astype(str) == hp_lbl).astype(int)
    df_lmm_lat['prob_num'] = pd.to_numeric(df_lmm_lat['HP_Distractor_Prob_numeric'], errors='coerce')
    
    sub_alpha_agg = df_lmm_lat.groupby(['subject_id', 'prob_num', 'is_violation'])['pre_stim_alpha_lateralization'].mean().reset_index()
    piv_alpha = sub_alpha_agg.pivot(index='subject_id', columns=['prob_num', 'is_violation'], values='pre_stim_alpha_lateralization')
    common_alpha_s = piv_alpha.dropna().index
    a_08_lp = piv_alpha.loc[common_alpha_s, (0.8, 0)].values
    a_08_hp = piv_alpha.loc[common_alpha_s, (0.8, 1)].values
    a_06_lp = piv_alpha.loc[common_alpha_s, (0.6, 0)].values
    a_06_hp = piv_alpha.loc[common_alpha_s, (0.6, 1)].values

    plot_paired_metric(
        ax_b, a_08_hp, a_08_lp, a_06_hp, a_06_lp,
        'Pre-Stim Alpha LI (Violation vs Standard)', 'Alpha LI (a.u.)'
    )
else:
    ax_b.text(0.5, 0.5, 'LMM single-trial alpha data unavailable', ha='center', va='center')

# Panel C: Cross-Subject Correlation: Pre-stim Alpha LI vs RT Suppression Cost
ax_c = axes4[2]
ax_c.scatter(li_08_shared, rt_cost_08_shared, color=COLOR_08_HP, s=45, alpha=0.7, label=f'80% (r = {r_rt_08["r"].iloc[0]:.2f}, p = {r_rt_08["p-val"].iloc[0]:.2f})')
ax_c.scatter(li_06_shared, rt_cost_06_shared, color=COLOR_06_HP, s=45, alpha=0.7, label=f'60% (r = {r_rt_06["r"].iloc[0]:.2f}, p = {r_rt_06["p-val"].iloc[0]:.2f})')

# Regression lines
sns.regplot(x=li_08_shared, y=rt_cost_08_shared, ax=ax_c, color=COLOR_08_HP, scatter=False, line_kws={'lw': 2, 'alpha': 0.8})
sns.regplot(x=li_06_shared, y=rt_cost_06_shared, ax=ax_c, color=COLOR_06_HP, scatter=False, line_kws={'lw': 2, 'alpha': 0.8})

ax_c.axhline(0, color='gray', lw=1.0, linestyle=':')
ax_c.axvline(0, color='gray', lw=1.0, linestyle=':')
ax_c.set_xlabel('Pre-Stim Alpha LI (a.u.)', fontsize=14)
ax_c.set_ylabel('RT Violation Cost (s)', fontsize=14)
ax_c.set_title('Proactive Alpha vs Behavioral Cost', fontsize=15, fontweight='bold', pad=10)
ax_c.legend(frameon=False, fontsize=11)
sns.despine(ax=ax_c)

fig4.tight_layout()
fig4_path = os.path.join(output_dir, 'statistical_rule_violation_alpha.png')
fig4.savefig(fig4_path, dpi=300)
print(f"[SUCCESS] Saved Figure 4 to {fig4_path}")

print("\n" + "="*80)
print("ANALYSIS EXECUTION COMPLETE!")
print(f"Summary Report: {report_path}")
print(f"Figures saved in: {output_dir}")
print("="*80)
