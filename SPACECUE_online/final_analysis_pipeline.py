"""
Final Confirmatory Analysis Pipeline for SPACECUE_online (Updated Paradigm).

Experimental Design & Statistical Model
---------------------------------------
The updated online paradigm uses a block-wise cueing architecture across 6 blocks:
- Block Type (Informative Cue Type): Target Cue, Distractor Cue, Control Cue
  (2 blocks per informative cue type).
- Cue Informativeness: Informative (50%) vs. Neutral (50%) interleaved within every block.
- Cue-Stimulus Delay Interval (Foreperiod): 0.5, 1.0, 1.5, 2.0 s (randomized per trial).

Primary Confirmatory Model (3 x 4 Repeated-Measures ANOVA on Difference Scores):
- For every subject, block type, and delay interval, the cueing effect is quantified
  against the delay-matched neutral baseline of the same block type:
      Delta = Informative(BlockType, Delay) - Neutral(BlockType, Delay)
- Factors:
  1. Informative Cue Type (3 levels: Target Cue, Distractor Cue, Control Cue)
  2. Cue-Stimulus Delay Interval (4 levels: 0.5, 1.0, 1.5, 2.0 s)
- Dependent Variables (evaluated across ALL trials, correct + incorrect pooled):
  1. Accuracy (Proportion Correct; missing responses/timeouts coded as 0)
  2. Response Time (s; all responded trials, correct + incorrect pooled)
- Planned & Post-hoc Inference:
  - Planned one-sample t-tests of delay-averaged Delta against 0 per Cue Type
    (Holm-Bonferroni corrected across the 3 cue types; powered for d_z = 0.5 at N = 50).
  - Post-hoc paired t-tests (`pingouin.pairwise_tests`, Holm-Bonferroni corrected)
    for main effects (Cue Type, Delay) and interaction simple comparisons
    (Cue Type comparisons within each Delay, and Delay comparisons within each Cue Type).
- Supplementary Model:
  - Full 3 (Block Type) x 2 (Informativeness) x 4 (Delay) rmANOVA decomposition,
    mathematically cross-checked against `statsmodels.stats.anova.AnovaRM`.
"""
import os
import re
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# Ensure project root is in python path for interactive console execution
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE_online
from SPACECUE.colors import COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG, COLOR_DIFF

# Publication-quality styling (strict adherence to workspace plotting conventions)
sns.set_theme(context="talk", style="ticks")
plt.ion()
plt.close('all')

# ============================================================
# 0. CONFIGURATION & CONSTANTS
# ============================================================
FINAL_SUBJECTS = list(SPACECUE_online.FINAL_SUBJECTS)
FINAL_PARADIGM_START = SPACECUE_online.FINAL_PARADIGM_START
DELAYS = list(SPACECUE_online.DELAYS)
CHANCE_LEVEL = SPACECUE_online.CHANCE_LEVEL
TARGET_N = SPACECUE_online.TARGET_N

MAX_NO_RESPONSE_RATE = 0.50   # subject exclusion threshold (> 50% missing responses)
MIN_ANTICIPATION_RT = 0.20    # trial exclusion threshold (< 0.20 s anticipation RTs)
ALPHA = 0.05

CUE_MAP = {
    'cue_target_location': 'Target Cue',
    'cue_distractor_location': 'Distractor Cue',
    'cue_nonsingleton_location': 'Control Cue',
    'cue_neutral': 'Neutral Cue',
}
INFORMATIVE_ORDER = ['Target Cue', 'Control Cue', 'Distractor Cue']

# Standard SPACECUE colors for Younger Adults (from SPACECUE/colors.py)
CUE_PALETTE = {
    'Target Cue': COLOR_TARGET_YOUNG,          # Light Green (#4CAF50)
    'Control Cue': COLOR_CONTROL_YOUNG,        # Light Grey (#9E9E9E)
    'Distractor Cue': COLOR_DISTRACTOR_YOUNG,  # Light Red (#EF5350)
    'Neutral Cue': COLOR_DIFF,                 # Black (#000000) for dashed baseline
}

data_dir = SPACECUE_online.get_concatenated_data_path()
out_dir = os.path.join(data_dir, 'final_pipeline')
fig_dir = os.path.join(out_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

LOG = []


def logprint(*args):
    """Print to console and append to persistent analysis log."""
    msg = ' '.join(str(a) for a in args)
    print(msg)
    LOG.append(msg)


# ============================================================
# 1. LOAD DATA & AUTOMATED FILENAME DATE CROSS-CHECK
# ============================================================
logprint('=' * 78)
logprint('   SPACECUE ONLINE - FINAL CONFIRMATORY ANALYSIS PIPELINE (3 x 4 DESIGN)')
logprint('=' * 78)

df_all = pd.read_csv(os.path.join(data_dir, 'concatenated_behavioral.csv'))

# Parse session date from source_file: sce-<sub_id>_block_<b_id>_data_<Month>_<DD>_<YYYY>_<HH>_<MM>_<SS>.csv
DATE_REGEX = re.compile(r"_data_([A-Za-z]+_\d{2}_\d{4})_")


def parse_file_date(fname):
    m = DATE_REGEX.search(str(fname))
    if not m:
        return pd.NaT
    return datetime.strptime(m.group(1), "%B_%d_%Y")


subj_dates = (df_all.groupby('subject_id')['source_file']
              .first().apply(parse_file_date).reset_index(name='session_date'))
cutoff_dt = datetime.strptime(FINAL_PARADIGM_START, "%Y-%m-%d")

newer_subjects = sorted(subj_dates.loc[subj_dates['session_date'] >= cutoff_dt, 'subject_id'].tolist())
unlisted_newer = sorted(set(newer_subjects) - set(FINAL_SUBJECTS))
listed_older = sorted(set(FINAL_SUBJECTS) - set(newer_subjects))

logprint(f"\nConfigured FINAL_SUBJECTS (n={len(FINAL_SUBJECTS)} / target N={TARGET_N}): {FINAL_SUBJECTS}")
logprint(f"Date cutoff check (>= {FINAL_PARADIGM_START}): matched subjects = {newer_subjects}")
if unlisted_newer:
    logprint(f"  WARNING: Subjects recorded on/after {FINAL_PARADIGM_START} NOT in FINAL_SUBJECTS: {unlisted_newer}")
if listed_older:
    logprint(f"  WARNING: Subjects in FINAL_SUBJECTS recorded BEFORE {FINAL_PARADIGM_START}: {listed_older}")
if not unlisted_newer and not listed_older:
    logprint("  Date cross-check PASSED: FINAL_SUBJECTS exactly matches all datasets >= cutoff date.\n")

df = df_all[df_all['subject_id'].isin(FINAL_SUBJECTS)].copy()

# ============================================================
# 2. DERIVE TRIAL-LEVEL VARIABLES & BLOCK-TYPE MAPPING
# ============================================================
df['Cue Type'] = df['CueInstruction'].map(CUE_MAP)
df['Informativeness'] = np.where(df['CueInstruction'] == 'cue_neutral', 'Neutral', 'Informative')
df['Delay'] = pd.to_numeric(df['cue_stim_delay_jitter'], errors='coerce')

# Each block pairs Neutral trials (50%) with exactly one Informative Cue Type (50%).
# Map every block to its Informative Cue Type ('Block Type') so neutral trials
# are strictly paired with the informative trials from the exact same block type.
block_type_Lookup = (df[df['CueInstruction'] != 'cue_neutral']
                     .groupby(['subject_id', 'block'])['Cue Type']
                     .first()
                     .rename('Block Type')
                     .reset_index())
df = df.merge(block_type_Lookup, on=['subject_id', 'block'], how='left')

# Dependent Variables (evaluated across ALL trials: correct + incorrect pooled)
df['responded'] = df['response'].notna()
df['IsCorrect'] = (df['responded'] & (df['response'] == df['TargetDigit']))
df['acc'] = df['IsCorrect'].astype(float)          # timeouts coded as 0.0 (incorrect)
df['rt_raw'] = pd.to_numeric(df['rt'], errors='coerce').where(df['responded'])

# ============================================================
# 3. PRE-SPECIFIED SUBJECT & TRIAL EXCLUSIONS
# ============================================================
logprint('=' * 78)
logprint('1. PRE-SPECIFIED SUBJECT & TRIAL EXCLUSION CHECKS')
logprint('=' * 78)
logprint(f"Criteria: (a) Overall accuracy > chance ({CHANCE_LEVEL:.4f}, one-sided binomial p < {ALPHA})")
logprint(f"          (b) No-response (timeout) rate <= {100 * MAX_NO_RESPONSE_RATE:.0f}%")
logprint(f"          (c) Trial-level anticipation filter: drop RT < {MIN_ANTICIPATION_RT:.2f} s (no SD trimming)\n")

excl_rows = []
for sub, d_sub in df.groupby('subject_id'):
    n_trials = len(d_sub)
    n_corr = int(d_sub['IsCorrect'].sum())
    acc_overall = n_corr / n_trials
    no_resp_rate = 1.0 - d_sub['responded'].mean()
    # One-sided binomial test against chance level 1/9
    binom_res = stats.binomtest(n_corr, n=n_trials, p=CHANCE_LEVEL, alternative='greater')
    p_binom = binom_res.pvalue
    n_anticip = int((d_sub['rt_raw'] < MIN_ANTICIPATION_RT).sum())

    pass_acc = p_binom < ALPHA
    pass_resp = no_resp_rate <= MAX_NO_RESPONSE_RATE
    include_sub = bool(pass_acc and pass_resp)
    excl_rows.append({
        'subject_id': int(sub),
        'n_trials': n_trials,
        'n_correct': n_corr,
        'acc_overall': acc_overall,
        'binom_p_vs_chance': p_binom,
        'no_response_rate': no_resp_rate,
        'n_anticipation_trials': n_anticip,
        'pass_accuracy': pass_acc,
        'pass_no_response': pass_resp,
        'included': include_sub,
    })

excl_df = pd.DataFrame(excl_rows)
excl_df.to_csv(os.path.join(out_dir, 'exclusions.csv'), index=False)
logprint(excl_df.to_string(index=False))

valid_subjects = excl_df.loc[excl_df['included'], 'subject_id'].tolist()
n_excluded_subs = len(FINAL_SUBJECTS) - len(valid_subjects)
logprint(f"\nIncluded subjects: {len(valid_subjects)}/{len(FINAL_SUBJECTS)} ({n_excluded_subs} excluded).")

df = df[df['subject_id'].isin(valid_subjects)].copy()

# Trial-level anticipation filter (< 0.20 s): exclude from RT, mark as invalid
n_anticip_total = int((df['rt_raw'] < MIN_ANTICIPATION_RT).sum())
df['rt_all'] = df['rt_raw'].where(df['rt_raw'] >= MIN_ANTICIPATION_RT)
logprint(f"Trial-level anticipation RTs (< {MIN_ANTICIPATION_RT:.2f} s) removed: {n_anticip_total} trials.\n")

# ============================================================
# 4. FACTORIAL DESIGN & CELL COUNT VERIFICATION
# ============================================================
logprint('=' * 78)
logprint('2. FACTORIAL DESIGN VERIFICATION (3 Block Types x 2 Informativeness x 4 Delays)')
logprint('=' * 78)

cell_counts = (df.groupby(['subject_id', 'Block Type', 'Informativeness', 'Delay'])
               .agg(n_trials=('acc', 'size'),
                    n_responded=('rt_all', 'count'))
               .reset_index())
cell_counts.to_csv(os.path.join(out_dir, 'design_cell_counts.csv'), index=False)

min_trials = int(cell_counts['n_trials'].min())
max_trials = int(cell_counts['n_trials'].max())
mean_trials = cell_counts['n_trials'].mean()
min_resp = int(cell_counts['n_responded'].min())
logprint(f"Total cells per subject: 24 (3 Block Types x 2 Informativeness x 4 Delays)")
logprint(f"Trials per cell across subjects: mean = {mean_trials:.1f}, min = {min_trials}, max = {max_trials}")
logprint(f"Responded trials per cell (for RT): min = {min_resp}, max = {int(cell_counts['n_responded'].max())}")
if (cell_counts['n_trials'] < 10).any():
    low_cells = cell_counts[cell_counts['n_trials'] < 10]
    logprint(f"  WARNING: {len(low_cells)} cells have < 10 total trials.")
if (cell_counts['n_responded'] == 0).any():
    empty_rt = cell_counts[cell_counts['n_responded'] == 0]
    logprint(f"  WARNING: {len(empty_rt)} cells have 0 responded trials (undefined RT cell mean).")
logprint('')

# ============================================================
# 5. SUBJECT-LEVEL CELL MEANS (24 RAW CELLS & 12 DELTA CELLS)
# ============================================================
# Aggregate across the 2 blocks of each Block Type per subject x delay
raw_cells = (df.groupby(['subject_id', 'Block Type', 'Informativeness', 'Delay'])
             .agg(acc=('acc', 'mean'),
                  rt=('rt_all', 'mean'),
                  n_trials=('acc', 'size'),
                  n_responded=('rt_all', 'count'))
             .reset_index())
raw_cells['Block Type'] = pd.Categorical(raw_cells['Block Type'], categories=INFORMATIVE_ORDER, ordered=True)
raw_cells = raw_cells.sort_values(['subject_id', 'Block Type', 'Informativeness', 'Delay']).reset_index(drop=True)
raw_cells.to_csv(os.path.join(out_dir, 'cell_means_raw_3x2x4.csv'), index=False)

# Compute 3 x 4 Difference Scores: Delta = Informative - Neutral (matched by Block Type and Delay)
inf_cells = (raw_cells[raw_cells['Informativeness'] == 'Informative']
             .rename(columns={'acc': 'acc_inf', 'rt': 'rt_inf',
                              'n_trials': 'n_trials_inf', 'n_responded': 'n_resp_inf'})
             .drop(columns=['Informativeness']))
neu_cells = (raw_cells[raw_cells['Informativeness'] == 'Neutral']
             .rename(columns={'acc': 'acc_neu', 'rt': 'rt_neu',
                              'n_trials': 'n_trials_neu', 'n_responded': 'n_resp_neu'})
             .drop(columns=['Informativeness']))

delta_cells = inf_cells.merge(neu_cells, on=['subject_id', 'Block Type', 'Delay'], how='left')
delta_cells['Cue Type'] = delta_cells['Block Type']
delta_cells['d_acc'] = delta_cells['acc_inf'] - delta_cells['acc_neu']
delta_cells['d_rt'] = delta_cells['rt_inf'] - delta_cells['rt_neu']
delta_cells.to_csv(os.path.join(out_dir, 'cell_means_delta_3x4.csv'), index=False)

# Delay-averaged cue effects per subject (for planned intercept tests & overview)
subj_cue_delta = (delta_cells.groupby(['subject_id', 'Cue Type'], observed=True)
                  .agg(d_acc=('d_acc', 'mean'),
                       d_rt=('d_rt', 'mean'),
                       acc_inf=('acc_inf', 'mean'),
                       acc_neu=('acc_neu', 'mean'),
                       rt_inf=('rt_inf', 'mean'),
                       rt_neu=('rt_neu', 'mean'))
                  .reset_index())
subj_cue_delta.to_csv(os.path.join(out_dir, 'subject_cue_effects_averaged_over_delay.csv'), index=False)

logprint('=' * 78)
logprint('3. DESCRIPTIVE OVERVIEW: DELAY-AVERAGED CUE EFFECTS PER SUBJECT')
logprint('=' * 78)
logprint(subj_cue_delta.pivot_table(index='subject_id', columns='Cue Type',
                                    values=['d_acc', 'd_rt'], observed=True).round(4).to_string())
logprint('\nCell-level Delta means across subjects (Cue Type x Delay):')
cell_summary = (delta_cells.groupby(['Cue Type', 'Delay'], observed=True)
                .agg(d_acc_mean=('d_acc', 'mean'), d_acc_sd=('d_acc', 'std'),
                     d_rt_mean=('d_rt', 'mean'), d_rt_sd=('d_rt', 'std'),
                     acc_inf_mean=('acc_inf', 'mean'), acc_neu_mean=('acc_neu', 'mean'),
                     rt_inf_mean=('rt_inf', 'mean'), rt_neu_mean=('rt_neu', 'mean'))
                .reset_index())
logprint(cell_summary.round(4).to_string(index=False))
logprint('')


# ============================================================
# 6. STATISTICAL INFERENCE ENGINE (3 x 4 rmANOVA + PAIRED t-TESTS)
# ============================================================
def run_confirmatory_pipeline(delta_df, raw_df, dv_delta, dv_raw, dv_label, unit_str):
    """
    Execute the complete pre-specified statistical pipeline for one DV:
      1. Cell-wise Shapiro-Wilk normality checks on the 12 Delta cells.
      2. Primary 3 (Cue Type) x 4 (Delay) repeated-measures ANOVA on Delta
         (`pingouin.rm_anova` with Mauchly's sphericity test & GG correction).
      3. Planned intercept tests: one-sample t-tests of delay-averaged Delta vs. 0
         per Cue Type (equivalent to paired t-test Informative vs. Neutral),
         with Holm-Bonferroni correction, Cohen's d_z, and achieved power.
      4. Post-hoc paired t-tests (`pingouin.pairwise_tests`, Holm-corrected):
         - Marginal paired t-tests across Cue Types (collapsed over Delay)
         - Marginal paired t-tests across Delays (collapsed over Cue Type)
         - Simple paired t-tests of Cue Types at each Delay level
         - Simple paired t-tests of Delays within each Cue Type
      5. Supplementary full 3 x 2 x 4 rmANOVA decomposition & AnovaRM verification.
    """
    logprint('=' * 78)
    logprint(f"STATISTICAL INFERENCE: {dv_label.upper()} [{unit_str}]")
    logprint('=' * 78)

    # Identify subjects with complete data across all 12 cells for this DV
    sub_valid = (delta_df.groupby('subject_id')[dv_delta]
                 .apply(lambda s: s.notna().all()))
    complete_subs = sub_valid[sub_valid].index.tolist()
    n_sub = len(complete_subs)
    d_clean = delta_df[delta_df['subject_id'].isin(complete_subs)].copy()
    r_clean = raw_df[raw_df['subject_id'].isin(complete_subs)].copy()

    logprint(f"Complete-case subjects for {dv_label}: n = {n_sub} (target N = {TARGET_N})")

    if n_sub < 3:
        logprint(f"  [INFO] Sample size n = {n_sub} < 3: Repeated-measures ANOVA, Mauchly's sphericity,")
        logprint("         Shapiro-Wilk normality, and group-level paired t-tests require n >= 3.")
        logprint("         Skipping group inference for current sample; descriptive cell tables and")
        logprint("         figures are generated. Full inferential output activates automatically at n >= 3.\n")
        return {
            'normality': pd.DataFrame(),
            'anova_3x4': pd.DataFrame(),
            'planned_vs_zero': pd.DataFrame(),
            'posthoc_marginal': pd.DataFrame(),
            'posthoc_simple_cue_at_delay': pd.DataFrame(),
            'posthoc_simple_delay_in_cue': pd.DataFrame(),
            'anova_3x2x4': pd.DataFrame(),
        }

    # --- 6.1 Assumption Check: Shapiro-Wilk Normality per Cell ----------------
    logprint('\n--- A. Assumption Check: Shapiro-Wilk Normality Test per Cell (3 x 4 Delta) ---')
    norm_rows = []
    for (cue, delay), grp in d_clean.groupby(['Cue Type', 'Delay'], observed=True):
        vals = grp[dv_delta].dropna().values
        if np.ptp(vals) == 0:
            W, p_sw = np.nan, np.nan
        else:
            W, p_sw = stats.shapiro(vals)
        norm_rows.append({
            'dv': dv_label, 'Cue Type': cue, 'Delay': delay, 'n': len(vals),
            'mean': vals.mean(), 'sd': vals.std(ddof=1),
            'shapiro_W': W, 'shapiro_p': p_sw
        })
    norm_df = pd.DataFrame(norm_rows)
    n_viol = int((norm_df['shapiro_p'] < ALPHA).sum())
    logprint(norm_df.to_string(index=False))
    logprint(f"Shapiro-Wilk summary: {n_viol}/12 cells with p < {ALPHA}.\n")

    # --- 6.2 Primary 3 x 4 Repeated-Measures ANOVA on Delta -------------------
    logprint('--- B. Primary 3 (Cue Type) x 4 (Delay) Repeated-Measures ANOVA on Delta ---')
    aov_3x4 = pg.rm_anova(
        data=d_clean, dv=dv_delta, within=['Cue Type', 'Delay'],
        subject='subject_id', correction=True, detailed=True, effsize='np2'
    )
    aov_3x4.insert(0, 'dv', dv_label)
    for _, row in aov_3x4.iterrows():
        src = row['Source']
        f_val = row['F']
        df1, df2 = int(row['ddof1']), int(row['ddof2'])
        p_unc = row['p-unc']
        np2 = row['np2']
        eps = row.get('eps', np.nan)
        p_gg = row.get('p-GG-corr', np.nan)
        w_mauch = row.get('W-spher', np.nan)
        p_mauch = row.get('p-spher', np.nan)
        logprint(f"  {src:<20s} | F({df1}, {df2}) = {f_val:.4f}, p_unc = {p_unc:.6f}, "
                 f"p_GG = {p_gg:.6f} (eps = {eps:.3f}), partial_eta2 = {np2:.4f} "
                 f"[Mauchly W = {w_mauch:.3f}, p = {p_mauch:.4f}]")

    # Grand intercept test on Delta (equivalent to Main Effect of Informativeness)
    sub_grand = d_clean.groupby('subject_id')[dv_delta].mean().values
    w_int, p_sw_int = stats.shapiro(sub_grand) if np.ptp(sub_grand) > 0 else (np.nan, np.nan)
    t_int = pg.ttest(sub_grand, 0.0)
    t_val_int = float(t_int['T'].iloc[0])
    p_val_int = float(t_int['p-val'].iloc[0])
    df_int = n_sub - 1
    dz_int = sub_grand.mean() / sub_grand.std(ddof=1)
    np2_int = (t_val_int ** 2) / (t_val_int ** 2 + df_int)
    logprint(f"  {'Intercept (Delta!=0)':<20s} | F(1, {df_int}) = {t_val_int**2:.4f} "
             f"(t = {t_val_int:+.4f}), p = {p_val_int:.6f}, partial_eta2 = {np2_int:.4f}, "
             f"d_z = {dz_int:+.3f} [Shapiro W = {w_int:.3f}, p = {p_sw_int:.4f}]\n")

    # --- 6.3 Planned Contrasts: Informative vs. Neutral per Cue Type ----------
    logprint('--- C. Planned Contrasts: Paired t-Tests Informative vs. Neutral per Cue Type (Delay-Averaged) ---')
    sub_cue = (d_clean.groupby(['subject_id', 'Cue Type'], observed=True)[dv_delta]
               .mean().reset_index())
    planned_rows = []
    raw_pvals = []
    for cue in INFORMATIVE_ORDER:
        x = sub_cue.loc[sub_cue['Cue Type'] == cue, dv_delta].values
        w_c, p_sw_c = stats.shapiro(x) if np.ptp(x) > 0 else (np.nan, np.nan)
        t_res = pg.ttest(x, 0.0)
        t_val = float(t_res['T'].iloc[0])
        p_val = float(t_res['p-val'].iloc[0])
        ci95 = t_res['CI95%'].iloc[0]
        dz = x.mean() / x.std(ddof=1)
        achieved_pwr = pg.power_ttest(d= abs(dz), n=n_sub, alpha=ALPHA, contrast='one-sample')
        raw_pvals.append(p_val)
        planned_rows.append({
            'dv': dv_label, 'Cue Type': cue, 'n': n_sub,
            'mean_delta': x.mean(), 'sd_delta': x.std(ddof=1),
            'sem_delta': stats.sem(x), 'ci95_low': ci95[0], 'ci95_high': ci95[1],
            't': t_val, 'df': n_sub - 1, 'p_unc': p_val,
            'cohens_dz': dz, 'achieved_power_alpha05': achieved_pwr,
            'shapiro_W': w_c, 'shapiro_p': p_sw_c
        })
    _, p_holm = pg.multicomp(raw_pvals, alpha=ALPHA, method='holm')
    for r, ph in zip(planned_rows, p_holm):
        r['p_holm'] = float(ph)
        logprint(f"  {r['Cue Type']:<15s} | mean Delta = {r['mean_delta']:+.4f} {unit_str} "
                 f"(SD = {r['sd_delta']:.4f}, 95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]), "
                 f"t({r['df']}) = {r['t']:+.4f}, p_unc = {r['p_unc']:.6f}, p_holm = {r['p_holm']:.6f}, "
                 f"d_z = {r['cohens_dz']:+.3f} [Shapiro W = {r['shapiro_W']:.3f}, p = {r['shapiro_p']:.4f}]")
    planned_df = pd.DataFrame(planned_rows)

    # --- 6.4 Post-Hoc Paired t-Tests (Main Effects & Interaction Simple Tests) -
    logprint('\n--- D. Post-Hoc Paired t-Tests (Marginal & Interaction Simple Effects, Holm-Corrected) ---')

    def compute_dz_paired(df_long, factor, a, b, val_col):
        piv = df_long.pivot_table(index='subject_id', columns=factor, values=val_col, observed=True)
        diff = (piv[a] - piv[b]).dropna().values
        return float(diff.mean() / diff.std(ddof=1))

    # 1. Marginal paired t-tests: Cue Type (collapsed over Delay) and Delay (collapsed over Cue Type)
    ph_marginal = pg.pairwise_tests(
        data=d_clean, dv=dv_delta, within=['Cue Type', 'Delay'],
        subject='subject_id', parametric=True, padjust='holm', effsize='hedges', marginal=True
    )
    ph_marginal.insert(0, 'dv', dv_label)
    # Add exact paired Cohen's d_z for transparency
    dz_list = []
    for _, row in ph_marginal.iterrows():
        fac = row['Contrast']
        if fac in ['Cue Type', 'Delay']:
            dz_list.append(compute_dz_paired(d_clean, fac, row['A'], row['B'], dv_delta))
        else:
            dz_list.append(np.nan)
    ph_marginal['cohens_dz'] = dz_list

    logprint('1) Marginal Paired t-Tests (Main Effects of Cue Type and Delay):')
    for _, row in ph_marginal[ph_marginal['Contrast'].isin(['Cue Type', 'Delay'])].iterrows():
        logprint(f"  [{row['Contrast']}] {str(row['A']):<15s} vs. {str(row['B']):<15s} | "
                 f"t({int(row['dof'])}) = {row['T']:+.4f}, p_unc = {row['p-unc']:.6f}, "
                 f"p_holm = {row.get('p-corr', row['p-unc']):.6f}, d_z = {row['cohens_dz']:+.3f}, "
                 f"hedges_g = {row['hedges']:+.3f}")

    # 2. Interaction simple paired t-tests: Cue Type comparisons within each Delay
    simple_cue_rows = []
    for delay_val, grp in d_clean.groupby('Delay', observed=True):
        pw = pg.pairwise_tests(
            data=grp, dv=dv_delta, within='Cue Type', subject='subject_id',
            parametric=True, padjust='holm', effsize='hedges'
        )
        pw.insert(0, 'Delay', delay_val)
        pw.insert(0, 'dv', dv_label)
        pw['cohens_dz'] = [compute_dz_paired(grp, 'Cue Type', r['A'], r['B'], dv_delta)
                           for _, r in pw.iterrows()]
        simple_cue_rows.append(pw)
    ph_simple_cue = pd.concat(simple_cue_rows, ignore_index=True)
    # Also compute Holm correction across all 12 (4 delays x 3 pairs) simple comparisons
    _, ph_simple_cue['p_holm_all12'] = pg.multicomp(ph_simple_cue['p-unc'].values, alpha=ALPHA, method='holm')

    logprint('\n2) Simple Paired t-Tests: Cue Type Comparisons at Each Delay Interval:')
    for _, row in ph_simple_cue.iterrows():
        logprint(f"  Delay {row['Delay']:.1f} s | {row['A']:<15s} vs. {row['B']:<15s} | "
                 f"t({int(row['dof'])}) = {row['T']:+.4f}, p_unc = {row['p-unc']:.6f}, "
                 f"p_holm(within-delay) = {row['p-corr']:.6f}, p_holm(all-12) = {row['p_holm_all12']:.6f}, "
                 f"d_z = {row['cohens_dz']:+.3f}")

    # 3. Interaction simple paired t-tests: Delay comparisons within each Cue Type
    simple_delay_rows = []
    for cue_val, grp in d_clean.groupby('Cue Type', observed=True):
        pw = pg.pairwise_tests(
            data=grp, dv=dv_delta, within='Delay', subject='subject_id',
            parametric=True, padjust='holm', effsize='hedges'
        )
        pw.insert(0, 'Cue Type', cue_val)
        pw.insert(0, 'dv', dv_label)
        pw['cohens_dz'] = [compute_dz_paired(grp, 'Delay', r['A'], r['B'], dv_delta)
                           for _, r in pw.iterrows()]
        simple_delay_rows.append(pw)
    ph_simple_delay = pd.concat(simple_delay_rows, ignore_index=True)
    _, ph_simple_delay['p_holm_all18'] = pg.multicomp(ph_simple_delay['p-unc'].values, alpha=ALPHA, method='holm')

    logprint('\n3) Simple Paired t-Tests: Delay Comparisons Within Each Cue Type:')
    for _, row in ph_simple_delay.iterrows():
        logprint(f"  {row['Cue Type']:<15s} | {row['A']:.1f} s vs. {row['B']:.1f} s | "
                 f"t({int(row['dof'])}) = {row['T']:+.4f}, p_unc = {row['p-unc']:.6f}, "
                 f"p_holm(within-cue) = {row['p-corr']:.6f}, p_holm(all-18) = {row['p_holm_all18']:.6f}, "
                 f"d_z = {row['cohens_dz']:+.3f}")

    # --- 6.5 Supplementary Full 3 x 2 x 4 rmANOVA & Mathematical Verification -
    logprint('\n--- E. Supplementary Full 3 (Block Type) x 2 (Informativeness) x 4 (Delay) rmANOVA ---')
    # Compute the 3 cue-independent terms (Block Type, Delay, Block Type x Delay)
    # on informative+neutral pooled means using pingouin (provides Mauchly & GG),
    # and cross-check all 7 terms against statsmodels AnovaRM.
    aov_bd = pg.rm_anova(
        data=r_clean, dv=dv_raw, within=['Block Type', 'Delay'],
        subject='subject_id', correction=True, detailed=True, effsize='np2'
    )
    # Cross-check with statsmodels AnovaRM on the full 3 x 2 x 4 table
    # (patsy requires valid Python identifiers without spaces for factor names)
    r_arm = r_clean.rename(columns={'Block Type': 'Block_Type'})
    arm = AnovaRM(data=r_arm, depvar=dv_raw, subject='subject_id',
                  within=['Block_Type', 'Informativeness', 'Delay']).fit()
    arm_tbl = arm.anova_table.reset_index().rename(columns={'index': 'Source', 'F Value': 'F_AnovaRM',
                                                            'Num DF': 'ddof1', 'Den DF': 'ddof2',
                                                            'Pr > F': 'p_unc_AnovaRM'})
    arm_tbl['Source'] = arm_tbl['Source'].str.replace('Block_Type', 'Block Type', regex=False)
    arm_tbl.insert(0, 'dv', dv_label)
    arm_tbl['partial_eta2'] = (arm_tbl['F_AnovaRM'] * arm_tbl['ddof1']) / (
        arm_tbl['F_AnovaRM'] * arm_tbl['ddof1'] + arm_tbl['ddof2']
    )

    # Verify exact mathematical equivalence between 3 x 4 on Delta and I-terms of 3 x 2 x 4
    f_map_arm = dict(zip(arm_tbl['Source'], arm_tbl['F_AnovaRM']))
    f_map_3x4 = dict(zip(aov_3x4['Source'], aov_3x4['F']))
    assert np.isclose(t_val_int ** 2, f_map_arm['Informativeness'], rtol=1e-5), "Equivalence check failed: I"
    assert np.isclose(f_map_3x4['Cue Type'], f_map_arm['Block Type:Informativeness'], rtol=1e-5), "Equivalence failed: B x I"
    assert np.isclose(f_map_3x4['Delay'], f_map_arm['Informativeness:Delay'], rtol=1e-5), "Equivalence failed: I x D"
    assert np.isclose(f_map_3x4['Cue Type * Delay'], f_map_arm['Block Type:Informativeness:Delay'], rtol=1e-5), "Equivalence failed: B x I x D"

    for _, row in arm_tbl.iterrows():
        logprint(f"  {row['Source']:<34s} | F({int(row['ddof1'])}, {int(row['ddof2'])}) = "
                 f"{row['F_AnovaRM']:.4f}, p_unc = {row['p_unc_AnovaRM']:.6f}, "
                 f"partial_eta2 = {row['partial_eta2']:.4f}")
    logprint("  [VERIFIED] All 4 Informativeness terms in 3x2x4 match the 3x4 Delta rmANOVA exactly.\n")

    return {
        'normality': norm_df,
        'anova_3x4': aov_3x4,
        'planned_vs_zero': planned_df,
        'posthoc_marginal': ph_marginal,
        'posthoc_simple_cue_at_delay': ph_simple_cue,
        'posthoc_simple_delay_in_cue': ph_simple_delay,
        'anova_3x2x4': arm_tbl,
    }


res_acc = run_confirmatory_pipeline(
    delta_cells, raw_cells, dv_delta='d_acc', dv_raw='acc',
    dv_label='Accuracy (All Trials)', unit_str='proportion'
)
res_rt = run_confirmatory_pipeline(
    delta_cells, raw_cells, dv_delta='d_rt', dv_raw='rt',
    dv_label='Response Time (All Responded Trials)', unit_str='s'
)

# Save all inferential tables if populated (n >= 3)
for key in ['normality', 'anova_3x4', 'planned_vs_zero', 'posthoc_marginal',
            'posthoc_simple_cue_at_delay', 'posthoc_simple_delay_in_cue', 'anova_3x2x4']:
    combined = pd.concat([res_acc[key], res_rt[key]], ignore_index=True)
    if not combined.empty:
        combined.to_csv(os.path.join(out_dir, f'stats_{key}.csv'), index=False)


# ============================================================
# 7. PUBLICATION-QUALITY FIGURES (WITH SEM SHADED ERROR BANDS)
# ============================================================
plt.close('all')


def plot_trajectory_with_sem(ax, df_sub, x_col, y_col, group_col, order, palette, linestyle='-'):
    """Plot group mean trajectory across delays with a semi-transparent SEM band (or SD if n=2)."""
    for grp in order:
        d_g = df_sub[df_sub[group_col] == grp]
        if d_g.empty:
            continue
        stats_g = (d_g.groupby(x_col, observed=True)[y_col]
                   .agg(mean='mean', sem=lambda s: stats.sem(s, nan_policy='omit') if len(s.dropna()) > 1 else 0.0)
                   .reindex(DELAYS)
                   .reset_index())
        x = stats_g[x_col].values.astype(float)
        y = stats_g['mean'].values.astype(float)
        err = stats_g['sem'].values.astype(float)
        color = palette[grp]
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=7, color=color,
                linestyle=linestyle, label=grp)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.3)


# --- Figure 1: Primary 3 x 4 Cueing Effects (Delta) across Delay Intervals ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle(f'Cue Effects Across Cue–Stimulus Delay Intervals (3 × 4 Design, N = {len(valid_subjects)})',
             fontsize=16, fontweight='bold')

plot_trajectory_with_sem(axes[0], delta_cells, 'Delay', 'd_acc', 'Cue Type', INFORMATIVE_ORDER, CUE_PALETTE)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1.2)
axes[0].set_xlabel('Cue–Stimulus Delay (s)')
axes[0].set_ylabel('Δ Accuracy (Proportion Correct)')
axes[0].set_title('Accuracy Effect (Informative − Neutral)')
axes[0].set_xticks(DELAYS)
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].legend(title='Informative Cue Type', frameon=False)

plot_trajectory_with_sem(axes[1], delta_cells, 'Delay', 'd_rt', 'Cue Type', INFORMATIVE_ORDER, CUE_PALETTE)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1.2)
axes[1].set_xlabel('Cue–Stimulus Delay (s)')
axes[1].set_ylabel('Δ Response Time (s)')
axes[1].set_title('Response Time Effect (Informative − Neutral)')
axes[1].set_xticks(DELAYS)
axes[1].spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig1_3x4_delta_by_delay.png'), dpi=200)
plt.close('all')

# --- Figure 2: Absolute Performance (Informative vs. Matched Neutral) ---------
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle(f'Absolute Performance Across Delay Intervals (N = {len(valid_subjects)})',
             fontsize=16, fontweight='bold')

raw_inf = raw_cells[raw_cells['Informativeness'] == 'Informative'].copy()
raw_neu_pooled = (raw_cells[raw_cells['Informativeness'] == 'Neutral']
                  .groupby(['subject_id', 'Delay'], observed=True)[['acc', 'rt']]
                  .mean().reset_index())
raw_neu_pooled['Block Type'] = 'Neutral Cue'

plot_trajectory_with_sem(axes[0], raw_inf, 'Delay', 'acc', 'Block Type', INFORMATIVE_ORDER, CUE_PALETTE)
plot_trajectory_with_sem(axes[0], raw_neu_pooled, 'Delay', 'acc', 'Block Type', ['Neutral Cue'],
                         CUE_PALETTE, linestyle='--')
axes[0].axhline(CHANCE_LEVEL, color='black', linestyle=':', linewidth=1.0)
axes[0].set_xlabel('Cue–Stimulus Delay (s)')
axes[0].set_ylabel('Accuracy (Proportion Correct)')
axes[0].set_title('Absolute Accuracy (All Trials)')
axes[0].set_xticks(DELAYS)
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].legend(title='Cue Condition', frameon=False)

plot_trajectory_with_sem(axes[1], raw_inf, 'Delay', 'rt', 'Block Type', INFORMATIVE_ORDER, CUE_PALETTE)
plot_trajectory_with_sem(axes[1], raw_neu_pooled, 'Delay', 'rt', 'Block Type', ['Neutral Cue'],
                         CUE_PALETTE, linestyle='--')
axes[1].set_xlabel('Cue–Stimulus Delay (s)')
axes[1].set_ylabel('Response Time (s)')
axes[1].set_title('Absolute Response Time (All Responded Trials)')
axes[1].set_xticks(DELAYS)
axes[1].spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig2_absolute_performance_by_delay.png'), dpi=200)
plt.close('all')

# --- Figure 3: Delay-Averaged Cue Effects (Planned Contrasts Overview) --------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Delay-Averaged Cue Effects per Cue Type (N = {len(valid_subjects)})',
             fontsize=16, fontweight='bold')

for ax, dv, ylab, title in [
    (axes[0], 'd_acc', 'Δ Accuracy (Proportion Correct)', 'Accuracy Effect (Informative − Neutral)'),
    (axes[1], 'd_rt', 'Δ Response Time (s)', 'Response Time Effect (Informative − Neutral)')
]:
    sns.barplot(data=subj_cue_delta, x='Cue Type', y=dv, order=INFORMATIVE_ORDER,
                palette=CUE_PALETTE, errorbar=('se', 1), alpha=0.65, ax=ax)
    # Connect each subject's 3 cue-type values to visualize within-subject paired structure
    for sub_id, d_s in subj_cue_delta.groupby('subject_id'):
        d_s_ord = d_s.set_index('Cue Type').reindex(INFORMATIVE_ORDER)
        ax.plot(range(len(INFORMATIVE_ORDER)), d_s_ord[dv].values,
                color='black', marker='o', markersize=6, linewidth=1.2, alpha=0.7,
                label=f'Sub {sub_id}')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel('')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)

if len(valid_subjects) <= 10:
    axes[0].legend(frameon=False, title='Subject')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig3_delay_averaged_cue_effects.png'), dpi=200)
plt.close('all')

# --- Figure 4: Trial Count Heatmap per Cell (Design Verification) -------------
mean_counts = (cell_counts.groupby(['Block Type', 'Informativeness', 'Delay'], observed=True)['n_trials']
               .mean().reset_index())
cond_order = [f"{cue} ({inf})" for cue in INFORMATIVE_ORDER for inf in ['Informative', 'Neutral']]
mean_counts['Condition'] = mean_counts['Block Type'].astype(str) + ' (' + mean_counts['Informativeness'] + ')'
piv_counts = (mean_counts.pivot(index='Condition', columns='Delay', values='n_trials')
              .reindex(index=cond_order, columns=DELAYS))

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(piv_counts, annot=False, cmap='viridis', cbar_kws={'label': 'Mean Trial Count'}, ax=ax)
# Manually annotate every cell to bypass the matplotlib 3.8 + seaborn 0.12 annotation bug
val_mid = 0.5 * (np.nanmin(piv_counts.values) + np.nanmax(piv_counts.values))
for r_idx in range(piv_counts.shape[0]):
    for c_idx in range(piv_counts.shape[1]):
        val = piv_counts.iloc[r_idx, c_idx]
        if pd.notna(val):
            txt_color = 'black' if val > val_mid else 'white'
            ax.text(c_idx + 0.5, r_idx + 0.5, f"{val:.1f}",
                    ha='center', va='center', color=txt_color, fontsize=14, fontweight='bold')
ax.set_title(f'Mean Trial Count per Design Cell (N = {len(valid_subjects)})')
ax.set_xlabel('Cue–Stimulus Delay (s)')
ax.set_ylabel('Block Type (Informativeness)')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig4_cell_counts_heatmap.png'), dpi=200)
plt.close('all')

# ============================================================
# 8. SAVE LOG
# ============================================================
log_file = os.path.join(out_dir, 'final_pipeline_log.txt')
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG))
logprint(f"Outputs and figures saved to: {out_dir}")
