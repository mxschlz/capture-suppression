"""
Cue effects on accuracy and response time in SPACECUE_online.

Design logic
------------
The experiment uses a BLOCK-wise cueing design (CueDesignStrategy == 'Block'):
each block contains ~50% neutral (non-informative) cues and ~50% cues of exactly
ONE informative type (target / distractor / control[nonsingleton]). Therefore,
cue effects are evaluated WITHIN blocks (informative - neutral of the same
block) and then averaged per subject.

Cohorts
-------
- subjects 01-13, 15 : 4 blocks, 120 trials, NO target cues (only distractor /
                       control cues) - old instruction
- subjects 21-29, 31 : 6 blocks, 80 trials, all three informative cue types
                       (target / distractor / control) - old instruction
- subjects 39, 49     : same 6-block design - NEW instruction (sanity check:
                       target cues must benefit performance). Subject 39 is
                       the first dataset saved to the new DataPipe location.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE_online
from SPACECUE.colors import COLOR_TARGET_YOUNG, COLOR_CONTROL_YOUNG, COLOR_DISTRACTOR_YOUNG, COLOR_DIFF

sns.set_theme(context="talk", style="ticks")
plt.ion()

# ============================================================
# PARAMETERS
# ============================================================
NEW_INSTRUCTION_SUBJECTS = SPACECUE_online.FINAL_SUBJECTS  # collected after the instruction change
NEW_LABEL = 'new instruction (subjects ' + ', '.join(str(s) for s in NEW_INSTRUCTION_SUBJECTS) + ')'
OLD_4BLOCK_SUBJECTS = list(range(1, 14)) + [15]   # 4-block design without target cues

CUE_MAP = {
    'cue_target_location': 'Target Cue',
    'cue_distractor_location': 'Distractor Cue',
    'cue_nonsingleton_location': 'Control Cue',
    'cue_neutral': 'Neutral Cue',
}
CUE_ORDER = ['Target Cue', 'Control Cue', 'Distractor Cue', 'Neutral Cue']
INFORMATIVE_ORDER = ['Target Cue', 'Control Cue', 'Distractor Cue']
CUE_PALETTE = {'Target Cue': COLOR_TARGET_YOUNG, 'Control Cue': COLOR_CONTROL_YOUNG,
               'Distractor Cue': COLOR_DISTRACTOR_YOUNG, 'Neutral Cue': COLOR_DIFF}
COHORT_PALETTE = {'old instruction': '#56B4E9', 'new instruction': '#D55E00'}
RESPONSE_PALETTE = {'Correct': '#0173B2', 'Distractor Intrusion': '#DE8F05',
                    'Control Intrusion': '#029E73', 'Other Digit': '#CC78BC',
                    'No Response': '#949494'}

LOG = []          # collected console output, written to a text file at the end


def logprint(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg)
    LOG.append(msg)


# ============================================================
# 1. LOAD & DERIVE TRIAL-LEVEL VARIABLES
# ============================================================
data_dir = SPACECUE_online.get_concatenated_data_path()
fig_dir = os.path.join(data_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(os.path.join(data_dir, 'concatenated_behavioral.csv'))
df['Cue Type'] = df['CueInstruction'].map(CUE_MAP)
df['instruction'] = np.where(df['subject_id'].isin(NEW_INSTRUCTION_SUBJECTS),
                             'new instruction', 'old instruction')
df['design'] = np.where(df['subject_id'].isin(OLD_4BLOCK_SUBJECTS),
                        '4-block', '6-block')

# --- Accuracy: the reported digit has to match the target digit -------------
# Missing responses (timeouts) count as incorrect.
df['responded'] = df['response'].notna()
df['IsCorrect'] = (df['responded'] & (df['response'] == df['TargetDigit']))
df['IsCorrect_f'] = df['IsCorrect'].astype(float)
# RT is only meaningful for responded trials; for RT analyses we additionally
# restrict to correct trials (see rt_valid below).
df['rt'] = df['rt'].where(df['responded'])
df['rt_valid'] = df['rt'].where(df['IsCorrect'])

# --- RT time-lock sanity check ----------------------------------------------
# In SPACECUE_explicit, RT was cue-locked and the cue-stimulus jitter had to be
# subtracted. Here, min(RT)=0.32 s is SHORTER than the minimum jitter (0.5 s),
# i.e. RT is already stimulus/response-locked and must NOT be corrected.
frac_would_be_negative = (df['rt'] < df['cue_stim_delay_jitter']).mean()
logprint(f"RT time-lock check: {100 * frac_would_be_negative:.1f}% of RTs are shorter "
         f"than the cue-stimulus jitter -> RT is stimulus-locked, no jitter correction applied.\n")

# --- Nonsingleton digits (a '0' means: no item at that slot) ----------------
ns_digits = df[['Non-Singleton1Digit', 'Non-Singleton2Digit']].apply(
    lambda r: tuple(d for d in r if d > 0), axis=1)


def classify_response(row, ns_digits):
    """Classify the reported digit to localize the source of errors."""
    if not row['responded']:
        return 'No Response'
    if row['response'] == row['TargetDigit']:
        return 'Correct'
    if row['response'] == row['SingletonDigit']:
        return 'Distractor Intrusion'
    if row['response'] in ns_digits:
        return 'Control Intrusion'
    return 'Other Digit'


df['response_type'] = [classify_response(r, ns) for r, ns in zip(df.to_dict('records'), ns_digits)]


# --- Digit at the cued location (cue-following diagnostic) ------------------
def cued_digit(row):
    """Digit of the item the (informative) cue points to."""
    if row['CueInstruction'] == 'cue_target_location':
        return row['TargetDigit']
    if row['CueInstruction'] == 'cue_distractor_location':
        return row['SingletonDigit']
    if row['CueInstruction'] == 'cue_nonsingleton_location':
        return row['Non-Singleton1Digit'] if row['Non-Singleton1Digit'] > 0 else row['Non-Singleton2Digit']
    return np.nan  # neutral cue -> no referent


df['cued_digit'] = df.apply(cued_digit, axis=1)
df['cue_followed'] = df['responded'] & (df['response'] == df['cued_digit'])

# ============================================================
# 2. BLOCK-LEVEL CONDITION MEANS (the unit of evaluation)
# ============================================================
sbc = (df.groupby(['subject_id', 'instruction', 'design', 'block', 'Cue Type'])
       .agg(acc=('IsCorrect_f', 'mean'),
            rt=('rt_valid', 'mean'),
            n=('trial_nr', 'size'))
       .reset_index())

# merge the neutral baseline of the SAME block (block-wise cueing design!)
neu = (sbc[sbc['Cue Type'] == 'Neutral Cue']
       .rename(columns={'acc': 'acc_neutral', 'rt': 'rt_neutral'})
       [['subject_id', 'block', 'acc_neutral', 'rt_neutral']])
sbc = sbc.merge(neu, on=['subject_id', 'block'], how='left')
sbc['d_acc'] = sbc['acc'] - sbc['acc_neutral']
sbc['d_rt'] = sbc['rt'] - sbc['rt_neutral']
sbc.to_csv(os.path.join(data_dir, 'subject_block_cue_means.csv'), index=False)

# --- Per-subject cue effects (block-level differences averaged per subject) --
eff = (sbc[sbc['Cue Type'] != 'Neutral Cue']
       .groupby(['subject_id', 'instruction', 'design', 'Cue Type'])[['d_acc', 'd_rt']]
       .mean().reset_index())
eff.to_csv(os.path.join(data_dir, 'cue_effects_per_subject.csv'), index=False)

# --- Per-subject absolute condition means (for overview plots) ---------------
subj_means = (sbc.groupby(['subject_id', 'instruction', 'Cue Type'])[['acc', 'rt']]
              .mean().reset_index())

logprint('Per-subject cue effects (informative - neutral, averaged over blocks):')
logprint(eff.pivot_table(index=['subject_id', 'instruction'], columns='Cue Type',
                         values=['d_acc', 'd_rt']).round(3).to_string())
logprint('')

# ============================================================
# 3. GROUP-LEVEL CUE EFFECTS + STATISTICS (old instruction cohort)
# ============================================================
logprint('=' * 70)
logprint('GROUP-LEVEL CUE EFFECTS - ONE-SAMPLE t-TESTS vs 0 (per cohort)')
logprint('=' * 70)

stats_rows = []
for cohort in ['old instruction', 'new instruction']:
    for cue in INFORMATIVE_ORDER:
        sub_eff = eff[(eff['instruction'] == cohort) & (eff['Cue Type'] == cue)]
        n = len(sub_eff)
        if n == 0:
            continue
        if n < 3:
            # too few subjects (new instruction): a t-test with df <= 1 is
            # uninformative and Shapiro-Wilk requires n >= 3 -> descriptives
            # only; within-subject / pooled trial-level tests follow in section 4
            for dv, name in [('d_acc', 'accuracy effect'), ('d_rt', 'RT effect')]:
                vals = ', '.join(f"sub-{int(s)}: {v:+.3f}"
                                 for s, v in zip(sub_eff['subject_id'], sub_eff[dv]))
                logprint(f"{cohort:>16s} | {cue:<15s} | {name}: n={n} -> mean="
                         f"{sub_eff[dv].mean():+.3f} [{vals}] (no group test possible)")
            continue

        for dv, name, unit in [('d_acc', 'accuracy effect', 'proportion'),
                               ('d_rt', 'RT effect', 's')]:
            x = sub_eff[dv].dropna().values
            # Assumption check: normality of the subject-level effects
            W, p_shapiro = stats.shapiro(x)
            t_res = pg.ttest(x, 0)
            t_val = t_res['T'].values[0]
            p_val = t_res['p-val'].values[0]
            dfree = n - 1
            d_z = x.mean() / x.std(ddof=1)          # Cohen's dz (subject-level averages)
            n_neg = int((x < 0).sum())
            logprint(f"{cohort:>16s} | {cue:<15s} | {name}: mean={x.mean():+.4f} {unit}, "
                     f"SD={x.std(ddof=1):.4f}, t({dfree})={t_val:+.3f}, p={p_val:.6f}, "
                     f"d_z={d_z:+.2f}, negative in {n_neg}/{n} subjects "
                     f"[Shapiro-Wilk: W={W:.3f}, p={p_shapiro:.3f}]")
            stats_rows.append({'cohort': cohort, 'cue': cue, 'dv': name,
                               'n_subjects': n, 'mean': x.mean(), 'sd': x.std(ddof=1),
                               't': t_val, 'df': dfree, 'p': p_val, 'd_z': d_z,
                               'shapiro_W': W, 'shapiro_p': p_shapiro})
logprint('')

pd.DataFrame(stats_rows).to_csv(os.path.join(data_dir, 'stats_cue_effects.csv'), index=False)

# ============================================================
# 4. NEW-INSTRUCTION SANITY CHECK (within-subject + pooled trial-level tests)
# ============================================================
# With N=2 a group-level test is not meaningful. We therefore
#   (a) test every new-instruction subject individually (trial level), and
#   (b) pool the trials of all new-instruction subjects in a fixed-effects
#       analysis STRATIFIED by subject x block, so that every informative cue is
#       only ever compared with the neutral trials of the same block (respects
#       the block-wise cueing design and removes between-subject / between-block
#       baseline differences). Inference generalizes to these trials/subjects
#       only, NOT to the population.
from statsmodels.stats.contingency_tables import StratifiedTable
import statsmodels.formula.api as smf


def within_subject_tests(d_sub, label):
    """Trial-level informative vs. neutral tests (same blocks) for one subject."""
    logprint(f"\nWithin-block informative vs. neutral tests ({label}):")
    rows = []
    for cue in INFORMATIVE_ORDER:
        cue_instr = [k for k, v in CUE_MAP.items() if v == cue][0]
        blocks_with_cue = d_sub.loc[d_sub['CueInstruction'] == cue_instr, 'block'].unique()
        d_sel = d_sub[d_sub['block'].isin(blocks_with_cue)]
        # --- accuracy: Fisher's exact test on the 2x2 table ---
        is_cue = d_sel['CueInstruction'] == cue_instr
        is_neu = d_sel['CueInstruction'] == 'cue_neutral'
        cued_corr = int(d_sel.loc[is_cue, 'IsCorrect'].sum())
        cued_inc = int((~d_sel.loc[is_cue, 'IsCorrect']).sum())
        neu_corr = int(d_sel.loc[is_neu, 'IsCorrect'].sum())
        neu_inc = int((~d_sel.loc[is_neu, 'IsCorrect']).sum())
        table = [[cued_corr, cued_inc], [neu_corr, neu_inc]]
        odds, p_fisher = stats.fisher_exact(table)
        chi2, p_chi2, dof, _ = stats.chi2_contingency(table, correction=True)
        acc_cue = cued_corr / (cued_corr + cued_inc)
        acc_neu = neu_corr / (neu_corr + neu_inc)
        # Cohen's h for the difference of two proportions
        h = 2 * np.arcsin(np.sqrt(acc_cue)) - 2 * np.arcsin(np.sqrt(acc_neu))
        # --- RT: Welch t-test + Mann-Whitney U (correct trials only) ---
        rt_cue = d_sel.loc[is_cue & d_sel['IsCorrect'], 'rt'].dropna()
        rt_neu = d_sel.loc[is_neu & d_sel['IsCorrect'], 'rt'].dropna()
        row = {'subject': label, 'cue': cue, 'acc_cue': acc_cue, 'acc_neutral': acc_neu,
               'odds_ratio': odds, 'p_fisher': p_fisher, 'chi2': chi2, 'chi2_df': dof,
               'p_chi2': p_chi2, 'cohens_h': h}
        if len(rt_cue) > 1 and len(rt_neu) > 1:
            # Assumption checks: normality (Shapiro-Wilk) and equal variances (Levene)
            _, p_sw_c = stats.shapiro(rt_cue)
            _, p_sw_n = stats.shapiro(rt_neu)
            _, p_lev = stats.levene(rt_cue, rt_neu)
            t_res = pg.ttest(rt_cue, rt_neu, correction=True)  # Welch (robust to unequal var.)
            U, p_mwu = stats.mannwhitneyu(rt_cue, rt_neu, alternative='two-sided')
            # Cohen's d for independent samples (pooled SD)
            sp = np.sqrt(((len(rt_cue) - 1) * rt_cue.var(ddof=1) +
                          (len(rt_neu) - 1) * rt_neu.var(ddof=1)) /
                         (len(rt_cue) + len(rt_neu) - 2))
            d_ind = (rt_cue.mean() - rt_neu.mean()) / sp
            rt_str = (f"RT: {rt_cue.mean():.3f} s vs {rt_neu.mean():.3f} s, "
                      f"Welch t({t_res['dof'].values[0]:.1f})={t_res['T'].values[0]:+.2f}, "
                      f"p={t_res['p-val'].values[0]:.6f}, d={d_ind:+.2f} "
                      f"[Mann-Whitney U={U:.0f}, p={p_mwu:.6f}; Shapiro p={p_sw_c:.3f}/{p_sw_n:.3f}, "
                      f"Levene p={p_lev:.3f}]")
            row.update({'rt_cue': rt_cue.mean(), 'rt_neutral': rt_neu.mean(),
                        't_welch': t_res['T'].values[0], 'df_welch': t_res['dof'].values[0],
                        'p_welch': t_res['p-val'].values[0], 'cohens_d': d_ind,
                        'U': U, 'p_mwu': p_mwu})
        else:
            rt_str = 'RT: not enough correct trials'
        logprint(f"  {cue:<15s} | Accuracy: {acc_cue:.3f} vs {acc_neu:.3f} (cue vs neutral), "
                 f"OR={odds:.2f}, Fisher's exact p={p_fisher:.6f}, chi2({dof})={chi2:.2f}, "
                 f"p={p_chi2:.6f}, h={h:+.2f} | {rt_str}")
        rows.append(row)
    return rows


logprint('=' * 70)
logprint(f'{NEW_LABEL.upper()} - SANITY CHECK')
logprint('=' * 70)
within_rows = []
for sub in NEW_INSTRUCTION_SUBJECTS:
    d_sub = df[df['subject_id'] == sub]
    logprint(f"\n--- Subject {sub} ---")
    logprint(f"Overall accuracy: {d_sub['IsCorrect_f'].mean():.3f}, "
             f"overall RT (correct): {d_sub['rt_valid'].mean():.3f} s, "
             f"no-response rate: {1 - d_sub['responded'].mean():.3f}, "
             f"n trials: {len(d_sub)}, color mapping: {', '.join(d_sub['Color'].unique())}")
    # which block contains which informative cue type?
    block_cues = (sbc[sbc['subject_id'] == sub]
                  .groupby('block')['Cue Type'].apply(lambda s: sorted(set(s) - {'Neutral Cue'})))
    logprint(f'Block -> informative cue type mapping (subject {sub}): ' +
             '; '.join(f"block {b}: {c[0] if c else 'none'}" for b, c in block_cues.items()))
    logprint(f'Per-block condition means (subject {sub}):')
    logprint(sbc[sbc['subject_id'] == sub]
             [['block', 'Cue Type', 'acc', 'rt', 'n', 'acc_neutral', 'd_acc', 'd_rt']]
             .round(3).to_string(index=False))
    within_rows += within_subject_tests(d_sub, f'subject {sub}')

# --- (b) pooled, subject x block stratified analysis --------------------------
logprint('\n' + '-' * 70)
logprint(f'POOLED TRIAL-LEVEL ANALYSIS ({NEW_LABEL}), stratified by subject x block')
logprint('-' * 70)
d_new = df[df['subject_id'].isin(NEW_INSTRUCTION_SUBJECTS)].copy()
d_new['stratum'] = d_new['subject_id'].astype(str) + '_' + d_new['block'].astype(str)
pooled_rows = []
for cue in INFORMATIVE_ORDER:
    cue_instr = [k for k, v in CUE_MAP.items() if v == cue][0]
    strata_with_cue = d_new.loc[d_new['CueInstruction'] == cue_instr, 'stratum'].unique()
    d_sel = d_new[d_new['stratum'].isin(strata_with_cue) &
                  d_new['CueInstruction'].isin([cue_instr, 'cue_neutral'])].copy()
    d_sel['informative'] = (d_sel['CueInstruction'] == cue_instr).astype(int)

    # --- accuracy: Cochran-Mantel-Haenszel test over subject x block strata ---
    tables = []
    for _, g in d_sel.groupby('stratum'):
        tables.append(np.array([[g.loc[g['informative'] == 1, 'IsCorrect'].sum(),
                                 (~g.loc[g['informative'] == 1, 'IsCorrect']).sum()],
                                [g.loc[g['informative'] == 0, 'IsCorrect'].sum(),
                                 (~g.loc[g['informative'] == 0, 'IsCorrect']).sum()]], dtype=float))
    st = StratifiedTable(tables)
    cmh = st.test_null_odds(correction=True)
    bd = st.test_equal_odds()           # Breslow-Day: homogeneity of OR across strata
    acc_cue = d_sel.loc[d_sel['informative'] == 1, 'IsCorrect_f'].mean()
    acc_neu = d_sel.loc[d_sel['informative'] == 0, 'IsCorrect_f'].mean()
    logprint(f"  {cue:<15s} | Accuracy: {acc_cue:.3f} vs {acc_neu:.3f} (cue vs neutral), "
             f"MH pooled OR={st.oddsratio_pooled:.2f}, CMH chi2(1)={cmh.statistic:.3f}, "
             f"p={cmh.pvalue:.6f} [Breslow-Day homogeneity chi2={bd.statistic:.2f}, "
             f"p={bd.pvalue:.3f}; {len(tables)} strata]")

    # --- RT: OLS rt ~ informative + stratum fixed effects (correct trials) ---
    d_rt = d_sel[d_sel['IsCorrect']].dropna(subset=['rt'])
    fit = smf.ols('rt ~ informative + C(stratum)', data=d_rt).fit()
    t_val, p_val, b = fit.tvalues['informative'], fit.pvalues['informative'], fit.params['informative']
    dfree = int(fit.df_resid)
    eta_p2 = t_val ** 2 / (t_val ** 2 + dfree)        # partial eta^2 of the cue effect
    _, p_sw_res = stats.shapiro(fit.resid)            # normality of residuals
    # robustness: log-RT (RT distributions are right-skewed)
    fit_log = smf.ols('np.log(rt) ~ informative + C(stratum)', data=d_rt).fit()
    logprint(f"  {'':<15s} | RT: b={b:+.4f} s, t({dfree})={t_val:+.3f}, p={p_val:.6f}, "
             f"partial eta2={eta_p2:.4f} [residual Shapiro p={p_sw_res:.4f}; log-RT: "
             f"t({int(fit_log.df_resid)})={fit_log.tvalues['informative']:+.3f}, "
             f"p={fit_log.pvalues['informative']:.6f}]")
    pooled_rows.append({'cue': cue, 'acc_cue': acc_cue, 'acc_neutral': acc_neu,
                        'mh_or': st.oddsratio_pooled, 'cmh_chi2': cmh.statistic, 'cmh_df': 1,
                        'cmh_p': cmh.pvalue, 'breslow_day_chi2': bd.statistic,
                        'breslow_day_p': bd.pvalue, 'rt_b': b, 'rt_t': t_val, 'rt_df': dfree,
                        'rt_p': p_val, 'rt_partial_eta2': eta_p2,
                        'rt_resid_shapiro_p': p_sw_res,
                        'logrt_t': fit_log.tvalues['informative'],
                        'logrt_p': fit_log.pvalues['informative']})
logprint('')
pd.DataFrame(within_rows).to_csv(os.path.join(data_dir, 'stats_new_instruction_within_subject.csv'), index=False)
pd.DataFrame(pooled_rows).to_csv(os.path.join(data_dir, 'stats_new_instruction_pooled.csv'), index=False)

# ============================================================
# 4b. RT ON ALL RESPONDED TRIALS (correct + incorrect pooled)
# ============================================================
# Complementary to the correct-only RT analyses above: RT is averaged over ALL
# trials with a response (correct + incorrect). Timeouts have no RT and remain
# excluded. Caveat: if a cue changes accuracy, the pooled RT also changes in its
# trial composition (correct vs. error RTs can differ) -> we therefore also report
# the correct-vs-error RT difference per cohort.
logprint('=' * 70)
logprint('RT ON ALL RESPONDED TRIALS (CORRECT + INCORRECT POOLED)')
logprint('=' * 70)

# --- Descriptive: RT of correct vs. incorrect (responded) trials ------------
rt_ce = (df[df['responded']]
         .assign(Correctness=lambda d: np.where(d['IsCorrect'], 'correct', 'incorrect'))
         .groupby(['subject_id', 'instruction', 'Correctness'])['rt'].mean()
         .unstack('Correctness').reset_index())
rt_ce['diff'] = rt_ce['incorrect'] - rt_ce['correct']
logprint('RT (s) of correct vs. incorrect responded trials, per subject (new instruction):')
logprint(rt_ce[rt_ce['instruction'] == 'new instruction'].round(3).to_string(index=False))
x = rt_ce.loc[rt_ce['instruction'] == 'old instruction', 'diff'].dropna().values
W, p_sw = stats.shapiro(x)
t_res = pg.ttest(x, 0)
logprint(f"Old instruction (n={len(x)}): incorrect - correct RT = {x.mean():+.4f} s "
         f"(SD={x.std(ddof=1):.4f}), t({len(x) - 1})={t_res['T'].values[0]:+.3f}, "
         f"p={t_res['p-val'].values[0]:.6f}, d_z={x.mean() / x.std(ddof=1):+.2f} "
         f"[Shapiro-Wilk: W={W:.3f}, p={p_sw:.3f}]\n")

# --- Block-level means + within-block cue effects ---------------------------
sbc_all = (df.groupby(['subject_id', 'instruction', 'design', 'block', 'Cue Type'])
           .agg(rt_all=('rt', 'mean'), n_resp=('rt', 'count')).reset_index())
neu_all = (sbc_all[sbc_all['Cue Type'] == 'Neutral Cue']
           .rename(columns={'rt_all': 'rt_all_neutral'})[['subject_id', 'block', 'rt_all_neutral']])
sbc_all = sbc_all.merge(neu_all, on=['subject_id', 'block'], how='left')
sbc_all['d_rt_all'] = sbc_all['rt_all'] - sbc_all['rt_all_neutral']
eff_all = (sbc_all[sbc_all['Cue Type'] != 'Neutral Cue']
           .groupby(['subject_id', 'instruction', 'design', 'Cue Type'])['d_rt_all']
           .mean().reset_index())
eff_all.to_csv(os.path.join(data_dir, 'cue_effects_rt_all_trials_per_subject.csv'), index=False)
subj_means_all = (sbc_all.groupby(['subject_id', 'instruction', 'Cue Type'])['rt_all']
                  .mean().reset_index())

logprint('Group-level RT effects (informative - neutral, all responded trials):')
stats_all_rows = []
for cohort in ['old instruction', 'new instruction']:
    for cue in INFORMATIVE_ORDER:
        sub_eff = eff_all[(eff_all['instruction'] == cohort) & (eff_all['Cue Type'] == cue)]
        n = len(sub_eff)
        if n == 0:
            continue
        if n < 3:
            vals = ', '.join(f"sub-{int(s)}: {v:+.3f}"
                             for s, v in zip(sub_eff['subject_id'], sub_eff['d_rt_all']))
            logprint(f"{cohort:>16s} | {cue:<15s} | RT effect (all): n={n} -> mean="
                     f"{sub_eff['d_rt_all'].mean():+.3f} s [{vals}] (no group test possible)")
            continue
        x = sub_eff['d_rt_all'].dropna().values
        W, p_sw = stats.shapiro(x)
        t_res = pg.ttest(x, 0)
        d_z = x.mean() / x.std(ddof=1)
        logprint(f"{cohort:>16s} | {cue:<15s} | RT effect (all): mean={x.mean():+.4f} s, "
                 f"SD={x.std(ddof=1):.4f}, t({n - 1})={t_res['T'].values[0]:+.3f}, "
                 f"p={t_res['p-val'].values[0]:.6f}, d_z={d_z:+.2f}, negative in "
                 f"{int((x < 0).sum())}/{n} subjects [Shapiro-Wilk: W={W:.3f}, p={p_sw:.3f}]")
        stats_all_rows.append({'cohort': cohort, 'cue': cue, 'dv': 'RT effect (all responded)',
                               'n_subjects': n, 'mean': x.mean(), 'sd': x.std(ddof=1),
                               't': t_res['T'].values[0], 'df': n - 1,
                               'p': t_res['p-val'].values[0], 'd_z': d_z,
                               'shapiro_W': W, 'shapiro_p': p_sw})
pd.DataFrame(stats_all_rows).to_csv(os.path.join(data_dir, 'stats_cue_effects_rt_all_trials.csv'),
                                    index=False)

# --- New instruction: within-subject + pooled stratified tests --------------
logprint(f'\nWithin-subject RT tests, all responded trials ({NEW_LABEL}):')
rt_all_rows = []
for sub in NEW_INSTRUCTION_SUBJECTS:
    d_sub = df[(df['subject_id'] == sub) & df['responded']]
    for cue in INFORMATIVE_ORDER:
        cue_instr = [k for k, v in CUE_MAP.items() if v == cue][0]
        blocks_with_cue = d_sub.loc[d_sub['CueInstruction'] == cue_instr, 'block'].unique()
        d_sel = d_sub[d_sub['block'].isin(blocks_with_cue)]
        rt_cue = d_sel.loc[d_sel['CueInstruction'] == cue_instr, 'rt'].dropna()
        rt_neu = d_sel.loc[d_sel['CueInstruction'] == 'cue_neutral', 'rt'].dropna()
        _, p_sw_c = stats.shapiro(rt_cue)
        _, p_sw_n = stats.shapiro(rt_neu)
        _, p_lev = stats.levene(rt_cue, rt_neu)
        t_res = pg.ttest(rt_cue, rt_neu, correction=True)   # Welch
        U, p_mwu = stats.mannwhitneyu(rt_cue, rt_neu, alternative='two-sided')
        sp = np.sqrt(((len(rt_cue) - 1) * rt_cue.var(ddof=1) +
                      (len(rt_neu) - 1) * rt_neu.var(ddof=1)) / (len(rt_cue) + len(rt_neu) - 2))
        d_ind = (rt_cue.mean() - rt_neu.mean()) / sp
        logprint(f"  subject {sub} | {cue:<15s} | RT: {rt_cue.mean():.3f} s (n={len(rt_cue)}) vs "
                 f"{rt_neu.mean():.3f} s (n={len(rt_neu)}), Welch t({t_res['dof'].values[0]:.1f})="
                 f"{t_res['T'].values[0]:+.2f}, p={t_res['p-val'].values[0]:.6f}, d={d_ind:+.2f} "
                 f"[Mann-Whitney U={U:.0f}, p={p_mwu:.6f}; Shapiro p={p_sw_c:.3f}/{p_sw_n:.3f}, "
                 f"Levene p={p_lev:.3f}]")
        rt_all_rows.append({'subject': sub, 'cue': cue, 'rt_cue': rt_cue.mean(),
                            'rt_neutral': rt_neu.mean(), 'n_cue': len(rt_cue), 'n_neutral': len(rt_neu),
                            't_welch': t_res['T'].values[0], 'df_welch': t_res['dof'].values[0],
                            'p_welch': t_res['p-val'].values[0], 'cohens_d': d_ind,
                            'U': U, 'p_mwu': p_mwu})

logprint(f'\nPooled RT (all responded trials, {NEW_LABEL}), OLS with subject x block fixed effects:')
for cue in INFORMATIVE_ORDER:
    cue_instr = [k for k, v in CUE_MAP.items() if v == cue][0]
    strata_with_cue = d_new.loc[d_new['CueInstruction'] == cue_instr, 'stratum'].unique()
    d_rt = d_new[d_new['stratum'].isin(strata_with_cue) & d_new['responded'] &
                 d_new['CueInstruction'].isin([cue_instr, 'cue_neutral'])].dropna(subset=['rt']).copy()
    d_rt['informative'] = (d_rt['CueInstruction'] == cue_instr).astype(int)
    fit = smf.ols('rt ~ informative + C(stratum)', data=d_rt).fit()
    t_val, p_val, b = fit.tvalues['informative'], fit.pvalues['informative'], fit.params['informative']
    dfree = int(fit.df_resid)
    eta_p2 = t_val ** 2 / (t_val ** 2 + dfree)
    _, p_sw_res = stats.shapiro(fit.resid)
    fit_log = smf.ols('np.log(rt) ~ informative + C(stratum)', data=d_rt).fit()
    # composition control: add correctness as covariate (does the cue effect on
    # pooled RT survive when the correct/error mix is accounted for?)
    d_rt['correct'] = d_rt['IsCorrect'].astype(int)
    fit_cov = smf.ols('rt ~ informative + correct + C(stratum)', data=d_rt).fit()
    logprint(f"  {cue:<15s} | RT: {d_rt.loc[d_rt['informative'] == 1, 'rt'].mean():.3f} s vs "
             f"{d_rt.loc[d_rt['informative'] == 0, 'rt'].mean():.3f} s, b={b:+.4f} s, "
             f"t({dfree})={t_val:+.3f}, p={p_val:.6f}, partial eta2={eta_p2:.4f} "
             f"[residual Shapiro p={p_sw_res:.4f}; log-RT: t({int(fit_log.df_resid)})="
             f"{fit_log.tvalues['informative']:+.3f}, p={fit_log.pvalues['informative']:.6f}; "
             f"+correctness covariate: b={fit_cov.params['informative']:+.4f} s, "
             f"t({int(fit_cov.df_resid)})={fit_cov.tvalues['informative']:+.3f}, "
             f"p={fit_cov.pvalues['informative']:.6f}]")
    rt_all_rows.append({'subject': 'pooled', 'cue': cue, 'rt_b': b, 'rt_t': t_val, 'rt_df': dfree,
                        'rt_p': p_val, 'rt_partial_eta2': eta_p2, 'rt_resid_shapiro_p': p_sw_res,
                        'logrt_t': fit_log.tvalues['informative'],
                        'logrt_p': fit_log.pvalues['informative'],
                        'cov_b': fit_cov.params['informative'],
                        'cov_t': fit_cov.tvalues['informative'],
                        'cov_p': fit_cov.pvalues['informative']})
logprint('')
pd.DataFrame(rt_all_rows).to_csv(os.path.join(data_dir, 'stats_new_instruction_rt_all_trials.csv'),
                                 index=False)

# ============================================================
# 5. DIAGNOSTICS: WHY ARE CUES DETRIMENTAL?
# ============================================================
logprint('=' * 70)
logprint('DIAGNOSTIC 1: RESPONSE TYPES (what digit is reported?)')
logprint('=' * 70)
resp_order = ['Correct', 'Distractor Intrusion', 'Control Intrusion', 'Other Digit', 'No Response']
for cohort, subs in [('old instruction (6-block)', (df['instruction'] == 'old instruction') & (df['design'] == '6-block')),
                     (NEW_LABEL, df['instruction'] == 'new instruction'),
                     ('old instruction (4-block)', (df['instruction'] == 'old instruction') & (df['design'] == '4-block'))]:
    counts = (df[subs].groupby(['Cue Type', 'response_type']).size()
              .reset_index(name='n'))
    counts['proportion'] = counts['n'] / counts.groupby('Cue Type')['n'].transform('sum')
    pivot = counts.pivot_table(index='Cue Type', columns='response_type',
                               values='proportion').reindex(columns=resp_order)
    logprint(f"\n{cohort}:")
    logprint(pivot.round(3).to_string())

logprint('\n' + '=' * 70)
logprint('DIAGNOSTIC 2: CUE FOLLOWING (is the digit at the CUED location reported?)')
logprint('=' * 70)
logprint('High cue-following for distractor/control cues means participants report the')
logprint('cued item -> guaranteed errors. Low cue-following for target cues means the')
logprint('cue meaning was not understood (or actively counteracted).')
cf = (df[df['CueInstruction'] != 'cue_neutral']
      .groupby(['instruction', 'design', 'Cue Type'])['cue_followed']
      .agg(['mean', 'std', 'count']).reset_index())
cf_group = (df[df['CueInstruction'] != 'cue_neutral']
            .groupby(['subject_id', 'instruction', 'Cue Type'])['cue_followed'].mean().reset_index())
cf_summary = (cf_group.groupby(['instruction', 'Cue Type'])['cue_followed']
              .agg(['mean', 'std', 'count']).reset_index())
logprint('\nProportion of trials in which the digit at the cued location was reported:')
logprint(cf_summary.round(3).to_string(index=False))

# ============================================================
# 6. FIGURES
# ============================================================
# --- Figure 1: overview of accuracy / RT and cue effects --------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Cue Effects on Accuracy and Response Time (Block-Level Evaluation)',
             fontsize=16, fontweight='bold')

# A: absolute accuracy
sns.barplot(data=subj_means, x='Cue Type', y='acc', hue='instruction',
            order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[0, 0])
sns.stripplot(data=subj_means, x='Cue Type', y='acc', hue='instruction',
              order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[0, 0])
axes[0, 0].set_ylabel('Accuracy (Proportion Correct)')
axes[0, 0].set_xlabel('')
axes[0, 0].set_title('Absolute Accuracy')

# B: absolute RT
sns.barplot(data=subj_means, x='Cue Type', y='rt', hue='instruction',
            order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[0, 1])
sns.stripplot(data=subj_means, x='Cue Type', y='rt', hue='instruction',
              order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[0, 1])
axes[0, 1].set_ylabel('Response Time (s)')
axes[0, 1].set_xlabel('')
axes[0, 1].set_title('Absolute Response Time (Correct Trials)')

# C: accuracy effects (informative - neutral)
sns.barplot(data=eff, x='Cue Type', y='d_acc', hue='instruction',
            order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[1, 0])
sns.stripplot(data=eff, x='Cue Type', y='d_acc', hue='instruction',
              order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[1, 0])
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_ylabel('Δ Accuracy (Proportion Correct)')
axes[1, 0].set_xlabel('')
axes[1, 0].set_title('Accuracy Effect (Informative − Neutral)')

# D: RT effects
sns.barplot(data=eff, x='Cue Type', y='d_rt', hue='instruction',
            order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[1, 1])
sns.stripplot(data=eff, x='Cue Type', y='d_rt', hue='instruction',
              order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[1, 1])
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_ylabel('Δ Response Time (s)')
axes[1, 1].set_xlabel('')
axes[1, 1].set_title('RT Effect (Informative − Neutral)')

for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', rotation=15)
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend(handles, labels, title='Instruction', frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig1_cue_effects_overview.png'), dpi=200)
plt.close('all')

# --- Figure 2: new-instruction subjects, block-level sanity check -----------
n_new = len(NEW_INSTRUCTION_SUBJECTS)
fig, axes = plt.subplots(n_new, 2, figsize=(15, 5.5 * n_new), squeeze=False)
fig.suptitle(f'{NEW_LABEL.capitalize()} - Block-Level Sanity Check', fontsize=16,
             fontweight='bold')
for row_i, sub in enumerate(NEW_INSTRUCTION_SUBJECTS):
    s_new = sbc[sbc['subject_id'] == sub].copy()
    block_cue_label = {b: (c[0] if len(c := sorted(set(v) - {'Neutral Cue'})) else 'None')
                       for b, v in (s_new.groupby('block')['Cue Type'].apply(list).items())}
    for ax, dv, ylab, title in [(axes[row_i, 0], 'acc', 'Accuracy (Proportion Correct)', 'Accuracy'),
                                (axes[row_i, 1], 'rt', 'Response Time (s)', 'Response Time (Correct Trials)')]:
        sns.barplot(data=s_new, x='block', y=dv, hue='Cue Type',
                    hue_order=['Neutral Cue'] + INFORMATIVE_ORDER,
                    palette=CUE_PALETTE, errorbar=None, alpha=0.7, ax=ax)
        ax.set_ylabel(ylab)
        ax.set_xlabel('Block')
        ax.set_title(f'Subject {sub}: {title}')
        blocks = sorted(s_new['block'].unique())
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels([f"{b}\n({block_cue_label[b].replace(' Cue', '')})" for b in blocks])
        ax.spines[['top', 'right']].set_visible(False)
        if not (row_i == 0 and dv == 'acc'):
            ax.legend_.remove()
axes[0, 0].legend(title='Cue', frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig2_new_instruction_block_level.png'), dpi=200)
plt.close('all')

# --- Figure 3: response types (error localization) ---------------------------
groups = [('old instruction (6-block)', (df['instruction'] == 'old instruction') & (df['design'] == '6-block')),
          (NEW_LABEL, df['instruction'] == 'new instruction'),
          ('old instruction (4-block)', (df['instruction'] == 'old instruction') & (df['design'] == '4-block'))]
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.suptitle('Response Types by Cue Condition (Error Localization)', fontsize=16, fontweight='bold')
for ax, (label, sel) in zip(axes, groups):
    d_sel = df[sel]
    counts = (d_sel.groupby(['subject_id', 'Cue Type', 'response_type']).size()
              .reset_index(name='n'))
    counts['proportion'] = counts['n'] / counts.groupby(['subject_id', 'Cue Type'])['n'].transform('sum')
    order = [c for c in CUE_ORDER if c in counts['Cue Type'].unique()]
    sns.barplot(data=counts, x='Cue Type', y='proportion', hue='response_type',
                order=order, hue_order=resp_order, palette=RESPONSE_PALETTE,
                errorbar=('se', 1), alpha=0.85, ax=ax)
    ax.set_title(label)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of Trials')
    ax.tick_params(axis='x', rotation=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend_.remove()
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, title='Response Type', frameon=False,
               bbox_to_anchor=(1.0, 1.0), loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig3_response_types.png'), dpi=200)
plt.close('all')

# --- Figure 4: cue following -------------------------------------------------
cf_plot = cf_group.copy()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=cf_plot, x='Cue Type', y='cue_followed', hue='instruction',
            order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=ax)
sns.stripplot(data=cf_plot, x='Cue Type', y='cue_followed', hue='instruction',
              order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=ax)
ax.axhline(1 / 9, color='black', linestyle=':', linewidth=1)
ax.text(2.45, 1 / 9 + 0.01, 'chance (1/9)', ha='right', fontsize=10)
ax.set_ylabel('Proportion of Trials')
ax.set_xlabel('')
ax.set_title('Cue Following: Digit at the Cued Location Was Reported')
ax.spines[['top', 'right']].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Instruction', frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig4_cue_following.png'), dpi=200)
plt.close('all')

# --- Figure 5: per-subject target-cue effects (sanity check overview) --------
eff6 = eff[eff['design'] == '6-block'].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Per-Subject Target-Cue Effects (6-Block Design)', fontsize=16, fontweight='bold')
for ax, dv, ylab in [(axes[0], 'd_acc', 'Δ Accuracy (Proportion Correct)'),
                     (axes[1], 'd_rt', 'Δ Response Time (s)')]:
    d_plot = eff6[eff6['Cue Type'] == 'Target Cue'].sort_values(dv)
    colors = [COHORT_PALETTE['new instruction'] if s in NEW_INSTRUCTION_SUBJECTS else '#949494'
              for s in d_plot['subject_id']]
    ax.bar(x=[str(s) for s in d_plot['subject_id']], height=d_plot[dv], color=colors)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel(ylab)
    ax.set_xlabel('Subject')
    ax.tick_params(axis='x', rotation=75)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_title('Accuracy Effect (Target − Neutral)')
axes[1].set_title('RT Effect (Target − Neutral)')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig5_target_effects_per_subject.png'), dpi=200)
plt.close('all')

# --- Figure 6: RT on all responded trials (correct + incorrect pooled) -------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Response Time - All Responded Trials (Correct + Incorrect)',
             fontsize=16, fontweight='bold')
sns.barplot(data=subj_means_all, x='Cue Type', y='rt_all', hue='instruction',
            order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[0])
sns.stripplot(data=subj_means_all, x='Cue Type', y='rt_all', hue='instruction',
              order=CUE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[0])
axes[0].set_ylabel('Response Time (s)')
axes[0].set_title('Absolute Response Time')
sns.barplot(data=eff_all, x='Cue Type', y='d_rt_all', hue='instruction',
            order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
            palette=COHORT_PALETTE, errorbar=('se', 1), alpha=0.6, ax=axes[1])
sns.stripplot(data=eff_all, x='Cue Type', y='d_rt_all', hue='instruction',
              order=INFORMATIVE_ORDER, hue_order=['old instruction', 'new instruction'],
              palette=COHORT_PALETTE, dodge=True, jitter=True, alpha=0.8,
              size=5, legend=False, ax=axes[1])
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_ylabel('Δ Response Time (s)')
axes[1].set_title('RT Effect (Informative − Neutral)')
for ax in axes:
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].legend_.remove()
axes[0].legend(title='Instruction', frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig6_rt_all_responded_trials.png'), dpi=200)
plt.close('all')

# ============================================================
# 7. SAVE LOG
# ============================================================
with open(os.path.join(data_dir, 'cue_effects_stats_summary.txt'), 'w') as f:
    f.write('\n'.join(LOG))
logprint(f"\nFigures saved to: {fig_dir}")
logprint(f"Log saved to: {os.path.join(data_dir, 'cue_effects_stats_summary.txt')}")
