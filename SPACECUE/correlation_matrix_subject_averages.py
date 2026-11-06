"""
SPACECUE: Correlation Matrix of Subject-Average Measures (LMM Data)
===================================================================

Creates publication-quality correlation-matrix heatmaps for the subject-level
measures in `spacecue_subject_averages_lmm.csv` (n = 36 subjects, 18 Young /
18 Old, 4 probability-condition rows per subject: DistractorProb x
HP_Distractor_Prob).

Beyond the 12 continuous measures, the matrix now includes:

  - Age (years): between-subject demographic influence on every measure.

  - Participant screening measures (merged from the participants
    spreadsheet by subject ID): PTA right / left ear (pure-tone average
    hearing threshold, dB HL), MoCA total score, and Digit Span forward /
    backward. The sheet contains many non-analysed subjects, placeholder
    rows (e.g. '5_B'), and '/' or blank cells for tests that were not
    administered; non-numeric cells become NaN and every pair is correlated
    pairwise-complete, so subjects without a given test simply drop out of
    the pairs involving it.

  - N2ac / Pd CONDITION-MODULATION scores: per-subject difference scores that
    capture how each ERP component depends on the probability conditions.
    These are computed for the console diagnostics (missingness, Shapiro-Wilk)
    but are deliberately NOT included in the correlation matrix or the stats
    tables, to keep the matrix visually comprehensible.

      * N2ac Amplitude/Latency: 80% - 60% block-probability contrast. N2ac is
        defined only on displays with a LATERAL target and a FRONTAL
        distractor (i.e. when the singleton is NOT at the high-probability
        location), so N2ac-valid displays occur only in the low-probability
        condition rows. An N2ac HP - LP (singleton-location) contrast is
        therefore structurally undefined and is not included.

      * Pd Amplitude/Latency: HP - LP singleton-location contrast and
        80% - 60% block-probability contrast. Pd is defined on displays with
        a LATERAL distractor and a FRONTAL target, which occur in both
        DistractorProb conditions.

    To keep the primary matrix visually comprehensible, only the AMPLITUDE
    modulation scores are shown; latency modulations are computed and
    reported in the console diagnostics but not plotted.

The matrix diagonal does not show the trivial r = 1; instead, each diagonal
cell displays the empirical distribution (histogram) of the corresponding
measure, giving an at-a-glance view of central tendency, spread and skew at
the respective analysis level. The upper triangle shows effect strength as
filled circles whose radius scales with |r|, coloured by the sign of r on
the same diverging scale as the heatmap (black outline = p < .05).

The lower triangle shows the raw data behind each correlation as a small
pairwise scatter plot (x = column variable, y = row variable,
pairwise-complete observations - exactly the pairs that entered Spearman rho).
The former r + significance-asterisk text annotations were removed because
they were redundant with the upper-triangle circles, which already encode
effect strength (size), direction (colour) and significance (black
outline). Exact r, p, df, and BH-FDR q-values are reported in the console
output and the saved stats CSV tables.

Variables are grouped and visually separated by measure type: task response
metrics, cognitive/hearing screening-test results, and EEG-derived neural
measures (Age is kept as its own demographic block).

Two matrices are produced:
  1. SUBJECT-LEVEL (primary, N = 36): each subject's 4 condition rows are
     averaged first (+ difference scores). This is the statistically
     appropriate level, because pooling the 144 condition rows would treat
     repeated measures within subjects as independent observations
     (pseudo-replication).
  2. POOLED-CONDITION (supplementary, N = 144): base measures + Age pooled
     across condition rows, for reference only (Age is subject-constant, so
     its correlations here are pseudo-replicated).

For every variable pair the script reports:
  - Spearman rho (PRIMARY: the correlation matrices are based on Spearman
    rho, a rank-based measure robust to the skewed / non-normal
    distributions flagged by the Shapiro-Wilk diagnostics), with exact
    (unrounded) p-value, degrees of freedom, and pairwise N
  - Benjamini-Hochberg FDR-adjusted q-values across all matrix pairs
  - Pearson r as a parametric companion (console + saved CSV tables)
  - Shapiro-Wilk normality diagnostics per variable (printed to console)

Identifier / categorical columns (subject_id, cohort, DistractorProb,
HP_Distractor_Prob) are excluded from the correlation matrix by design:
subject_id is an arbitrary code, and the remaining three are condition labels
whose "correlations" would merely re-describe the balanced design. The
influence of these probability conditions on N2ac and Pd is quantified via
the difference scores (console diagnostics only, see above).
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib

# Ensure Unicode characters (μV, en-dash) print correctly on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Use 'Agg' backend unless we are in an interactive console (sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro

sns.set_theme(context="talk", style="ticks")

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = r"G:\Meine Ablage\PhD\data\SPACECUE\concatenated\spacecue_subject_averages_lmm.csv"
# Output directory: the project's SPACECUE plots folder
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback when running in an interactive console without __file__
    _script_dir = r"C:\Users\Max\PycharmProjects\capture-suppression\SPACECUE"
OUT_DIR = os.path.join(_script_dir, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Continuous variables to correlate, with clean display labels + units
VARIABLE_LABELS = {
    'rt':                            'Response Time (s)',
    'IsCorrect':                     'Accuracy (Prop. Correct)',
    'target_towardness':             'Target Towardness (%)',
    'distractor_towardness':         'Distractor Towardness (%)',
    'control_towardness':            'Control Towardness (%)',
    'pre_stim_alpha_lateralization': 'Pre-Stim Alpha LI (8\u201312 Hz)',
    'pre_stim_alpha_total_power':    'Pre-Stim Alpha Total Power (ln)',
    'n2ac_latency_ms':               'N2ac Latency (ms)',
    'n2ac_mean_amp':                 'N2ac Amplitude (\u03bcV)',
    'pd_latency_ms':                 'PD Latency (ms)',
    'pd_mean_amp':                   'PD Amplitude (\u03bcV)',
}

VARIABLES = list(VARIABLE_LABELS.keys())

# Participant screening spreadsheet (audiometry + neuropsychology). This is
# a messy manual export: it contains many non-analysed subjects, placeholder
# rows like '5_B', and uses '/' or blank cells when a test was not
# administered. Only rows whose 'Subject ID' parses as a number are kept, and
# only the subjects present in the LMM data are used (join below).
PARTICIPANTS_CSV_PATH = r"C:\Users\Max\Downloads\SPACECUE Participants  - Tabellenblatt1.csv"

PARTICIPANT_LABELS = {
    'PTA r':               'PTA Right Ear (dB HL)',
    'PTA l':               'PTA Left Ear (dB HL)',
    'Moca':                'MoCA (Total Score)',
    'Digit Span forward':  'Digit Span Forward (Digits)',
    'Digit Span backward': 'Digit Span Backward (Digits)',
}
PARTICIPANT_MEASURES = list(PARTICIPANT_LABELS.keys())

# Subject-level N2ac / Pd condition-modulation difference scores (see module
# docstring for why an N2ac HP - LP location contrast is not included)
MODULATION_LABELS = {
    'n2ac_mean_amp_mod_prob':   'N2ac Amplitude: 80%\u221260% (\u03bcV)',
    'n2ac_latency_ms_mod_prob': 'N2ac Latency: 80%\u221260% (ms)',
    'pd_mean_amp_mod_loc':      'Pd Amplitude: HP\u2212LP Loc (\u03bcV)',
    'pd_latency_ms_mod_loc':    'Pd Latency: HP\u2212LP Loc (ms)',
    'pd_mean_amp_mod_prob':     'Pd Amplitude: 80%\u221260% (\u03bcV)',
    'pd_latency_ms_mod_prob':   'Pd Latency: 80%\u221260% (ms)',
}

# Full label map: Age + base measures + participant screening + modulation scores
LABELS = {'Age': 'Age (Years)', **VARIABLE_LABELS, **PARTICIPANT_LABELS, **MODULATION_LABELS}

# Measure-type categories for the correlation matrix: variables are grouped
# and visually separated by block (thick boundary lines, coloured edge bands,
# category names, and category-coloured tick labels). Behavioural measures are
# subdivided into task response metrics and screening-test results (cognitive
# tests + pure-tone audiometry); the EEG-derived measures form their own block.
VARIABLE_CATEGORIES = [
    ('Age',                        ['Age']),
    ('Response metrics',           ['rt', 'IsCorrect', 'target_towardness',
                                    'distractor_towardness', 'control_towardness']),
    ('Cognitive & hearing tests',
                                   ['Moca', 'Digit Span forward',
                                    'Digit Span backward', 'PTA r', 'PTA l']),
    ('Neural correlates',          ['pre_stim_alpha_lateralization',
                                    'pre_stim_alpha_total_power',
                                    'n2ac_mean_amp', 'n2ac_latency_ms',
                                    'pd_mean_amp', 'pd_latency_ms']),
]
# Single canonical ordering of the matrix variables (grouped by category)
CATEGORY_VAR_ORDER = [v for _, vs in VARIABLE_CATEGORIES for v in vs]
assert set(CATEGORY_VAR_ORDER) == set(['Age'] + VARIABLES + PARTICIPANT_MEASURES), \
    "VARIABLE_CATEGORIES must cover exactly the matrix variables (no more, no less)"

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("SPACECUE SUBJECT-AVERAGE CORRELATION MATRIX")
print("=" * 70)
print(f"\nLoading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# The regenerated CSV must carry the per-subject Age column (written by
# create_lmm_data.py since 21.10.2026); fail loudly rather than silently NaN out.
if 'Age' not in df.columns:
    raise ValueError(
        "'Age' column missing from the subject-averages CSV. "
        "Regenerate it with: python SPACECUE/create_lmm_data.py --subject-averages-only"
    )

# Coerce all analysis columns to numeric (guards against stray string codes)
for col in ['Age'] + VARIABLES:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"  Raw rows: {len(df)}  |  Subjects: {df['subject_id'].nunique()}")
print(f"  Cohort counts: {df.groupby('subject_id')['cohort'].first().value_counts().to_dict()}")

# --- Participant screening sheet (PTA / MoCA / Digit Span) -----------------
if not os.path.exists(PARTICIPANTS_CSV_PATH):
    raise FileNotFoundError(
        f"Participant screening sheet not found:\n  {PARTICIPANTS_CSV_PATH}\n"
        "Update PARTICIPANTS_CSV_PATH at the top of this script."
    )
print(f"\nLoading participant measures: {PARTICIPANTS_CSV_PATH}")
part_df = pd.read_csv(PARTICIPANTS_CSV_PATH)
# Keep rows whose 'Subject ID' parses as a number (drops placeholder rows
# such as '5_B' and empty separator rows). '/' and blank test cells become
# NaN via the numeric coercion below.
part_df['subject_id'] = pd.to_numeric(part_df['Subject ID'], errors='coerce')
part_df = (part_df.dropna(subset=['subject_id'])
                  .drop_duplicates(subset='subject_id', keep='first'))
part_df['subject_id'] = part_df['subject_id'].astype(int)
for col in PARTICIPANT_MEASURES:
    part_df[col] = pd.to_numeric(part_df[col], errors='coerce')
part_measures = part_df.set_index('subject_id')[PARTICIPANT_MEASURES]
print(f"  Rows with a numeric subject ID: {len(part_measures)} "
      f"(only the LMM subjects are joined below)")

# ----------------------------------------------------------------------------
# 2. SUBJECT-LEVEL TABLE (primary analysis level, N = 36)
# ----------------------------------------------------------------------------
# Per-subject mean across the 4 probability-condition rows (pairwise-complete
# columns such as N2ac simply average their available observations), plus the
# N2ac / Pd condition-modulation difference scores.
BASE_VARIABLES = VARIABLES
MODULATION_VARS = list(MODULATION_LABELS.keys())
sub_df = df.groupby('subject_id')[BASE_VARIABLES + ['Age']].mean()
N_SUB = len(sub_df)


def condition_mean(g, cond_col, cond_val, col):
    """Mean of `col` within one condition cell (NaN-safe)."""
    vals = g.loc[g[cond_col] == cond_val, col]
    return vals.mean() if vals.notna().any() else np.nan


mod_rows = []
for sub_id, g in df.groupby('subject_id'):
    r = {'subject_id': sub_id}
    # 80% - 60% block-probability contrasts. For N2ac only the low-probability
    # rows contain N2ac-valid displays, so the 80% selection automatically
    # reduces to the 80% vs 60% contrast within the low-probability condition
    # (exactly the relevant modulation for N2ac).
    for col in ['n2ac_mean_amp', 'n2ac_latency_ms', 'pd_mean_amp', 'pd_latency_ms']:
        r[col + '_mod_prob'] = (condition_mean(g, 'HP_Distractor_Prob', '80%', col)
                                - condition_mean(g, 'HP_Distractor_Prob', '60%', col))
    # HP - LP singleton-location contrast (Pd only: N2ac is structurally
    # undefined in high-probability rows, see module docstring). A difference
    # score is NaN whenever either contrast cell is undefined for the subject.
    for col in ['pd_mean_amp', 'pd_latency_ms']:
        r[col + '_mod_loc'] = (condition_mean(g, 'DistractorProb', 'high-probability', col)
                               - condition_mean(g, 'DistractorProb', 'low-probability', col))
    mod_rows.append(r)
sub_df = sub_df.join(pd.DataFrame(mod_rows).set_index('subject_id'))

# Merge participant screening measures onto the subject level. Tests that
# were not administered (e.g. the first four pilot subjects have no
# audiometry / neuropsychology at all) stay NaN; correlations are computed
# pairwise-complete, see correlation_table().
sub_df = sub_df.join(part_measures, how='left')

# Propagate the participant measures onto the condition rows as well, so the
# pooled supplementary level can include them (subject-constant, exactly like
# Age there -> pseudo-replicated, reference only).
df = df.merge(part_measures, left_on='subject_id', right_index=True, how='left')

# Consistency check: screening-sheet Age vs the LMM Age column
age_sheet = pd.to_numeric(part_df.set_index('subject_id')['Age'], errors='coerce')
age_cmp = sub_df[['Age']].join(age_sheet.rename('Age_sheet'))
age_diff = (age_cmp['Age'] - age_cmp['Age_sheet']).abs()
n_age_bad = int((age_diff > 0.5).sum())
if n_age_bad:
    bad_ids = list(age_diff.index[age_diff > 0.5])
    print(f"  WARNING: Age mismatch between screening sheet and LMM data for "
          f"{n_age_bad} subject(s): {bad_ids}")
else:
    print(f"  Age cross-check (screening sheet vs LMM): OK for all {N_SUB} subjects")

print(f"\nSubject-level aggregation (incl. Age + modulation scores): N = {N_SUB} subjects")

# Modulation difference scores are computed for the console diagnostics only
# (see missingness / Shapiro-Wilk sections above); they are NOT included in
# the correlation matrix, to keep the matrix visually comprehensible.
subject_matrix_vars = list(CATEGORY_VAR_ORDER)  # grouped by measure type
print("  NOTE: N2ac/Pd difference (modulation) scores are not shown in the "
      "matrix (visual clarity); console diagnostics only.")

# ============================================================================
# 3. NORMALITY DIAGNOSTICS (Shapiro-Wilk per variable)
# ============================================================================
print("\n--- Missingness per variable ---")
for col in BASE_VARIABLES:
    n_missing = int(df[col].isna().sum())
    print(f"  {LABELS[col]:<34}: {n_missing}/{len(df)} condition rows missing "
          f"({100 * n_missing / len(df):.1f}%)")
for col in MODULATION_VARS:
    n_ok = int(sub_df[col].notna().sum())
    print(f"  {LABELS[col]:<34}: {n_ok}/{N_SUB} subjects with complete contrast cells")

print("  --- Participant screening measures (subject level) ---")
for col in PARTICIPANT_MEASURES:
    n_ok = int(sub_df[col].notna().sum())
    print(f"  {LABELS[col]:<34}: {n_ok}/{N_SUB} subjects with a valid score")

print("\n--- Shapiro-Wilk Normality Checks (subject-level values) ---")
print(f"  {'Variable':<34} {'W':>8} {'p':>10}  Normal (p >= .05)?")
for col in subject_matrix_vars:
    vals = sub_df[col].dropna()
    if len(vals) >= 3:
        w_stat, p_val = shapiro(vals)
        verdict = 'yes' if p_val >= 0.05 else 'NO (consider Spearman rho)'
    else:
        w_stat, p_val, verdict = np.nan, np.nan, 'insufficient data'
    print(f"  {LABELS[col]:<34} {w_stat:8.4f} {p_val:10.5f}  {verdict}")

# ============================================================================
# 4. PAIRWISE CORRELATION ENGINES (Pearson + Spearman, exact p-values)
# ============================================================================
def correlation_table(data, variables, method='pearson'):
    """Pairwise-complete correlations with exact p-values and df."""
    rows = []
    for i, c1 in enumerate(variables):
        for c2 in variables[i + 1:]:
            pair = data[[c1, c2]].dropna()
            n = len(pair)
            if n < 4 or pair[c1].nunique() < 2 or pair[c2].nunique() < 2:
                rows.append({'var1': c1, 'var2': c2, 'n': n,
                             'r': np.nan, 'p': np.nan, 'df': np.nan})
                continue
            func = pearsonr if method == 'pearson' else spearmanr
            r_val, p_val = func(pair[c1], pair[c2])
            rows.append({'var1': c1, 'var2': c2, 'n': n,
                         'r': r_val, 'p': p_val, 'df': n - 2})
    out = pd.DataFrame(rows)
    return out


def benjamini_hochberg(pvals):
    """BH FDR adjustment (returns q-values, NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    idx = np.flatnonzero(valid)          # original positions of finite p-values
    m = len(idx)
    if m == 0:
        return q
    p_valid = p[idx]
    order = np.argsort(p_valid, kind='stable')   # ranks within the finite subset
    ranked = p_valid[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # step-up monotonicity
    q[idx[order]] = np.clip(ranked, 0, 1)
    return q


def p_to_stars(p):
    """Significance asterisks from an (uncorrected) p-value."""
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def matrix_from_table(table, variables, value_col='r'):
    """Reshape a long correlation table into a symmetric matrix."""
    mat = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for _, row in table.iterrows():
        mat.loc[row['var1'], row['var2']] = row[value_col]
        mat.loc[row['var2'], row['var1']] = row[value_col]
    return mat


# ============================================================================
# 5. COMPUTE STATISTICS (both analysis levels)
# ============================================================================
LEVELS = [
    ('subject_averaged', sub_df[subject_matrix_vars].astype(float)),
    ('pooled_conditions', df[CATEGORY_VAR_ORDER].astype(float)),
]
stat_tables = {}
for level, data in LEVELS:
    level_vars = list(data.columns)
    tab_p = correlation_table(data, level_vars, method='pearson')
    tab_s = correlation_table(data, level_vars, method='spearman')
    tab_p['q_fdr'] = benjamini_hochberg(tab_p['p'].values)
    tab_s['q_fdr'] = benjamini_hochberg(tab_s['p'].values)
    stat_tables[level] = {'pearson': tab_p, 'spearman': tab_s}

    print(f"\n{'=' * 70}")
    print(f"CORRELATIONS ({level.replace('_', ' ').upper()}, N_rows = {len(data)})")
    print(f"{'=' * 70}")
    # Spearman rho is the PRIMARY method (basis of the plotted matrices);
    # Pearson r is reported as a parametric companion.
    for method_name, tab in [('SPEARMAN RHO (primary, shown in the matrices)', tab_s),
                             ('PEARSON r (parametric companion)', tab_p)]:
        print(f"\n  --- {method_name} ---")
        print(f"  {'Pair':<70} {'rho':>7} {'df':>4} {'p':>10} {'q_FDR':>8}")
        sig = tab[np.isfinite(tab['p']) & (tab['p'] < 0.05)].sort_values('p')
        print(f"  --- Uncorrected p < .05 ({len(sig)}/{len(tab)} pairs) ---")
        for _, row in sig.iterrows():
            pair_name = f"{LABELS[row['var1']]}  vs  {LABELS[row['var2']]}"
            print(f"  {pair_name:<70} {row['r']:+7.3f} {int(row['df']):>4} "
                  f"{row['p']:10.6f} {row['q_fdr']:8.4f} {p_to_stars(row['p'])}")
        sig_fdr = tab[np.isfinite(tab['q_fdr']) & (tab['q_fdr'] < 0.05)]
        print(f"  --- Surviving BH-FDR (q < .05): {len(sig_fdr)}/{len(tab)} pairs ---")
        for _, row in sig_fdr.sort_values('q_fdr').iterrows():
            print(f"      q = {row['q_fdr']:.4f} | rho = {row['r']:+.3f} | "
                  f"{LABELS[row['var1']]} vs {LABELS[row['var2']]}")

    # Save long-format statistics tables
    tab_p_out = tab_p.copy()
    tab_p_out['label_var1'] = tab_p_out['var1'].map(LABELS)
    tab_p_out['label_var2'] = tab_p_out['var2'].map(LABELS)
    tab_p_out.to_csv(os.path.join(OUT_DIR, f"correlation_stats_pearson_{level}.csv"), index=False)
    tab_s_out = tab_s.copy()
    tab_s_out['label_var1'] = tab_s_out['var1'].map(LABELS)
    tab_s_out['label_var2'] = tab_s_out['var2'].map(LABELS)
    tab_s_out.to_csv(os.path.join(OUT_DIR, f"correlation_stats_spearman_{level}.csv"), index=False)

# ============================================================================
# 6. PLOTTING (lower-triangle scatter correlogram)
# ============================================================================
def plot_correlation_matrix(r_matrix, p_table, n_total, level_label, filename, data,
                            corr_label='Pearson r'):
    """Publication-quality correlogram: lower-triangle pairwise scatter
    plots, upper-triangle effect-strength circles, and diagonal distribution
    histograms.

    - Lower triangle: small pairwise scatter plots of the raw data (x =
      column variable, y = row variable, pairwise-complete observations -
      exactly the pairs that entered the correlation, here Spearman rho).
      These replace the former r + stars text annotations, which were
      redundant with the upper-triangle circles.
    - Upper triangle: filled circles whose radius scales with |r| and whose
      colour encodes the sign of r (same diverging scale as the heatmap);
      a black outline marks p < .05.
    - Diagonal: instead of the trivial r = 1, each cell contains a small
      histogram of the corresponding variable's distribution (drawn from
      `data`, i.e. at the same analysis level as the correlations).
    - Blocks: variables are grouped by measure type (VARIABLE_CATEGORIES);
      blocks are separated by thick boundary lines and marked with coloured
      edge bands, category names, and category-coloured tick labels.
    """
    # Wrap long labels onto (at most) two lines so the tick labels stay
    # legible when the figure is downscaled to page width in a document
    labels = [textwrap.fill(LABELS[c], width=18) for c in r_matrix.columns]
    n_var = len(labels)

    # Significance-star matrix aligned with r_matrix
    star_mat = pd.DataFrame('', index=r_matrix.index, columns=r_matrix.columns)
    for _, row in p_table.iterrows():
        stars = p_to_stars(row['p'])
        star_mat.loc[row['var1'], row['var2']] = stars
        star_mat.loc[row['var2'], row['var1']] = stars

    # Mask the ENTIRE matrix: the lower triangle is replaced by pairwise
    # scatter plots and the diagonal by distribution histograms (inset axes
    # below), so no heatmap cells are shown at all. The fully masked mesh is
    # still created by sns.heatmap because it serves as the colorbar source
    # for the shared 'vlag' diverging scale that also colours the
    # upper-triangle effect circles.
    mask = np.ones((n_var, n_var), dtype=bool)

    # Scale figure with the matrix dimension, but keep the canvas compact
    # relative to the font sizes: the figure is typically downscaled to page
    # width in a word processor, so a smaller canvas + larger fonts = much
    # better readability after downscaling.
    side = max(12.5, 0.78 * n_var + 2.0)
    annot_fs, tick_fs = (12.5, 13.5) if n_var <= 14 else (11, 12.5)

    fig, ax = plt.subplots(figsize=(side, side * 0.86))
    sns.heatmap(
        r_matrix, mask=mask, annot=False, cmap='vlag',
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.6, linecolor='white',
        cbar_kws={'label': corr_label, 'shrink': 0.55,
                  'ticks': np.linspace(-1, 1, 9)},
        ax=ax
    )

    # Keep the colourbar legible after downscaling in a document
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fs * 0.85)
    cbar.set_label(corr_label, fontsize=tick_fs)

    # Lower triangle: pairwise scatter plots of the true values. The former
    # r + stars text annotations were redundant with the upper-triangle
    # circles (size = |r|, colour = sign, black outline = p < .05), so each
    # lower-triangle cell now shows the underlying data itself: a small
    # inset axes (positioned in data coordinates, like the diagonal
    # histograms) containing the pairwise-complete scatter of
    # x = column variable vs. y = row variable -- exactly the observation
    # pairs that entered the Spearman rho of that cell.
    scatter_color = '.25'
    for i in range(n_var):
        for j in range(n_var):
            if i <= j:  # upper triangle and diagonal handled elsewhere
                continue
            x_vals = pd.to_numeric(data[r_matrix.columns[j]], errors='coerce')
            y_vals = pd.to_numeric(data[r_matrix.index[i]], errors='coerce')
            pair = pd.concat([x_vals, y_vals], axis=1).dropna()
            cell_ax = ax.inset_axes([j + 0.10, i + 0.10, 0.80, 0.80],
                                    transform=ax.transData)
            cell_ax.set_facecolor('white')
            if len(pair) >= 3:
                # Shrink the markers for the dense pooled-condition matrix
                # (N = 144) so points stay distinguishable within a cell
                ms = 13 if len(pair) <= 60 else 7
                cell_ax.scatter(pair.iloc[:, 0], pair.iloc[:, 1], s=ms,
                                color=scatter_color, alpha=0.7, linewidths=0)
                # Small margins so edge points are not clipped by the spines
                cell_ax.margins(0.12)
            else:
                cell_ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                             fontsize=annot_fs * 0.7,
                             transform=cell_ax.transAxes, color='.3')
            cell_ax.set_xticks([])
            cell_ax.set_yticks([])
            for spine in cell_ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color('.7')

    # Diagonal: distribution of each variable (histogram) instead of the
    # redundant r = 1 cells. A small inset axes is placed inside each
    # diagonal heatmap cell (positioned in data coordinates so it stays
    # aligned regardless of figure size / aspect) and shows the empirical
    # distribution of that measure at the current analysis level.
    diag_color = sns.color_palette('colorblind')[0]
    for k, var in enumerate(r_matrix.columns):
        vals = pd.to_numeric(data[var], errors='coerce').dropna().values
        diag_ax = ax.inset_axes([k + 0.07, k + 0.07, 0.86, 0.86],
                                transform=ax.transData)
        diag_ax.set_facecolor('white')
        if len(vals) >= 3 and np.ptp(vals) > 0:
            # Square-root rule for bin count, clamped to a sensible range
            n_bins = int(np.clip(np.ceil(np.sqrt(len(vals))), 4, 15))
            diag_ax.hist(vals, bins=n_bins, color=diag_color, alpha=0.85,
                         edgecolor='white', linewidth=0.4)
        elif len(vals) >= 3:
            diag_ax.text(0.5, 0.5, 'constant', ha='center', va='center',
                         fontsize=annot_fs * 0.8, transform=diag_ax.transAxes,
                         color='.3')
        else:
            diag_ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                         fontsize=annot_fs * 0.8, transform=diag_ax.transAxes,
                         color='.3')
        diag_ax.set_xticks([])
        diag_ax.set_yticks([])
        for spine in diag_ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('.7')

    # Upper triangle: effect-strength circles. The circle radius scales with
    # |r| (radius = 0.47 * |r|), the fill colour encodes the sign of r on the
    # same diverging scale as the heatmap / colourbar, and pairs with
    # p < .05 receive a black outline. This fills the previously empty upper
    # triangle with an at-a-glance overview of effect magnitude and direction.
    try:
        cmap = matplotlib.colormaps['vlag']
    except AttributeError:  # matplotlib < 3.6 fallback
        cmap = plt.get_cmap('vlag')
    norm = plt.Normalize(vmin=-1, vmax=1)
    for i in range(n_var):
        for j in range(n_var):
            if j <= i:  # lower triangle and diagonal handled elsewhere
                continue
            r_val = r_matrix.iloc[i, j]
            if not np.isfinite(r_val):
                continue
            circle = Circle((j + 0.5, i + 0.5), 0.47 * abs(r_val),
                            facecolor=cmap(norm(r_val)),
                            edgecolor='white', linewidth=0.5)
            if star_mat.iloc[i, j]:  # significant pair (p < .05)
                circle.set_edgecolor('black')
                circle.set_linewidth(1.2)
            ax.add_patch(circle)

    # Category blocks: variables are grouped by measure type (see
    # VARIABLE_CATEGORIES). Each block is visually separated by thick black
    # boundary lines, marked with a coloured band along the top and left
    # edges, labelled with the category name above the top band, and its
    # tick labels are tinted with the category colour.
    col_to_cat = {v: cat for cat, vs in VARIABLE_CATEGORIES for v in vs}
    cat_colors = dict(zip([c for c, _ in VARIABLE_CATEGORIES],
                          sns.color_palette('colorblind',
                                            len(VARIABLE_CATEGORIES))))
    blocks = []  # (category, start column, end column), end exclusive
    start = 0
    for k in range(1, n_var + 1):
        if k == n_var or (col_to_cat[r_matrix.columns[k]]
                          != col_to_cat[r_matrix.columns[start]]):
            blocks.append((col_to_cat[r_matrix.columns[start]], start, k))
            start = k
    band = 0.30  # edge-band width in cell units
    for cat, s, e in blocks:
        color = cat_colors[cat]
        # Coloured bands along the top and left edges, just outside the axes
        ax.add_patch(Rectangle((s, -band), e - s, band, facecolor=color,
                               edgecolor='white', linewidth=0.8,
                               clip_on=False, zorder=7))
        ax.add_patch(Rectangle((-band, s), band, e - s, facecolor=color,
                               edgecolor='white', linewidth=0.8,
                               clip_on=False, zorder=7))
        # Category name centred above the top band
        ax.text((s + e) / 2.0, -band - 0.45, cat, ha='center', va='center',
                fontsize=tick_fs * 0.95, fontweight='semibold', color=color,
                clip_on=False, zorder=7)
    # Thick separators between category blocks, spanning the full matrix
    for _, s, _ in blocks[1:]:
        ax.plot([s, s], [0, n_var], color='black', lw=2.8, zorder=6)
        ax.plot([0, n_var], [s, s], color='black', lw=2.8, zorder=6)

    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(labels, rotation=0, fontsize=tick_fs)
    # Tint the tick labels with their category colour and push the y labels
    # clear of the left edge band
    for tick, col in zip(ax.get_xticklabels(), r_matrix.columns):
        tick.set_color(cat_colors[col_to_cat[col]])
    for tick, col in zip(ax.get_yticklabels(), r_matrix.columns):
        tick.set_color(cat_colors[col_to_cat[col]])
    ax.tick_params(axis='y', pad=22)

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {out_path}")
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)


# Primary: subject-averaged matrix (N = 36 subjects), based on Spearman rho:
# both the circle/matrix values AND the significance stars / black outlines
# come from the Spearman table, and the colourbar is labelled accordingly.
plot_correlation_matrix(
    matrix_from_table(stat_tables['subject_averaged']['spearman'], subject_matrix_vars),
    stat_tables['subject_averaged']['spearman'],
    N_SUB,
    f"Subject Averages (N = {N_SUB})",
    "correlation_matrix_subject_averages.png",
    data=sub_df[subject_matrix_vars].astype(float),
    corr_label='Spearman \u03c1'
)

# Supplementary: pooled condition rows (N = 144; repeated measures, reference
# only), likewise based on Spearman rho throughout
plot_correlation_matrix(
    matrix_from_table(stat_tables['pooled_conditions']['spearman'], CATEGORY_VAR_ORDER),
    stat_tables['pooled_conditions']['spearman'],
    len(df),
    f"All Condition Rows Pooled (N = {len(df)}, repeated measures)",
    "correlation_matrix_pooled_conditions.png",
    data=df[CATEGORY_VAR_ORDER].astype(float),
    corr_label='Spearman \u03c1'
)

print("\n" + "=" * 70)
print(f"ANALYSIS COMPLETE \u2014 figures and stats tables saved to:\n  {OUT_DIR}")
print("=" * 70)
