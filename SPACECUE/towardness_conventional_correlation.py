"""
Within-subject correlations between cursor towardness and conventional
behavioral metrics (accuracy, response time), based on the concatenated
SPACECUE behavioral table.

For every subject, trial-level towardness scores (target / distractor /
control) are correlated (Pearson r; for the binary accuracy metric this is
equivalent to the point-biserial correlation) with

    1. trial accuracy (IsCorrect, 0/1; all valid trials)
    2. trial response time (rt in seconds; all trials with a valid rt)

Each within-subject r is Fisher z transformed (z = arctanh(r), r clipped to
+/-0.9999 to avoid infinite values). Group-level inference is then performed
on the z values:

    - one-sample t-tests against 0 (overall and per age cohort), with
      Shapiro-Wilk normality checks and Cohen's dz
    - cohort comparisons (Welch t / Mann-Whitney with Shapiro checks,
      Cohen's d pooled)
    - Holm-Bonferroni correction across tests within each family

Outputs:
    - printed statistical report (t, df, exact p, effect sizes)
    - subject-level results CSV in <data_path>/concatenated/
    - publication figure in SPACECUE/plots/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback when running directly in interactive python console
    project_root = os.path.abspath(os.getcwd())

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE

sns.set_theme(context="talk", style="ticks")
plt.close('all')

# --- analysis configuration -------------------------------------------------
COHORTS = ['Younger adults', 'Older adults']
COHORT_COLORS = {COHORTS[0]: sns.color_palette('colorblind')[0],
                 COHORTS[1]: sns.color_palette('colorblind')[1]}
TOWARDNESS_COLS = {
    'Target': 'target_towardness',
    'Distractor': 'distractor_towardness',
    'Control': 'control_towardness',
}
CONVENTIONAL_COLS = {
    'Accuracy': 'IsCorrect',   # 0/1; Pearson r == point-biserial r
    'RT': 'rt',                # seconds; correct trials only
}
MIN_TRIALS = 30       # minimum trials per cell for a within-subject correlation


def stars_from_p(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def fisher_z(r):
    """Fisher z transform with clipping to avoid infinite values."""
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r_clipped)


def within_subject_correlations(df):
    """Trial-level Pearson r (and Fisher z) per subject for all metric pairs.

    All correlations use all valid trials: accuracy on trials with a valid
    IsCorrect, RT on all trials with a valid rt. Subjects with fewer than
    MIN_TRIALS valid trials in a cell get NaN.
    """
    results = []
    for (sub_id, cohort), sub_df in df.groupby(['subject_id', 'cohort']):
        for tow_label, tow_col in TOWARDNESS_COLS.items():
            for conv_label, conv_col in CONVENTIONAL_COLS.items():
                cell = sub_df.dropna(subset=[tow_col, conv_col])
                n = len(cell)
                r = np.nan
                if n >= MIN_TRIALS:
                    x = cell[tow_col].to_numpy(dtype=float)
                    y = cell[conv_col].to_numpy(dtype=float)
                    # guard against zero variance in either variable
                    if x.std() > 0 and y.std() > 0:
                        r = np.corrcoef(x, y)[0, 1]
                results.append({
                    'subject_id': sub_id, 'cohort': cohort,
                    'towardness': tow_label, 'conventional': conv_label,
                    'n_trials': n, 'r': r, 'fisher_z': fisher_z(r),
                })
    return pd.DataFrame(results)



def one_sample_report(z_vals, label_pair, cohort_label):
    """One-sample test against 0 on Fisher z values (with assumption check).

    Shapiro-Wilk on the z values: if normality holds (p >= .05) a one-sample
    t-test is run, otherwise a Wilcoxon signed-rank test as fallback.
    Effect size is Cohen's dz = mean(z) / sd(z) on the raw z values.
    """
    z_vals = np.asarray(z_vals, dtype=float)
    z_vals = z_vals[~np.isnan(z_vals)]
    shapiro_p = float(pg.normality(z_vals)['pval'].iloc[0])
    mean_z = float(z_vals.mean())
    if shapiro_p >= 0.05:
        t_df = pg.ttest(z_vals, 0)
        res = {'test': 't', 'stat': float(t_df['T'].iloc[0]),
               'df': int(t_df['dof'].iloc[0]), 'p': float(t_df['p-val'].iloc[0])}
    else:
        w_df = pg.wilcoxon(z_vals)
        res = {'test': 'W', 'stat': float(w_df['W-val'].iloc[0]),
               'df': np.nan, 'p': float(w_df['p-val'].iloc[0])}
    res.update({
        'cohort': cohort_label, 'pair': label_pair, 'n': z_vals.size,
        'mean_z': mean_z,
        'mean_r_backtransformed': float(np.tanh(mean_z)),  # r back-transformed from mean z
        'dz': float(mean_z / z_vals.std(ddof=1)),
        'shapiro_p': shapiro_p,
    })
    return res



def cohort_comparison(z_young, z_old):
    """Independent-samples comparison of Fisher z between cohorts (Welch t / MWU)."""
    z_young = np.asarray(z_young, dtype=float)
    z_old = np.asarray(z_old, dtype=float)
    z_young, z_old = z_young[~np.isnan(z_young)], z_old[~np.isnan(z_old)]
    sh_a = float(pg.normality(z_young)['pval'].iloc[0])
    sh_b = float(pg.normality(z_old)['pval'].iloc[0])
    if sh_a >= 0.05 and sh_b >= 0.05:
        t_df = pg.ttest(z_young, z_old, correction='auto')
        res = {'test': 't', 'stat': float(t_df['T'].iloc[0]),
               'df': float(t_df['dof'].iloc[0]), 'p': float(t_df['p-val'].iloc[0])}
    else:
        mw = pg.mwu(z_young, z_old)
        res = {'test': 'U', 'stat': float(mw['U-val'].iloc[0]),
               'df': np.nan, 'p': float(mw['p-val'].iloc[0])}
    na, nb = z_young.size, z_old.size
    sp = np.sqrt(((na - 1) * z_young.var(ddof=1) + (nb - 1) * z_old.var(ddof=1))
                 / (na + nb - 2))
    res.update({'d': float((z_young.mean() - z_old.mean()) / sp),
                'shapiro_p_young': sh_a, 'shapiro_p_old': sh_b})
    return res


def main():
    print("=====================================================================")
    print("   WITHIN-SUBJECT TOWARDNESS <-> CONVENTIONAL METRIC CORRELATIONS   ")
    print("=====================================================================\n")

    # 1. Load data -----------------------------------------------------------
    data_path = SPACECUE.get_data_path()
    csv_path = os.path.join(data_path, 'concatenated', 'concatenated_behavioral.csv')
    print(f"Loading concatenated behavioral data from: {csv_path}")
    if not os.path.exists(csv_path):
        print("Error: Concatenated behavioral data file not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)

    from SPACECUE.subjects import subject_ids
    df = df[df['subject_id'].isin(subject_ids)].copy()

    # 2. Cohort assignment ---------------------------------------------------
    df['cohort'] = np.where(df['Age'] <= 35, COHORTS[0], COHORTS[1])
    print(f"Subjects: {df['subject_id'].nunique()} | trials: {len(df)}\n")

    # 3. Within-subject correlations ------------------------------------------
    print("Computing within-subject Pearson r (Fisher z transformed) ...")
    sub_corr = within_subject_correlations(df)
    print(f"  -> {len(sub_corr)} correlation cells "
          f"({df['subject_id'].nunique()} subjects x {len(TOWARDNESS_COLS)} x {len(CONVENTIONAL_COLS)})\n")

    PAIRS = [(t, c) for t in TOWARDNESS_COLS for c in CONVENTIONAL_COLS]

    # 4. Group-level inference on Fisher z ------------------------------------
    report_rows = []
    cohort_comp_rows = []
    for tow_label, conv_label in PAIRS:
        pair_name = f"{tow_label} towardness ~ {conv_label}"
        sel = sub_corr[(sub_corr['towardness'] == tow_label)
                       & (sub_corr['conventional'] == conv_label)]

        # overall + per-cohort one-sample tests (Holm-corrected within pair)
        family_rows = []
        for cohort_label in [None] + COHORTS:
            z = sel['fisher_z'] if cohort_label is None \
                else sel.loc[sel['cohort'] == cohort_label, 'fisher_z']
            row = one_sample_report(z, pair_name, cohort_label or 'Overall')
            family_rows.append(row)
            report_rows.append(row)

        _, p_holm = pg.multicomp([r['p'] for r in family_rows], alpha=0.05, method='holm')
        for row, p_h in zip(family_rows, p_holm):
            row['p_holm'] = float(p_h)

        # cohort comparison on Fisher z
        comp = cohort_comparison(sel.loc[sel['cohort'] == COHORTS[0], 'fisher_z'],
                                 sel.loc[sel['cohort'] == COHORTS[1], 'fisher_z'])
        comp.update({'pair': pair_name})
        cohort_comp_rows.append(comp)

    report_df = pd.DataFrame(report_rows)
    comp_df = pd.DataFrame(cohort_comp_rows)

    # 5. Print full statistical report ----------------------------------------
    print("\n=== One-sample tests on Fisher z (vs. 0), Holm-corrected within each pair ===")
    for pair_name in [f"{t} towardness ~ {c}" for t, c in PAIRS]:
        print(f"\n--- {pair_name} ---")
        for _, r in report_df[report_df['pair'] == pair_name].iterrows():
            stars = stars_from_p(r['p_holm'])
            if r['test'] == 't':
                stat_str = f"t({r['df']}) = {r['stat']:.3f}"
            else:
                stat_str = f"W = {r['stat']:.1f}"
            print(f"  {r['cohort']:<15s} n = {r['n']:2d} | "
                  f"mean r = {r['mean_r_backtransformed']:+.3f} (mean z = {r['mean_z']:+.3f}) | "
                  f"{stat_str}, p = {r['p']:.4g}, "
                  f"p_holm = {r['p_holm']:.4g} {stars}, dz = {r['dz']:.2f}, "
                  f"Shapiro-W p = {r['shapiro_p']:.3g}")

    print("\n=== Cohort comparisons on Fisher z (Younger vs. Older) ===")
    for _, r in comp_df.iterrows():
        if r['test'] == 't':
            stat_str = f"t({r['df']:.1f}) = {r['stat']:.3f}"
        else:
            stat_str = f"U = {r['stat']:.1f}"
        print(f"  {r['pair']}: {stat_str}, p = {r['p']:.4g}, d = {r['d']:.2f}, "
              f"Shapiro p (Y/O) = {r['shapiro_p_young']:.3g} / {r['shapiro_p_old']:.3g}")

    # 6. Save results ----------------------------------------------------------
    out_dir = os.path.join(data_path, 'concatenated')
    results_file = os.path.join(out_dir, 'towardness_conventional_correlation_subject_level.csv')
    report_file = os.path.join(out_dir, 'towardness_conventional_correlation_group_report.csv')
    sub_corr.to_csv(results_file, index=False)
    report_df.to_csv(report_file, index=False)
    print(f"\nSaved subject-level results to: {results_file}")
    print(f"Saved group-level report to: {report_file}")

    plot_and_save_figure(sub_corr, report_df)


def plot_and_save_figure(sub_corr, report_df):
    """Publication figure: Fisher z summary (A) and target-towardness r map (B)."""
    PAIRS = [(t, c) for t in TOWARDNESS_COLS for c in CONVENTIONAL_COLS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.3})

    # ---- Panel A: mean Fisher z per pair and cohort -----------------------
    ax_a = axes[0]
    x_pos = np.arange(len(PAIRS))
    pair_labels = [f"{t}\n~ {c}" for t, c in PAIRS]
    hue_offset = 0.16  # horizontal offset between the two cohorts

    np.random.seed(42)
    for (tow_label, conv_label), cohort in [(p, c) for p in PAIRS for c in COHORTS]:
        sel = sub_corr[(sub_corr['towardness'] == tow_label)
                       & (sub_corr['conventional'] == conv_label)
                       & (sub_corr['cohort'] == cohort)]
        xi = x_pos[PAIRS.index((tow_label, conv_label))] \
            + (hue_offset if cohort == COHORTS[1] else -hue_offset)
        color = COHORT_COLORS[cohort]

        # subject-level dots
        ax_a.scatter(np.full(len(sel), xi) + np.random.uniform(-0.04, 0.04, len(sel)),
                     sel['fisher_z'], color=color, alpha=0.45, s=28, zorder=3,
                     edgecolor='none')

        # cohort mean +/- SEM
        m, sem = sel['fisher_z'].mean(), sel['fisher_z'].sem()
        ax_a.errorbar(xi, m, yerr=sem, fmt='o', color=color, ecolor=color,
                      elinewidth=2.2, capsize=5, capthick=1.8, markersize=8, zorder=6)

        # Holm-corrected significance of the one-sample test vs. 0
        row = report_df[(report_df['pair'] == f"{tow_label} towardness ~ {conv_label}")
                        & (report_df['cohort'] == cohort)].iloc[0]
        y_star = m + sem + (0.12 if m >= 0 else -0.12)
        ax_a.text(xi, y_star, stars_from_p(row['p_holm']), ha='center',
                  va='bottom' if m >= 0 else 'top',
                  fontsize=10, fontweight='bold', color=color, zorder=7)

    ax_a.axhline(0, color='#9E9E9E', linestyle='--', lw=1.2, zorder=1)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(pair_labels, fontsize=10)
    ax_a.set_ylabel('Within-subject Fisher $z$')
    ax_a.set_title('Towardness-behavior correlations', loc='left', pad=12,
                   fontweight='bold', fontsize=13, color='#212121')
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=COHORT_COLORS[c], label=c)
               for c in COHORTS]
    ax_a.legend(handles=handles, frameon=False, fontsize=10, loc='upper right')
    sns.despine(ax=ax_a)

    # ---- Panel B: subject-level r for the target towardness pairs ---------
    ax_b = axes[1]
    for cohort in COHORTS:
        sel_y = sub_corr[(sub_corr['towardness'] == 'Target')
                         & (sub_corr['conventional'] == 'Accuracy')
                         & (sub_corr['cohort'] == cohort)]
        sel_r = sub_corr[(sub_corr['towardness'] == 'Target')
                         & (sub_corr['conventional'] == 'RT')
                         & (sub_corr['cohort'] == cohort)]
        merged = sel_y.merge(sel_r, on='subject_id', suffixes=('_acc', '_rt'))
        ax_b.scatter(merged['r_acc'], merged['r_rt'], color=COHORT_COLORS[cohort],
                     alpha=0.75, s=60, label=cohort, zorder=4,
                     edgecolor='white', linewidth=0.8)

    ax_b.axhline(0, color='#9E9E9E', linestyle='--', lw=1.2, zorder=1)
    ax_b.axvline(0, color='#9E9E9E', linestyle='--', lw=1.2, zorder=1)
    ax_b.set_xlabel('r (Target towardness ~ Accuracy)')
    ax_b.set_ylabel('r (Target towardness ~ RT (s))')
    ax_b.set_title('Target towardness: accuracy vs. speed', loc='left', pad=12,
                   fontweight='bold', fontsize=13, color='#212121')
    ax_b.legend(frameon=False, fontsize=10, loc='best')
    sns.despine(ax=ax_b)

    # save ------------------------------------------------------------------
    plots_dir = os.path.join(project_root, 'SPACECUE', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    png_path = os.path.join(plots_dir, 'towardness_conventional_correlations.png')
    svg_path = os.path.join(plots_dir, 'towardness_conventional_correlations.svg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved figure to {png_path} and {svg_path}")
    plt.close(fig)  # memory guardrail


if __name__ == "__main__":
    main()


