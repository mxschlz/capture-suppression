import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pingouin as pg
from scipy.stats import sem

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback when running directly in interactive python console
    project_root = os.path.abspath(os.getcwd())

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
from SPACECUE.colors import (
    COLOR_TARGET_YOUNG, COLOR_TARGET_OLD,
    COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD,
    COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD,
)


# ----------------------------------------------------------------------------
# Effect-size helpers
# ----------------------------------------------------------------------------
def cohens_dz(high, low):
    """Cohen's dz for paired designs: mean(diff) / SD(diff).

    NOTE: pingouin's 'cohen-d' for paired t-tests is d_av (mean difference
    divided by the average SD of the two conditions), which ignores the
    within-subject correlation. It is NOT dz and must not be labelled as such.
    """
    diff = (high - low).dropna()
    return diff.mean() / diff.std(ddof=1)


def main():
    print("=========================================================")
    print("   SPACECUE BEHAVIORAL COMPARISON: YOUNG VS. OLD COHORTS  ")
    print("=========================================================\n")

    # 1. Load data
    data_path = SPACECUE.get_data_path()
    csv_path = os.path.join(data_path, "concatenated", "concatenated_behavioral.csv")
    print(f"Loading concatenated behavioral data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Concatenated behavioral data file not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Filter by subjects.py
    from SPACECUE.subjects import subject_ids
    df = df[df['subject_id'].isin(subject_ids)].copy()
    
    # Exclude frontal targets (performance drops hugely here, confound in LP distractor locations)
    df = df[df['TargetLoc'] != 'Front'].copy()
    
    # 2. Cohort assignment
    # Young: Age <= 35 (sub 1-4), Old: Age > 35 (sub 6-11)
    df['cohort'] = df['Age'].apply(lambda x: 'Young' if x <= 35 else 'Old')
    
    # Report subject demographic characteristics
    sub_demographics = df.groupby('subject_id')[['Age', 'Gender', 'cohort']].first()
    print("--- Demographics ---")
    print(sub_demographics)
    print()

    # 4. Subject-Level Aggregation
    # Compute mean metrics for each subject
    sub_perf = df.groupby('subject_id')['IsCorrect'].mean().reset_index()
    sub_rt = df.groupby('subject_id')['rt'].mean().reset_index()
    sub_target_tow = df.groupby('subject_id')['target_towardness'].mean().reset_index()
    sub_dist_tow = df.groupby('subject_id')['distractor_towardness'].mean().reset_index()
    sub_ctrl_tow = df.groupby('subject_id')['control_towardness'].mean().reset_index()

    # Merge subject level metrics
    sub_metrics = sub_perf.merge(sub_rt, on='subject_id') \
                           .merge(sub_target_tow, on='subject_id') \
                           .merge(sub_dist_tow, on='subject_id') \
                           .merge(sub_ctrl_tow, on='subject_id')
    
    # Append cohort info
    sub_metrics = sub_metrics.merge(sub_demographics.reset_index(), on='subject_id')
    
    # 5. Cohort Summary Stats
    cohort_summary = sub_metrics.groupby('cohort')[['IsCorrect', 'rt', 'target_towardness', 'distractor_towardness', 'control_towardness']].agg(['mean', 'std', 'sem'])
    
    # 6. Statistical Comparisons between Groups (Independent t-tests)
    stats_results = {}
    variables = {
        'IsCorrect': 'Accuracy',
        'rt': 'Reaction Time',
        'target_towardness': 'Target Towardness',
        'distractor_towardness': 'Distractor Towardness',
        'control_towardness': 'Control Towardness'
    }
    
    print("--- Cohort Comparisons (Independent samples t-tests) ---")
    for var, label in variables.items():
        young_data = sub_metrics[sub_metrics['cohort'] == 'Young'][var]
        old_data = sub_metrics[sub_metrics['cohort'] == 'Old'][var]
        
        ttest_res = pg.ttest(young_data, old_data, paired=False)
        stats_results[var] = ttest_res
        
        print(f"\n{label}:")
        print(f"  Young Mean: {young_data.mean():.4f} ± {young_data.std():.4f}")
        print(f"  Old Mean:   {old_data.mean():.4f} ± {old_data.std():.4f}")
        print(f"  t({ttest_res['dof'].values[0]:.1f}) = {ttest_res['T'].values[0]:.4f}, p = {ttest_res['p-val'].values[0]:.4f}, Cohen's d = {ttest_res['cohen-d'].values[0]:.4f}")

    # 7. Spatial Suppression Learning Analysis (High vs. Low Distractor Location Probability)
    # We aggregate RT, Accuracy, and Towardness metrics per subject for high-probability and low-probability trials.
    sub_prob_data = {}
    sub_prob_data['rt'] = df.groupby(['subject_id', 'cohort', 'DistractorProb'])['rt'].mean().reset_index()
    sub_prob_data['IsCorrect'] = df.groupby(['subject_id', 'cohort', 'DistractorProb'])['IsCorrect'].mean().reset_index()
    sub_prob_data['target_towardness'] = df.groupby(['subject_id', 'cohort', 'DistractorProb'])['target_towardness'].mean().reset_index()
    sub_prob_data['distractor_towardness'] = df.groupby(['subject_id', 'cohort', 'DistractorProb'])['distractor_towardness'].mean().reset_index()
    sub_prob_data['control_towardness'] = df.groupby(['subject_id', 'cohort', 'DistractorProb'])['control_towardness'].mean().reset_index()
    
    # Run Mixed ANOVAs to test for interactions between Cohort (between-subject) and Distractor Probability (within-subject)
    print("\n--- Mixed ANOVAs (Cohort x Distractor Probability) ---")
    
    anova_results = {}
    for var, label in zip(['rt', 'IsCorrect', 'target_towardness', 'distractor_towardness', 'control_towardness'],
                          ['Reaction Time', 'Accuracy', 'Target Towardness', 'Distractor Towardness', 'Control Towardness']):
        anova_res = pg.mixed_anova(data=sub_prob_data[var], dv=var, between='cohort', within='DistractorProb', subject='subject_id')
        anova_results[var] = anova_res
        print(f"\n{label} Mixed ANOVA:")
        print(anova_res[['Source', 'SS', 'DF1', 'DF2', 'F', 'p-unc', 'np2']])
    
    # 7b. Follow-up Paired Comparisons within Cohorts (High vs. Low Distractor Probability)
    print("\n--- Follow-up Paired Comparisons (High vs. Low Distractor Probability within Cohorts) ---")
    paired_stats_results = {}
    for var, label in zip(['target_towardness', 'distractor_towardness', 'control_towardness'],
                          ['Target Towardness', 'Distractor Towardness', 'Control Towardness']):
        y_high = sub_prob_data[var][(sub_prob_data[var]['cohort'] == 'Young') & (sub_prob_data[var]['DistractorProb'] == 'high-probability')].set_index('subject_id')[var]
        y_low = sub_prob_data[var][(sub_prob_data[var]['cohort'] == 'Young') & (sub_prob_data[var]['DistractorProb'] == 'low-probability')].set_index('subject_id')[var]
        tt_y = pg.ttest(y_high, y_low, paired=True)
        
        o_high = sub_prob_data[var][(sub_prob_data[var]['cohort'] == 'Old') & (sub_prob_data[var]['DistractorProb'] == 'high-probability')].set_index('subject_id')[var]
        o_low = sub_prob_data[var][(sub_prob_data[var]['cohort'] == 'Old') & (sub_prob_data[var]['DistractorProb'] == 'low-probability')].set_index('subject_id')[var]
        tt_o = pg.ttest(o_high, o_low, paired=True)
        
        paired_stats_results[var] = {
            'Young': tt_y, 'Old': tt_o, 
            'y_high': y_high, 'y_low': y_low, 
            'o_high': o_high, 'o_low': o_low
        }
        
        print(f"\n{label} Paired Comparisons (High vs. Low Distractor Location Probability):")
        print(f"  Young (N={len(y_high)}): High={y_high.mean():.2f} ± {y_high.std():.2f}, Low={y_low.mean():.2f} ± {y_low.std():.2f}")
        print(f"    t({tt_y['dof'].values[0]:.1f}) = {tt_y['T'].values[0]:.4f}, p = {tt_y['p-val'].values[0]:.4f}, Cohen's dz = {cohens_dz(y_high, y_low):.4f}")
        print(f"  Old   (N={len(o_high)}): High={o_high.mean():.2f} ± {o_high.std():.2f}, Low={o_low.mean():.2f} ± {o_low.std():.2f}")
        print(f"    t({tt_o['dof'].values[0]:.1f}) = {tt_o['T'].values[0]:.4f}, p = {tt_o['p-val'].values[0]:.4f}, Cohen's dz = {cohens_dz(o_high, o_low):.4f}")

    def draw_paired_bracket(ax, x1, x2, y_base, h, text, color, lw=1.4):
        """Draws a clean paired comparison bracket with label."""
        ax.plot([x1, x1, x2, x2], [y_base, y_base + h, y_base + h, y_base], 
                color=color, lw=lw, clip_on=False)
        ax.text((x1 + x2) * 0.5, y_base + h * 1.3, text, 
                ha='center', va='bottom', color=color, 
                fontweight='bold', fontsize=10)

    # 8. Plotting Summary Figure
    print("\nGenerating behavioral summary figures...")
    sns.set_theme(style="ticks", context="talk")
    # Narrow format: compressing the x-axis steepens the High-vs-Low
    # probability slopes in the bottom row so those effects stand out
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    
    # SPACECUE standard palette (SPACECUE/colors.py): light shade = Young,
    # dark shade = Old. Green family for general behavioral metrics
    # (consistent with the cohort colors in plot_mean_cursor_trajectories.py).
    cohort_colors = {'Young': COLOR_TARGET_YOUNG, 'Old': COLOR_TARGET_OLD}
    
    # Per-stimulus-family cohort shades for the towardness panels
    family_cohort_colors = {
        'target_towardness': {'Young': COLOR_TARGET_YOUNG, 'Old': COLOR_TARGET_OLD},
        'distractor_towardness': {'Young': COLOR_DISTRACTOR_YOUNG, 'Old': COLOR_DISTRACTOR_OLD},
        'control_towardness': {'Young': COLOR_CONTROL_YOUNG, 'Old': COLOR_CONTROL_OLD},
    }
    
    # Panel A: Reaction Time
    sns.barplot(data=sub_metrics, x='cohort', y='rt', ax=axes[0, 0], palette=cohort_colors, errorbar='se', capsize=0.1, alpha=0.3)
    sns.stripplot(data=sub_metrics, x='cohort', y='rt', ax=axes[0, 0], hue='cohort', palette=cohort_colors, size=8, jitter=0.1, linewidth=1, edgecolor='gray', legend=False)
    axes[0, 0].set_title("Response Time", fontweight='bold')
    axes[0, 0].set_ylabel("RT (s)")
    axes[0, 0].set_xlabel("Age Cohort")
    rt_p = stats_results['rt']['p-val'].values[0]
    axes[0, 0].text(0.5, axes[0, 0].get_ylim()[1]*0.9, f"p = {rt_p:.3f}\nd = {stats_results['rt']['cohen-d'].values[0]:.2f}", ha='center', fontweight='bold')
    sns.despine(ax=axes[0, 0])
    
    # Panel B: Accuracy
    sns.barplot(data=sub_metrics, x='cohort', y='IsCorrect', ax=axes[0, 1], palette=cohort_colors, errorbar='se', capsize=0.1, alpha=0.3)
    sns.stripplot(data=sub_metrics, x='cohort', y='IsCorrect', ax=axes[0, 1], hue='cohort', palette=cohort_colors, size=8, jitter=0.1, linewidth=1, edgecolor='gray', legend=False)
    axes[0, 1].set_title("Accuracy", fontweight='bold')
    axes[0, 1].set_ylabel("Proportion correct")
    axes[0, 1].set_xlabel("Age Cohort")
    acc_p = stats_results['IsCorrect']['p-val'].values[0]
    axes[0, 1].text(0.5, axes[0, 1].get_ylim()[1]*0.9, f"p = {acc_p:.3f}\nd = {stats_results['IsCorrect']['cohen-d'].values[0]:.2f}", ha='center', fontweight='bold')
    sns.despine(ax=axes[0, 1])
    
    # Panel C: Towardness Scores Grouped Barplot
    # Melt towardness columns for plotting
    toward_melt = sub_metrics.melt(id_vars=['subject_id', 'cohort'], 
                                   value_vars=['target_towardness', 'distractor_towardness', 'control_towardness'],
                                   var_name='metric', value_name='score')
    toward_melt['metric'] = toward_melt['metric'].replace({
        'target_towardness': 'Target',
        'distractor_towardness': 'Distractor',
        'control_towardness': 'Control'
    })
    
    # Grouped bars with the SPACECUE stimulus-family cohort shades
    # (light = Young, dark = Old per stimulus family; SPACECUE/colors.py)
    bar_family_colors = [
        ('Target', COLOR_TARGET_YOUNG, COLOR_TARGET_OLD),
        ('Distractor', COLOR_DISTRACTOR_YOUNG, COLOR_DISTRACTOR_OLD),
        ('Control', COLOR_CONTROL_YOUNG, COLOR_CONTROL_OLD),
    ]
    for x_idx, (metric, c_young, c_old) in enumerate(bar_family_colors):
        sub_m = toward_melt[toward_melt['metric'] == metric]
        for cohort, col, x_off in [('Young', c_young, -0.19), ('Old', c_old, 0.19)]:
            vals = sub_m.loc[sub_m['cohort'] == cohort, 'score'].dropna()
            axes[0, 2].bar(x_idx + x_off, vals.mean(), width=0.36, color=col,
                           alpha=0.7, yerr=sem(vals), capsize=4,
                           error_kw=dict(lw=1.5, capthick=1.5))
    axes[0, 2].set_xticks([0, 1, 2])
    axes[0, 2].set_xticklabels(['Target', 'Distractor', 'Control'])
    axes[0, 2].set_title("Towardness by Cohort", fontweight='bold', fontsize=13)
    axes[0, 2].set_ylabel("Towardness (%)")
    axes[0, 2].set_xlabel("Stimulus Dimension")
    axes[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5, lw=1)
    axes[0, 2].legend(handles=[Patch(facecolor=COLOR_TARGET_YOUNG, alpha=0.7, label='Young'),
                               Patch(facecolor=COLOR_TARGET_OLD, alpha=0.7, label='Old')],
                      title='Cohort', loc='upper right')
    sns.despine(ax=axes[0, 2])
    
    # Add stats comparison on all Towardness metrics (Young vs Old cohorts)
    axes[0, 2].set_ylim(-38, 55)
    for x_idx, metric_key in enumerate(['target_towardness', 'distractor_towardness', 'control_towardness']):
        p_val = stats_results[metric_key]['p-val'].values[0]
        cohen_d = stats_results[metric_key]['cohen-d'].values[0]
        star = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        if metric_key == 'distractor_towardness':
            y_pos = -35
            va = 'bottom'
        elif metric_key == 'target_towardness':
            y_pos = 42
            va = 'bottom'
        else:
            y_pos = 12
            va = 'bottom'
            
        axes[0, 2].text(x_idx, y_pos, f"{star}\np = {p_val:.3f}\nd = {cohen_d:.2f}", 
                        ha='center', va=va, fontweight='bold', fontsize=11, color='black',
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray', boxstyle='round,pad=0.2', lw=0.5))

    # Panels D, E, F: Towardness Interactions (Target, Distractor, Control)
    interaction_vars = {
        'target_towardness': ('Target Towardness', axes[1, 0]),
        'distractor_towardness': ('Distractor Towardness', axes[1, 1]),
        'control_towardness': ('Control Towardness', axes[1, 2])
    }
    
    for var, (label, ax) in interaction_vars.items():
        # SPACECUE cohort shades for this stimulus family (light = Young, dark = Old)
        cohort_colors = family_cohort_colors[var]
        
        # Rename for plotting
        plot_df = sub_prob_data[var].copy()
        plot_df['DistractorProb'] = plot_df['DistractorProb'].replace({
            'high-probability': 'High Prob',
            'low-probability': 'Low Prob'
        })
        
        sns.pointplot(data=plot_df, x='DistractorProb', y=var, hue='cohort', ax=ax, palette=cohort_colors, markers=['o', 's'], linestyles=['-', '--'], dodge=0.1, errorbar='se', capsize=0.1)

        # Thicken the cohort mean lines (they span the two x categories,
        # unlike the short error-bar caps) so the High-vs-Low probability
        # effect dominates over the thin per-subject lines
        for ln in ax.lines:
            xd = np.asarray(ln.get_xdata())
            if len(xd) >= 2 and (np.max(xd) - np.min(xd)) > 0.5:
                ln.set_linewidth(3.0)

        # Plot individual subject lines to show within-subject changes
        for sub_id in plot_df['subject_id'].unique():
            sub_data = plot_df[plot_df['subject_id'] == sub_id]
            cohort = sub_data['cohort'].iloc[0]
            color = cohort_colors[cohort]
            # Map categories to x coordinate
            x_coords = [0, 1] if sub_data['DistractorProb'].iloc[0] == 'High Prob' else [1, 0]
            # Apply small dodge
            dodge_val = -0.05 if cohort == 'Young' else 0.05
            ax.plot([x + dodge_val for x in x_coords], sub_data[var], color=color, alpha=0.15, lw=1.0)
            
        ax.set_title(f"{label} x Cohort", fontweight='bold', pad=12, fontsize=13)
        ax.set_ylabel(f"{label} (%)")
        ax.set_xlabel("Distractor Probability")
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, lw=1)
        sns.despine(ax=ax)
        
        # Paired t-tests within cohorts
        tt_y = paired_stats_results[var]['Young']
        tt_o = paired_stats_results[var]['Old']
        
        p_y = tt_y['p-val'].values[0]
        dz_y = cohens_dz(paired_stats_results[var]['y_high'], paired_stats_results[var]['y_low'])
        p_o = tt_o['p-val'].values[0]
        dz_o = cohens_dz(paired_stats_results[var]['o_high'], paired_stats_results[var]['o_low'])
        
        star_y = '***' if p_y < 0.001 else ('**' if p_y < 0.01 else ('*' if p_y < 0.05 else 'n.s.'))
        star_o = '***' if p_o < 0.001 else ('**' if p_o < 0.01 else ('*' if p_o < 0.05 else 'n.s.'))
        
        # Calculate dynamic bracket positions based on data range
        y_max_data = max(plot_df[var].max(), 0 if var == 'distractor_towardness' else plot_df[var].max())
        y_min_data = plot_df[var].min()
        y_span = y_max_data - y_min_data
        
        y_b1 = y_max_data + y_span * 0.05
        h1 = y_span * 0.03
        y_b2 = y_b1 + y_span * 0.14
        h2 = y_span * 0.03
        
        top_lim = y_b2 + y_span * 0.15
        bot_lim = y_min_data - y_span * 0.05
        ax.set_ylim(bot_lim, top_lim)
        
        # Young bracket (solid line, Young blue color)
        text_y = f"Young: {star_y} (p = {p_y:.3f}, $d$ = {dz_y:.2f})"
        draw_paired_bracket(ax, -0.05, 0.95, y_b1, h1, text_y, cohort_colors['Young'])
        
        # Old bracket (solid line, Old orange color)
        text_o = f"Old: {star_o} (p = {p_o:.3f}, $d$ = {dz_o:.2f})"
        draw_paired_bracket(ax, 0.05, 1.05, y_b2, h2, text_o, cohort_colors['Old'])
        
        # Remove the pointplot legends: in the narrow format they would
        # collide with the per-subject lines, and the colored cohort
        # bracket labels ("Young:" / "Old:") already identify the two means
        legend = ax.get_legend()
        if legend:
            legend.remove()
    
    plt.tight_layout()
    
    plots_dir = os.path.join(project_root, "spacecue", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "behavioral_age_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()
    print(f"Saved summary plot to: {plot_path}")
    
    # 9. Save detailed statistical report
    report_path = os.path.join(plots_dir, "behavioral_age_comparison_report.txt")
    
    report_lines = [
        "=========================================================",
        "     SPACECUE BEHAVIORAL AGE COMPARISON REPORT           ",
        "=========================================================\n",
        "This report compares behavioral performance, reaction times,",
        "and mouse towardness metrics between younger and older adults.\n",
        f"Data Sourced: {csv_path}",
        f"Subjects Included:",
        f"  Young Cohort (N={len(sub_demographics[sub_demographics['cohort']=='Young'])}): {list(sub_demographics[sub_demographics['cohort']=='Young'].index)}",
        f"  Old Cohort (N={len(sub_demographics[sub_demographics['cohort']=='Old'])}): {list(sub_demographics[sub_demographics['cohort']=='Old'].index)}\n",
        "---------------------------------------------------------",
        "PART 1: OVERALL BEHAVIORAL COMPARISONS (Independent t-tests)",
        "---------------------------------------------------------"
    ]
    
    for var, label in variables.items():
        young_data = sub_metrics[sub_metrics['cohort'] == 'Young'][var]
        old_data = sub_metrics[sub_metrics['cohort'] == 'Old'][var]
        ttest_res = stats_results[var]
        
        report_lines.extend([
            f"\nVariable: {label}",
            f"  Young Cohort Mean: {young_data.mean():.4f} ± {young_data.std():.4f} (SEM: {sem(young_data):.4f})",
            f"  Old Cohort Mean:   {old_data.mean():.4f} ± {old_data.std():.4f} (SEM: {sem(old_data):.4f})",
            f"  T-statistic:       t({ttest_res['dof'].values[0]:.1f}) = {ttest_res['T'].values[0]:.4f}",
            f"  p-value:           p = {ttest_res['p-val'].values[0]:.4f}",
            f"  Effect Size:       Cohen's d = {ttest_res['cohen-d'].values[0]:.4f}",
            f"  Statistical Power: Power = {ttest_res['power'].values[0]:.4f}"
        ])
        
    report_lines.extend([
        "\n---------------------------------------------------------",
        "PART 2: DISTRACTOR PROBABILITY EFFECTS (Mixed ANOVA)",
        "---------------------------------------------------------",
        "\nReaction Time (RT) ANOVA:",
        anova_results['rt'][['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].to_string(index=False),
        "\nAccuracy (IsCorrect) ANOVA:",
        anova_results['IsCorrect'][['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].to_string(index=False),
        "\nTarget Towardness ANOVA:",
        anova_results['target_towardness'][['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].to_string(index=False),
        "\nDistractor Towardness ANOVA:",
        anova_results['distractor_towardness'][['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].to_string(index=False),
        "\nControl Towardness ANOVA:",
        anova_results['control_towardness'][['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].to_string(index=False),
        "\n---------------------------------------------------------",
        "PART 3: PAIRED COMPARISONS WITHIN COHORTS (High vs. Low Prob)",
        "---------------------------------------------------------"
    ])
    
    for var, label in zip(['target_towardness', 'distractor_towardness', 'control_towardness'],
                          ['Target Towardness', 'Distractor Towardness', 'Control Towardness']):
        y_info = paired_stats_results[var]
        tt_y = y_info['Young']
        tt_o = y_info['Old']
        yh = y_info['y_high']
        yl = y_info['y_low']
        oh = y_info['o_high']
        ol = y_info['o_low']
        
        report_lines.extend([
            f"\nVariable: {label}",
            f"  Young Cohort (N={len(yh)}):",
            f"    High Prob Mean:  {yh.mean():.4f} ± {yh.std():.4f} (SEM: {sem(yh):.4f})",
            f"    Low Prob Mean:   {yl.mean():.4f} ± {yl.std():.4f} (SEM: {sem(yl):.4f})",
            f"    Mean Difference: {(yh - yl).mean():.4f} ± {(yh - yl).std():.4f}",
            f"    T-statistic:     t({tt_y['dof'].values[0]:.1f}) = {tt_y['T'].values[0]:.4f}",
            f"    p-value:         p = {tt_y['p-val'].values[0]:.4f}",
            f"    Effect Size:     Cohen's dz = {cohens_dz(yh, yl):.4f}",
            f"    Power:           Power = {tt_y['power'].values[0]:.4f}",
            f"  Old Cohort (N={len(oh)}):",
            f"    High Prob Mean:  {oh.mean():.4f} ± {oh.std():.4f} (SEM: {sem(oh):.4f})",
            f"    Low Prob Mean:   {ol.mean():.4f} ± {ol.std():.4f} (SEM: {sem(ol):.4f})",
            f"    Mean Difference: {(oh - ol).mean():.4f} ± {(oh - ol).std():.4f}",
            f"    T-statistic:     t({tt_o['dof'].values[0]:.1f}) = {tt_o['T'].values[0]:.4f}",
            f"    p-value:         p = {tt_o['p-val'].values[0]:.4f}",
            f"    Effect Size:     Cohen's dz = {cohens_dz(oh, ol):.4f}",
            f"    Power:           Power = {tt_o['power'].values[0]:.4f}"
        ])
        
    report_text = "\n".join(report_lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
        
    print(f"\nSaved statistical report to: {report_path}")
    print("\n=========================================================")
    print("                    ANALYSIS COMPLETE                    ")
    print("=========================================================")

if __name__ == '__main__':
    main()
