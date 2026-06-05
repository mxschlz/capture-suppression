import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr
import pingouin as pg
import statsmodels.formula.api as smf
import SPACECUE_explicit
plt.ion()


sns.set_theme("talk", "ticks")

# ============================================================
# PARAMETERS
# ============================================================
COMPARISON_CORRECTION = "fdr_bh"  # multiple comparison correction (e.g. 'holm', 'bonferroni', 'fdr_bh')
ERRORBAR = ("se", 1)                 # error band for bar plots (e.g. ("se", 1), ("ci", 95), ("sd", 1))
N_DELAY_BINS = 3                      # number of quantile bins for cue-stimulus delay
YLIM_RT_EFFECT = (-0.2, 0.2)          # y-axis limits for RT effect plots
YLIM_ACC_EFFECT = (-0.1, 0.1)         # y-axis limits for accuracy effect plots
EFFECT_PALETTE = {'target_effect': 'g', 'distractor_effect': 'r',
                  'target_rt_benefit': 'g', 'distractor_rt_cost': 'r'}
COHORT_PALETTE = {'young': 'lightblue', 'old': 'darkblue'}
STRATEGY_PALETTE = {'Trial': 'black', 'Block': 'grey'}
PVALUE_THRESHOLDS = [[1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1.0, 'ns']]
# ============================================================

# ============================================================
# HELPER FOR ONE-SAMPLE TESTS AGAINST ZERO
# ============================================================
def add_zero_test_annotations(ax, data, x, y, hue=None, order=None, hue_order=None, correction=None):
    """
    Adds significance stars inside the bars for one-sample t-tests against 0.
    """
    rects = [p for p in ax.patches if isinstance(p, patches.Rectangle) and p.get_width() > 0]
    
    if order is not None:
        x_levels = order
    else:
        x_levels = [t.get_text() for t in ax.get_xticklabels()]
        if not x_levels:
            x_levels = list(data[x].unique())
            
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    tests = []
    
    if hue is not None:
        if hue_order is not None:
            hue_levels = hue_order
        else:
            handles, labels = ax.get_legend_handles_labels()
            hue_levels = labels if labels else list(data[hue].unique())
            
        if len(rects) >= len(x_levels) * len(hue_levels):
            idx = 0
            for h in hue_levels:
                for cat in x_levels:
                    rect = rects[idx]
                    idx += 1
                    group_data = data[(data[x] == cat) & (data[hue] == h)][y].dropna()
                    if len(group_data) >= 2:
                        p = pg.ttest(group_data, 0)['p-val'].values[0]
                        tests.append({'rect': rect, 'p': p})
    else:
        if len(rects) >= len(x_levels):
            for idx, cat in enumerate(x_levels):
                rect = rects[idx]
                group_data = data[data[x] == cat][y].dropna()
                if len(group_data) >= 2:
                    p = pg.ttest(group_data, 0)['p-val'].values[0]
                    tests.append({'rect': rect, 'p': p})
                    
    if not tests:
        return
        
    pvals = [t['p'] for t in tests]
    if correction:
        reject, pvals_corr = pg.multicomp(pvals, method=correction)
    else:
        pvals_corr = pvals
        
    for t, p_corr in zip(tests, pvals_corr):
        star = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'n.s.'
        rect = t['rect']
        x_pos = rect.get_x() + rect.get_width() / 2
        bbox = rect.get_bbox()
        bar_val = bbox.y1 if abs(bbox.y1) > abs(bbox.y0) else bbox.y0
        y_pos = min(bar_val / 2, y_range * 0.05) if bar_val > 0 else max(bar_val / 2, -y_range * 0.05)
            
        font_size = 16 if star != 'n.s.' else 12
        ax.text(x_pos, y_pos, star, ha='center', va='center', 
                fontweight='bold', color='white', fontsize=font_size,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

df = pd.read_csv(f"{SPACECUE_explicit.get_data_path()}\\concatenated\\data_for_jamovi_single_trials.csv")

df_mean = df.groupby(["subject_id", "CueInstruction", "CueDesignStrategy", "cohort", 'age'])[["select_target", "rt"]].mean().reset_index()
df_mean["select_target"] = df_mean["select_target"].astype(float)

# --- Descriptive Statistics ---
print("Participant counts per cohort:")
cohort_counts = df_mean.groupby('cohort')['subject_id'].nunique()
print(cohort_counts)
print(f"\nTotal unique participants: {df_mean['subject_id'].nunique()}")
print(f"Sum of cohorts: {cohort_counts.sum()}")
print("\n--- Descriptive Statistics (df_mean) ---")
print(df_mean[["rt", "select_target"]].describe().round(3))

# ============================================================
# COMPUTE EFFECTS (relative to neutral)
# ============================================================
id_vars = ['subject_id', 'CueDesignStrategy', 'cohort', 'age']

df_pivot_acc = df_mean.pivot_table(
    index=id_vars, columns='CueInstruction', values='select_target'
).reset_index()
df_pivot_acc.columns.name = None
df_pivot_acc['target_effect'] = df_pivot_acc['cue_target_location'] - df_pivot_acc['cue_neutral']
df_pivot_acc['distractor_effect'] = df_pivot_acc['cue_distractor_location'] - df_pivot_acc['cue_neutral']

df_pivot_rt = df_mean.pivot_table(
    index=id_vars, columns='CueInstruction', values='rt'
).reset_index()
df_pivot_rt.columns.name = None
df_pivot_rt['target_rt_benefit'] = df_pivot_rt['cue_target_location'] - df_pivot_rt['cue_neutral']
df_pivot_rt['distractor_rt_cost'] = df_pivot_rt['cue_distractor_location'] - df_pivot_rt['cue_neutral']

df_acc_effects = pd.melt(df_pivot_acc, id_vars=id_vars,
                          value_vars=['target_effect', 'distractor_effect'],
                          var_name='effect_type', value_name='accuracy_effect')

df_rt_effects = pd.melt(df_pivot_rt, id_vars=id_vars,
                         value_vars=['target_rt_benefit', 'distractor_rt_cost'],
                         var_name='effect_type', value_name='rt_effect')

# ============================================================
# STATISTICS ON EFFECTS
# ============================================================

for effect in ['target_effect', 'distractor_effect']:
    print(f"\n--- One-sample t-test: {effect} vs 0 ---")
    print(pg.print_table(pg.ttest(df_pivot_acc[effect].dropna(), 0).round(3)))

for effect in ['target_rt_benefit', 'distractor_rt_cost']:
    print(f"\n--- One-sample t-test: {effect} vs 0 ---")
    print(pg.print_table(pg.ttest(df_pivot_rt[effect].dropna(), 0).round(3)))

print("\n--- Independent T-Test: target_effect by CueDesignStrategy ---")
print(pg.print_table(pg.ttest(
    df_pivot_acc.query("CueDesignStrategy=='Trial'")['target_effect'],
    df_pivot_acc.query("CueDesignStrategy=='Block'")['target_effect']
).round(3)))

print("\n--- Independent T-Test: distractor_effect by CueDesignStrategy ---")
print(pg.print_table(pg.ttest(
    df_pivot_acc.query("CueDesignStrategy=='Trial'")['distractor_effect'],
    df_pivot_acc.query("CueDesignStrategy=='Block'")['distractor_effect']
).round(3)))

print("\n--- Independent T-Test: target_effect by cohort ---")
print(pg.print_table(pg.ttest(
    df_pivot_acc.query("cohort=='young'")['target_effect'],
    df_pivot_acc.query("cohort=='old'")['target_effect']
).round(3)))

print("\n--- Independent T-Test: distractor_effect by cohort ---")
print(pg.print_table(pg.ttest(
    df_pivot_acc.query("cohort=='young'")['distractor_effect'],
    df_pivot_acc.query("cohort=='old'")['distractor_effect']
).round(3)))

print("\n--- Pearson Correlations: Age vs Effects ---")
for effect_name, df_eff, col in [
    ("Target Accuracy Effect", df_pivot_acc, "target_effect"),
    ("Distractor Accuracy Effect", df_pivot_acc, "distractor_effect"),
    ("Target RT Benefit", df_pivot_rt, "target_rt_benefit"),
    ("Distractor RT Cost", df_pivot_rt, "distractor_rt_cost")
]:
    valid_data = df_eff.dropna(subset=['age', col])
    r, p = pearsonr(valid_data["age"], valid_data[col])
    print(f"{effect_name}: r = {r:.3f}, p = {p:.3f}")

# ============================================================
# PLOTS: ACCURACY EFFECTS
# ============================================================

# Plot 1: Accuracy effects (overall)
plt.figure(figsize=(6, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            errorbar=ERRORBAR, palette=EFFECT_PALETTE, ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
#ax.set_title('Accuracy Cueing Effect (relative to neutral)')
ax.set_ylabel('Accuracy difference (%)')
Annotator(ax=ax, pairs=[('target_effect', 'distractor_effect')],
          data=df_acc_effects, x='effect_type', y='accuracy_effect').configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_acc_effects, x='effect_type', y='accuracy_effect', 
                          correction=COMPARISON_CORRECTION)
ax.set_xticklabels(['Target', 'Distractor'])
ax.set_xlabel("Cue")
plt.tight_layout()
sns.despine()

# Plot 2: Accuracy effects by CueDesignStrategy
pairs_strategy = [
    (('target_effect', 'Block'), ('target_effect', 'Trial')),
    (('distractor_effect', 'Block'), ('distractor_effect', 'Trial')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            hue='CueDesignStrategy', errorbar=ERRORBAR, ax=ax, palette=STRATEGY_PALETTE)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect by Design Strategy')
ax.set_ylabel('Difference in Proportion correct')
Annotator(ax=ax, pairs=pairs_strategy, data=df_acc_effects,
          x='effect_type', y='accuracy_effect', hue='CueDesignStrategy').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_acc_effects, x='effect_type', y='accuracy_effect', 
                          hue='CueDesignStrategy', correction=COMPARISON_CORRECTION)
plt.tight_layout()

# Plot 3: Accuracy effects by cohort
pairs_cohort = [
    (('target_effect', 'young'), ('target_effect', 'old')),
    (('distractor_effect', 'young'), ('distractor_effect', 'old')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            hue='cohort', errorbar=ERRORBAR, ax=ax, palette=COHORT_PALETTE)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect by Cohort')
ax.set_ylabel('Difference in Proportion correct')
Annotator(ax=ax, pairs=pairs_cohort, data=df_acc_effects,
          x='effect_type', y='accuracy_effect', hue='cohort').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_acc_effects, x='effect_type', y='accuracy_effect', 
                          hue='cohort', correction=COMPARISON_CORRECTION)
plt.tight_layout()

# ============================================================
# PLOTS: RT EFFECTS
# ============================================================

# Plot 4: RT effects (overall)
plt.figure(figsize=(6, 5))
ax = plt.gca()
sns.barplot(data=df_rt_effects, x='effect_type', y='rt_effect',
            errorbar=ERRORBAR, palette=EFFECT_PALETTE, ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect (relative to neutral)')
ax.set_ylabel('Response Time Effect (s)')
Annotator(ax=ax, pairs=[('target_rt_benefit', 'distractor_rt_cost')],
          data=df_rt_effects, x='effect_type', y='rt_effect').configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_rt_effects, x='effect_type', y='rt_effect', 
                          correction=COMPARISON_CORRECTION)
plt.tight_layout()

# Plot 5: RT effects by CueDesignStrategy
pairs_strategy_rt = [
    (('target_rt_benefit', 'Block'), ('target_rt_benefit', 'Trial')),
    (('distractor_rt_cost', 'Block'), ('distractor_rt_cost', 'Trial')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_rt_effects, x='effect_type', y='rt_effect',
            hue='CueDesignStrategy', errorbar=ERRORBAR, ax=ax, palette=STRATEGY_PALETTE)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect by Design Strategy')
ax.set_ylabel('Response Time Effect (s)')
Annotator(ax=ax, pairs=pairs_strategy_rt, data=df_rt_effects,
          x='effect_type', y='rt_effect', hue='CueDesignStrategy').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_rt_effects, x='effect_type', y='rt_effect', 
                          hue='CueDesignStrategy', correction=COMPARISON_CORRECTION)
plt.tight_layout()

# Plot 6: RT effects by cohort
pairs_cohort_rt = [
    (('target_rt_benefit', 'young'), ('target_rt_benefit', 'old')),
    (('distractor_rt_cost', 'young'), ('distractor_rt_cost', 'old')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_rt_effects, x='effect_type', y='rt_effect',
            hue='cohort', errorbar=ERRORBAR, ax=ax, palette=COHORT_PALETTE)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect by Cohort')
ax.set_ylabel('Response Time Effect (s)')
Annotator(ax=ax, pairs=pairs_cohort_rt, data=df_rt_effects,
          x='effect_type', y='rt_effect', hue='cohort').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION,
    pvalue_thresholds=PVALUE_THRESHOLDS
).apply_and_annotate()
add_zero_test_annotations(ax, df_rt_effects, x='effect_type', y='rt_effect', 
                          hue='cohort', correction=COMPARISON_CORRECTION)
plt.tight_layout()

# Plot 7: Age Effects (4-panel figure)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Age Effects on Accuracy and Response Times', fontsize=16, fontweight='bold')

plot_configs = [
    (axes[0, 0], df_pivot_acc, 'target_effect', 'Age vs Target Accuracy Effect', 'Difference in Accuracy', 'g'),
    (axes[0, 1], df_pivot_acc, 'distractor_effect', 'Age vs Distractor Accuracy Effect', 'Difference in Accuracy', 'r'),
    (axes[1, 0], df_pivot_rt, 'target_rt_benefit', 'Age vs Target RT Benefit', 'Response Time Effect (s)', 'g'),
    (axes[1, 1], df_pivot_rt, 'distractor_rt_cost', 'Age vs Distractor RT Cost', 'Response Time Effect (s)', 'r')
]

for ax, data_df, y_col, title, ylabel, color in plot_configs:
    sns.regplot(data=data_df, x='age', y=y_col, ax=ax, scatter_kws={'alpha': 0.6}, color=color)
    ax.set_title(title)
    ax.set_xlabel('Age')
    ax.set_ylabel(ylabel)
    
    valid_data = data_df.dropna(subset=['age', y_col])
    r_val, p_val = pearsonr(valid_data['age'], valid_data[y_col])
    star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    
    ax.annotate(f"r = {r_val:.3f}\np = {p_val:.3f} ({star})", xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

plt.tight_layout()

# ============================================================
# SECTION 4: CUE-STIMULUS DELAY ANALYSIS
# ============================================================

df['rt_corrected'] = df['rt'] - df['cue_stim_delay_jitter']

negative_rts = df[df['rt_corrected'] < 0]
if not negative_rts.empty:
    print(f"\nWARNING: Found {len(negative_rts)} trials ({len(negative_rts)/len(df)*100:.2f}%) with negative corrected RTs.")
print("-" * 40)

df['delay_bin'] = pd.qcut(df['cue_stim_delay_jitter'], q=N_DELAY_BINS, labels=['short', 'medium', 'long'])
print("\nValue counts for each delay bin:")
print(df['delay_bin'].value_counts())
print("-" * 40)

df_delay_mean = df.groupby(
    ["subject_id", "CueInstruction", "delay_bin", "cohort"]
)[["select_target", "rt_corrected"]].mean().reset_index().dropna()
df_delay_mean["select_target"] = df_delay_mean["select_target"].astype(float)

# Compute delay effects relative to neutral
delay_id_vars = ['subject_id', 'delay_bin', 'cohort']

df_delay_pivot_acc = df_delay_mean.pivot_table(
    index=delay_id_vars, columns='CueInstruction', values='select_target'
).reset_index()
df_delay_pivot_acc.columns.name = None
df_delay_pivot_acc['target_effect'] = df_delay_pivot_acc['cue_target_location'] - df_delay_pivot_acc['cue_neutral']
df_delay_pivot_acc['distractor_effect'] = df_delay_pivot_acc['cue_distractor_location'] - df_delay_pivot_acc['cue_neutral']

df_delay_pivot_rt = df_delay_mean.pivot_table(
    index=delay_id_vars, columns='CueInstruction', values='rt_corrected'
).reset_index()
df_delay_pivot_rt.columns.name = None
df_delay_pivot_rt['target_rt_benefit'] = df_delay_pivot_rt['cue_target_location'] - df_delay_pivot_rt['cue_neutral']
df_delay_pivot_rt['distractor_rt_cost'] = df_delay_pivot_rt['cue_distractor_location'] - df_delay_pivot_rt['cue_neutral']

df_delay_acc_effects = pd.melt(df_delay_pivot_acc, id_vars=delay_id_vars,
                                value_vars=['target_effect', 'distractor_effect'],
                                var_name='effect_type', value_name='accuracy_effect')

df_delay_rt_effects = pd.melt(df_delay_pivot_rt, id_vars=delay_id_vars,
                               value_vars=['target_rt_benefit', 'distractor_rt_cost'],
                               var_name='effect_type', value_name='rt_effect')

# Stats: LMM on effects
print("\n--- LMM: Accuracy Effect (effect_type x delay_bin) ---")
lmm_acc = smf.mixedlm(
    "accuracy_effect ~ C(effect_type) * C(delay_bin)",
    data=df_delay_acc_effects, groups="subject_id"
).fit()
print(lmm_acc.summary())

print("\n--- LMM: RT Effect (effect_type x delay_bin x cohort) ---")
lmm_rt = smf.mixedlm(
    "rt_effect ~ C(effect_type) * C(delay_bin) * C(cohort)",
    data=df_delay_rt_effects, groups="subject_id"
).fit()
print(lmm_rt.summary())

# Plot: Accuracy effects by delay_bin
hue_order_delay = ['short', 'medium', 'long']
pairs_delay_acc = [
    (('short', 'target_effect'), ('medium', 'target_effect')),
    (('medium', 'target_effect'), ('long', 'target_effect')),
    (('short', 'distractor_effect'), ('medium', 'distractor_effect')),
    (('medium', 'distractor_effect'), ('long', 'distractor_effect')),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle('Accuracy Cueing Effect by Cue-Stimulus Delay and Cohort', fontsize=16, fontweight='bold')

for ax, cohort in zip(axes, ['young', 'old']):
    cohort_data = df_delay_acc_effects[df_delay_acc_effects['cohort'] == cohort]
    sns.barplot(data=cohort_data, x='delay_bin', y='accuracy_effect',
                hue='effect_type', errorbar=ERRORBAR, order=hue_order_delay,
                palette=EFFECT_PALETTE, ax=ax)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylim(YLIM_ACC_EFFECT)
    ax.set_title(f'{cohort.capitalize()} Adults')
    ax.set_ylabel('Difference in Proportion correct' if ax == axes[0] else '')
    ax.set_xlabel('Cue-Stimulus Delay')
    Annotator(ax=ax, pairs=pairs_delay_acc, data=cohort_data,
              x='delay_bin', y='accuracy_effect', hue='effect_type',
              order=hue_order_delay).configure(
        test='t-test_paired', text_format='star', loc='inside', verbose=0,
        comparisons_correction=COMPARISON_CORRECTION,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()
    add_zero_test_annotations(ax, cohort_data, x='delay_bin', y='accuracy_effect', 
                              hue='effect_type', order=hue_order_delay, 
                              correction=COMPARISON_CORRECTION)
    if ax == axes[0]:
        ax.get_legend().remove()
    else:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()

# Plot: RT effects by delay_bin
pairs_delay_rt = [
    (('short', 'target_rt_benefit'), ('medium', 'target_rt_benefit')),
    (('medium', 'target_rt_benefit'), ('long', 'target_rt_benefit')),
    (('short', 'distractor_rt_cost'), ('medium', 'distractor_rt_cost')),
    (('medium', 'distractor_rt_cost'), ('long', 'distractor_rt_cost')),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle('RT Cueing Effect by Cue-Stimulus Delay and Cohort', fontsize=16, fontweight='bold')

for ax, cohort in zip(axes, ['young', 'old']):
    cohort_data = df_delay_rt_effects[df_delay_rt_effects['cohort'] == cohort]
    sns.barplot(data=cohort_data, x='delay_bin', y='rt_effect',
                hue='effect_type', errorbar=ERRORBAR, order=hue_order_delay,
                palette=EFFECT_PALETTE, ax=ax)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylim(YLIM_RT_EFFECT)
    ax.set_title(f'{cohort.capitalize()} Adults')
    ax.set_ylabel('Response Time Effect (s)' if ax == axes[0] else '')
    ax.set_xlabel('Cue-Stimulus Delay')
    Annotator(ax=ax, pairs=pairs_delay_rt, data=cohort_data,
              x='delay_bin', y='rt_effect', hue='effect_type',
              order=hue_order_delay).configure(
        test='t-test_paired', text_format='star', loc='inside', verbose=0,
        comparisons_correction=COMPARISON_CORRECTION,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()
    add_zero_test_annotations(ax, cohort_data, x='delay_bin', y='rt_effect',
                              hue='effect_type', order=hue_order_delay,
                              correction=COMPARISON_CORRECTION)
    if ax == axes[0]:
        ax.get_legend().remove()
    else:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()

# ============================================================
# SECTION 5: REPEATED DISTRACTOR LOCATION ANALYSIS
# ============================================================

# NOTE: Update `DISTRACTOR_LOC_COL` to the actual column name in your dataset
# that tracks the distractor's physical location (e.g., 'distractor_location', 'distractor_pos')
DISTRACTOR_LOC_COL = 'SingletonLoc'

print(f"\nAnalyzing repeated distractor locations using column: {DISTRACTOR_LOC_COL}")

# Assuming 'df' retains chronological trial presentation order.
# (If your data is split by blocks, you could group by ['subject_id', 'block_num'] instead)
df['prev_distractor_loc'] = df.groupby('subject_id')[DISTRACTOR_LOC_COL].shift(1)

# A trial has a "repeated" distractor if its location matches the previous trial's location (2 or more in a row)
df['is_distractor_repeated'] = (df[DISTRACTOR_LOC_COL] == df['prev_distractor_loc']) & df[DISTRACTOR_LOC_COL].notna()

# Focus on trials where participants were explicitly cued to the distractor location
#df_dist_cued = df[df['CueInstruction'] == 'cue_distractor_location'].copy()
df_dist_cued = df.copy()
df_dist_cued['Repetition_Status'] = df_dist_cued['is_distractor_repeated'].map({True: 'Repeated', False: 'New'})

# Group performance by subject, cohort, and repetition status
df_rep_perf = df_dist_cued.groupby(['subject_id', 'cohort', 'Repetition_Status'])[['select_target', 'rt']].mean().reset_index()
df_rep_perf['select_target'] = df_rep_perf['select_target'].astype(float)

print("\n--- Descriptive Statistics: Repeated vs. New Distractor Location ---")
print(df_rep_perf.groupby('Repetition_Status')[['select_target', 'rt']].describe().round(3))

# Reshape for paired statistics
df_rep = df_rep_perf[df_rep_perf['Repetition_Status'] == 'Repeated'].set_index(['subject_id', 'cohort'])
df_new = df_rep_perf[df_rep_perf['Repetition_Status'] == 'New'].set_index(['subject_id', 'cohort'])

# Join to make sure we only run stats on subjects who experienced both states
paired_df = df_rep.join(df_new, lsuffix='_rep', rsuffix='_new', how='inner')

if not paired_df.empty:
    paired_df['acc_rep_benefit'] = paired_df['select_target_rep'] - paired_df['select_target_new']
    paired_df['rt_rep_benefit'] = paired_df['rt_rep'] - paired_df['rt_new']
    paired_df = paired_df.reset_index()

    print("\n--- Paired T-Test: Accuracy (Repeated vs. New Distractor Location) ---")
    print(pg.print_table(pg.ttest(paired_df['select_target_rep'], paired_df['select_target_new'], paired=True).round(3)))

    print("\n--- Paired T-Test: RT (Repeated vs. New Distractor Location) ---")
    print(pg.print_table(pg.ttest(paired_df['rt_rep'], paired_df['rt_new'], paired=True).round(3)))

    # Plotting Accuracy & RT for Repeated vs New Distractor Location
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Performance: Repeated vs. New Distractor Location', fontsize=16, fontweight='bold')

    sns.barplot(data=df_rep_perf, x='Repetition_Status', y='select_target', errorbar=ERRORBAR,
                ax=axes[0], palette='Set2', order=['New', 'Repeated'])
    axes[0].set_title('Accuracy')
    axes[0].set_ylabel('Proportion correct')
    Annotator(ax=axes[0], pairs=[('New', 'Repeated')], data=df_rep_perf,
              x='Repetition_Status', y='select_target', order=['New', 'Repeated']
              ).configure(
        test='t-test_paired', text_format='star', loc='inside', verbose=0,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()

    sns.barplot(data=df_rep_perf, x='Repetition_Status', y='rt', errorbar=ERRORBAR,
                ax=axes[1], palette='Set2', order=['New', 'Repeated'])
    axes[1].set_title('Response Time')
    axes[1].set_ylabel('RT (s)')
    Annotator(ax=axes[1], pairs=[('New', 'Repeated')], data=df_rep_perf,
              x='Repetition_Status', y='rt', order=['New', 'Repeated']
              ).configure(
        test='t-test_paired', text_format='star', loc='inside', verbose=0,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()

    plt.tight_layout()

    print("\n--- Independent T-Test: Accuracy Repetition Benefit (Young vs Old) ---")
    print(pg.print_table(pg.ttest(
        paired_df.query("cohort=='young'")['acc_rep_benefit'],
        paired_df.query("cohort=='old'")['acc_rep_benefit']
    ).round(3)))

    print("\n--- Independent T-Test: RT Repetition Benefit (Young vs Old) ---")
    print(pg.print_table(pg.ttest(
        paired_df.query("cohort=='young'")['rt_rep_benefit'],
        paired_df.query("cohort=='old'")['rt_rep_benefit']
    ).round(3)))

    # Plotting Repetition Benefit (Repeated - New) by Cohort
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.suptitle('Repetition Benefit (Repeated minus New) by Cohort', fontsize=16, fontweight='bold')

    sns.barplot(data=paired_df, x='cohort', y='acc_rep_benefit', errorbar=ERRORBAR,
                ax=axes2[0], palette=COHORT_PALETTE)
    axes2[0].set_title('Accuracy Benefit')
    axes2[0].set_ylabel('Difference in Accuracy')
    axes2[0].axhline(0, color='black', linestyle='--')
    Annotator(ax=axes2[0], pairs=[('young', 'old')], data=paired_df,
              x='cohort', y='acc_rep_benefit').configure(
        test='t-test_ind', text_format='star', loc='inside', verbose=0,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()
    add_zero_test_annotations(axes2[0], paired_df, x='cohort', y='acc_rep_benefit')

    sns.barplot(data=paired_df, x='cohort', y='rt_rep_benefit', errorbar=ERRORBAR,
                ax=axes2[1], palette=COHORT_PALETTE)
    axes2[1].set_title('RT Benefit')
    axes2[1].set_ylabel('Difference in RT (s)')
    axes2[1].axhline(0, color='black', linestyle='--')
    Annotator(ax=axes2[1], pairs=[('young', 'old')], data=paired_df,
              x='cohort', y='rt_rep_benefit').configure(
        test='t-test_ind', text_format='star', loc='inside', verbose=0,
        pvalue_thresholds=PVALUE_THRESHOLDS
    ).apply_and_annotate()
    add_zero_test_annotations(axes2[1], paired_df, x='cohort', y='rt_rep_benefit')

    plt.tight_layout()