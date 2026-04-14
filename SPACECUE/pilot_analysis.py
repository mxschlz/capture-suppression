import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr
import pingouin as pg
import statsmodels.formula.api as smf
import SPACECUE
plt.ion()


sns.set_theme("talk", "ticks")

# ============================================================
# PARAMETERS
# ============================================================
COMPARISON_CORRECTION = "bonferroni"  # multiple comparison correction (e.g. 'holm', 'bonferroni', 'fdr_bh')
ERRORBAR = ("ci", 95)                 # error band for bar plots (e.g. ("se", 1), ("ci", 95), ("sd", 1))
N_DELAY_BINS = 3                      # number of quantile bins for cue-stimulus delay
YLIM_RT_EFFECT = (-0.2, 0.2)          # y-axis limits for RT effect plots
YLIM_ACC_EFFECT = (-0.1, 0.1)         # y-axis limits for accuracy effect plots
# ============================================================

df = pd.read_csv(f"{SPACECUE.get_data_path()}\\concatenated\\data_for_jamovi_single_trials.csv")

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
df_pivot_rt['target_rt_benefit'] = df_pivot_rt['cue_neutral'] - df_pivot_rt['cue_target_location']
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

print("\n--- Pearson Correlation: Age vs RT ---")
r, p = pearsonr(df_pivot_rt["age"], df_pivot_rt["target_rt_benefit"])
print(f"r = {r:.3f}, p = {p:.3f}")

# ============================================================
# PLOTS: ACCURACY EFFECTS
# ============================================================

# Plot 1: Accuracy effects (overall)
plt.figure(figsize=(6, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            errorbar=ERRORBAR, palette=['g', 'r'], ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect (relative to neutral)')
ax.set_ylabel('Difference in Target Selection Rate')
Annotator(ax=ax, pairs=[('target_effect', 'distractor_effect')],
          data=df_acc_effects, x='effect_type', y='accuracy_effect').configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# Plot 2: Accuracy effects by CueDesignStrategy
pairs_strategy = [
    (('target_effect', 'Block'), ('target_effect', 'Trial')),
    (('distractor_effect', 'Block'), ('distractor_effect', 'Trial')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            hue='CueDesignStrategy', errorbar=ERRORBAR, ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect by Design Strategy')
ax.set_ylabel('Difference in Target Selection Rate')
Annotator(ax=ax, pairs=pairs_strategy, data=df_acc_effects,
          x='effect_type', y='accuracy_effect', hue='CueDesignStrategy').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# Plot 3: Accuracy effects by cohort
pairs_cohort = [
    (('target_effect', 'young'), ('target_effect', 'old')),
    (('distractor_effect', 'young'), ('distractor_effect', 'old')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_acc_effects, x='effect_type', y='accuracy_effect',
            hue='cohort', errorbar=ERRORBAR, ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect by Cohort')
ax.set_ylabel('Difference in Target Selection Rate')
Annotator(ax=ax, pairs=pairs_cohort, data=df_acc_effects,
          x='effect_type', y='accuracy_effect', hue='cohort').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# ============================================================
# PLOTS: RT EFFECTS
# ============================================================

# Plot 4: RT effects (overall)
plt.figure(figsize=(6, 5))
ax = plt.gca()
sns.barplot(data=df_rt_effects, x='effect_type', y='rt_effect',
            errorbar=ERRORBAR, palette=['g', 'r'], ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect (relative to neutral)')
ax.set_ylabel('Response Time Effect (s)')
Annotator(ax=ax, pairs=[('target_rt_benefit', 'distractor_rt_cost')],
          data=df_rt_effects, x='effect_type', y='rt_effect').configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# Plot 5: RT effects by CueDesignStrategy
pairs_strategy_rt = [
    (('target_rt_benefit', 'Block'), ('target_rt_benefit', 'Trial')),
    (('distractor_rt_cost', 'Block'), ('distractor_rt_cost', 'Trial')),
]
plt.figure(figsize=(7, 5))
ax = plt.gca()
sns.barplot(data=df_rt_effects, x='effect_type', y='rt_effect',
            hue='CueDesignStrategy', errorbar=ERRORBAR, ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect by Design Strategy')
ax.set_ylabel('Response Time Effect (s)')
Annotator(ax=ax, pairs=pairs_strategy_rt, data=df_rt_effects,
          x='effect_type', y='rt_effect', hue='CueDesignStrategy').configure(
    test='t-test_ind', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# Plot 6: Age vs RT benefit
ax_lm = sns.lmplot(data=df_pivot_rt, x="age", y="target_rt_benefit")
ax_lm.ax.set_title("Age vs Target RT Benefit")
r_age, p_age = pearsonr(df_pivot_rt["age"].dropna(), df_pivot_rt["target_rt_benefit"].dropna())
ax_lm.ax.annotate(f"r = {r_age:.3f}, p = {p_age:.3f}", xy=(0.05, 0.92), xycoords='axes fraction', fontsize=12)

# ============================================================
# SECTION 3: PRIMING ANALYSIS
# ============================================================

if "Priming" in df.columns:
    df_priming_mean = df.groupby(["subject_id", "Priming"])[["rt", "select_target"]].mean().reset_index()
    priming_categories = sorted(df_priming_mean["Priming"].unique())

    if len(priming_categories) >= 2:
        pairs_priming = []
        if len(priming_categories) == 2:
            pairs_priming.append((priming_categories[0], priming_categories[1]))
        elif len(priming_categories) > 2:
            pairs_priming.append((priming_categories[0], priming_categories[1]))
            pairs_priming.append((priming_categories[1], priming_categories[2]))
            if len(priming_categories) >= 3:
                pairs_priming.append((priming_categories[0], priming_categories[2]))

        plt.figure(figsize=(6, 5))
        ax_rt_priming = plt.gca()
        sns.barplot(data=df_priming_mean, x="Priming", y="rt", ax=ax_rt_priming,
                    errorbar=ERRORBAR, order=priming_categories)
        plt.title("RT by Priming")
        if pairs_priming:
            Annotator(ax=ax_rt_priming, pairs=pairs_priming, data=df_priming_mean,
                      x="Priming", y="rt", order=priming_categories).configure(
                test='t-test_paired', text_format='star', loc='inside', verbose=0,
                comparisons_correction=COMPARISON_CORRECTION
            ).apply_and_annotate()
        plt.tight_layout()

        plt.figure(figsize=(6, 5))
        ax_sel_priming = plt.gca()
        sns.barplot(data=df_priming_mean, x="Priming", y="select_target", ax=ax_sel_priming,
                    errorbar=ERRORBAR, order=priming_categories)
        plt.title("Target Selection by Priming")
        if pairs_priming:
            Annotator(ax=ax_sel_priming, pairs=pairs_priming, data=df_priming_mean,
                      x="Priming", y="select_target", order=priming_categories).configure(
                test='t-test_paired', text_format='star', loc='inside', verbose=0,
                comparisons_correction=COMPARISON_CORRECTION
            ).apply_and_annotate()
        plt.tight_layout()
    else:
        print("Not enough 'Priming' categories to compare.")
else:
    print("'Priming' column not found in DataFrame. Skipping priming plots.")

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
df_delay_pivot_rt['target_rt_benefit'] = df_delay_pivot_rt['cue_neutral'] - df_delay_pivot_rt['cue_target_location']
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
plt.figure(figsize=(8, 6))
ax = plt.gca()
sns.barplot(data=df_delay_acc_effects, x='delay_bin', y='accuracy_effect',
            hue='effect_type', errorbar=ERRORBAR, order=hue_order_delay,
            palette=['g', 'r'], ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_ACC_EFFECT)
ax.set_title('Accuracy Cueing Effect by Cue-Stimulus Delay')
ax.set_ylabel('Difference in Target Selection Rate')
ax.set_xlabel('Cue-Stimulus Delay')
Annotator(ax=ax, pairs=pairs_delay_acc, data=df_delay_acc_effects,
          x='delay_bin', y='accuracy_effect', hue='effect_type',
          order=hue_order_delay).configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()

# Plot: RT effects by delay_bin
pairs_delay_rt = [
    (('short', 'target_rt_benefit'), ('medium', 'target_rt_benefit')),
    (('medium', 'target_rt_benefit'), ('long', 'target_rt_benefit')),
    (('short', 'distractor_rt_cost'), ('medium', 'distractor_rt_cost')),
    (('medium', 'distractor_rt_cost'), ('long', 'distractor_rt_cost')),
]
plt.figure(figsize=(8, 6))
ax = plt.gca()
sns.barplot(data=df_delay_rt_effects, x='delay_bin', y='rt_effect',
            hue='effect_type', errorbar=ERRORBAR, order=hue_order_delay,
            palette=['g', 'r'], ax=ax)
ax.axhline(0, color='black', linestyle='--')
ax.set_ylim(YLIM_RT_EFFECT)
ax.set_title('RT Cueing Effect by Cue-Stimulus Delay')
ax.set_ylabel('Response Time Effect (s)')
ax.set_xlabel('Cue-Stimulus Delay')
Annotator(ax=ax, pairs=pairs_delay_rt, data=df_delay_rt_effects,
          x='delay_bin', y='rt_effect', hue='effect_type',
          order=hue_order_delay).configure(
    test='t-test_paired', text_format='star', loc='inside', verbose=0,
    comparisons_correction=COMPARISON_CORRECTION
).apply_and_annotate()
plt.tight_layout()
