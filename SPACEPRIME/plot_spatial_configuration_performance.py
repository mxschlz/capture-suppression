import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
import SPACEPRIME
import pingouin as pg
sns.set_theme(style="ticks", context="talk")
plt.ion()


# --- 0. Configuration ---
# Define an output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


# --- 1. Data Loading & Preprocessing ---
OUTLIER_RT_THRESHOLD = 2.0

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
PRIMING_COL = 'Priming'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_INT_COL = 'select_target_int'
BLOCK_COL = "block"
# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])

# --- Preprocessing Steps (largely unchanged) ---
if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
# Map categorical variables to strings
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map(TARGET_LOC_MAP)
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(DISTRACTOR_LOC_MAP)
df[PRIMING_COL] = pd.to_numeric(df[PRIMING_COL], errors='coerce').map(PRIMING_MAP)
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(int).astype(str)


target_towardness = SPACEPRIME.load_concatenated_csv("target_towardness.csv", index_col=0)

merge_cols = [SUBJECT_ID_COL, BLOCK_COL, 'trial_nr']

# It's good practice to ensure the key columns are the same data type before merging
# to prevent silent failures.
target_towardness[SUBJECT_ID_COL] = target_towardness[SUBJECT_ID_COL].astype(int).astype(str)
target_towardness = target_towardness[["subject_id", "block", "trial_nr", "target_towardness"]]
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(int).astype(str)

df_final = pd.merge(
    df,
    target_towardness,
    on=merge_cols,
    how='left'
)

# transform to subject averages
df_mean = df_final.groupby(["subject_id", "TargetLoc", "SingletonLoc"])[["target_towardness", "rt", "select_target"]].mean().reset_index()

# Create a new column to distinguish between distractor present and absent trials
df_mean['distractor_presence'] = np.where(df_mean[DISTRACTOR_COL] == 'absent', 'absent', 'present')

# --- 3a. Overall ANOVA: Target Location x Distractor Presence ---
print("\n--- 3x2 Repeated Measures ANOVA (Target Location x Distractor Presence) ---")
# This ANOVA tests the main effects of target location and distractor presence,
# and their interaction.
aov_presence = pg.rm_anova(
    data=df_mean,
    dv='target_towardness',
    within=['TargetLoc', 'distractor_presence'],
    subject=SUBJECT_ID_COL,
    detailed=True,
    effsize="np2"  # Generalized eta-squared is a good effect size for RM designs
)
print("ANOVA Results (Target Location x Distractor Presence):")
print(aov_presence)

# --- 3a-ii. Post-hoc tests for the 3x2 ANOVA ---
print("\n--- Post-hoc Tests for 3x2 ANOVA ---")

# Check if the interaction is significant, as this guides interpretation
interaction_p_val = aov_presence.loc[aov_presence['Source'] == 'TargetLoc * distractor_presence', 'p-unc'].iloc[0]

if interaction_p_val < 0.05:
    print("NOTE: The interaction is significant (p < .05). Main effects should be interpreted with caution.")
    print("The most informative follow-up tests are for the *simple effects* (e.g., comparing target locations")
    print("separately for distractor-present and distractor-absent conditions), which are already performed in section 3c.\n")

# --- Post-hocs for Main Effects ---
# These are run for completeness, but be mindful of any significant interaction.

# Post-hoc for the main effect of TargetLoc
print("Post-hoc for Main Effect of Target Location (Bonferroni-corrected):")
posthoc_targetloc = pg.pairwise_tests(
    data=df_mean,
    dv='target_towardness',
    within='TargetLoc',
    subject=SUBJECT_ID_COL,
    padjust='bonf',
    effsize='cohen'
)
# Display a cleaner version of the results table
print(posthoc_targetloc[['A', 'B', 'T', 'p-corr', 'cohen']])

print("Post-hoc for Main Effect of Target Location (Bonferroni-corrected):")
posthoc_targetloc_absent = pg.pairwise_tests(
    data=df_mean.query("distractor_presence=='absent'"),
    dv='target_towardness',
    within='TargetLoc',
    subject=SUBJECT_ID_COL,
    padjust='bonf',
    effsize='cohen'
)
# Display a cleaner version of the results table
print(posthoc_targetloc_absent[['A', 'B', 'T', 'p-corr', 'cohen']])


# Post-hoc for the main effect of distractor_presence
# Note: Since this factor only has 2 levels, this t-test is equivalent to the
# F-test in the ANOVA table and the t-test you run for Panel E.
print("\nPost-hoc for Main Effect of Distractor Presence:")
posthoc_presence = pg.pairwise_tests(
    data=df_mean,
    dv='target_towardness',
    within='distractor_presence',
    subject=SUBJECT_ID_COL,
    effsize='cohen'
)
# Display a cleaner version of the results table
print(posthoc_presence[['A', 'B', 'T', 'p-unc', 'cohen']])

# --- 3b. Focused ANOVA: Target Location x Distractor Location (Present Trials Only) ---
print("\n--- 3x3 Repeated Measures ANOVA (Target Location x Distractor Location - Present Trials Only) ---")
# This ANOVA focuses only on trials where a distractor was present to see how
# the specific locations of the target and distractor interact.
df_present_only = df_mean[df_mean['distractor_presence'] == 'present'].copy()

aov_locations = pg.rm_anova(
    data=df_present_only,
    dv='target_towardness',
    within=['TargetLoc', 'SingletonLoc'],
    subject=SUBJECT_ID_COL,
    detailed=True,
    effsize="np2"
)
print("ANOVA Results (Target Location x Distractor Location on Present Trials):")
print(aov_locations)

# --- 3. Statistical Analysis ---
# Helper function to convert p-values to significance stars
def p_to_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'

# --- 3a. Pairwise tests comparing Target Locations within each Distractor condition ---
stats_df = df_mean.groupby(DISTRACTOR_COL).apply(
    lambda df: pg.pairwise_tests(
        data=df.reset_index(),
        dv='target_towardness',
        within=TARGET_COL,
        subject=SUBJECT_ID_COL,
        padjust='bonf',
	    effsize="cohen"
    )
).reset_index()

print("--- Pairwise T-test Results ---")
print(stats_df)

# --- 3b. Pairwise tests comparing Distractor Locations within each Target condition ---
stats_sloc_df = df_mean.groupby(TARGET_COL).apply(
    lambda df: pg.pairwise_tests(
        data=df.reset_index(),
        dv='target_towardness',
        within=DISTRACTOR_COL,
        subject=SUBJECT_ID_COL,
        padjust='bonf',
	    effsize="cohen"
    )
).reset_index()

print("\n--- Pairwise T-test Results (Singleton Location) ---")
print(stats_sloc_df)

# --- 4. Reliability Analysis (Cronbach's Alpha over Blocks) ---
print("\n--- 4. Reliability Analysis (Cronbach's Alpha over Blocks) ---")
print("Assessing the internal consistency reliability of metrics across experimental blocks.")

# Ensure the columns used for reliability analysis are numeric
# This step is crucial to handle any non-numeric values that might have
# crept into these columns during data loading or earlier processing.
for col in ['target_towardness', 'rt', 'select_target']:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# 1. Aggregate data by subject and block for each metric
# We need to go back to df_final as df_mean has already averaged over blocks and conditions.
df_block_means = df_final.groupby([SUBJECT_ID_COL, BLOCK_COL])[
    ['target_towardness', 'rt', 'select_target']
].mean().reset_index()

# 2. Pivot the aggregated data for each metric
# This creates a wide format where rows are subjects and columns are blocks.

# Target Towardness
df_tt_wide = df_block_means.pivot_table(
    index=SUBJECT_ID_COL,
    columns=BLOCK_COL,
    values='target_towardness'
)

# Reaction Time
df_rt_wide = df_block_means.pivot_table(
    index=SUBJECT_ID_COL,
    columns=BLOCK_COL,
    values='rt'
)

# Select Target (Accuracy)
df_acc_wide = df_block_means.pivot_table(
    index=SUBJECT_ID_COL,
    columns=BLOCK_COL,
    values='select_target'
)


# 3. Calculate Cronbach's Alpha for each metric
# pingouin.cronbach_alpha handles NaN values by default (drops subjects with missing block data).

# Cronbach's Alpha for Target Towardness
alpha_tt = pg.cronbach_alpha(data=df_tt_wide)
print(f"\nCronbach's Alpha for 'target_towardness':")
print(f"  Alpha: {alpha_tt[0]:.3f} (95% CI: {alpha_tt[1][0]:.3f}, {alpha_tt[1][1]:.3f})")

# Cronbach's Alpha for Reaction Time
alpha_rt = pg.cronbach_alpha(data=df_rt_wide)
print(f"\nCronbach's Alpha for 'rt' (Reaction Time):")
print(f"  Alpha: {alpha_rt[0]:.3f} (95% CI: {alpha_rt[1][0]:.3f}, {alpha_rt[1][1]:.3f})")

# Cronbach's Alpha for Select Target (Accuracy)
alpha_acc = pg.cronbach_alpha(data=df_acc_wide)
print(f"\nCronbach's Alpha for 'select_target' (Accuracy):")
print(f"  Alpha: {alpha_acc[0]:.3f} (95% CI: {alpha_acc[1][0]:.3f}, {alpha_acc[1][1]:.3f})")

# --- 4b. Reliability of Priming Effects (Cronbach's Alpha) ---
print("\n--- 4b. Reliability of Priming Effects (Difference Scores) ---")
print("Assessing the internal consistency of negative and positive priming effects across blocks.")

# 1. Aggregate data by subject, block, and priming condition
df_priming_block_means = df_final.groupby([SUBJECT_ID_COL, BLOCK_COL, PRIMING_COL])[
    ['target_towardness', 'rt', 'select_target']
].mean()  # No reset_index() needed, pivot_table works well with multi-index

# 2. Pivot to get priming conditions as columns
df_priming_pivot = df_priming_block_means.unstack(level=PRIMING_COL)

# 3. Calculate difference scores (priming effects) for each metric
metrics_to_test = ['target_towardness', 'rt', 'select_target']

for metric in metrics_to_test:
    # Extract columns for the current metric, handling potential missing data
    no_p_col = (metric, 'no-p')
    np_col = (metric, 'np')
    pp_col = (metric, 'pp')

    # Calculate difference scores. If a priming condition is missing for a subject/block, the result will be NaN.
    df_priming_pivot[(f'{metric}_np_effect', '')] = df_priming_pivot.get(np_col, np.nan) - df_priming_pivot.get(
        no_p_col, np.nan)
    df_priming_pivot[(f'{metric}_pp_effect', '')] = df_priming_pivot.get(pp_col, np.nan) - df_priming_pivot.get(
        no_p_col, np.nan)

    # --- Negative Priming Effect Reliability ---
    # Create a wide-format DataFrame for the NP effect
    df_np_effect_wide = df_priming_pivot[(f'{metric}_np_effect', '')].unstack(level=BLOCK_COL)

    # Calculate and print Cronbach's Alpha
    alpha_np = pg.cronbach_alpha(data=df_np_effect_wide)
    print(f"\nCronbach's Alpha for '{metric}' NP effect (np - no-p):")
    if alpha_np[0] is not None:
        print(f"  Alpha: {alpha_np[0]:.3f} (95% CI: {alpha_np[1][0]:.3f}, {alpha_np[1][1]:.3f})")
    else:
        print("  Could not be computed (likely insufficient data).")

    # --- Positive Priming Effect Reliability ---
    # Create a wide-format DataFrame for the PP effect
    df_pp_effect_wide = df_priming_pivot[(f'{metric}_pp_effect', '')].unstack(level=BLOCK_COL)

    # Calculate and print Cronbach's Alpha
    alpha_pp = pg.cronbach_alpha(data=df_pp_effect_wide)
    print(f"Cronbach's Alpha for '{metric}' PP effect (pp - no-p):")
    if alpha_pp[0] is not None:
        print(f"  Alpha: {alpha_pp[0]:.3f} (95% CI: {alpha_pp[1][0]:.3f}, {alpha_pp[1][1]:.3f})")
    else:
        print("  Could not be computed (likely insufficient data).")

# --- 4. Combined Plotting: Mosaic Layout ---
# Helper function for annotating significance
def annotate_significance(ax, stats, x_order, p_col='p-corr', d_col='cohen'):
    """
    Annotates the axes with significance brackets based on the provided statistics DataFrame.
    """
    if stats is None or stats.empty:
        return

    # Filter for significance (p < 0.05)
    sig_stats = stats[stats[p_col] < 0.05].copy()
    if sig_stats.empty:
        return

    # Map x-axis labels to positions
    x_map = {label: i for i, label in enumerate(x_order)}
    
    # Calculate positions
    sig_stats['x1'] = sig_stats['A'].map(x_map)
    sig_stats['x2'] = sig_stats['B'].map(x_map)
    # Filter out any rows where mapping failed
    sig_stats = sig_stats.dropna(subset=['x1', 'x2'])
    
    sig_stats['dist'] = abs(sig_stats['x1'] - sig_stats['x2'])
    # Sort by distance (shortest first) to nest brackets
    sig_stats = sig_stats.sort_values('dist')

    # Determine starting y-position based on the highest bar in the plot
    max_h = 0
    for patch in ax.patches:
        if np.isfinite(patch.get_height()):
            max_h = max(max_h, patch.get_height())
    
    # Start slightly above the highest bar
    y_curr = max_h + 0.05
    y_step = 0.08  # Vertical space per bracket

    for _, row in sig_stats.iterrows():
        x1, x2 = row['x1'], row['x2']
        p_val = row[p_col]
        d_val = row[d_col]

        # Draw bracket
        h = 0.01
        ax.plot([x1, x1, x2, x2], [y_curr, y_curr + h, y_curr + h, y_curr], lw=1.5, c='k')

        # Add text
        label = f"p={p_val:.3f}\nd={d_val:.2f}"
        ax.text((x1 + x2) * 0.5, y_curr + h + 0.005, label, ha='center', va='bottom', fontsize=9)

        y_curr += y_step

    # Adjust y-limits to fit annotations if needed
    current_ylim = ax.get_ylim()
    if current_ylim[1] < y_curr + 0.02:
        ax.set_ylim(current_ylim[0], y_curr + 0.02)

# Create a figure using subplot_mosaic for a more intuitive layout.
# 'e' is the new summary plot. 'a' is distractor absent. 'b', 'c', 'd' are distractor present.
fig, axes = plt.subplot_mosaic(
    mosaic="""
    ea.
    bcd
    """,
    figsize=(16, 11),
    sharey=True,
    gridspec_kw={'width_ratios': [1, 1, 1]} # Ensure columns have equal width
)

# --- Define which axis is for which plot ---
# With mosaic, axes is a dictionary. We access each subplot by its label.
ax_e = axes['e']
ax_a = axes['a']
axes_b = [axes['b'], axes['c'], axes['d']] # Group the bottom axes for iteration

# --- 4a. Plotting Panel A: Distractor Absent ---
df_absent = df_mean[df_mean[DISTRACTOR_COL] == 'absent']
target_order = ["left", "mid", "right"]

axes['a'].sharex(axes['b']) # Share x-axis between a and bcd
sns.barplot(
    data=df_absent,
    x=TARGET_COL,
    y='target_towardness',
    order=target_order,
    color="darkgreen",
    errorbar=('se', 1),
    ax=ax_a
)
ax_a.set_title("A: Distractor Absent", fontsize=14, weight='bold')

# Add annotations for Panel A
stats_absent = stats_df[stats_df[DISTRACTOR_COL] == 'absent']
annotate_significance(ax_a, stats_absent, target_order)

# --- Print Stats for Panel A ---
print("\n--- Panel A: Distractor Absent Stats ---")
for loc in target_order:
    data = df_absent[df_absent[TARGET_COL] == loc]['target_towardness']
    mean = data.mean()
    sem = data.sem()
    n = len(data)
    print(f"  Target: {loc.capitalize()} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

# --- 4aa. Plotting Panel E: Distractor Present vs. Absent ---
df_presence_summary = df_mean.groupby(['subject_id', 'distractor_presence'])['target_towardness'].mean().reset_index()

sns.barplot(
    data=df_presence_summary,
    x='distractor_presence',
    y='target_towardness',
    order=['absent', 'present'],
    palette={"absent": "darkgreen", "present": "darkred"},
    errorbar=('se', 1),
    ax=ax_e
)
ax_e.set_title("E: Distractor Presence", fontsize=14, weight='bold')
ax_e.set_xlabel("Distractor Presence", fontsize=12)

# --- Stats for Panel E ---
print("\n--- Panel E: Distractor Presence Stats ---")
data_abs = df_presence_summary[df_presence_summary['distractor_presence'] == 'absent']['target_towardness']
data_pres = df_presence_summary[df_presence_summary['distractor_presence'] == 'present']['target_towardness']

ttest_presence = pg.ttest(data_abs, data_pres, paired=True)
print("  Paired T-test (Absent vs. Present):")
print(f"    t = {ttest_presence['T'].iloc[0]:.3f}, p = {ttest_presence['p-val'].iloc[0]:.3f}, Cohen's d = {ttest_presence['cohen-d'].iloc[0]:.3f}")

# Add annotations for Panel E
stats_presence = pd.DataFrame({
    'A': ['absent'],
    'B': ['present'],
    'p-val': ttest_presence['p-val'].values,
    'cohen': ttest_presence['cohen-d'].values
})
annotate_significance(ax_e, stats_presence, ['absent', 'present'], p_col='p-val', d_col='cohen')

for cond in ['absent', 'present']:
    data = df_presence_summary[df_presence_summary['distractor_presence'] == cond]['target_towardness']
    print(f"  Condition: {cond.capitalize():<7} | Mean: {data.mean():.3f}, SEM: {data.sem():.3f}, N: {len(data)}")



# --- 4b. Plotting Panels B, C, and D: Distractor Present ---
df_present = df_mean[df_mean[DISTRACTOR_COL] != 'absent']
distractor_locations_present = ["left", "mid", "right"]
panel_labels = ["B: Distractor Left", "C: Distractor Mid", "D: Distractor Right"]

# --- Print Stats for Distractor Present Conditions ---
print("\n--- Distractor Present Stats ---")

# Loop through the bottom axes to create the distractor-present plots
for ax, dist_loc, panel_label in zip(axes_b, distractor_locations_present, panel_labels):
    df_subplot = df_present[df_present[DISTRACTOR_COL] == dist_loc]

    # --- Print stats for this subplot ---
    print(f"  Condition: {dist_loc.capitalize()}")
    for target_loc in target_order:
        data = df_subplot[df_subplot[TARGET_COL] == target_loc]['target_towardness']
        mean = data.mean()
        sem = data.sem()
        n = len(data)
        print(f"    Target: {target_loc.capitalize():<5} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

    # Print significant stats to console
    stats_subplot = stats_df[stats_df[DISTRACTOR_COL] == dist_loc]
    print(f"  Pairwise Differences ({dist_loc}) [Plotting only p < 0.05]:")
    if not stats_subplot.empty:
        print(stats_subplot[['A', 'B', 'p-corr', 'cohen']])
    else:
        print("    No stats computed.")

    # Create the bar plot
    sns.barplot(
        data=df_subplot,
        x=TARGET_COL,
        y='target_towardness',
        order=target_order,
        color="darkred",
        errorbar=('se', 1),
        ax=ax
    )

    ax.set_title(panel_label, fontsize=14, weight='bold')
    
    # Add annotations
    annotate_significance(ax, stats_subplot, target_order)


# --- 5. Final Figure-Wide Touches ---
# Set shared axis labels for the entire figure
fig.supxlabel("Target Location", fontsize=14)
fig.supylabel("Target Towardness", fontsize=14)

# Set a consistent Y-limit for all plots by accessing the dictionary's values
#plt.setp(list(axes.values()), ylim=(-0.05, 0.55))

# Clean up the overall figure appearance
sns.despine(fig=fig)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.98]) # Adjust layout for super-labels

# Save the combined figure with a new name reflecting the layout
fig.savefig(os.path.join(output_dir, "combined_spatial_performance_mosaic.svg"))

# --- 6. Distribution of Target Towardness by Accuracy ---
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df_final, x='target_towardness', hue=ACCURACY_COL, fill=True, common_norm=False)
plt.title('Distribution of Target Towardness by Response Accuracy')
plt.xlabel('Target Towardness')
plt.ylabel('Density')
sns.despine()
plt.tight_layout()
