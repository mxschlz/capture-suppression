import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from scipy.stats import ttest_rel
import matplotlib.gridspec as gridspec
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

# Create a new column to distinguish between distractor present and absent trials
df_final['distractor_presence'] = np.where(df_final[DISTRACTOR_COL] == 'absent', 'absent', 'present')

# transform to subject averages for 3x2 ANOVA
df_3x2 = df_final.groupby(["subject_id", "TargetLoc", "distractor_presence"])[["target_towardness"]].mean().reset_index()

# --- Save Dataframes for External Statistics (e.g., Jamovi) ---
print(f"Saving aggregated dataframes to {output_dir}...")
metric_map = {'target_towardness': 'TT', 'rt': 'RT', 'select_target': 'Acc'}
df_3x2.to_csv(os.path.join(f"{get_data_path()}concatenated", "stats_df_3x2.csv"), index=False)

# --- 3a. Overall ANOVA: Target Location x Distractor Presence ---
print("\n" + "="*60)
print(f"{'3x2 REPEATED MEASURES ANOVA (TargetLoc x DistractorPresence)':^60}")
print("="*60)

aov = pg.rm_anova(
    data=df_3x2,
    dv='target_towardness',
    within=['TargetLoc', 'distractor_presence'],
    subject=SUBJECT_ID_COL,
    detailed=True,
    effsize="np2"  # Generalized eta-squared is a good effect size for RM designs
)
print(aov)

# --- Post-hoc tests ---
print("\n" + "-"*60)
print(f"{'POST-HOC ANALYSES (Bonferroni-corrected Paired T-tests)':^60}")
print("-"*60)

# 2. Simple Effects of Distractor Presence within each Target Location
print("\n>>> Simple Effects: Distractor Presence within Target Location")
posthoc_presence = []
for target in ['left', 'mid', 'right']:
    res = pg.pairwise_tests(
        data=df_3x2[df_3x2['TargetLoc'] == target],
        dv='target_towardness',
        within='distractor_presence',
        subject=SUBJECT_ID_COL,
        padjust='bonf',
        effsize='cohen'
    )
    res.insert(0, 'TargetLoc', target)
    posthoc_presence.append(res)

posthoc_presence_df = pd.concat(posthoc_presence)
# If only 1 comparison is made (e.g. 2 levels), pingouin returns 'p-unc' but not 'p-corr'
p_col = 'p-corr' if 'p-corr' in posthoc_presence_df.columns else 'p-unc'
print(posthoc_presence_df[['TargetLoc', 'A', 'B', 'T', p_col, 'cohen']].to_string(index=False))

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
df_block_wide = df_block_means.pivot(index=SUBJECT_ID_COL, columns=BLOCK_COL, values=['target_towardness', 'rt', 'select_target'])
df_block_wide.columns = [f"{metric_map.get(c[0], c[0])}_Block-{c[1]}" for c in df_block_wide.columns]
df_block_wide = df_block_wide.reset_index()
df_block_wide.to_csv(os.path.join(f"{get_data_path()}concatenated", "stats_df_reliability_blocks_wide.csv"), index=False)

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

# --- 5. Plotting ---
print("\n--- 5. Generating Summary Plots ---")

# Prepare data for Panel 3 (Distractor Location breakdown)
df_panel3 = df_final[df_final['distractor_presence'] == 'present'].groupby(
    [SUBJECT_ID_COL, DISTRACTOR_COL, TARGET_COL]
)['target_towardness'].mean().reset_index()

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

def plot_sig_bar(ax, x1, x2, p, y_start, d=None, bf=None, h_factor=0.05, color='k'):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * h_factor
    y = y_start + h
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    s = "ns"
    if p < 0.001: s = "***"
    elif p < 0.01: s = "**"
    elif p < 0.05: s = "*"
    if d is not None:
        s += f"\n$d$={d:.2f}"
    if p >= 0.05 and bf is not None:
        s += f"\nBF$_{{10}}$={bf:.2f}"
    ax.text((x1+x2)*.5, y+h, s, ha='center', va='bottom', color=color)
    return y + h * 2

# --- Panel 1: Absent vs Present ---
ax1 = fig.add_subplot(gs[0, 0])
df_p1 = df_3x2.groupby([SUBJECT_ID_COL, 'distractor_presence'])['target_towardness'].mean().reset_index()
sns.boxplot(data=df_p1, x='distractor_presence', y='target_towardness', order=['absent', 'present'], ax=ax1)
sns.stripplot(data=df_p1, x='distractor_presence', y='target_towardness', order=['absent', 'present'], ax=ax1, color='k', alpha=0.3, jitter=True)
ax1.set_title("Distractor Presence")
ax1.set_xlabel("")
ax1.set_ylabel("Target Towardness")

# Stats P1
absent_vals = df_p1[df_p1['distractor_presence']=='absent'].set_index(SUBJECT_ID_COL)['target_towardness']
present_vals = df_p1[df_p1['distractor_presence']=='present'].set_index(SUBJECT_ID_COL)['target_towardness']
common = absent_vals.index.intersection(present_vals.index)
t_p1, p_p1 = ttest_rel(absent_vals.loc[common], present_vals.loc[common])
y_max = df_p1['target_towardness'].max()
d_p1 = pg.compute_effsize(absent_vals.loc[common], present_vals.loc[common], paired=True, eftype='cohen')
bf_p1 = float(pg.ttest(absent_vals.loc[common], present_vals.loc[common], paired=True)['BF10'].values[0])
plot_sig_bar(ax1, 0, 1, p_p1, y_max, d=d_p1, bf=bf_p1)
print("\n--- Panel 1 Stats: Distractor Absent vs. Present ---")
print(f"t({len(common)-1}) = {t_p1:.3f}, p = {p_p1:.4f}, d = {d_p1:.3f}, BF10 = {bf_p1:.3f}")

# --- Panel 2: Target Loc in Absent ---
ax2 = fig.add_subplot(gs[0, 1:], sharey=ax1)
df_p2 = df_3x2[df_3x2['distractor_presence']=='absent']
order_p2 = ['left', 'mid', 'right']
sns.boxplot(data=df_p2, x='TargetLoc', y='target_towardness', order=order_p2, ax=ax2)
sns.stripplot(data=df_p2, x='TargetLoc', y='target_towardness', order=order_p2, ax=ax2, color='k', alpha=0.3)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_title("Target Location (Distractor Absent)")
ax2.set_xlabel("Target Location")
ax2.set_ylabel("")

# Stats P2
print("\n--- Panel 2 Stats: Target Location (Distractor Absent) ---")
pairs = [('left', 'mid'), ('mid', 'right'), ('left', 'right')]
y_curr = df_p2['target_towardness'].max()
for pair in pairs:
    d1 = df_p2[df_p2['TargetLoc']==pair[0]].set_index(SUBJECT_ID_COL)['target_towardness']
    d2 = df_p2[df_p2['TargetLoc']==pair[1]].set_index(SUBJECT_ID_COL)['target_towardness']
    common = d1.index.intersection(d2.index)
    t, p = ttest_rel(d1.loc[common], d2.loc[common])
    d_val = pg.compute_effsize(d1.loc[common], d2.loc[common], paired=True, eftype='cohen')
    bf_val = float(pg.ttest(d1.loc[common], d2.loc[common], paired=True)['BF10'].values[0])
    x1 = order_p2.index(pair[0])
    x2 = order_p2.index(pair[1])
    y_curr = plot_sig_bar(ax2, x1, x2, p, y_curr, d=d_val, bf=bf_val)
    print(f"{pair[0]} vs {pair[1]}: t({len(common)-1}) = {t:.3f}, p = {p:.4f}, d = {d_val:.3f}, BF10 = {bf_val:.3f}")

# --- Panel 3: Distractor Locs (Left, Mid, Right) ---
print("\n--- Panel 3 Stats: Target Location by Distractor Location ---")
dist_locs = ['left', 'mid', 'right']
target_opts = {'left': ['mid', 'right'], 'mid': ['left', 'right'], 'right': ['left', 'mid']}
full_order = ['left', 'mid', 'right']

for i, d_loc in enumerate(dist_locs):
    ax = fig.add_subplot(gs[1, i], sharey=ax1)
    curr_df = df_panel3[df_panel3[DISTRACTOR_COL] == d_loc]
    curr_targets = target_opts[d_loc]
    sns.boxplot(data=curr_df, x=TARGET_COL, y='target_towardness', order=full_order, ax=ax)
    sns.stripplot(data=curr_df, x=TARGET_COL, y='target_towardness', order=full_order, ax=ax, color='k', alpha=0.3)

    # Force x-axis to show all 3 positions and label the distractor
    ax.set_xlim(-0.5, 2.5)
    ax.text(full_order.index(d_loc), np.mean(ax.get_ylim()), 'Distractor', ha='center', va='center', rotation=90, color='gray', alpha=0.5)

    ax.set_title(f"Distractor: {d_loc.capitalize()}")
    ax.set_xlabel("Target Location")
    if i == 0: ax.set_ylabel("Target Towardness")
    else:
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)
    
    if len(curr_targets) == 2:
        d1 = curr_df[curr_df[TARGET_COL]==curr_targets[0]].set_index(SUBJECT_ID_COL)['target_towardness']
        d2 = curr_df[curr_df[TARGET_COL]==curr_targets[1]].set_index(SUBJECT_ID_COL)['target_towardness']
        common = d1.index.intersection(d2.index)
        if len(common) > 1:
            t, p = ttest_rel(d1.loc[common], d2.loc[common])
            d_val = pg.compute_effsize(d1.loc[common], d2.loc[common], paired=True, eftype='cohen')
            bf_val = float(pg.ttest(d1.loc[common], d2.loc[common], paired=True)['BF10'].values[0])
            idx1 = full_order.index(curr_targets[0])
            idx2 = full_order.index(curr_targets[1])
            plot_sig_bar(ax, idx1, idx2, p, curr_df['target_towardness'].max(), d=d_val, bf=bf_val)
            print(f"Distractor {d_loc}: {curr_targets[0]} vs {curr_targets[1]}: t({len(common)-1}) = {t:.3f}, p = {p:.4f}, d = {d_val:.3f}, BF10 = {bf_val:.3f}")

plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(output_dir, "spatial_configuration_performance.svg"))
plt.show()
