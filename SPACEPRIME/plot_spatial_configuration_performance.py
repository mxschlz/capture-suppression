import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Rectangle
import os
import numpy as np
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
import SPACEPRIME
import pingouin as pg
plt.ion()


# --- 1. Data Loading & Preprocessing ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2

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
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
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
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(int).astype(str)

df_final = pd.merge(
    df,
    target_towardness,
    on=merge_cols,
    how='left'
)

# transform to subject averages
df_mean = df_final.groupby(["subject_id", "TargetLoc", "SingletonLoc"])[["target_towardness", "rt", "select_target"]].mean().reset_index()


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

# Perform pairwise t-tests for each distractor location condition
stats_results = []
singleton_locations = df_mean[DISTRACTOR_COL].unique()
target_locations = ["left", "mid", "right"]

for s_loc in singleton_locations:
    df_distractor = df_mean[df_mean[DISTRACTOR_COL] == s_loc]

    # Define pairs for comparison
    pairs = [("left", "mid"), ("left", "right"), ("mid", "right")]
    n_comparisons_top = len(pairs)

    for t_loc1, t_loc2 in pairs:
        # Prepare data for paired t-test
        data1 = df_distractor[df_distractor[TARGET_COL] == t_loc1].set_index(SUBJECT_ID_COL)['target_towardness']
        data2 = df_distractor[df_distractor[TARGET_COL] == t_loc2].set_index(SUBJECT_ID_COL)['target_towardness']

        # Align subjects
        common_subjects = data1.index.intersection(data2.index)
        if len(common_subjects) < 2: # Need at least 2 pairs for a t-test
            continue

        x = data1.loc[common_subjects]
        y = data2.loc[common_subjects]

        # Perform t-test
        ttest_res = pg.ttest(x, y, paired=True)
        p_val = ttest_res['p-val'].iloc[0]
        p_val_corrected = min(p_val * n_comparisons_top, 1.0)
        cohen_d = ttest_res["cohen-d"].iloc[0]

        stats_results.append({
            DISTRACTOR_COL: s_loc,
            'pair': (t_loc1, t_loc2),
            'p_val': p_val,
            'p_val_corrected': p_val_corrected,
            'cohen_d': cohen_d
        })

stats_df = pd.DataFrame(stats_results)
print("--- Pairwise T-test Results ---")
print(stats_df)

# --- 3b. Statistical Analysis for Singleton Location comparison ---
stats_results_sloc = []
target_locations = ["left", "mid", "right"] # Use a fixed order to match plotting
singleton_locations_with_absent = ["absent", "left", "mid", "right"]

for t_loc in target_locations:
    df_target = df_mean[df_mean[TARGET_COL] == t_loc]

    # Define pairs for comparison
    pairs = [("absent", "left"), ("absent", "right"), ("absent", "mid"),
             ("left", "right"), ("left", "mid"), ("right", "mid")]
    n_comparisons_sloc = len(pairs)

    for s_loc1, s_loc2 in pairs:
        # Prepare data for paired t-test
        data1 = df_target[df_target[DISTRACTOR_COL] == s_loc1].set_index(SUBJECT_ID_COL)['target_towardness']
        data2 = df_target[df_target[DISTRACTOR_COL] == s_loc2].set_index(SUBJECT_ID_COL)['target_towardness']

        # Align subjects
        common_subjects = data1.index.intersection(data2.index)
        if len(common_subjects) < 2: # Need at least 2 pairs for a t-test
            continue

        x = data1.loc[common_subjects]
        y = data2.loc[common_subjects]

        # Perform t-test
        ttest_res = pg.ttest(x, y, paired=True)
        p_val = ttest_res['p-val'].iloc[0]
        p_val_corrected = min(p_val * n_comparisons_sloc, 1.0)
        stats_results_sloc.append({
            TARGET_COL: t_loc,
            'pair': (s_loc1, s_loc2),
            'p_val_corrected': p_val_corrected,
            'cohen_d': ttest_res["cohen-d"].iloc[0]
        })

stats_sloc_df = pd.DataFrame(stats_results_sloc)
print("\n--- Pairwise T-test Results (Singleton Location) ---")
print(stats_sloc_df)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- 4. Plotting Setup ---
# --- Panel A: Distractor Absent ---
fig_a, ax_a = plt.subplots(figsize=(8, 6))

# Filter data for distractor-absent trials
df_absent = df_mean[df_mean[DISTRACTOR_COL] == 'absent']

# Define order for bars
bar_order = ["left", "mid", "right"]

# Create the bar plots
sns.barplot(
    data=df_absent,
    x=TARGET_COL, # This is the x-axis for Panel A
    y='target_towardness',
    order=bar_order,
    color="green",
    errorbar=('se', 1), # Show standard error
    ax=ax_a
)

# --- Print Stats for Panel A ---
print("\n--- Panel A: Distractor Absent Stats ---")
for loc in bar_order:
    data = df_absent[df_absent[TARGET_COL] == loc]['target_towardness']
    mean = data.mean()
    sem = data.sem()
    n = len(data)
    print(f"  Target: {loc.capitalize()} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

# --- Final Touches for Panel A ---
ax_a.set_title("Panel A: Distractor Absent", fontsize=14, weight='bold')
ax_a.set_xlabel("Target Location", fontsize=12)
ax_a.set_ylabel("Target Towardness", fontsize=12)
ax_a.set_ylim(0, 0.5) # Set symmetrical y-axis as requested
ax_a.axhline(0, color='grey', linestyle='--', lw=1)
sns.despine(ax=ax_a)
fig_a.tight_layout()

# --- Panel B: Custom Bar Plots for Distractor-Present Trials ---

# Filter for distractor-present trials only
df_present = df_mean[df_mean[DISTRACTOR_COL] != 'absent'].copy()

# Create a new figure and subplot for Panel B
fig_b, ax_b1 = plt.subplots(figsize=(8, 6))

# --- Create the three specific bars as requested ---

# 1. Calculate means and standard errors for each specific condition
bar_data = [
    # Bar 1: All left target trials
    df_present[df_present[TARGET_COL] == 'left']['target_towardness'],
    # Bar 2: All mid distractor trials
    df_present[df_present[DISTRACTOR_COL] == 'mid']['target_towardness'],
    # Bar 3: All right target trials
    df_present[df_present[TARGET_COL] == 'right']['target_towardness']
]
bar_means = [d.mean() for d in bar_data]
bar_errors = [d.sem() for d in bar_data]
bar_labels = ['Target: Left', 'Distractor: Mid', 'Target: Right']
bar_colors = ['green', 'red', 'green']

# 2. Plot the bars manually
ax_b1.bar(bar_labels, bar_means, yerr=bar_errors, capsize=5, color=bar_colors)

# --- Print Stats for Panel B, Plot 1 ---
print("\n--- Panel B (Set 1) Stats ---")
for i, label in enumerate(bar_labels):
    mean = bar_means[i]
    sem = bar_errors[i]
    n = len(bar_data[i]) # Number of subjects
    print(f"  {label} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

# --- Final Touches for the new Panel B subplot ---
ax_b1.set_ylabel("Target Towardness", fontsize=12)
ax_b1.set_ylim(0, 0.5) # Set symmetrical y-axis as requested
ax_b1.axhline(0, color='grey', linestyle='--', lw=1)
ax_b1.tick_params(axis='x', rotation=45)
ax_b1.set_title("Panel B (Set 1)", fontsize=14, weight='bold')
sns.despine(ax=ax_b1)
fig_b.tight_layout()

# --- Panel B, Plot 2 ---
fig_b2, ax_b2 = plt.subplots(figsize=(8, 6))

# --- Create the three specific bars for the second plot ---
bar_data_2 = [
    # Bar 1: All left distractor trials
    df_present[df_present[DISTRACTOR_COL] == 'left']['target_towardness'],
    # Bar 2: All mid target trials
    df_present[df_present[TARGET_COL] == 'mid']['target_towardness'],
    # Bar 3: All right target trials
    df_present[df_present[TARGET_COL] == 'right']['target_towardness']
]
bar_means_2 = [d.mean() for d in bar_data_2]
bar_errors_2 = [d.sem() for d in bar_data_2]
bar_labels_2 = ['Distractor: Left', 'Target: Mid', 'Target: Right']
bar_colors_2 = ['red', 'green', 'green']

# Plot the bars
ax_b2.bar(bar_labels_2, bar_means_2, yerr=bar_errors_2, capsize=5, color=bar_colors_2)

# --- Print Stats for Panel B, Plot 2 ---
print("\n--- Panel B (Set 2) Stats ---")
for i, label in enumerate(bar_labels_2):
    mean = bar_means_2[i]
    sem = bar_errors_2[i]
    n = len(bar_data_2[i]) # Number of subjects
    print(f"  {label} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

# Final touches for the plot
ax_b2.set_ylabel("Target Towardness", fontsize=12)
ax_b2.axhline(0, color='grey', linestyle='--', lw=1)
ax_b2.set_ylim(0, 0.5) # Set symmetrical y-axis as requested
ax_b2.tick_params(axis='x', rotation=45)
ax_b2.set_title("Panel B (Set 2)", fontsize=14, weight='bold')
sns.despine(ax=ax_b2)
fig_b2.tight_layout()

# --- Panel B, Plot 3 ---
fig_b3, ax_b3 = plt.subplots(figsize=(8, 6))

# --- Create the three specific bars for the third plot ---
bar_data_3 = [
    # Bar 1: All left target trials
    df_present[df_present[TARGET_COL] == 'left']['target_towardness'],
    # Bar 2: All mid target trials
    df_present[df_present[TARGET_COL] == 'mid']['target_towardness'],
    # Bar 3: All right distractor trials
    df_present[df_present[DISTRACTOR_COL] == 'right']['target_towardness']
]
bar_means_3 = [d.mean() for d in bar_data_3]
bar_errors_3 = [d.sem() for d in bar_data_3]
bar_labels_3 = ['Target: Left', 'Target: Mid', 'Distractor: Right']
bar_colors_3 = ['green', 'green', 'red']

# Plot the bars
ax_b3.bar(bar_labels_3, bar_means_3, yerr=bar_errors_3, capsize=5, color=bar_colors_3)

# --- Print Stats for Panel B, Plot 3 ---
print("\n--- Panel B (Set 3) Stats ---")
for i, label in enumerate(bar_labels_3):
    mean = bar_means_3[i]
    sem = bar_errors_3[i]
    n = len(bar_data_3[i]) # Number of subjects
    print(f"  {label} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")

# Final touches for the plot
ax_b3.set_ylabel("Target Towardness", fontsize=12)
ax_b3.axhline(0, color='grey', linestyle='--', lw=1)
ax_b3.set_ylim(0, 0.5) # Set symmetrical y-axis as requested
ax_b3.tick_params(axis='x', rotation=45)
ax_b3.set_title("Panel B (Set 3)", fontsize=14, weight='bold')
sns.despine(ax=ax_b3)
fig_b3.tight_layout()
