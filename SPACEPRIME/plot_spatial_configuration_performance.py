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


# --- 0. Configuration ---
# Define an output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


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

# --- 4. Combined Plotting: Mosaic Layout ---
# Create a figure using subplot_mosaic for a more intuitive layout.
# 'a' will be the top-center plot, with 'b', 'c', and 'd' along the bottom row.
fig, axes = plt.subplot_mosaic(
    mosaic="""
    .a.
    bcd
    """,
    figsize=(12, 10),
    sharex=True,
    sharey=True
)

# --- Define which axis is for which plot ---
# With mosaic, axes is a dictionary. We access each subplot by its label.
ax_a = axes['a']
axes_b = [axes['b'], axes['c'], axes['d']] # Group the bottom axes for iteration

# --- 4a. Plotting Panel A: Distractor Absent ---
df_absent = df_mean[df_mean[DISTRACTOR_COL] == 'absent']
target_order = ["left", "mid", "right"]

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
ax_a.axhline(0, color='grey', linestyle='--', lw=1)

# --- Print Stats for Panel A ---
print("\n--- Panel A: Distractor Absent Stats ---")
for loc in target_order:
    data = df_absent[df_absent[TARGET_COL] == loc]['target_towardness']
    mean = data.mean()
    sem = data.sem()
    n = len(data)
    print(f"  Target: {loc.capitalize()} | Mean: {mean:.3f}, SEM: {sem:.3f}, N: {n}")


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
    ax.axhline(0, color='grey', linestyle='--', lw=1)


# --- 5. Final Figure-Wide Touches ---
# Set shared axis labels for the entire figure
fig.supxlabel("Target Location", fontsize=14)
fig.supylabel("Target Towardness", fontsize=14)

# Set a consistent Y-limit for all plots by accessing the dictionary's values
plt.setp(list(axes.values()), ylim=(0, 0.5))

# Clean up the overall figure appearance
sns.despine(fig=fig)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.98]) # Adjust layout for super-labels

# Save the combined figure with a new name reflecting the layout
fig.savefig(os.path.join(output_dir, "combined_spatial_performance_mosaic.svg"))

plt.show() # Display the final combined plot