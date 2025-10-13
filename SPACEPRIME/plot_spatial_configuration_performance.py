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

# Create a new column to distinguish between distractor present and absent trials
df_mean['distractor_presence'] = np.where(df_mean[DISTRACTOR_COL] == 'absent', 'absent', 'present')



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

# --- 4. Combined Plotting: Mosaic Layout ---
# Create a figure using subplot_mosaic for a more intuitive layout.
# 'e' is the new summary plot. 'a' is distractor absent. 'b', 'c', 'd' are distractor present.
fig, axes = plt.subplot_mosaic(
    mosaic="""
    eaa
    bcd
    """,
    figsize=(15, 10),
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
ax_a.axhline(0, color='grey', linestyle='--', lw=1)

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
ax_e.axhline(0, color='grey', linestyle='--', lw=1)

# --- Stats for Panel E ---
print("\n--- Panel E: Distractor Presence Stats ---")
data_abs = df_presence_summary[df_presence_summary['distractor_presence'] == 'absent']['target_towardness']
data_pres = df_presence_summary[df_presence_summary['distractor_presence'] == 'present']['target_towardness']

ttest_presence = pg.ttest(data_abs, data_pres, paired=True)
print("  Paired T-test (Absent vs. Present):")
print(f"    t = {ttest_presence['T'].iloc[0]:.3f}, p = {ttest_presence['p-val'].iloc[0]:.3f}, Cohen's d = {ttest_presence['cohen-d'].iloc[0]:.3f}")

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
plt.setp(list(axes.values()), ylim=(0.0, 0.5))

# Clean up the overall figure appearance
sns.despine(fig=fig)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.98]) # Adjust layout for super-labels

# Save the combined figure with a new name reflecting the layout
fig.savefig(os.path.join(output_dir, "combined_spatial_performance_mosaic.svg"))

plt.show() # Display the final combined plot
