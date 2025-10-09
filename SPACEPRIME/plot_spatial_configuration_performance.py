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

# --- 4a. Color Palette Definition ---
location_colors = {
    "left": "#AA3377",   # Purple
    "mid": "#888888",    # Dark Grey
    "right": "#CCBB44",  # Yellow
    "absent": "#EEEEEE"  # Lighter Grey
}

# --- 4. Combined Plotting on a Single Canvas ---
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 15), sharey=True)

# --- Plot 1: Top Row (TargetLoc on X-axis) ---
col_order = ["left", "right", "mid"]
x_order_target = ["left", "mid", "right"]

# --- Pre-calculate annotation heights for top row for consistency across panels ---
offset = 0.05
pairs_pos_top = [
    (("left", "mid"), (0, 1)),
    (("left", "right"), (0, 2)),
    (("mid", "right"), (1, 2))
]

# Find the max bar height for each comparison across ALL relevant panels
max_heights_top = []
for pair, _ in pairs_pos_top:
    max_val = df_mean[df_mean[TARGET_COL].isin(pair)].groupby(DISTRACTOR_COL)['target_towardness'].max().max()
    max_heights_top.append(max_val)

y_bracket_levels_top = [h + offset * 1.5 for h in max_heights_top]
if len(y_bracket_levels_top) > 1:
    y_bracket_levels_top[1] = max(y_bracket_levels_top[0], y_bracket_levels_top[2]) + offset * 2.0

for i, s_loc in enumerate(col_order):
    ax = axes[0, i]
    df_plot = df_mean[df_mean[DISTRACTOR_COL] == s_loc]

    sns.barplot(data=df_plot, x=TARGET_COL, y="target_towardness",
                   order=x_order_target, palette=location_colors, ax=ax, errorbar=('se', 1),
                   width=1.0)

    ax.set_title(f"{DISTRACTOR_COL} = {s_loc}")
    ax.set_xlabel(TARGET_COL)
    if i > 0:
        ax.set_ylabel("")

    # --- Annotations for the top row (using pre-calculated heights) ---
    for j, (pair, (x1, x2)) in enumerate(pairs_pos_top):
        stats = stats_df[(stats_df[DISTRACTOR_COL] == s_loc) & (stats_df['pair'] == pair)]
        if stats.empty:
            continue

        p_str = p_to_stars(stats['p_val_corrected'].iloc[0])
        y_bracket = y_bracket_levels_top[j]

        ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket + offset, y_bracket + offset, y_bracket], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y_bracket + offset * 1.5, p_str, ha='center', va='bottom', color='k')

    # Set a uniform Y-limit for the entire row after calculating all heights
    ax.set_ylim(top=max(y_bracket_levels_top) + offset * 2.5)

# --- Plot 2: Bottom Row (SingletonLoc on X-axis) ---
col_order_sloc = ["left", "right", "mid"]
x_order_sloc = ["absent", "left", "mid", "right"]

# --- Pre-calculate annotation heights for bottom row ---
pairs_pos_bottom = [
    (("absent", "left"), (0, 1)),   # Level 1
    (("left", "right"), (1, 3)),    # Level 1
    (("absent", "right"), (0, 3)),  # Level 2
    (("left", "mid"), (1, 2)),      # Level 2
    (("right", "mid"), (3, 2)),     # Level 1
    (("absent", "mid"), (0, 2)),     # Level 3
]

max_heights_bottom = []
for pair, _ in pairs_pos_bottom: # TODO: this is not working as intended
    max_val = df_mean[df_mean[DISTRACTOR_COL].isin(pair)].groupby(TARGET_COL)['target_towardness'].max().max()
    max_heights_bottom.append(max_val)

y_bracket_levels_bottom = []
current_y = df_mean['target_towardness'].max() + offset * 0.5 # Start from global max with a small offset
for h in max_heights_bottom:
    current_y = max(current_y, h) + offset * 0.5
    y_bracket_levels_bottom.append(current_y)

for i, t_loc in enumerate(col_order_sloc):
    ax = axes[1, i]
    df_plot = df_mean[df_mean[TARGET_COL] == t_loc]

    sns.barplot(data=df_plot, x=DISTRACTOR_COL, y="target_towardness",
                   order=x_order_sloc, palette=location_colors, ax=ax, errorbar=('se', 1),
                   width=1.0)

    ax.set_title(f"{TARGET_COL} = {t_loc}")
    ax.set_xlabel(DISTRACTOR_COL)
    if i > 0:
        ax.set_ylabel("")

    # --- Annotations for the bottom row (using pre-calculated heights) ---
    for j, (pair, (x1, x2)) in enumerate(pairs_pos_bottom):
        stats = stats_sloc_df[(stats_sloc_df[TARGET_COL] == t_loc) & (stats_sloc_df['pair'] == pair)]
        if stats.empty:
            continue

        p_str = p_to_stars(stats['p_val_corrected'].iloc[0])
        y_bracket = y_bracket_levels_bottom[j]

        ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket + offset, y_bracket + offset, y_bracket], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y_bracket + offset * 1.5, p_str, ha='center', va='bottom', color='k')

    # Set a uniform Y-limit for the entire row
    ax.set_ylim(top=max(y_bracket_levels_bottom) + offset * 3.0)


# --- 5. Final Touches ---
sns.despine(fig=fig)
# Use tight_layout to adjust for titles and labels.
# We need to draw the figure first to allow tight_layout to calculate the correct positions.
fig.canvas.draw()

# Get the positions of the rightmost subplots in figure coordinates
ax_top_right = axes[0, 2]
ax_bottom_right = axes[1, 2]

bbox_top = ax_top_right.get_position()
bbox_bottom = ax_bottom_right.get_position()

# Define the rectangle that encloses both subplots
rect_x0 = bbox_bottom.x0 - 0.03
rect_y0 = bbox_bottom.y0 - 0.05
rect_width = bbox_top.x1 - rect_x0 + 0.03
rect_height = bbox_top.y1 - rect_y0 + 0.05

# Add the rectangle to the figure
rect = Rectangle((rect_x0, rect_y0), rect_width, rect_height,
                 transform=fig.transFigure,
                 facecolor='none',
                 edgecolor='grey', lw=2, linestyle='--', zorder=0)
fig.add_artist(rect)

plt.show()
