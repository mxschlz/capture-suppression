import pandas as pd
from stats import remove_outliers
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from utils import calculate_trajectory_projections, get_vector_length, resample_all_trajectories # Assuming utils is importable
import glob
import numpy as np
from scipy.optimize import curve_fit



sns.set_theme(context="talk", style="ticks")

OUTLIER_THRESH = 2
PALETTE = {"High": "green", "Low": "red", "Absent": "gray"}
TRAJECTORY_BOUNDARY_DVA = 2
RESAMP_FREQ = 50
ROLLING_WINDOW = 40

# Define the locations of the stimuli in degrees of visual angle (dva)
# This is crucial for calculating trajectory projections.
locations_map = {
    7: (-0.6, -0.6), 8: (0, -0.6), 9: (0.6, -0.6),
    4: (-0.6, 0), 5: (0, 0), 6: (0.6, 0),
    1: (-0.6, 0.6), 2: (0, 0.6), 3: (0.6, 0.6),
}

# --- Data Loading Logic (from implicit_learning_effect.py) ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
experiment_folder = "pilot/distractor"

# Load all CSV files in the directory
df = pd.concat([pd.read_csv(f"{data_path}{experiment_folder}/{file}") for file in os.listdir(f"{data_path}{experiment_folder}")], ignore_index=True)

if "distractor-absent" in experiment_folder:
    # Ensure key columns are of integer type for merging/grouping
    df['Subject ID'] = df['subject_id'].astype(int, errors="ignore")
    df['Block'] = df['block'].astype(int, errors="ignore")
    df['Trial Nr'] = df['trial_nr'].astype(int, errors="ignore")

    # Calculate accuracy
    df["IsCorrect"] = df["select_target"].astype(float, errors="ignore")

    if 'target_loc' in df.columns:
        df['TargetLoc'] = df['target_loc']

    # Map SingletonLoc values to labels
    # 0: Absent, 1: Left, 2: Front, 3: Right
    df['SingletonLoc'] = df['SingletonLoc'].map({0: 'Absent', 1: 'Left', 2: 'Front', 3: 'Right'})
    df['TargetLoc'] = df['TargetLoc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

# Determine location column based on experiment design
loc_col = "Non-Singleton2Loc" if "control" in experiment_folder else "SingletonLoc"

# Ensure IsCorrect is numeric
if "IsCorrect" not in df.columns and "select_target" in df.columns:
    df["IsCorrect"] = df["select_target"]
if "IsCorrect" in df.columns:
    df["IsCorrect"] = df["IsCorrect"].replace({'True': 1, 'False': 0, True: 1, False: 0})
    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors='coerce')

# Remove outliers
df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID")

# Ensure key columns are of integer type for merging with mouse data
df['Subject ID'] = df['Subject ID'].astype(int, errors="ignore")
df['Block'] = df['Block'].astype(int, errors="ignore")
df['Trial Nr'] = df['Trial Nr'].astype(int, errors="ignore")
df["IsCorrect"] = df["IsCorrect"].astype(float, errors="ignore")

# Create snake_case columns for consistency
df['subject_id'] = df['Subject ID'].astype(int, errors='ignore')
df['block'] = df['Block'].astype(int, errors='ignore')
df['trial_nr'] = df['Trial Nr'].astype(int, errors='ignore')

# Define probability condition based on Subject ID and Location
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # Even subjects: Left is High, others (Right, Front) are Low
    # Odd subjects: Right is High, others (Left, Front) are Low
    is_even = row['subject_id'] % 2 == 0
    high_loc = 'Left' if is_even else 'Right'
    return 'High' if row[loc_col] == high_loc else 'Low'

df['Probability'] = df.apply(get_probability, axis=1)
df['DistractorProb'] = df['Probability']

# --- Analysis: Performance by Singleton Location per Subject ---
print("Generating plots for Response Time and Accuracy per subject...")

# Create dataframe for distractor analysis (removing Front targets)
df_distractor = df[(df["TargetLoc"] != "Front")].copy()

# --- Sanity Check: Target Location Distribution ---
print("Generating Target Location distribution histograms...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df_distractor, x="TargetLoc", hue="DistractorProb",
              order=["Left", "Front", "Right"], hue_order=["High", "Low", "Absent"],
              palette=PALETTE)
plt.title("Target Location Distribution per Condition")
sns.despine()
plt.show()

def plot_perf(data, x, y, **kwargs):
    order = ["Absent", "Left", "Right"]
    # Draw connecting line
    sns.pointplot(data=data, x=x, y=y, order=order,
                  color="black", markers="", linestyles="-", errorbar=None)
    # Draw points colored by Probability
    sns.pointplot(data=data, x=x, y=y, hue="Probability", order=order,
                  palette=PALETTE, join=False, errorbar=None, dodge=False)

# 1. Response Time Visualization
# Using FacetGrid to create a subplot for each subject
g_rt = sns.FacetGrid(df_distractor, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_rt.map_dataframe(plot_perf, x=loc_col, y="rt")
g_rt.set_titles("Sub {col_name}")
g_rt.set_axis_labels(loc_col, "RT (s)")
plt.show()

# 2. Accuracy Visualization
g_acc = sns.FacetGrid(df_distractor, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_acc.map_dataframe(plot_perf, x=loc_col, y="IsCorrect")
g_acc.set_titles("Sub {col_name}")
g_acc.set_axis_labels(loc_col, "Accuracy (%)")
plt.show()

# 3. Visualization by Distractor Probability
print("Generating plots for Response Time and Accuracy by Distractor Probability...")

def plot_prob_perf(data, x, y, **kwargs):
    order = ["High", "Low", "Absent"]
    sns.pointplot(data=data, x=x, y=y, order=order,
                  palette=PALETTE, join=False, errorbar=None)

g_rt_prob = sns.FacetGrid(df_distractor, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_rt_prob.map_dataframe(plot_prob_perf, x="DistractorProb", y="rt")
g_rt_prob.set_titles("Sub {col_name}")
g_rt_prob.set_axis_labels("Distractor Prob", "RT (s)")
plt.show()

g_acc_prob = sns.FacetGrid(df_distractor, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_acc_prob.map_dataframe(plot_prob_perf, x="DistractorProb", y="IsCorrect")
g_acc_prob.set_titles("Sub {col_name}")
g_acc_prob.set_axis_labels("Distractor Prob", "Accuracy (%)")
plt.show()

# 4. Print aggregated data for inspection
agg_df = df_distractor.groupby(['subject_id', loc_col])[['rt', 'IsCorrect']].mean()
print("\n--- Aggregated Performance (Mean) ---")
print(agg_df)

# 5. Aggregate Bar Plots (All Subjects)
print("Generating aggregate bar plots...")

def get_cohens_d(df, subj, cond, val, c1, c2):
    """Calculates paired Cohen's d using pingouin."""
    try:
        wide = df[df[cond].isin([c1, c2])].pivot(index=subj, columns=cond, values=val).dropna()
        if c1 in wide and c2 in wide:
            return pg.compute_effsize(wide[c1], wide[c2], paired=True, eftype='cohen')
    except Exception:
        pass
    return None

# Calculate subject means for correct error bars and individual lines
subject_means = df_distractor.groupby(['subject_id', 'DistractorProb'])[['rt', 'IsCorrect']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
order = ["High", "Low", "Absent"]

sns.barplot(data=subject_means, x="DistractorProb", y="rt", palette=PALETTE, errorbar=("se", 1), ax=axes[0], alpha=0.5)
sns.lineplot(data=subject_means, x="DistractorProb", y="rt", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[0])
axes[0].set_ylabel("RT (s)")
d_rt_hl = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'rt', 'High', 'Low')
d_rt_la = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'rt', 'Low', 'Absent')
t_rt = []
if d_rt_hl is not None: t_rt.append(f"H-L cohens d={d_rt_hl:.2f}")
if d_rt_la is not None: t_rt.append(f"L-A cohens d={d_rt_la:.2f}")
if t_rt: axes[0].set_title(", ".join(t_rt))

sns.barplot(data=subject_means, x="DistractorProb", y="IsCorrect", palette=PALETTE, errorbar=("se", 1), ax=axes[1], alpha=0.5)
sns.lineplot(data=subject_means, x="DistractorProb", y="IsCorrect", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[1])
axes[1].set_ylabel("Accuracy (%)")
d_acc_hl = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'IsCorrect', 'High', 'Low')
d_acc_la = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'IsCorrect', 'Low', 'Absent')
t_acc = []
if d_acc_hl is not None: t_acc.append(f"H-L cohens d={d_acc_hl:.2f}")
if d_acc_la is not None: t_acc.append(f"L-A cohens d={d_acc_la:.2f}")
if t_acc: axes[1].set_title(", ".join(t_acc))

plt.tight_layout()
sns.despine()
plt.show()

# 7. Analysis: Learning Effect (Block-wise)
print("Generating plots for Learning Effect (Block-wise)...")

# Calculate subject means per block and condition
block_means = df_distractor.groupby(['subject_id', 'block', 'DistractorProb'])[['rt', 'IsCorrect']].mean().reset_index()

# Filter only High and Low
block_means = block_means[block_means['DistractorProb'].isin(['High', 'Low'])]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.lineplot(data=block_means, x="block", y="rt", hue="DistractorProb", palette=PALETTE, errorbar=("se", 1), ax=axes[0], marker="o")
axes[0].set_ylabel("RT (s)")

sns.lineplot(data=block_means, x="block", y="IsCorrect", hue="DistractorProb", palette=PALETTE, errorbar=("se", 1), ax=axes[1], marker="o")
axes[1].set_ylabel("Accuracy (%)")

plt.tight_layout()
sns.despine()
plt.show()

# 8. Analysis: Temporal Evolution (Rolling Average)
print("Generating plots for Temporal Evolution (Rolling Average)...")

# Sort data to ensure temporal order
df_distractor = df_distractor.sort_values(['subject_id', 'block', 'trial_nr'])
df_distractor['abs_trial'] = df_distractor.groupby('subject_id').cumcount() + 1

# Calculate rolling averages and difference (High - Low)
diff_data = []
for sub_id, sub_df in df_distractor.groupby('subject_id'):
    # Create a complete index of trials for this subject
    trial_index = sub_df.set_index('abs_trial').index

    # Calculate rolling means for High condition
    high_df = sub_df[sub_df['DistractorProb'] == 'High'].set_index('abs_trial')
    high_rt = high_df['rt'].rolling(ROLLING_WINDOW, min_periods=5, center=True).mean()
    high_acc = high_df['IsCorrect'].rolling(ROLLING_WINDOW, min_periods=5, center=True).mean()

    # Calculate rolling means for Low condition
    low_df = sub_df[sub_df['DistractorProb'] == 'Low'].set_index('abs_trial')
    low_rt = low_df['rt'].rolling(ROLLING_WINDOW, min_periods=5, center=True).mean()
    low_acc = low_df['IsCorrect'].rolling(ROLLING_WINDOW, min_periods=5, center=True).mean()

    # Combine into a single dataframe indexed by abs_trial
    tmp = pd.DataFrame(index=trial_index)
    tmp['rt_high'] = high_rt
    tmp['acc_high'] = high_acc
    tmp['rt_low'] = low_rt
    tmp['acc_low'] = low_acc

    # Forward fill to propagate the last known state
    tmp = tmp.ffill()

    # Calculate difference
    tmp['rt_diff'] = tmp['rt_high'] - tmp['rt_low']
    tmp['acc_diff'] = tmp['acc_high'] - tmp['acc_low']
    tmp['subject_id'] = sub_id

    diff_data.append(tmp.reset_index())

diff_df = pd.concat(diff_data).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.lineplot(data=diff_df, x="abs_trial", y="rt_diff", color="purple", errorbar=("se", 1), ax=axes[0])
axes[0].set_ylabel(f"RT Diff (s) (High - Low) ({ROLLING_WINDOW}-trial Avg)")
axes[0].set_xlabel("Trial Number")
axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0].set_title("RT Difference")

sns.lineplot(data=diff_df, x="abs_trial", y="acc_diff", color="purple", errorbar=("se", 1), ax=axes[1])
axes[1].set_ylabel(f"Accuracy Diff (High - Low) ({ROLLING_WINDOW}-trial Avg)")
axes[1].set_xlabel("Trial Number")
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].set_title("Accuracy Difference")

plt.tight_layout()
sns.despine()
plt.show()

# 8b. Analysis: Pinpoint Learning Effect (Piecewise Linear Fit)
print("Calculating learning stabilization point...")

learning_curve = diff_df.groupby('abs_trial')['rt_diff'].mean().reset_index().dropna()
x_data = learning_curve['abs_trial'].values
y_data = learning_curve['rt_diff'].values

# Define Piecewise Linear Model (Broken Stick: Linear decay -> Plateau)
def piecewise_linear(x, x_break, y_plateau, slope):
    # Model: y = y_plateau + slope * (x - x_break)  for x < x_break
    #        y = y_plateau                          for x >= x_break
    # Note: slope should be negative to represent a decrease
    return y_plateau + slope * np.minimum(0, x - x_break)

# Fit the model
# Initial guesses: break at trial 50, plateau at mean of last 100 trials, slope -0.001
p0 = [50, np.mean(y_data[-100:]), -0.001]
try:
    popt, pcov = curve_fit(piecewise_linear, x_data, y_data, p0=p0)
    x_break_est, y_plateau_est, slope_est = popt
    print(f"Estimated stabilization trial (Break Point): {x_break_est:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_data, y_data, label='Group Mean RT Diff', alpha=0.5)
    plt.plot(x_data, piecewise_linear(x_data, *popt), 'r--', lw=2, label=f'Fit (Break at {x_break_est:.0f})')
    plt.axvline(x_break_est, color='k', linestyle=':', label='Stabilization')
    plt.xlabel("Trial Number")
    plt.ylabel("RT Difference (s)")
    plt.legend()
    plt.title("Learning Effect Stabilization (RT Diff)")
    sns.despine()
    plt.show()
except Exception as e:
    print(f"Fitting failed: {e}")

# 6. Analysis: Target Location Probability
print("Generating plots for Target Location Probability...")

# Create a copy for target analysis and remove Front targets
df_target = df[(df["TargetLoc"] != "Front")].copy()

def get_target_prob(row):
    # Even subjects: Left is High-Prob Singleton (Suppressed)
    # Odd subjects: Right is High-Prob Singleton (Suppressed)
    is_even = row['subject_id'] % 2 == 0
    high_prob_loc = 'Left' if is_even else 'Right'
    
    if row['TargetLoc'] == high_prob_loc:
        return 'High' # Target at Suppressed Location
    return 'Low' # Target at Non-Suppressed Location

df_target['Target_at_HP_distractor_loc'] = df_target.apply(get_target_prob, axis=1)

# Aggregate plots for Target Prob
subject_means_target = df_target.groupby(['subject_id', 'Target_at_HP_distractor_loc'])[['rt', 'IsCorrect']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
order_target = ["High", "Low"]

sns.barplot(data=subject_means_target, x="Target_at_HP_distractor_loc", y="rt", order=order_target, palette=PALETTE, errorbar=("se", 1), ax=axes[0], alpha=0.5)
sns.lineplot(data=subject_means_target, x="Target_at_HP_distractor_loc", y="rt", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[0])
axes[0].set_ylabel("RT (s)")
d_rt_t = get_cohens_d(subject_means_target, 'subject_id', 'Target_at_HP_distractor_loc', 'rt', 'High', 'Low')
if d_rt_t is not None: axes[0].set_title(f"H-L cohens d={d_rt_t:.2f}")

sns.barplot(data=subject_means_target, x="Target_at_HP_distractor_loc", y="IsCorrect", order=order_target, palette=PALETTE, errorbar=("se", 1), ax=axes[1], alpha=0.5)
sns.lineplot(data=subject_means_target, x="Target_at_HP_distractor_loc", y="IsCorrect", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[1])
axes[1].set_ylabel("Accuracy (%)")
d_acc_t = get_cohens_d(subject_means_target, 'subject_id', 'Target_at_HP_distractor_loc', 'IsCorrect', 'High', 'Low')
if d_acc_t is not None: axes[1].set_title(f"H-L cohens d={d_acc_t:.2f}")

plt.tight_layout()
sns.despine()
plt.show()

# --- Load and process raw trajectory data for towardness analysis ---
print("Loading and processing raw trajectory data for towardness analysis...")

# Find all subject folders for raw mouse data
raw_data_base_path = os.path.join(SPACECUE_implicit.get_data_path(), 'sourcedata', 'raw')
subject_folders = glob.glob(f"{raw_data_base_path}/sci-*")

df_mouse_list = []
for subject_folder in subject_folders:
    mouse_file = glob.glob(f"{subject_folder}/beh/*mouse_data.csv")
    if mouse_file:
        temp_df = pd.read_csv(mouse_file[0])
        sub_id_str = os.path.basename(subject_folder)
        temp_df['subject_id'] = int(sub_id_str.split('-')[1])
        df_mouse_list.append(temp_df)

df_mouse = pd.concat(df_mouse_list, ignore_index=True)
# Corrected line:
df_mouse['block'] = df_mouse.groupby('subject_id')['trial_nr'].transform(
    lambda x: ((x == 0) & (x.shift(1) != 0)).cumsum() - 1)

# resample data
df_mouse = resample_all_trajectories(df_mouse, RESAMP_FREQ)

# --- Filter out noisy trajectories ---
print(f"Filtering out noisy trajectories beyond {TRAJECTORY_BOUNDARY_DVA} dva...")
max_coords_per_trial = df_mouse.groupby(['subject_id', 'block', 'trial_nr']).agg(
    max_abs_x=('x', lambda s: s.abs().max()),
    max_abs_y=('y', lambda s: s.abs().max())
).reset_index()

noisy_mask = (max_coords_per_trial['max_abs_x'] > TRAJECTORY_BOUNDARY_DVA) | \
             (max_coords_per_trial['max_abs_y'] > TRAJECTORY_BOUNDARY_DVA)
noisy_trials_df = max_coords_per_trial[noisy_mask]

noisy_trial_identifiers = set(zip(
    noisy_trials_df['subject_id'],
    noisy_trials_df['block'],
    noisy_trials_df['trial_nr']
))

initial_count = len(df)
df = df[~df.set_index(['subject_id', 'block', 'trial_nr']).index.isin(noisy_trial_identifiers)].copy()
print(f"Removed {initial_count - len(df)} noisy trials.")

# Calculate the average trajectory vector for each trial
avg_trajectory_vectors_df = df_mouse.groupby(['subject_id', 'block', 'trial_nr']).agg(
    avg_x_dva=('x', 'mean'),
    avg_y_dva=('y', 'mean')
).reset_index()

# Ensure subject_id, block, trial_nr are consistent types for merging
avg_trajectory_vectors_df['subject_id'] = avg_trajectory_vectors_df['subject_id'].astype(int)
avg_trajectory_vectors_df['block'] = avg_trajectory_vectors_df['block'].astype(int)
avg_trajectory_vectors_df['trial_nr'] = avg_trajectory_vectors_df['trial_nr'].astype(int)

# Merge the average vectors into the main behavioral dataframe
df = pd.merge(df, avg_trajectory_vectors_df, on=['subject_id', 'block', 'trial_nr'], how='left')

# Calculate the towardness scores
towardness_scores = df.apply(calculate_trajectory_projections, axis=1, locations_map=locations_map)
tt_df = pd.concat([df, towardness_scores], axis=1)
tt_df = tt_df[tt_df["TargetDigit"] != 5]
#tt_df.dropna(subset=['proj_target'], inplace=True) # Drop trials where score couldn't be computed

# Normalize by vector length
tt_df['target_vec_length'] = tt_df['TargetDigit'].apply(get_vector_length, locations_map=locations_map)
tt_df['target_towardness'] = tt_df['proj_target'] / tt_df['target_vec_length']

tt_df_distractor = tt_df[(tt_df["TargetLoc"] != "Front")]
df_mean = tt_df_distractor.groupby(["subject_id", "DistractorProb"])[["rt", "IsCorrect", "target_towardness"]].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
order = ["High", "Low", "Absent"]

# 1. Response Time
sns.barplot(data=df_mean, x="DistractorProb", y="rt", order=order, palette=PALETTE, errorbar=("se", 1), ax=axes[0], alpha=0.5)
sns.lineplot(data=df_mean, x="DistractorProb", y="rt", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[0])
axes[0].set_ylabel("RT (s)")
d_rt = get_cohens_d(df_mean, 'subject_id', 'DistractorProb', 'rt', 'High', 'Low')
if d_rt is not None: axes[0].set_title(f"H-L d={d_rt:.2f}")

# 2. Accuracy
sns.barplot(data=df_mean, x="DistractorProb", y="IsCorrect", order=order, palette=PALETTE, errorbar=("se", 1), ax=axes[1], alpha=0.5)
sns.lineplot(data=df_mean, x="DistractorProb", y="IsCorrect", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[1])
axes[1].set_ylabel("Accuracy")
d_acc = get_cohens_d(df_mean, 'subject_id', 'DistractorProb', 'IsCorrect', 'High', 'Low')
if d_acc is not None: axes[1].set_title(f"H-L d={d_acc:.2f}")

# 3. Target Towardness
sns.barplot(data=df_mean, x="DistractorProb", y="target_towardness", order=order, palette=PALETTE, errorbar=("se", 1), ax=axes[2], alpha=0.5)
sns.lineplot(data=df_mean, x="DistractorProb", y="target_towardness", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[2])
axes[2].set_ylabel("Target Towardness")
d_tt = get_cohens_d(df_mean, 'subject_id', 'DistractorProb', 'target_towardness', 'High', 'Low')
if d_tt is not None: axes[2].set_title(f"H-L d={d_tt:.2f}")

plt.tight_layout()
sns.despine()
plt.show()

# 9. Analysis: Target Towardness by Target Location Probability
print("Generating plot for Target Towardness by Target Location Probability...")

# Filter out Front targets for this analysis
tt_df_target = tt_df[(tt_df["TargetLoc"] != "Front")].copy()

tt_df_target['Target_at_HP_distractor_loc'] = tt_df_target.apply(get_target_prob, axis=1)

target_towardness_means = tt_df_target.groupby(['subject_id', 'Target_at_HP_distractor_loc'])['target_towardness'].mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
order_target = ["High", "Low"]

sns.barplot(data=target_towardness_means, x="Target_at_HP_distractor_loc", y="target_towardness", order=order_target, palette=PALETTE, errorbar=("se", 1), ax=ax, alpha=0.5)
sns.lineplot(data=target_towardness_means, x="Target_at_HP_distractor_loc", y="target_towardness", units="subject_id", estimator=None, color="black", alpha=0.2, ax=ax)
ax.set_ylabel("Target Towardness")
d_tt_t = get_cohens_d(target_towardness_means, 'subject_id', 'Target_at_HP_distractor_loc', 'target_towardness', 'High', 'Low')
if d_tt_t is not None: ax.set_title(f"H-L d={d_tt_t:.2f}")

plt.tight_layout()
sns.despine()
plt.show()
