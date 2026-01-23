import pandas as pd
from stats import remove_outliers
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import glob
from utils import calculate_trajectory_projections, get_vector_length # Assuming utils is importable
sns.set_theme(context="talk", style="ticks")


OUTLIER_THRESH = 2
TRAJECTORY_BOUNDARY_DVA = 2

# Define the locations of the stimuli in degrees of visual angle (dva)
# This is crucial for calculating trajectory projections.
locations_map = {
    7: (-0.6, -0.6), 8: (0, -0.6), 9: (0.6, -0.6),
    4: (-0.6, 0), 5: (0, 0), 6: (0.6, 0),
    1: (-0.6, 0.6), 2: (0, 0.6), 3: (0.6, 0.6),
}

def p_to_asterisks(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

data_path = SPACECUE_implicit.get_data_path()
df = pd.concat([pd.read_csv(f"{data_path}pilot/distractor/{file}") for file in os.listdir(f"{data_path}pilot/distractor")])

#df = df.query('SingletonLoc != 1.0')
df = df.query("SingletonLoc != 'Front'")

# Ensure key columns are of integer type for merging with mouse data
df['Subject ID'] = df['Subject ID'].astype(int, errors="ignore")
df['Block'] = df['Block'].astype(int, errors="ignore")
df['Trial Nr'] = df['Trial Nr'].astype(int, errors="ignore")

df["IsCorrect"] = df["IsCorrect"].astype(float, errors="ignore")
df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID")

# Create snake_case columns for merging and filtering
df['subject_id'] = df['Subject ID'].astype(int, errors='ignore')
df['block'] = df['Block'].astype(int, errors='ignore')
df['trial_nr'] = df['Trial Nr'].astype(int, errors='ignore')


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

df_mean = tt_df.groupby(["subject_id", "DistractorProb"])[["rt", "IsCorrect", "target_towardness"]].mean().reset_index()

df_mean['DistractorProb'] = pd.Categorical(df_mean['DistractorProb'], categories=["distractor-absent", "low-probability", "high-probability"], ordered=True)

fig, ax = plt.subplots()

order = ["low-probability", "high-probability"]
sns.barplot(data=df_mean, x="DistractorProb", y="rt", ax=ax,
            errorbar=("se", 1), facecolor=(0, 0, 0, 0), edgecolor=".2", order=order)

sns.lineplot(data=df_mean, x="DistractorProb", y="rt",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax, palette="tab10")

# --- Pairwise comparisons ---
pairwise_tests = pg.pairwise_tests(data=df_mean, dv='rt', within='DistractorProb', subject='subject_id', effsize='cohen')

y_max = df_mean['rt'].max()
y_pos = y_max + 0.05
h = 0.02

# Comparison: low-probability vs high-probability
stat2 = pairwise_tests[(pairwise_tests.A == 'low-probability') & (pairwise_tests.B == 'high-probability')].iloc[0]
x1, x2 = 0, 1
y, col = y_pos, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, f"d={stat2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax.set_ylim(top=y_pos + 0.2)

ax.set_ylabel("Reaction Time (s)") # Corrected label
ax.set_xlabel("Distractor Probability")
plt.show()

fig, ax = plt.subplots()

order = ["low-probability", "high-probability"]
sns.barplot(data=df_mean, x="DistractorProb", y="IsCorrect", ax=ax,
            errorbar=("se", 1), facecolor=(0, 0, 0, 0), edgecolor=".2", order=order)

sns.lineplot(data=df_mean, x="DistractorProb", y="IsCorrect",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax, palette="tab10", legend=False)

# --- Pairwise comparisons ---
pairwise_tests = pg.pairwise_tests(data=df_mean, dv='IsCorrect', within='DistractorProb', subject='subject_id', effsize='cohen')

y_max = df_mean['IsCorrect'].max()
y_pos = y_max + 0.05
h = 0.02

# Comparison: low-probability vs high-probability
stat2 = pairwise_tests[(pairwise_tests.A == 'low-probability') & (pairwise_tests.B == 'high-probability')].iloc[0]
x1, x2 = 0, 1
y, col = y_pos, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, f"d={stat2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax.set_ylim(top=y_pos + 0.2)

ax.set_ylabel("Accuracy (%)") # Corrected label
ax.set_xlabel("Distractor Probability")
plt.show()

# --- Performance when target is at the high-probability distractor location ---

# NOTE: This assumes a column 'high_prob_loc' exists in your dataframe,
# which we will create now.

# 1. Find the single high-probability location for each subject.
# We can get this by looking at the 'SingletonLoc' on any 'high-probability' trial for that subject.
high_prob_loc_map = tt_df[tt_df['DistractorProb'] == 'high-probability'].drop_duplicates('subject_id').set_index('subject_id')['SingletonLoc']

# 2. Map this location back to all trials for each subject.
tt_df['high_prob_loc'] = tt_df['subject_id'].map(high_prob_loc_map)

# 3. Now, correctly identify trials where the target appeared at that high-probability location.
tt_df["target_at_high_prob_loc"] = tt_df["TargetLoc"] == tt_df["high_prob_loc"]
# Calculate the mean performance for each subject under each condition
df_targetloc_mean = tt_df.groupby(["subject_id", "target_at_high_prob_loc"])["IsCorrect"].mean().reset_index()

# Print the overall mean accuracy for each condition
print("Mean accuracy based on target location relative to high-probability distractor location:")
print(df_targetloc_mean.groupby("target_at_high_prob_loc")["IsCorrect"].mean())

# Create a new figure to visualize the results
fig2, ax2 = plt.subplots(figsize=(6, 5))

order2 = [False, True]
sns.barplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="IsCorrect",
            ax=ax2, errorbar=("ci", 95), capsize=.1, palette="viridis", order=order2)

sns.lineplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="IsCorrect",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax2, palette="tab10")

# --- Pairwise comparisons ---
pairwise_tests2 = pg.pairwise_tests(data=df_targetloc_mean, dv='IsCorrect', within='target_at_high_prob_loc', subject='subject_id', effsize='cohen')
stat_2 = pairwise_tests2.iloc[0]
x1, x2 = 0, 1
y_max2 = df_targetloc_mean['IsCorrect'].max()
y, h, col = y_max2 + 0.01, 0.01, 'k'
ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax2.text((x1+x2)*.5, y+h, f"d={stat_2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax2.set_ylim(top=y_max2 + 0.05)

ax2.set_xlabel("Target at High-Probability Distractor Location")
ax2.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# --- Performance when target is at the high-probability distractor location ---

# NOTE: This assumes a column 'high_prob_loc' exists in your dataframe,
# which we will create now.

# 1. Find the single high-probability location for each subject.
# We can get this by looking at the 'SingletonLoc' on any 'high-probability' trial for that subject.
high_prob_loc_map = tt_df[tt_df['DistractorProb'] == 'high-probability'].drop_duplicates('subject_id').set_index('subject_id')['SingletonLoc']

# 2. Map this location back to all trials for each subject.
tt_df['high_prob_loc'] = tt_df['subject_id'].map(high_prob_loc_map)

# 3. Now, correctly identify trials where the target appeared at that high-probability location.
tt_df["distractor_at_high_prob_loc"] = tt_df["SingletonLoc"] == tt_df["high_prob_loc"]
# Calculate the mean performance for each subject under each condition
df_distractorloc_mean = tt_df.groupby(["subject_id", "distractor_at_high_prob_loc"])["IsCorrect"].mean().reset_index()

# Print the overall mean accuracy for each condition
print("Mean accuracy based on distractor location relative to high-probability distractor location:")
print(df_distractorloc_mean.groupby("distractor_at_high_prob_loc")["IsCorrect"].mean())

# Create a new figure to visualize the results
fig2, ax2 = plt.subplots(figsize=(6, 5))

order2 = [False, True]
sns.barplot(data=df_distractorloc_mean, x="distractor_at_high_prob_loc", y="IsCorrect",
            ax=ax2, errorbar=("ci", 95), capsize=.1, palette="viridis", order=order2)

sns.lineplot(data=df_distractorloc_mean, x="distractor_at_high_prob_loc", y="IsCorrect",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax2, palette="tab10")

# --- Pairwise comparisons ---
pairwise_tests2 = pg.pairwise_tests(data=df_distractorloc_mean, dv='IsCorrect', within='distractor_at_high_prob_loc', subject='subject_id', effsize='cohen')
stat_2 = pairwise_tests2.iloc[0]
x1, x2 = 0, 1
y_max2 = df_distractorloc_mean['IsCorrect'].max()
y, h, col = y_max2 + 0.01, 0.01, 'k'
ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax2.text((x1+x2)*.5, y+h, f"d={stat_2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax2.set_ylim(top=y_max2 + 0.05)

ax2.set_xlabel("distractor at High-Probability Distractor Location")
ax2.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# ===================================================================
#       TARGET TOWARDNESS ANALYSIS
# ===================================================================
print("\n--- Analyzing Target Towardness by Distractor Probability ---")

# Aggregate the new towardness score by subject and condition
df_towardness_mean = tt_df.groupby(["subject_id", "DistractorProb"])["target_towardness"].mean().reset_index()
df_towardness_mean['DistractorProb'] = pd.Categorical(df_towardness_mean['DistractorProb'], categories=["low-probability", "high-probability"], ordered=True)

# --- Create the plot for Target Towardness ---
fig_toward, ax_toward = plt.subplots()

sns.barplot(data=df_towardness_mean, x="DistractorProb", y="target_towardness", ax=ax_toward,
            errorbar=("se", 1), facecolor=(0, 0, 0, 0), edgecolor=".2", order=order)

sns.lineplot(data=df_towardness_mean, x="DistractorProb", y="target_towardness",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax_toward, palette="tab10", legend=False)

# --- Pairwise comparisons for Target Towardness ---
pairwise_tests_toward = pg.pairwise_tests(data=df_towardness_mean, dv='target_towardness', within='DistractorProb', subject='subject_id', effsize='cohen')

y_max_toward = df_towardness_mean['target_towardness'].max()
y_pos_toward = y_max_toward + 0.05
h_toward = 0.02 # Use a distinct variable name

# Comparison: low-probability vs high-probability
stat_toward = pairwise_tests_toward[(pairwise_tests_toward.A == 'low-probability') & (pairwise_tests_toward.B == 'high-probability')].iloc[0]
x1_toward, x2_toward = 0, 1
y_toward, col_toward = y_pos_toward, 'k'
ax_toward.plot([x1_toward, x1_toward, x2_toward, x2_toward], [y_toward, y_toward+h_toward, y_toward+h_toward, y_toward], lw=1.5, c=col_toward)
ax_toward.text((x1_toward+x2_toward)*.5, y_toward+h_toward, f"d={stat_toward['cohen']:.2f}", ha='center', va='bottom', color=col_toward)

ax_toward.set_ylim(top=y_pos_toward + 0.2)
ax_toward.set_title("Target Towardness by Distractor Probability")
ax_toward.set_ylabel("Target Towardness (dva)")
ax_toward.set_xlabel("Distractor Probability")
plt.show()
