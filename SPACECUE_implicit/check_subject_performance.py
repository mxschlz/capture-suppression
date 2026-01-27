import pandas as pd
from stats import remove_outliers
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

sns.set_theme(context="talk", style="ticks")

OUTLIER_THRESH = 2
PALETTE = {"High": "green", "Low": "red", "Absent": "gray"}

# --- Data Loading Logic (from implicit_learning_effect.py) ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
experiment_folder = "pilot/distractor"

# Load all CSV files in the directory
df = pd.concat([pd.read_csv(f"{data_path}{experiment_folder}/{file}") for file in os.listdir(f"{data_path}{experiment_folder}")])

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

# Remove Front location
df = df[df[loc_col] != "Front"]

# Remove outliers
df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID")

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
g_rt = sns.FacetGrid(df, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_rt.map_dataframe(plot_perf, x=loc_col, y="rt")
g_rt.set_titles("Sub {col_name}")
g_rt.set_axis_labels(loc_col, "RT (s)")
plt.show()

# 2. Accuracy Visualization
g_acc = sns.FacetGrid(df, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
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

g_rt_prob = sns.FacetGrid(df, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_rt_prob.map_dataframe(plot_prob_perf, x="DistractorProb", y="rt")
g_rt_prob.set_titles("Sub {col_name}")
g_rt_prob.set_axis_labels("Distractor Prob", "RT (s)")
plt.show()

g_acc_prob = sns.FacetGrid(df, col="subject_id", col_wrap=5, sharey=False, height=3.5, aspect=1)
g_acc_prob.map_dataframe(plot_prob_perf, x="DistractorProb", y="IsCorrect")
g_acc_prob.set_titles("Sub {col_name}")
g_acc_prob.set_axis_labels("Distractor Prob", "Accuracy (%)")
plt.show()

# 4. Print aggregated data for inspection
agg_df = df.groupby(['subject_id', loc_col])[['rt', 'IsCorrect']].mean()
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
subject_means = df.groupby(['subject_id', 'DistractorProb'])[['rt', 'IsCorrect']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
order = ["High", "Low", "Absent"]

sns.barplot(data=subject_means, x="DistractorProb", y="rt", order=order, palette=PALETTE, errorbar=("se", 1), ax=axes[0], alpha=0.5)
sns.lineplot(data=subject_means, x="DistractorProb", y="rt", units="subject_id", estimator=None, color="black", alpha=0.2, ax=axes[0])
axes[0].set_ylabel("RT (s)")
d_rt_hl = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'rt', 'High', 'Low')
d_rt_la = get_cohens_d(subject_means, 'subject_id', 'DistractorProb', 'rt', 'Low', 'Absent')
t_rt = []
if d_rt_hl is not None: t_rt.append(f"H-L cohens d={d_rt_hl:.2f}")
if d_rt_la is not None: t_rt.append(f"L-A cohens d={d_rt_la:.2f}")
if t_rt: axes[0].set_title(", ".join(t_rt))

sns.barplot(data=subject_means, x="DistractorProb", y="IsCorrect", order=order, palette=PALETTE, errorbar=("se", 1), ax=axes[1], alpha=0.5)
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
block_means = df.groupby(['subject_id', 'block', 'DistractorProb'])[['rt', 'IsCorrect']].mean().reset_index()

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

# 6. Analysis: Target Location Probability
print("Generating plots for Target Location Probability...")

def get_target_prob(row):
    # Even subjects: Left is High-Prob Singleton (Suppressed)
    # Odd subjects: Right is High-Prob Singleton (Suppressed)
    is_even = row['subject_id'] % 2 == 0
    high_prob_loc = 'Left' if is_even else 'Right'
    
    if row['TargetLoc'] == high_prob_loc:
        return 'High' # Target at Suppressed Location
    return 'Low' # Target at Non-Suppressed Location

df['Target_at_HP_distractor_loc'] = df.apply(get_target_prob, axis=1)

# Aggregate plots for Target Prob
subject_means_target = df.groupby(['subject_id', 'Target_at_HP_distractor_loc'])[['rt', 'IsCorrect']].mean().reset_index()

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
