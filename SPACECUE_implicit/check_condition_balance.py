import pandas as pd
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
from stats import remove_outliers

sns.set_theme(context="talk", style="ticks")

# --- Configuration ---
OUTLIER_THRESH = 2
experiment_folder = "pilot/distractor"

# --- Data Loading ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
full_path = os.path.join(data_path, experiment_folder)

files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(os.path.join(full_path, f)) for f in files], ignore_index=True)

# --- Preprocessing ---
# Ensure types
if 'Subject ID' in df.columns:
    df['subject_id'] = df['Subject ID'].astype(int, errors="ignore")
elif 'subject_id' in df.columns:
    df['subject_id'] = df['subject_id'].astype(int, errors="ignore")

# Map Locations
if 'SingletonLoc' in df.columns:
    if pd.api.types.is_numeric_dtype(df['SingletonLoc']):
        df['SingletonLoc'] = df['SingletonLoc'].map({0: 'Absent', 1: 'Left', 2: 'Front', 3: 'Right'})
if 'target_loc' in df.columns:
    df['TargetLoc'] = df['target_loc']
if pd.api.types.is_numeric_dtype(df['TargetLoc']):
    df['TargetLoc'] = df['TargetLoc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

# Determine location column
loc_col = "Non-Singleton2Loc" if "control" in experiment_folder else "SingletonLoc"

# Define Probability Logic
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # Even subjects: Left is High, others are Low
    # Odd subjects: Right is High, others are Low
    is_even = row['subject_id'] % 2 == 0
    high_loc = 'Left' if is_even else 'Right'
    return 'High' if row[loc_col] == high_loc else 'Low'

df['DistractorProb'] = df.apply(get_probability, axis=1)

# Remove outliers (to match analysis conditions)
if "rt" in df.columns:
    df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="subject_id")

# --- Analysis: Systematic Confound Check (Spatial x Condition) ---
print("\nPlotting Systematic Spatial Confounds...")
# 1. Determine High Probability Location per Subject
df['HighProbLoc'] = df['subject_id'].apply(lambda x: 'Left' if x % 2 == 0 else 'Right')

# 2. Determine Control Location
def get_control_loc(row):
    all_locs = {'Left', 'Front', 'Right'}
    occupied = {row['TargetLoc'], row[loc_col]}
    remaining = list(all_locs - occupied)
    return remaining[0] if remaining else None

df['ControlLoc'] = df.apply(get_control_loc, axis=1)

# 3. Restructure Data (Melt) for Plotting
# Target Data
df_t = df[['subject_id', 'rt', 'IsCorrect', 'TargetLoc', 'HighProbLoc']].rename(columns={'TargetLoc': 'Location'})
df_t['SoundType'] = 'Target'
# Distractor Data
df_d = df[['subject_id', 'rt', 'IsCorrect', loc_col, 'HighProbLoc']].rename(columns={loc_col: 'Location'})
df_d['SoundType'] = 'Distractor'
# Control Data
df_c = df[['subject_id', 'rt', 'IsCorrect', 'ControlLoc', 'HighProbLoc']].rename(columns={'ControlLoc': 'Location'})
df_c['SoundType'] = 'Control'

df_plot = pd.concat([df_t, df_d, df_c], ignore_index=True)
df_plot = df_plot[df_plot['Location'].isin(['Left', 'Front', 'Right'])]

# 4. Define Conditions
def get_plot_condition(row):
    loc = row['Location']
    stype = row['SoundType']
    high_loc = row['HighProbLoc']

    if stype == 'Control':
        return "Control"
    elif stype == 'Distractor':
        return "Distractor Expected" if loc == high_loc else "Distractor Unexpected"
    elif stype == 'Target':
        return "Target Unexpected" if loc == high_loc else "Target Expected"
    return "Other"

df_plot['Condition'] = df_plot.apply(get_plot_condition, axis=1)

# --- Aggregate per subject ---
print("Aggregating data per subject...")
df_agg = df_plot.groupby(['subject_id', 'Location', 'Condition'])[['rt', 'IsCorrect']].mean().reset_index()

# 5. Plot
plt.figure(figsize=(12, 7))
order_cond = [
    "Distractor Expected", "Distractor Unexpected",
    "Target Expected", "Target Unexpected", "Control"
]
# Filter order to existing conditions
order_cond = [c for c in order_cond if c in df_plot['Condition'].unique()]

sns.barplot(
    data=df_agg, x="Location", y="rt", hue="Condition",
    order=["Left", "Front", "Right"], hue_order=order_cond,
    palette="tab10", errorbar=("se", 1)
)
plt.title("Performance by Spatial Location and Condition Context")
plt.ylabel("Reaction Time (s)")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Condition")
sns.despine()
plt.tight_layout()
plt.show()

if 'IsCorrect' in df_agg.columns:
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_agg, x="Location", y="IsCorrect", hue="Condition",
        order=["Left", "Front", "Right"], hue_order=order_cond,
        palette="tab10", errorbar=("se", 1)
    )
    plt.title("Accuracy by Spatial Location and Condition Context")
    plt.ylabel("Proportion Correct")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Condition")
    sns.despine()
    plt.tight_layout()
    plt.show()
