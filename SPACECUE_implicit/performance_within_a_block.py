import pandas as pd
import numpy as np
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
from stats import remove_outliers
from scipy.stats import linregress, ttest_1samp

sns.set_theme(context="talk", style="ticks")

# --- Configuration ---
OUTLIER_THRESH = 2
experiment_folder = "pilot/distractor-switch"

# --- Data Loading ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
full_path = os.path.join(data_path, experiment_folder)

files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(os.path.join(full_path, f)) for f in files], ignore_index=True)
# --- Data Cleaning & Filtering ---
# 1. Drop rows with missing Subject ID, ensure it's an integer, and then filter.
df.dropna(subset=['Subject ID'], inplace=True)
df['Subject ID'] = df['Subject ID'].astype(int)
df = df[~df['Subject ID'].isin([900, 901])]

# Calculate TrialInBlock BEFORE filtering to preserve temporal structure (0-359)
if "Block" in df.columns and "Subject ID" in df.columns:
    df['TrialInBlock'] = df.groupby(['Subject ID', "Block"]).cumcount()

df = df[df["TargetLoc"] != "Front"]


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

df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

# Determine location column
loc_col = "SingletonLoc"

# Define probability condition based on Subject ID and Location
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # The HP_distractor_loc switches dynamically
    return 'High' if row[loc_col] == row['HP_Distractor_Loc'] else 'Low'

df['Probability'] = df.apply(get_probability, axis=1)
df['DistractorProb'] = df['Probability']

# --- Block Analysis Prep ---
block_col = "Block"

# Ensure TrialInBlock exists (it should have been calculated before filtering)
if block_col and 'TrialInBlock' not in df.columns:
    df['TrialInBlock'] = df.groupby(['subject_id', block_col]).cumcount()

# --- Data Splitting ---
df_acc = df.copy()

# Remove outliers (to match analysis conditions) - Only for RT analysis
if "rt" in df.columns:
    pass
    #df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID").reset_index(drop=True)

# --- Analysis: Performance within a block ---
if block_col:
    print(f"Analyzing performance within blocks (using '{block_col}')...")

    # --- Subject-Level Analysis ---
    # This is a more robust approach than a single regression on pooled data.
    # We calculate the trend for each subject, then test if the average trend is significant.
    rt_slopes, rt_intercepts = [], []
    acc_slopes, acc_intercepts = [], []
    subjects = df['Subject ID'].unique()

    # Ensure IsCorrect is numeric before looping
    df_acc['IsCorrect'] = pd.to_numeric(df_acc['IsCorrect'], errors='coerce')

    for subject in subjects:
        subject_df = df[df['Subject ID'] == subject]
        subject_df_acc = df_acc[df_acc['Subject ID'] == subject]

        # RT analysis for this subject
        rt_clean_subj = subject_df[['TrialInBlock', 'rt']].dropna()
        if len(rt_clean_subj) > 2:  # Need at least 2 points for a line
            slope, intercept, _, _, _ = linregress(rt_clean_subj['TrialInBlock'], rt_clean_subj['rt'])
            rt_slopes.append(slope)
            rt_intercepts.append(intercept)

        # Accuracy analysis for this subject
        acc_clean_subj = subject_df_acc[['TrialInBlock', 'IsCorrect']].dropna()
        if len(acc_clean_subj) > 2:
            slope, intercept, _, _, _ = linregress(acc_clean_subj['TrialInBlock'], acc_clean_subj['IsCorrect'])
            acc_slopes.append(slope)
            acc_intercepts.append(intercept)

    # --- Statistical tests on the distribution of slopes ---
    # RT
    if rt_slopes:
        mean_rt_slope = np.mean(rt_slopes)
        mean_rt_intercept = np.mean(rt_intercepts)
        t_stat_rt, p_val_rt = ttest_1samp(rt_slopes, 0)
        print(f"RT Trend (Subject-Level): Mean Slope = {mean_rt_slope * 1000:.2f} ms/trial (t={t_stat_rt:.2f}, p={p_val_rt:.3f})")
    else:
        mean_rt_slope, mean_rt_intercept = 0, 0
        print("RT Trend (Subject-Level): Not enough data to calculate.")

    # Accuracy
    if acc_slopes:
        mean_acc_slope = np.mean(acc_slopes)
        mean_acc_intercept = np.mean(acc_intercepts)
        t_stat_acc, p_val_acc = ttest_1samp(acc_slopes, 0)
        print(f"Acc Trend (Subject-Level): Mean Slope = {mean_acc_slope:.5f} /trial (t={t_stat_acc:.2f}, p={p_val_acc:.3f})")
    else:
        mean_acc_slope, mean_acc_intercept = 0, 0
        print("Acc Trend (Subject-Level): Not enough data to calculate.")

    # --- Aggregated data for plotting (to visualize the overall trend) ---
    rt_stats = df.groupby('TrialInBlock')['rt'].agg(['mean', 'sem']).reset_index()
    acc_stats = df_acc.groupby('TrialInBlock')['IsCorrect'].agg(['mean', 'sem']).reset_index()

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # RT
    color = 'tab:blue'
    ax1.set_xlabel('Trial in Block')
    ax1.set_ylabel('RT (s)', color=color)
    ax1.plot(rt_stats['TrialInBlock'], rt_stats['mean'], color=color, label='RT')
    ax1.fill_between(rt_stats['TrialInBlock'], rt_stats['mean'] - rt_stats['sem'], rt_stats['mean'] + rt_stats['sem'], color=color, alpha=0.2)
    ax1.plot(rt_stats['TrialInBlock'], mean_rt_intercept + mean_rt_slope * rt_stats['TrialInBlock'], color=color, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    # Acc
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_stats['TrialInBlock'], acc_stats['mean'], color=color, label='Accuracy')
    ax2.fill_between(acc_stats['TrialInBlock'], acc_stats['mean'] - acc_stats['sem'], acc_stats['mean'] + acc_stats['sem'], color=color, alpha=0.2)
    ax2.plot(acc_stats['TrialInBlock'], mean_acc_intercept + mean_acc_slope * acc_stats['TrialInBlock'], color=color, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Within-Block Performance (Tiredness Check)")
    plt.tight_layout()
    plt.show()
else:
    print("Block column not found. Cannot perform within-block analysis.")
