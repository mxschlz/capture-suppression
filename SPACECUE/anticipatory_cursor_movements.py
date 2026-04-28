import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SPACECUE

# --- 0. Configuration ---
sns.set_theme(context="talk", style="ticks")
plt.ion()

SUBJECT_IDS = [1]

# --- 1. Data Loading Logic ---
print("Loading behavioral data...")
data_path = SPACECUE.get_data_path()
experiment_folder = "/derivatives/preprocessing"

beh_data_base_path = f"{data_path}{experiment_folder}"
subject_folders = glob.glob(f"{beh_data_base_path}/sci-*")

df_list = []
for subject_folder in subject_folders:
    sub_id_str = os.path.basename(subject_folder)
    try:
        sub_id = int(sub_id_str.split('-')[1])
    except (IndexError, ValueError):
        continue

    if sub_id in SUBJECT_IDS:
        beh_files = glob.glob(f"{subject_folder}/beh/*.csv")
        for file in beh_files:
            df_list.append(pd.read_csv(file))

df_beh = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

if df_beh.empty:
    raise ValueError("No behavioral data found. Check your SUBJECT_IDS and paths.")

# Clean and format behavioral columns
df_beh['subject_id'] = df_beh['Subject ID'].astype(int, errors="ignore")
df_beh['block'] = df_beh['Block'].astype(int, errors="ignore")
df_beh['trial_nr'] = df_beh['Trial Nr'].astype(int, errors="ignore")
df_beh['HP_Distractor_Loc'] = df_beh['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

# Filter for the conditions of interest
df_hp = df_beh[df_beh['HP_Distractor_Loc'].isin(['Left', 'Right'])].copy()

print("Loading and cleaning raw mouse trajectory data...")
raw_data_base_path = os.path.join(data_path, 'sourcedata', 'raw')
df_mouse_list = []

for sub_id in SUBJECT_IDS:
    sub_str = f"sci-{sub_id}" if sub_id < 10 else f"sci-{sub_id}"
    subject_folder = os.path.join(raw_data_base_path, sub_str)
    mouse_files = glob.glob(f"{subject_folder}/beh/*mouse_data.csv")

    for file in mouse_files:
        temp_df = pd.read_csv(file)
        temp_df['subject_id'] = sub_id

        # 1. Detect blocks (increments whenever trial_nr resets to 0)
        temp_df['block'] = ((temp_df['trial_nr'] == 0) & (temp_df['trial_nr'].shift(1).fillna(-1) != 0)).cumsum() - 1

        # 2. Detect Phases FIRST, as they are the basis for continuous segments
        group_cols = ['subject_id', 'block', 'trial_nr']
        temp_df['phase'] = temp_df.groupby(group_cols)['time'].transform(lambda x: (x.diff() < -0.1).cumsum())

        # 3. Calculate Kinematics PER PHASE
        # This is more robust as it calculates velocity only on continuous trajectory segments.
        phase_group_cols = ['subject_id', 'block', 'trial_nr', 'phase']
        temp_df['dt'] = temp_df.groupby(phase_group_cols)['time'].diff()
        temp_df['dx'] = temp_df.groupby(phase_group_cols)['x'].diff()

        # Safely calculate velocity, avoiding division by zero, and fill NaNs for first samples
        dt_safe = temp_df['dt'].replace(0, np.nan)
        temp_df['vx'] = (temp_df['dx'] / dt_safe).fillna(0)

        # Apply a median filter to remove 1-frame cursor reset/teleportation artifacts
        temp_df['vx'] = temp_df.groupby(phase_group_cols)['vx'].transform(
            lambda x: x.rolling(window=3, center=True, min_periods=1).median()
        )

        # Clean up intermediate columns
        temp_df = temp_df.drop(columns=['dt', 'dx'])

        df_mouse_list.append(temp_df)

df_mouse = pd.concat(df_mouse_list, ignore_index=True)

# --- 2. Data Merging and Time Normalization ---
print("Merging data and normalizing time...")

plot_data = pd.merge(
    df_mouse,
    df_hp[['subject_id', 'block', 'trial_nr', 'HP_Distractor_Loc']],
    on=['subject_id', 'block', 'trial_nr'],
    how='inner'
)

print("Mapping phases dynamically and binning time...")
def map_phase_by_time(group):
    t_min = group['time'].min()
    # Based on your timers: Stimulus ~ -0.25, Response ~ -1.75
    if t_min >= -0.5:
        group['phase_name'] = 'Stimulus'
        # Remove the first 5 samples to clear any lingering teleportation/reset artifacts
        group = group.iloc[2:]
    elif t_min <= -1.5:
        group['phase_name'] = 'Response'
    else:
        group['phase_name'] = 'ITI'
    return group

plot_data = plot_data.groupby(['subject_id', 'block', 'trial_nr', 'phase'], group_keys=False).apply(map_phase_by_time)

# Bin real time for smooth visualization (bins of 50ms)
plot_data['time_bin'] = (plot_data['time'] * 20).round() / 20

# --- 3. Plotting Phase Time-course ---
print("Generating phase plots...")
phases = ['Stimulus', 'Response', 'ITI']
phase_labels = {'Stimulus': 'Stimulus (250ms)', 'Response': 'Response Interval', 'ITI': 'ITI'}

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for i, phase_name in enumerate(phases):
    ax = axes[i]
    phase_subset = plot_data[plot_data['phase_name'] == phase_name]

    if not phase_subset.empty:
        sns.lineplot(
            data=phase_subset,
            x='time_bin', y='vx',
            hue='HP_Distractor_Loc',
            palette={'Left': 'blue', 'Right': 'red'},
            ax=ax, errorbar='se'
        )

    ax.set_title(phase_labels[phase_name])
    ax.set_xlabel('Time to Phase End (s)')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    if phase_name != 'ITI':
        legend = ax.get_legend()
        if legend: legend.remove()
    else:
        ax.legend(title='HP Distractor', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Corrected Horizontal Velocity (Global Kinematics)')
plt.tight_layout()
plt.show()

# --- 4. Plotting Anticipatory Bias ---
print("Generating ITI bias plot...")

# Extract the final 20% of the actual time duration for the ITI phase
iti_data = plot_data[plot_data['phase_name'] == 'ITI']
iti_end_data = iti_data.groupby(['subject_id', 'block', 'trial_nr'], group_keys=False).apply(
    lambda x: x[x['time'] >= x['time'].min() + 0.9 * (x['time'].max() - x['time'].min())]
)

if not iti_end_data.empty:
    iti_bias = iti_end_data.groupby(['subject_id', 'block', 'trial_nr', 'HP_Distractor_Loc'])['vx'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=iti_bias, x='HP_Distractor_Loc', y='vx', hue='HP_Distractor_Loc',
                palette={'Left': 'blue', 'Right': 'red'}, ax=ax, order=['Left', 'Right'])
    ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax.set_title('Anticipatory Bias (End of ITI)')
    plt.tight_layout()
    plt.show()

print("Done.")