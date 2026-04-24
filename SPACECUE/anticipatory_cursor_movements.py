import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SPACECUE

sns.set_theme(context="talk", style="ticks")
plt.ion()

SUBJECT_IDS = [1, 2, 3, 4]

# --- 1. Data Loading Logic ---
print("Loading behavioral data...")
data_path = SPACECUE.get_data_path()
experiment_folder = "/derivatives/preprocessing"

# Load behavioral data
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

df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

if df.empty:
    raise ValueError("No behavioral data found. Check your SUBJECT_IDS and paths.")

# Clean and format necessary columns
df['subject_id'] = df['Subject ID'].astype(int, errors="ignore")
df['block'] = df['Block'].astype(int, errors="ignore")
df['trial_nr'] = df['Trial Nr'].astype(int, errors="ignore")
df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})
df['HP_Distractor_Prob'] = df.get('HP_Distractor_Prob', 1.0).astype(float, errors="ignore")

# We only care about blocks where the HP Distractor was specifically Left or Right
df_hp = df[df['HP_Distractor_Loc'].isin(['Left', 'Right'])].copy()

print("Loading raw mouse trajectory data...")
raw_data_base_path = os.path.join(data_path, 'sourcedata', 'raw')
subject_folders = glob.glob(f"{raw_data_base_path}/sci-*")

df_mouse_list = []
for subject_folder in subject_folders:
    mouse_file = glob.glob(f"{subject_folder}/beh/*mouse_data.csv")
    if mouse_file:
        sub_id_str = os.path.basename(subject_folder)
        try:
            sub_id = int(sub_id_str.split('-')[1])
        except (IndexError, ValueError):
            continue

        if sub_id not in SUBJECT_IDS:
            continue

        temp_df = pd.read_csv(mouse_file[0])
        temp_df['subject_id'] = sub_id

        # Calculate block and trial to ensure uniqueness
        temp_df['block'] = ((temp_df['trial_nr'] == 0) & (temp_df['trial_nr'].shift(1).fillna(-1) != 0)).cumsum() - 1
        
        # Detect phases based on time resets (negative diff implies reset)
        # 0: Stimulus, 1: Response, 2: ITI
        temp_df['phase'] = temp_df.groupby(['block', 'trial_nr'])['time'].transform(lambda x: (x.diff() < -0.5).cumsum())
        
        df_mouse_list.append(temp_df)

df_mouse = pd.concat(df_mouse_list, ignore_index=True) if df_mouse_list else pd.DataFrame()

if df_mouse.empty:
    raise ValueError("No mouse data found.")

# --- 2. Data Merging and Time Normalization ---
print("Merging data and normalizing time phases...")

# Merge HP condition info into mouse data
plot_data = pd.merge(
    df_mouse,
    df_hp[['subject_id', 'block', 'trial_nr', 'HP_Distractor_Loc', 'HP_Distractor_Prob']],
    on=['subject_id', 'block', 'trial_nr'],
    how='inner'
)

# Normalize time within each trial's phase to 0.0 - 1.0 so we can average across trials properly
# We will also bin it to smooth the time-series plotting
def normalize_time(group):
    t_min = group['time'].min()
    t_max = group['time'].max()
    if t_max > t_min:
        group['t_norm'] = (group['time'] - t_min) / (t_max - t_min)
    else:
        group['t_norm'] = 0.0
    return group

plot_data = plot_data.groupby(['subject_id', 'block', 'trial_nr', 'phase'], group_keys=False).apply(normalize_time)

# Create ~20 time bins for smooth plotting (0.0, 0.05, 0.10 ... 1.0)
plot_data['t_bin'] = (plot_data['t_norm'] * 20).round() / 20

# Calculate instantaneous horizontal velocity
def calculate_kinematics(group):
    group = group.sort_values('time')
    group['dt'] = group['time'].diff()
    group['dx'] = group['x'].diff()
    # Safely divide, avoiding division by zero if two points share a timestamp
    dt_safe = group['dt'].replace(0, np.nan)
    group['vx'] = (group['dx'] / dt_safe).fillna(0)
    return group

print("Calculating movement metrics (velocity)...")
plot_data = plot_data.groupby(['subject_id', 'block', 'trial_nr', 'phase'], group_keys=False).apply(calculate_kinematics)

# --- 3. Plotting: Time-course of horizontal cursor position ---
print("Generating phase plots...")

phases = [0, 1, 2]
phase_labels = {0: 'Stimulus Presentation', 1: 'Response Interval', 2: 'Inter-Trial Interval (ITI)'}

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for phase_val in phases:
    ax = axes[phase_val]
    phase_data = plot_data[plot_data['phase'] == phase_val]
    
    if phase_data.empty:
        ax.set_title(f"{phase_labels[phase_val]} (No Data)")
        continue

    sns.lineplot(
        data=phase_data, 
        x='t_bin',
        y='vx', 
        hue='HP_Distractor_Loc',
        palette={'Left': 'blue', 'Right': 'red'},
        ax=ax,
        errorbar='se'
    )
    
    ax.set_title(phase_labels[phase_val])
    ax.set_xlabel('Normalized Phase Time (0 = Start, 1 = End)')
    if phase_val == 0:
        ax.set_ylabel('Horizontal Velocity (units/s)')
    
    # Add horizontal reference line at center (x=0)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Cleanup legends
    if phase_val == 2:
        ax.legend(title='HP Distractor', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()

plt.suptitle('Cursor Horizontal Velocity Over Trial Phases by HP Distractor Location')
plt.tight_layout()
plt.show()

# --- 4. Plotting: Anticipatory bias right before stimulus onset (End of ITI) ---
print("Generating ITI anticipatory bias plot...")

# Focus exclusively on the final 20% of the ITI (t_norm >= 0.8)
iti_end_data = plot_data[(plot_data['phase'] == 2) & (plot_data['t_norm'] >= 0.8)].copy()

if not iti_end_data.empty:
    # Calculate average terminal X velocity per trial
    iti_bias = iti_end_data.groupby(
        ['subject_id', 'block', 'trial_nr', 'HP_Distractor_Loc']
    )['vx'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.barplot(
        data=iti_bias,
        x='HP_Distractor_Loc',
        y='vx',
        hue='HP_Distractor_Loc',
        palette={'Left': 'blue', 'Right': 'red'},
        ax=ax,
        order=['Left', 'Right']
    )
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax.set_title('Anticipatory Cursor Velocity (End of ITI)')
    ax.set_xlabel('High Probability Distractor Location')
    ax.set_ylabel('Average Final ITI Horizontal Velocity')
    
    plt.tight_layout()
    plt.show()
else:
    print("No sufficient ITI phase data found to calculate terminal anticipatory bias.")
    
print("Done.")