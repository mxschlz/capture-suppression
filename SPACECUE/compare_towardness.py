import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import SPACECUE
from utils import calculate_trajectory_projections

sns.set_theme(context="talk", style="ticks")
plt.ion()

SUBJECT_IDS = [2]

# --- Data Loading Logic ---
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

df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
df = df[df["Subject ID"].isin(SUBJECT_IDS)]

df['subject_id'] = df['Subject ID'].astype(int, errors='ignore')
df['block'] = df['Block'].astype(int, errors='ignore')
df['trial_nr'] = df['Trial Nr'].astype(int, errors='ignore')

# --- Load and process raw trajectory data ---
print("Loading raw trajectory data...")
raw_data_base_path = os.path.join(SPACECUE.get_data_path(), 'sourcedata', 'raw')
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
        temp_df['block'] = ((temp_df['trial_nr'] == 0) & (temp_df['trial_nr'].shift(1).fillna(-1) != 0)).cumsum() - 1
        df_mouse_list.append(temp_df)

df_mouse = pd.concat(df_mouse_list, ignore_index=True)

# --- Align trajectories ---
def align_trial_coordinates(trial_group):
    try:
        target_pos = np.array([trial_group['Target_pos_x'].iloc[0], trial_group['Target_pos_y'].iloc[0]])
    except IndexError:
        return None
    traj_points = trial_group[['x', 'y']].values
    target_angle = np.arctan2(target_pos[1], target_pos[0])
    rotation_angle = np.pi / 2 - target_angle
    cos_rot = np.cos(rotation_angle)
    sin_rot = np.sin(rotation_angle)
    rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
    
    traj_rotated = np.dot(traj_points, rot_matrix.T)
    target_rotated = np.dot(rot_matrix, target_pos)
    
    trial_group['x_aligned'] = traj_rotated[:, 0]
    trial_group['y_aligned'] = traj_rotated[:, 1]
    trial_group['target_x_aligned'] = target_rotated[0]
    trial_group['target_y_aligned'] = target_rotated[1]
    return trial_group

print("Aligning trajectories...")
location_cols = ['subject_id', 'block', 'trial_nr', 'Target_pos_x', 'Target_pos_y']
df_mouse_with_locs = pd.merge(df_mouse, df[location_cols], on=['subject_id', 'block', 'trial_nr'], how='left')
df_mouse_with_locs.dropna(subset=['Target_pos_x'], inplace=True)

df_mouse_aligned = df_mouse_with_locs.groupby(['subject_id', 'block', 'trial_nr'], group_keys=False).apply(align_trial_coordinates)

# Aggregating aligned and unaligned
agg_dict = {
    'avg_x_dva': ('x_aligned', 'mean'),
    'avg_y_dva': ('y_aligned', 'mean'),
    'avg_orig_x_dva': ('x', 'mean'),
    'avg_orig_y_dva': ('y', 'mean'),
    'target_x_aligned': ('target_x_aligned', 'first'),
    'target_y_aligned': ('target_y_aligned', 'first'),
}
avg_trajectory_vectors_df = df_mouse_aligned.groupby(['subject_id', 'block', 'trial_nr']).agg(**agg_dict).reset_index()

df = pd.merge(df, avg_trajectory_vectors_df, on=['subject_id', 'block', 'trial_nr'], how='left')

def get_aligned_towardness(row):
    locs = {1: (row['target_x_aligned'], row['target_y_aligned'])}
    row_copy = row.copy()
    row_copy['TargetDigit'] = 1
    scores = calculate_trajectory_projections(row_copy, locations_map=locs)
    return scores.get('proj_target', np.nan)

def get_original_towardness(row):
    locs = {1: (row['Target_pos_x'], row['Target_pos_y'])}
    row_copy = row.copy()
    row_copy['TargetDigit'] = 1
    row_copy['avg_x_dva'] = row['avg_orig_x_dva']
    row_copy['avg_y_dva'] = row['avg_orig_y_dva']
    scores = calculate_trajectory_projections(row_copy, locations_map=locs)
    return scores.get('proj_target', np.nan)

df['aligned_target_towardness'] = df.apply(get_aligned_towardness, axis=1)
df['original_target_towardness'] = df.apply(get_original_towardness, axis=1)

print("Plotting...")
plot_df = df.sort_values(['subject_id', 'block', 'trial_nr']).reset_index(drop=True)
plot_df['absolute_trial'] = plot_df.groupby('subject_id').cumcount()

plt.figure(figsize=(15, 6))
sns.lineplot(data=plot_df, x='absolute_trial', y='aligned_target_towardness', label='Aligned', alpha=0.8)
sns.lineplot(data=plot_df, x='absolute_trial', y='original_target_towardness', label='Original', alpha=0.8, linestyle='--')
plt.title('Target Towardness: Aligned vs Original')
plt.xlabel('Absolute Trial Number (Overall Sequence)')
plt.ylabel('Target Towardness Score')
plt.legend()
plt.tight_layout()
plt.show()