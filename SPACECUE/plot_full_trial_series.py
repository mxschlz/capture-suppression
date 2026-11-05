import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SPACECUE
from utils import calculate_trajectory_projections
import utils

sns.set_theme(context="talk", style="ticks")
plt.ion()

# --- Analysis & Plotting Parameters ---
SUBJECT_IDS = [9]
EXPERIMENT_FOLDER = "/derivatives/preprocessing"

# Rolling Average Parameters for Towardness Metrics
USE_ROLLING_AVERAGE = True
ROLLING_WINDOW = 45
CHANCE_LEVEL = 1/3

# --- Data Loading Logic ---
print("Loading data...")
data_path = SPACECUE.get_data_path()

# Load behavioral data for single subjects according to BIDS format
beh_data_base_path = f"{data_path}{EXPERIMENT_FOLDER}"
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

# Determine location column based on experiment design
loc_col = "Non-Singleton2Loc" if "control" in EXPERIMENT_FOLDER else "SingletonLoc"

if "IsCorrect" in df.columns:
    df["IsCorrect"] = df["IsCorrect"].replace({'True': 1, 'False': 0, True: 1, False: 0})
    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors='coerce')

# Ensure key columns are of integer type
df['Subject ID'] = df['Subject ID'].astype(int, errors="ignore")
df['Block'] = df['Block'].astype(int, errors="ignore")
df['Trial Nr'] = df['Trial Nr'].astype(int, errors="ignore")
df["IsCorrect"] = df["IsCorrect"].astype(float, errors="ignore")
df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})
df['HP_Distractor_Prob'] = df['HP_Distractor_Prob'].astype(float, errors="ignore")

# Create snake_case columns for consistency
df['subject_id'] = df['Subject ID'].astype(int, errors='ignore')
df['block'] = df['Block'].astype(int, errors='ignore')
df['trial_nr'] = df['Trial Nr'].astype(int, errors='ignore')

# --- Load and process raw trajectory data ---
print("Loading raw trajectory data...")
df_mouse = utils.load_raw_mouse_trajectories(SUBJECT_IDS, SPACECUE.get_data_path(), calculate_phase=False)

print("Aligning trajectories...")
df_mouse_aligned = utils.align_raw_mouse_trajectories(df_mouse, df)

agg_dict = {
    'avg_x_dva': ('x_aligned', 'mean'),
    'avg_y_dva': ('y_aligned', 'mean'),
    'target_x_aligned': ('target_x_aligned', 'first'),
    'target_y_aligned': ('target_y_aligned', 'first'),
    'distractor_x_aligned': ('distractor_x_aligned', 'first'),
    'distractor_y_aligned': ('distractor_y_aligned', 'first'),
    'control_x_aligned': ('control_x_aligned', 'first'),
    'control_y_aligned': ('control_y_aligned', 'first'),
}
avg_trajectory_vectors_df = df_mouse_aligned.groupby(['subject_id', 'block', 'trial_nr']).agg(**agg_dict).reset_index()

avg_trajectory_vectors_df['subject_id'] = avg_trajectory_vectors_df['subject_id'].astype(int)
avg_trajectory_vectors_df['block'] = avg_trajectory_vectors_df['block'].astype(int)
avg_trajectory_vectors_df['trial_nr'] = avg_trajectory_vectors_df['trial_nr'].astype(int)

df = pd.merge(df, avg_trajectory_vectors_df, on=['subject_id', 'block', 'trial_nr'], how='left')

towardness_scores_list = []
for idx, row in df.iterrows():
    locations_map = {
        1: (row['target_x_aligned'], row['target_y_aligned']),
        2: (row['distractor_x_aligned'], row['distractor_y_aligned']),
        3: (row['control_x_aligned'], row['control_y_aligned'])
    }
    
    row_copy = row.copy()
    row_copy['TargetDigit'] = 1
    row_copy['SingletonDigit'] = 2
    row_copy['SingletonPresent'] = 1
    row_copy['Non-Singleton2Digit'] = 3

    scores = calculate_trajectory_projections(row_copy, locations_map=locations_map)
    scores['target_towardness'] = scores['proj_target']
    scores['distractor_towardness'] = scores['proj_distractor']
    scores['control_towardness'] = scores['proj_control_avg']
    
    towardness_scores_list.append(scores)

towardness_scores = pd.DataFrame(towardness_scores_list, index=df.index)
tt_df = pd.concat([df, towardness_scores], axis=1)

if USE_ROLLING_AVERAGE:
    print(f"Applying rolling average (window={ROLLING_WINDOW})...")
    metrics_to_smooth = ['target_towardness', 'distractor_towardness', 'control_towardness']
    tt_df = tt_df.sort_values(['subject_id', 'block', 'trial_nr'])
    for m in metrics_to_smooth:
        tt_df[f'{m}_smooth'] = tt_df.groupby('subject_id')[m].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )


# --- Plotting the Full Trial Series ---
print("Generating full trial series plots...")
df_corr = tt_df[tt_df['HP_Distractor_Loc'].isin(['Left', 'Right'])].copy()
df_corr = df_corr.sort_values(['subject_id', 'block', 'trial_nr'])

metrics = ['target_towardness_smooth', 'distractor_towardness_smooth', 'control_towardness_smooth']
colors = {'target_towardness_smooth': 'green', 
          'distractor_towardness_smooth': 'red', 
          'control_towardness_smooth': 'grey'}
titles = {'target_towardness_smooth': 'Target', 
          'distractor_towardness_smooth': 'Distractor', 
          'control_towardness_smooth': 'Control'}

for sub_id, subject_full_df in df_corr.groupby('subject_id'):
    # Adjust figsize here for presentation slides
    fig_full, ax_full = plt.subplots(figsize=(10, 5))

    subject_full_df = subject_full_df.sort_values(['block', 'trial_nr']).reset_index(drop=True)

    # Plot the full behavioral signals (raw)
    for m in metrics:
        ax_full.plot(subject_full_df.index, subject_full_df[m], color=colors[m], label=titles[m], alpha=0.7, lw=2.5)

    direction = subject_full_df['HP_Distractor_Loc'].map({'Left': 1, 'Right': -1})
    full_hp_signal = (direction * subject_full_df['HP_Distractor_Prob']) / CHANCE_LEVEL

    ax_twin_full = ax_full.twinx()
    ax_twin_full.step(subject_full_df.index, full_hp_signal, where='mid', color='black', 
                      linestyle='--', label='HP Signal')

    # Highlight the probability blocks using background color
    subject_full_df['prob_block_id'] = (subject_full_df['HP_Distractor_Prob'].diff() != 0).cumsum()
    added_labels = set()
    for _, block_df in subject_full_df.groupby('prob_block_id'):
        prob_val = block_df['HP_Distractor_Prob'].iloc[0]
        color = 'lightsteelblue' if prob_val == 0.8 else 'aliceblue'
        label = f'{int(prob_val*100)}% Blocks' if prob_val not in added_labels else None
        ax_full.axvspan(block_df.index.min(), block_df.index.max(), facecolor=color, alpha=0.3, zorder=-1)
        if label:
            ax_full.fill_between([], [], color=color, alpha=0.3, label=label)  # Dummy for legend
            added_labels.add(prob_val)
            
    # Mark side switches (changes in HP_Distractor_Loc)
    # fillna prevents the very first trial from being falsely flagged as a switch
    is_switch = subject_full_df['HP_Distractor_Loc'] != subject_full_df['HP_Distractor_Loc'].shift(1).fillna(subject_full_df['HP_Distractor_Loc'].iloc[0])
    switch_indices = subject_full_df.index[is_switch]
    
    added_adapt_label = False
    for idx in switch_indices:
        # Highlight an expected "Adaptation Window" (e.g., first 10 trials post-switch)
        adapt_label = 'Side Switch Window' if not added_adapt_label else None
        ax_full.axvspan(idx, min(idx + 10, len(subject_full_df) - 1), 
                        color='gold', alpha=0.3, zorder=0, label=adapt_label)

        added_adapt_label = True
        
    # Mark probability block switches (changes in HP_Distractor_Prob)
    is_prob_switch = subject_full_df['HP_Distractor_Prob'] != subject_full_df['HP_Distractor_Prob'].shift(1).fillna(subject_full_df['HP_Distractor_Prob'].iloc[0])
    # Exclude instances where a side switch also occurs to avoid overlapping windows
    pure_prob_switch = is_prob_switch & ~is_switch
    prob_switch_indices = subject_full_df.index[pure_prob_switch]
    
    added_prob_adapt_label = False
    for idx in prob_switch_indices:
        adapt_label = 'Prob Switch Window' if not added_prob_adapt_label else None
        # Using 'cyan' to contrast with the 'gold' side switch window
        ax_full.axvspan(idx, min(idx + 10, len(subject_full_df) - 1), 
                        color='cyan', alpha=0.3, zorder=0, label=adapt_label)
        added_prob_adapt_label = True

    # --- Schematic Axes Modifications ---
    
    # 1. Left Y-Axis (Towardness)
    ax_full.set_ylabel("Towardness")  # Remove original label
    y_min, y_max = ax_full.get_ylim()
    ax_full.set_yticks([y_min, y_max])
    ax_full.set_yticklabels(['Low', 'High'], fontsize=14)
    ax_full.tick_params(axis='y', length=0) # Hide tick marks

    # 2. Right Y-Axis (HP Signal)
    ax_twin_full.set_ylabel("HP Distractor Location", fontsize=16, labelpad=20)
    hp_max_full = full_hp_signal.abs().max()
    if pd.isna(hp_max_full) or hp_max_full == 0:
        hp_max_full = 1.0
    ax_twin_full.set_ylim(-hp_max_full * 1.1, hp_max_full * 1.1)
    ax_twin_full.set_yticks([-hp_max_full, hp_max_full])
    ax_twin_full.set_yticklabels(['Right', 'Left'], fontsize=14)
    ax_twin_full.tick_params(axis='y', length=0) # Hide tick marks

    # 3. X-Axis
    ax_full.set_xlabel("Trial #", fontsize=16, labelpad=10)
    ax_full.tick_params(axis='x', which='major', labelsize=14)

    # --- General Formatting ---
    #ax_full.set_title(f"Full Trial Series for Subject {sub_id}", fontsize=18, fontweight='bold', pad=15)
    
    # Construct the legend
    lines, labels = ax_full.get_legend_handles_labels()
    lines2, labels2 = ax_twin_full.get_legend_handles_labels()
    # Remove the "HP Signal" from the main legend as the axis is now self-explanatory
    if "HP Signal" in labels2:
        idx = labels2.index("HP Signal")
        del lines2[idx]
        del labels2[idx]
    # ax_full.legend(lines + lines2, labels + labels2, loc='best', fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    sns.despine(right=False)
    plt.show()
    
    os.makedirs("spacecue/plots", exist_ok=True)
    # You can uncomment this line to automatically save the figures:
    fig_full.savefig(f"spacecue/plots/full_trial_series_sub_{sub_id}.png", dpi=300, bbox_inches='tight')