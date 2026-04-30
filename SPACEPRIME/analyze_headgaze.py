import mne
import os
import numpy as np
import pandas as pd
import glob
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from SPACEPRIME.subjects import subject_ids

sns.set_theme(context="talk", style="ticks")
plt.ion()

# Define Paths
project_data_dir = r"G:\Meine Ablage\PhD\data\SPACEPRIME"
derivatives_dir = os.path.join(project_data_dir, "derivatives")
epoching_dir = os.path.join(derivatives_dir, "epoching")
video_dir = os.path.join(derivatives_dir, "preprocessing")
base_hg_dir = video_dir

def plot_circular_euler_group(euler_left_dict, euler_right_dict, times, contrast_name):
    """Generates a group-level circular polar plot for all three Euler angles."""
    if not euler_left_dict['Yaw'] or not euler_right_dict['Yaw']:
        return
        
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
    angles = ['Yaw', 'Pitch', 'Roll']

    for idx, angle in enumerate(angles):
        ax = axes[idx]
        angle_left_dict = euler_left_dict[angle]
        angle_right_dict = euler_right_dict[angle]

        # Plot individual subjects (thin, semi-transparent lines)
        for sub_id in angle_left_dict.keys():
            ax.plot(np.deg2rad(angle_left_dict[sub_id]), times, color='b', linewidth=1, alpha=0.3)
            ax.plot(np.deg2rad(angle_right_dict[sub_id]), times, color='r', linewidth=1, alpha=0.3)

        # Plot Grand Average across subjects (thick lines)
        avg_left = np.mean(list(angle_left_dict.values()), axis=0)
        avg_right = np.mean(list(angle_right_dict.values()), axis=0)
        ax.plot(np.deg2rad(avg_left), times, label='Left (Group Avg)', color='b', linewidth=3)
        ax.plot(np.deg2rad(avg_right), times, label='Right (Group Avg)', color='r', linewidth=3)

        ax.set_theta_zero_location('N')  
        ax.set_theta_direction(-1)  
        ax.set_thetamin(-10)
        ax.set_thetamax(10)
        ax.set_rlabel_position(0)  
        ax.set_ylabel("Time (s)", labelpad=30)
        ax.set_title(f'{contrast_name}: Head {angle}', va='bottom', y=1.1)
        
        if idx == 2: # Only draw legend on the far right panel
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
            
    fig.suptitle(f'{contrast_name} - Group Euler Angles Comparison', y=1.05)
    plt.tight_layout()

def plot_euler_bar_and_stats(euler_left_dict, euler_right_dict, contrast_name):
    """Plots subject-averaged Euler angles as bar plots and performs paired t-tests."""
    if not euler_left_dict['Yaw'] or not euler_right_dict['Yaw']:
        return

    angles = ['Yaw', 'Pitch', 'Roll']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, angle in enumerate(angles):
        ax = axes[idx]
        left_dict = euler_left_dict[angle]
        right_dict = euler_right_dict[angle]
        
        subjects = list(left_dict.keys())
        # Calculate temporal mean for each subject
        left_means = [np.mean(left_dict[sub]) for sub in subjects]
        right_means = [np.mean(right_dict[sub]) for sub in subjects]

        df = pd.DataFrame({
            'Condition': ['Left'] * len(subjects) + ['Right'] * len(subjects),
            'Mean Angle': left_means + right_means
        })

        # Paired T-test using pingouin
        stats = pg.ttest(left_means, right_means, paired=True)
        print(f"\n--- Paired T-Test Results ({contrast_name}: Left vs. Right {angle}) ---")
        print(stats)

        sns.barplot(data=df, x='Condition', y='Mean Angle', ax=ax, capsize=0.1, errorbar='se', alpha=0.6)
        for i in range(len(subjects)):
            ax.plot([0, 1], [left_means[i], right_means[i]], color='black', marker='o', alpha=0.4)

        ax.set_ylabel(f'Average {angle} (degrees)')
        ax.set_title(f'{contrast_name}:\nSubject-Averaged {angle}')
        
        # Add p-value annotation
        p_val = stats['p-val'].values[0]
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        ax.text(0.5, 0.95, p_text, ha='center', va='bottom', transform=ax.transAxes, fontsize=12)

    fig.suptitle(f'{contrast_name} - Subject-Averaged Euler Angles', y=1.05)
    plt.tight_layout()

def plot_gaze_bar_and_stats(x_left_dict, x_right_dict, y_left_dict, y_right_dict, contrast_name):
    """Plots subject-averaged gaze X and Y as bar plots and performs paired t-tests."""
    if not x_left_dict or not x_right_dict:
        return

    subjects = list(x_left_dict.keys())
    
    # Calculate temporal mean for each subject
    x_left_means = [np.mean(x_left_dict[sub]) for sub in subjects]
    x_right_means = [np.mean(x_right_dict[sub]) for sub in subjects]
    y_left_means = [np.mean(y_left_dict[sub]) for sub in subjects]
    y_right_means = [np.mean(y_right_dict[sub]) for sub in subjects]

    # T-tests using pingouin
    stats_x = pg.ttest(x_left_means, x_right_means, paired=True)
    print(f"\n--- Paired T-Test Results ({contrast_name}: Left vs. Right Gaze X) ---")
    print(stats_x)

    stats_y = pg.ttest(y_left_means, y_right_means, paired=True)
    print(f"\n--- Paired T-Test Results ({contrast_name}: Left vs. Right Gaze Y) ---")
    print(stats_y)

    # DataFrames for plotting
    df_x = pd.DataFrame({
        'Condition': ['Left'] * len(subjects) + ['Right'] * len(subjects),
        'Mean Gaze X': x_left_means + x_right_means
    })
    df_y = pd.DataFrame({
        'Condition': ['Left'] * len(subjects) + ['Right'] * len(subjects),
        'Mean Gaze Y': y_left_means + y_right_means
    })

    # Plotting X and Y side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # X Plot
    sns.barplot(data=df_x, x='Condition', y='Mean Gaze X', ax=axes[0], capsize=0.1, errorbar='se', alpha=0.6)
    for i in range(len(subjects)):
        axes[0].plot([0, 1], [x_left_means[i], x_right_means[i]], color='black', marker='o', alpha=0.4)
    axes[0].set_ylabel('Evoked Change in Iris X')
    axes[0].set_title(f'{contrast_name}:\nSubject-Averaged Gaze X')
    p_val_x = stats_x['p-val'].values[0]
    axes[0].text(0.5, 0.95, f"p = {p_val_x:.3f}" if p_val_x >= 0.001 else "p < 0.001", ha='center', va='bottom', transform=axes[0].transAxes, fontsize=12)

    # Y Plot
    sns.barplot(data=df_y, x='Condition', y='Mean Gaze Y', ax=axes[1], capsize=0.1, errorbar='se', alpha=0.6)
    for i in range(len(subjects)):
        axes[1].plot([0, 1], [y_left_means[i], y_right_means[i]], color='black', marker='o', alpha=0.4)
    axes[1].set_ylabel('Evoked Change in Iris Y')
    axes[1].set_title(f'{contrast_name}:\nSubject-Averaged Gaze Y')
    p_val_y = stats_y['p-val'].values[0]
    axes[1].text(0.5, 0.95, f"p = {p_val_y:.3f}" if p_val_y >= 0.001 else "p < 0.001", ha='center', va='bottom', transform=axes[1].transAxes, fontsize=12)

    fig.suptitle(f'{contrast_name} - Subject-Averaged Gaze', y=1.05)
    plt.tight_layout()

def plot_gaze_spatial(evoked_left, evoked_right, sub_id):
    """Generates a 2D spatial density/heatmap comparing averaged X and Y gaze coordinates."""
    gaze_chs = ["Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy", "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy"]
    if not all(ch in evoked_left.ch_names for ch in gaze_chs):
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))

    # Left condition
    l_dx_l = evoked_left.get_data(picks="Left Iris Relative Pos Dx").flatten()
    r_dx_l = evoked_left.get_data(picks="Right Iris Relative Pos Dx").flatten()
    l_dy_l = evoked_left.get_data(picks="Left Iris Relative Pos Dy").flatten()
    r_dy_l = evoked_left.get_data(picks="Right Iris Relative Pos Dy").flatten()
    x_left = ((l_dx_l - l_dx_l[0]) + (r_dx_l - r_dx_l[0])) / 2.0
    y_left = ((l_dy_l - l_dy_l[0]) + (r_dy_l - r_dy_l[0])) / 2.0

    # Right condition
    l_dx_r = evoked_right.get_data(picks="Left Iris Relative Pos Dx").flatten()
    r_dx_r = evoked_right.get_data(picks="Right Iris Relative Pos Dx").flatten()
    l_dy_r = evoked_right.get_data(picks="Left Iris Relative Pos Dy").flatten()
    r_dy_r = evoked_right.get_data(picks="Right Iris Relative Pos Dy").flatten()
    x_right = ((l_dx_r - l_dx_r[0]) + (r_dx_r - r_dx_r[0])) / 2.0
    y_right = ((l_dy_r - l_dy_r[0]) + (r_dy_r - r_dy_r[0])) / 2.0

    # 2D Kernel Density Estimation (Heatmap)
    sns.kdeplot(x=x_left, y=y_left, ax=ax, cmap="Blues", fill=True, alpha=0.5, thresh=0.05)
    sns.kdeplot(x=x_right, y=y_right, ax=ax, cmap="Reds", fill=True, alpha=0.5, thresh=0.05)
    
    # Mark origin point (t=0)
    ax.plot(0, 0, 'k*', markersize=15, label='t=0 (Onset)')

    # Custom legend for KDE density patches
    left_patch = mpatches.Patch(color='blue', alpha=0.5, label='Left Distractor Density')
    right_patch = mpatches.Patch(color='red', alpha=0.5, label='Right Distractor Density')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [left_patch, right_patch])

    ax.invert_yaxis() # Invert Y so up is screen-up 
    ax.set_xlabel("Evoked Change in Iris X")
    ax.set_ylabel("Evoked Change in Iris Y")
    ax.set_title(f'Spatial Gaze Density - Subject {sub_id}')
    plt.tight_layout()

def extract_saccade_metrics(epochs, keys, tmin=0):
    """Calculates direction-specific saccade counts (leftward vs rightward) at the single-trial level."""
    # Crop to the post-stimulus period
    ep = epochs[keys].copy().crop(tmin=tmin)
    sfreq = ep.info['sfreq']
    
    ch_x_l, ch_y_l = "Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy"
    ch_x_r, ch_y_r = "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy"
    
    # Extract data shape: (trials, times)
    x_l_trials = ep.get_data(picks=ch_x_l)[:, 0, :]
    y_l_trials = ep.get_data(picks=ch_y_l)[:, 0, :]
    x_r_trials = ep.get_data(picks=ch_x_r)[:, 0, :]
    y_r_trials = ep.get_data(picks=ch_y_r)[:, 0, :]
    
    # Conjugate gaze
    x = ((x_l_trials - x_l_trials[:, 0, np.newaxis]) + (x_r_trials - x_r_trials[:, 0, np.newaxis])) / 2.0
    y = ((y_l_trials - y_l_trials[:, 0, np.newaxis]) + (y_r_trials - y_r_trials[:, 0, np.newaxis])) / 2.0
    
    # Compute 2D Velocity (change per second)
    vx = np.gradient(x, axis=-1) * sfreq
    vy = np.gradient(y, axis=-1) * sfreq
    vel = np.sqrt(vx**2 + vy**2)
    
    # Trial-Averaged Directional Saccade Counts
    sac_left_counts = []
    sac_right_counts = []
    
    for i in range(len(vel)):
        v = vel[i]
        pos_x = x[i]
        
        median_v = np.median(v)
        mad_v = np.median(np.abs(v - median_v))
        # Standard Engbert-Kliegl threshold (Median + 5 * MAD)
        th = median_v + 5 * mad_v
        if th < 1e-5: 
            th = 1e-5 # Prevent zero-thresholds on completely flat data
            
        is_sac = v > th
        diff_sac = np.diff(is_sac.astype(int))
        
        # Find start and end indices of the suprathreshold velocity bursts
        starts = np.where(diff_sac == 1)[0] + 1
        ends = np.where(diff_sac == -1)[0] + 1
        
        if is_sac[0]: starts = np.insert(starts, 0, 0)
        if is_sac[-1]: ends = np.append(ends, len(v) - 1)
        
        left_c = 0
        right_c = 0
        
        for s, e in zip(starts, ends):
            # Calculate horizontal spatial displacement during the saccade
            dx = pos_x[e] - pos_x[s]
            if dx < 0:
                left_c += 1
            elif dx > 0:
                right_c += 1
                
        sac_left_counts.append(left_c)
        sac_right_counts.append(right_c)
        
    return np.mean(sac_left_counts), np.mean(sac_right_counts)

def plot_saccade_direction_stats(towards_left_dict, towards_right_dict, away_left_dict, away_right_dict, contrast_name):
    """Plots trial-averaged directional saccade metrics as bar plots and performs paired t-tests."""
    if not towards_left_dict or not towards_right_dict:
        return

    subjects = list(towards_left_dict.keys())
    towards_l = [towards_left_dict[sub] for sub in subjects]
    towards_r = [towards_right_dict[sub] for sub in subjects]
    away_l = [away_left_dict[sub] for sub in subjects]
    away_r = [away_right_dict[sub] for sub in subjects]

    stats_t = pg.ttest(towards_l, towards_r, paired=True)
    print(f"\n--- Paired T-Test Results ({contrast_name}: Saccades Towards Item) ---")
    print(stats_t)

    stats_a = pg.ttest(away_l, away_r, paired=True)
    print(f"\n--- Paired T-Test Results ({contrast_name}: Saccades Away from Item) ---")
    print(stats_a)
    
    combined_towards = np.mean([towards_l, towards_r], axis=0)
    combined_away = np.mean([away_l, away_r], axis=0)
    stats_combined = pg.ttest(combined_towards, combined_away, paired=True)
    print(f"\n--- OVERALL DIRECTION T-TEST ({contrast_name}: Towards vs Away) ---")
    print(stats_combined)

    df_t = pd.DataFrame({
        'Condition': ['Left Item'] * len(subjects) + ['Right Item'] * len(subjects), 
        'Saccade Count': towards_l + towards_r
    })
    df_a = pd.DataFrame({
        'Condition': ['Left Item'] * len(subjects) + ['Right Item'] * len(subjects), 
        'Saccade Count': away_l + away_r
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Towards Plot
    sns.barplot(data=df_t, x='Condition', y='Saccade Count', ax=axes[0], capsize=0.1, errorbar='se', alpha=0.6)
    for i in range(len(subjects)): 
        axes[0].plot([0, 1], [towards_l[i], towards_r[i]], color='black', marker='o', alpha=0.4)
    axes[0].set_ylabel('Saccades per Trial')
    axes[0].set_title(f'{contrast_name}:\nSaccades TOWARDS Item')
    p_val_t = stats_t['p-val'].values[0]
    axes[0].text(0.5, 0.95, f"p = {p_val_t:.3f}" if p_val_t >= 0.001 else "p < 0.001", ha='center', va='bottom', transform=axes[0].transAxes, fontsize=12)

    # Away Plot
    sns.barplot(data=df_a, x='Condition', y='Saccade Count', ax=axes[1], capsize=0.1, errorbar='se', alpha=0.6)
    for i in range(len(subjects)): 
        axes[1].plot([0, 1], [away_l[i], away_r[i]], color='black', marker='o', alpha=0.4)
    axes[1].set_ylabel('Average Saccades per Trial')
    axes[1].set_title(f'{contrast_name}:\nSaccades AWAY from Item')
    p_val_a = stats_a['p-val'].values[0]
    axes[1].text(0.5, 0.95, f"p = {p_val_a:.3f}" if p_val_a >= 0.001 else "p < 0.001", ha='center', va='bottom', transform=axes[1].transAxes, fontsize=12)

    fig.suptitle(f'{contrast_name} - Saccade Directionality', y=1.05)
    plt.tight_layout()

def get_avg_coords(sub_id, metadata_df, base_hg_dir):
    """Helper function to calculate average landmark coordinates for a given set of trials."""
    # Find all unique blocks to load CSVs efficiently
    if 'block' not in metadata_df.columns:
        print("  -> 'block' column not found in metadata. Cannot find CSVs.")
        return None
    blocks = metadata_df['block'].unique()
    blocks = blocks[~np.isnan(blocks)]

    # Load all relevant CSVs once
    block_dfs = {}
    for block_idx in blocks:
        # Find the correct CSV for this block
        hg_csv_pattern = os.path.join(base_hg_dir, f"sub-{sub_id}", "headgaze", f"sub-{sub_id}_None_eye_tracking_log_*.csv")
        hg_csv_files = sorted(glob.glob(hg_csv_pattern))
        if len(hg_csv_files) > int(block_idx):
            csv_path = hg_csv_files[int(block_idx)]
            if os.path.exists(csv_path):
                block_dfs[int(block_idx)] = pd.read_csv(csv_path)
            else:
                 print(f"Warning: Headgaze CSV not found at {csv_path}")
        else:
            print(f"Warning: Could not find headgaze CSV for block {block_idx}")
    
    if not block_dfs:
        print("  -> No headgaze CSVs could be loaded.")
        return None

    # Find landmark columns (all columns ending in _x or _y)
    sample_df = next(iter(block_dfs.values()))
    landmark_cols_x = sorted([c for c in sample_df.columns if c.endswith('_x')])
    landmark_cols_y = sorted([c for c in sample_df.columns if c.endswith('_y')])
    
    if not landmark_cols_x or len(landmark_cols_x) != len(landmark_cols_y):
        print("  -> Could not find matching landmark_x/landmark_y columns in CSV. Skipping landmark plot.")
        return None

    accumulated_coords = np.zeros((len(landmark_cols_x), 2), dtype=np.float32)
    trial_count = 0

    if 'onset' not in metadata_df.columns:
        print("  -> 'onset' column not found in metadata. Cannot find trial timestamps.")
        return None

    for _, row in metadata_df.iterrows():
        block_idx = int(row['block'])
        if block_idx not in block_dfs:
            continue
        
        block_df = block_dfs[block_idx]
        
        # Find closest timestamp
        target_ms = row['onset'] * 1000
        time_diffs = np.abs(block_df['Timestamp (ms)'] - target_ms)
        closest_idx = time_diffs.idxmin()
        
        # Get landmark data for that row
        landmark_row = block_df.iloc[closest_idx]
        
        # Extract X and Y coordinates and stack them into a (N_landmarks, 2) array
        coords = np.vstack([landmark_row[landmark_cols_x].values, landmark_row[landmark_cols_y].values]).T
        
        accumulated_coords += coords.astype(np.float32)
        trial_count += 1
        
    if trial_count > 0:
        return accumulated_coords / trial_count
    return None

def plot_average_landmarks(sub_id, metadata_left, metadata_right, base_hg_dir):
    """
    Averages facial landmark coordinates from the raw CSV files for different conditions and plots them.
    """
    print("\nProcessing facial landmarks from CSV...")
    avg_coords_left = get_avg_coords(sub_id, metadata_left, base_hg_dir)
    avg_coords_right = get_avg_coords(sub_id, metadata_right, base_hg_dir)
    
    if avg_coords_left is None and avg_coords_right is None:
        print("Could not generate average landmark coordinates for any condition.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    
    if avg_coords_left is not None:
        ax.scatter(avg_coords_left[:, 0], avg_coords_left[:, 1], s=15, color='b', label='Left Distractor', alpha=0.7)
    
    if avg_coords_right is not None:
        ax.scatter(avg_coords_right[:, 0], avg_coords_right[:, 1], s=15, color='r', label='Right Distractor', alpha=0.7)
        
    ax.set_title(f'Average Facial Landmarks - Subject {sub_id}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.invert_yaxis()
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

# --- Analysis Execution Setup ---
contrasts = {
    'Lateral Targets': {
        'left_keys_query': 'Target-1-Singleton-2',   # Target Left, Midline Singleton
        'right_keys_query': 'Target-3-Singleton-2'   # Target Right, Midline Singleton
    },
    'Lateral Singletons': {
        'left_keys_query': 'Target-2-Singleton-1',   # Midline Target, Singleton Left
        'right_keys_query': 'Target-2-Singleton-3'   # Midline Target, Singleton Right
    }
}

# Data structure to hold data for both contrasts independently
group_data = {}
for c_name in contrasts.keys():
    group_data[c_name] = {
        'euler_left': {angle: {} for angle in ['Yaw', 'Pitch', 'Roll']},
        'euler_right': {angle: {} for angle in ['Yaw', 'Pitch', 'Roll']},
        'gaze_x_left': {}, 'gaze_x_right': {},
        'gaze_y_left': {}, 'gaze_y_right': {},
        'sac_towards_left': {}, 'sac_towards_right': {},
        'sac_away_left': {}, 'sac_away_right': {}
    }

time_axis = None

for sub_id in subject_ids:
    epochs_path = os.path.join(epoching_dir, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-spaceprime_headgaze-epo.fif")
    
    if not os.path.exists(epochs_path):
        print(f"Skipping Subject {sub_id} - Merged epochs file not found.")
        continue
        
    print(f"Loading data for Subject {sub_id}...")
    epochs = mne.read_epochs(epochs_path, preload=True)

    for c_name, c_queries in contrasts.items():
        left_keys = [k for k in epochs.event_id.keys() if c_queries['left_keys_query'] in k]
        right_keys = [k for k in epochs.event_id.keys() if c_queries['right_keys_query'] in k]
        
        if not left_keys or not right_keys:
            print(f"  -> Missing events for {c_name} in subject {sub_id}. Skipping contrast.")
            continue
            
        evoked_left = epochs[left_keys].average(picks='all').crop(tmin=0)
        evoked_right = epochs[right_keys].average(picks='all').crop(tmin=0)
        
        if time_axis is None:
            time_axis = evoked_left.times
            
        # 1. Collect Circular Euler Angles Data (Yaw, Pitch, Roll)
        for angle in ['Yaw', 'Pitch', 'Roll']:
            if angle in evoked_left.ch_names:
                group_data[c_name]['euler_left'][angle][sub_id] = evoked_left.get_data(picks=angle).flatten()
                group_data[c_name]['euler_right'][angle][sub_id] = evoked_right.get_data(picks=angle).flatten()
        
        # 2. Collect Spatial Eye Gaze Data
        gaze_chs = ["Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy", "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy"]
        if all(ch in evoked_left.ch_names for ch in gaze_chs):
            group_data[c_name]['gaze_x_left'][sub_id] = evoked_left.get_data(picks="Left Iris Relative Pos Dx").flatten()
            group_data[c_name]['gaze_y_left'][sub_id] = evoked_left.get_data(picks="Left Iris Relative Pos Dy").flatten()
            group_data[c_name]['gaze_x_right'][sub_id] = evoked_right.get_data(picks="Left Iris Relative Pos Dx").flatten()
            group_data[c_name]['gaze_y_right'][sub_id] = evoked_right.get_data(picks="Left Iris Relative Pos Dy").flatten()

            # 3. Collect Single-Trial Saccade Metrics (Directional)
            left_sacs_l, right_sacs_l = extract_saccade_metrics(epochs, left_keys)
            left_sacs_r, right_sacs_r = extract_saccade_metrics(epochs, right_keys)
            
            # Left condition (lateral item is on the left)
            group_data[c_name]['sac_towards_left'][sub_id] = left_sacs_l
            group_data[c_name]['sac_away_left'][sub_id] = right_sacs_l
            
            # Right condition (lateral item is on the right)
            group_data[c_name]['sac_towards_right'][sub_id] = right_sacs_r
            group_data[c_name]['sac_away_right'][sub_id] = left_sacs_r

# --- Plotting Group Data per Contrast ---
for c_name, data in group_data.items():
    if data['euler_left']['Yaw'] and data['euler_right']['Yaw'] and time_axis is not None:
        plot_circular_euler_group(data['euler_left'], data['euler_right'], time_axis, c_name)
        plot_euler_bar_and_stats(data['euler_left'], data['euler_right'], c_name)

    if data['gaze_x_left'] and data['gaze_x_right']:
        plot_gaze_bar_and_stats(data['gaze_x_left'], data['gaze_x_right'], data['gaze_y_left'], data['gaze_y_right'], c_name)
        
    if data['sac_towards_left'] and data['sac_towards_right']:
        plot_saccade_direction_stats(
            data['sac_towards_left'], data['sac_towards_right'], 
            data['sac_away_left'], data['sac_away_right'], 
            c_name
        )