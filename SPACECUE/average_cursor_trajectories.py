import seaborn as sns
import pandas as pd
import os
import SPACECUE
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from utils import calculate_trajectory_projections


TRAJECTORY_BOUNDARY_DVA = 999
#OUTLIER_THRESH = 2
DWELL_TIME_FILTER_RADIUS = 0.0
SUBJECT_IDS = [1]

# --- Data Loading Logic (from implicit_learning_effect.py) ---
print("Loading data...")
data_path = SPACECUE.get_data_path()
experiment_folder = "/derivatives/preprocessing"

# Load behavioral data for single subjects according to BIDS format
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
df = df[df["TargetLoc"] != "Front"]

# Determine location column based on experiment design
loc_col = "Non-Singleton2Loc" if "control" in experiment_folder else "SingletonLoc"

if "IsCorrect" in df.columns:
    df["IsCorrect"] = df["IsCorrect"].replace({'True': 1, 'False': 0, True: 1, False: 0})
    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors='coerce')

# Remove outliers
#df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID")

# Ensure key columns are of integer type for merging with mouse data
df['Subject ID'] = df['Subject ID'].astype(int, errors="ignore")
df['Block'] = df['Block'].astype(int, errors="ignore")
df['Trial Nr'] = df['Trial Nr'].astype(int, errors="ignore")
df["IsCorrect"] = df["IsCorrect"].astype(float, errors="ignore")
df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})
df['HP_Distractor_Prob'] = df.get('HP_Distractor_Prob', 1.0).astype(float, errors="ignore")

# Create snake_case columns for consistency
df['subject_id'] = df['Subject ID'].astype(int, errors='ignore')
df['block'] = df['Block'].astype(int, errors='ignore')
df['trial_nr'] = df['Trial Nr'].astype(int, errors='ignore')

# Define probability condition based on Subject ID and Location
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # The HP_distractor_loc switches dynamically
    return 'High' if row[loc_col] == row['HP_Distractor_Loc'] else 'Low'

df['Probability'] = df.apply(get_probability, axis=1)
df['DistractorProb'] = df['Probability']

# --- Load and process raw trajectory data for towardness analysis ---
print("Loading and processing raw trajectory data for towardness analysis...")

# Find all subject folders for raw mouse data
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

        # Calculate block to differentiate trials with same number across blocks
        # 0 at start of file, or when trial_nr resets to 0 from non-zero
        temp_df['block'] = ((temp_df['trial_nr'] == 0) & (temp_df['trial_nr'].shift(1).fillna(-1) != 0)).cumsum() - 1
        
        # Detect phases based on time jumps (0 -> negative) within each trial
        # 0: Stimulus, 1: Response, 2: ITI
        temp_df['phase'] = temp_df.groupby(['block', 'trial_nr'])['time'].transform(lambda x: (x.diff() < -0.5).cumsum())
        
        df_mouse_list.append(temp_df)

df_mouse = pd.concat(df_mouse_list, ignore_index=True)

# resample data
# df_mouse = resample_all_trajectories(df_mouse, RESAMP_FREQ, trial_cols=['subject_id', 'block', 'trial_nr', 'phase'])

# --- Align all trajectories to a canonical reference frame ---
# The goal is to rotate/flip each trial's coordinate system so that:
# 1. The target is always on the positive y-axis (i.e., "up").
# 2. The distractor is always on the left side of the y-axis.
# This allows for averaging trajectories across trials with different spatial layouts.

def align_trial_coordinates(trial_group):
    """
    Rotates and flips a single trial's trajectory and its item locations.
    """
    # Get locations for this trial (they are constant within the group)
    try:
        target_pos = np.array([trial_group['Target_pos_x'].iloc[0], trial_group['Target_pos_y'].iloc[0]])
        distractor_pos = np.array([trial_group['Distractor_pos_x'].iloc[0], trial_group['Distractor_pos_y'].iloc[0]])
        control_pos = np.array([trial_group['Control2_pos_x'].iloc[0], trial_group['Control2_pos_y'].iloc[0]])
    except IndexError:
        return None # Should not happen if data is clean

    # Get trajectory points for this trial
    traj_points = trial_group[['x', 'y']].values

    # --- 1. Rotation: Align target vector with positive y-axis (90 degrees) ---
    target_angle = np.arctan2(target_pos[1], target_pos[0])
    rotation_angle = np.pi / 2 - target_angle

    cos_rot = np.cos(rotation_angle)
    sin_rot = np.sin(rotation_angle)
    rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    # Rotate all coordinates for the trial
    traj_rotated = np.dot(traj_points, rot_matrix.T)
    target_rotated = np.dot(rot_matrix, target_pos)
    distractor_rotated = np.dot(rot_matrix, distractor_pos)
    control_rotated = np.dot(rot_matrix, control_pos)

    # --- 2. Reflection: Ensure distractor is on the left (negative x-coordinate) ---
    flip_factor = -1.0 if distractor_rotated[0] > 0 else 1.0

    # Apply flip to the x-coordinates
    trial_group['x_aligned'] = traj_rotated[:, 0] * flip_factor
    trial_group['y_aligned'] = traj_rotated[:, 1]

    # Store the fully aligned locations for this trial
    trial_group['target_x_aligned'] = target_rotated[0] * flip_factor
    trial_group['target_y_aligned'] = target_rotated[1]
    trial_group['distractor_x_aligned'] = distractor_rotated[0] * flip_factor
    trial_group['distractor_y_aligned'] = distractor_rotated[1]
    trial_group['control_x_aligned'] = control_rotated[0] * flip_factor
    trial_group['control_y_aligned'] = control_rotated[1]

    return trial_group

print("Aligning trajectories to a canonical reference frame...")
# Merge location data from the main dataframe into the mouse data
location_cols = ['subject_id', 'block', 'trial_nr', 'Target_pos_x', 'Target_pos_y', 'Distractor_pos_x', 'Distractor_pos_y', 'Control2_pos_x', 'Control2_pos_y']
df_mouse_with_locs = pd.merge(df_mouse, df[location_cols], on=['subject_id', 'block', 'trial_nr'], how='left')
df_mouse_with_locs.dropna(subset=['Target_pos_x'], inplace=True)

# Apply the alignment function to each trial
df_mouse_aligned = df_mouse_with_locs.groupby(['subject_id', 'block', 'trial_nr'], group_keys=False).apply(align_trial_coordinates)

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

# Calculate the average trajectory vector for each trial using ALIGNED coordinates
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

# Ensure subject_id, block, trial_nr are consistent types for merging
avg_trajectory_vectors_df['subject_id'] = avg_trajectory_vectors_df['subject_id'].astype(int)
avg_trajectory_vectors_df['block'] = avg_trajectory_vectors_df['block'].astype(int)
avg_trajectory_vectors_df['trial_nr'] = avg_trajectory_vectors_df['trial_nr'].astype(int)

# Merge the average vectors into the main behavioral dataframe
df = pd.merge(df, avg_trajectory_vectors_df, on=['subject_id', 'block', 'trial_nr'], how='left')

# Calculate the towardness scores using the aligned data
def get_trial_towardness(row):
    # Build the locations_map from the ALIGNED coordinates.
    # These coordinates are now consistent across all trials.
    # The avg_x_dva and avg_y_dva columns are also from the aligned data.
    locations_map = {
        1: (row['target_x_aligned'], row['target_y_aligned']),
        2: (row['distractor_x_aligned'], row['distractor_y_aligned']),
        3: (row['control_x_aligned'], row['control_y_aligned'])
    }

    # Create a copy of the row to add the digit keys needed by calculate_trajectory_projections
    row_copy = row.copy()
    row_copy['TargetDigit'] = 1
    row_copy['SingletonDigit'] = 2
    row_copy['SingletonPresent'] = 1
    # Assuming the third location is always the second non-singleton (control)
    row_copy['Non-Singleton2Digit'] = 3

    # Calculate projections using the aligned average vector and aligned locations map
    scores = calculate_trajectory_projections(row_copy, locations_map=locations_map)

    # Calculate towardness as percentage of distance to item
    #target_vec_length = get_vector_length(1, locations_map)
    scores['target_towardness'] = scores['proj_target']

    #distractor_vec_length = get_vector_length(2, locations_map)
    scores['distractor_towardness'] = scores['proj_distractor']

    #control_vec_length = get_vector_length(3, locations_map)
    scores['control_towardness'] = scores['proj_control_avg']

    return scores

towardness_scores = df.apply(get_trial_towardness, axis=1)
tt_df = pd.concat([df, towardness_scores], axis=1)

# Display the average target towardness in a heatmap
plt.figure(figsize=(10, 8))
sns.barplot(data=tt_df, x='Probability', y='target_towardness', errorbar=None)

# --- Plot Average Aligned Trajectories ---
def calculate_dva_heatmap(data, bounds, bin_size, sigma):
    """Calculates a smoothed 2D histogram of trajectory points."""
    min_dva, max_dva = bounds
    n_bins = int((max_dva - min_dva) / bin_size)

    # Create 2D histogram
    # Note: histogram2d returns H[y_bin, x_bin] if we pass y first, which matches imshow origin='lower'
    hist, _, _ = np.histogram2d(
        data['y_aligned'],
        data['x_aligned'],
        bins=n_bins,
        range=[[min_dva, max_dva], [min_dva, max_dva]]
    )

    # Normalize by number of trials to get average density per trial
    n_trials = data[['subject_id', 'block', 'trial_nr']].drop_duplicates().shape[0]
    if n_trials > 0:
        hist = hist / n_trials

    return gaussian_filter(hist, sigma=sigma)

# This plot visualizes the average movement path for different conditions,
# demonstrating the effect of the alignment.

def plot_average_aligned_trajectories_correctness(df, aligned_mouse_df):
    """
    Plots the average aligned trajectories, separated by correctness.
    """
    # Use the provided dataframe directly
    df_sub = df.copy()
    if df_sub.empty:
        print("No data found to plot average trajectories.")
        return

    # Merge condition info (IsCorrect) into the aligned mouse data
    plot_data = pd.merge(
        aligned_mouse_df,
        df_sub[['subject_id', 'block', 'trial_nr', 'IsCorrect']],
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'
    )

    if plot_data.empty:
        print("No trajectory data available after merging.")
        return

    # --- Filter for the response interval (Phase 1) ---
    plot_data = plot_data[plot_data['phase'] == 1].copy()
    print("Plotting heatmap data only for response interval (Phase 1)")

    if plot_data.empty:
        print("Warning: No data remains after filtering for the response interval. The plot will be empty.")

    # Define heatmap parameters
    bounds = (-4, 4)  # DVA range
    bin_size = 0.1     # DVA per bin
    sigma = 2.0         # Smoothing factor

    conditions = [1, 0]
    condition_labels = {1: 'Correct', 0: 'Incorrect'}
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    
    # Calculate canonical locations for overlay
    canonical_locs = df_sub[['target_x_aligned', 'target_y_aligned', 'distractor_x_aligned', 'distractor_y_aligned', 'control_x_aligned', 'control_y_aligned']].mean()

    histograms = {}

    for i, cond_val in enumerate(conditions):
        ax = axes[i]
        label = condition_labels[cond_val]
        group = plot_data[plot_data['IsCorrect'] == cond_val]
        
        if group.empty:
            ax.set_title(f'{label} (No Data)')
            histograms[label] = None
            continue

        # Generate heatmap
        hist = calculate_dva_heatmap(group, bounds, bin_size, sigma)
        histograms[label] = hist
        
        # Plot heatmap
        extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
        im = ax.imshow(hist, origin='lower', extent=extent, cmap='magma', vmin=0)
        
        # Overlay canonical locations
        ax.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, label='Target', markeredgecolor='white')
        ax.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, label='Distractor', markeredgecolor='white')
        ax.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, label='Control', markeredgecolor='white')
        
        # Formatting
        ax.set_title(f'Condition: {label}')
        ax.set_xlabel('X (dva)')
        if i == 0:
            ax.set_ylabel('Y (dva)')
        ax.grid(False)
        ax.axhline(0, color='white', lw=0.5, alpha=0.5)
        ax.axvline(0, color='white', lw=0.5, alpha=0.5)
        
        # Add start point
        ax.plot(0, 0, '+', color='white', markersize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Difference Plot ---
    ax_diff = axes[2]
    if histograms.get('Correct') is not None and histograms.get('Incorrect') is not None:
        diff_hist = histograms['Correct'] - histograms['Incorrect']
        
        # Determine symmetric scale for difference
        max_abs = np.max(np.abs(diff_hist))
        if max_abs == 0: max_abs = 1
        
        extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
        im_diff = ax_diff.imshow(diff_hist, origin='lower', extent=extent, cmap='seismic', vmin=-max_abs, vmax=max_abs)
        
        ax_diff.set_title('Difference (Correct - Incorrect)')
        ax_diff.set_xlabel('X (dva)')
        ax_diff.grid(False)
        ax_diff.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax_diff.axvline(0, color='black', lw=0.5, alpha=0.5)
        
        # Overlay canonical locations
        ax_diff.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, markeredgecolor='black')
        ax_diff.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, markeredgecolor='black')
        ax_diff.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, markeredgecolor='black')

        #plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label='Density Difference')
    else:
        ax_diff.set_title('Difference (Insufficient Data)')
        ax_diff.axis('off')

    #plt.suptitle('Average Aligned Trajectory Density by Correctness (Response Interval) for Subject 905')
    plt.tight_layout()
    plt.show()

def plot_average_aligned_trajectories_phase(df, aligned_mouse_df):
    """
    Plots the average aligned trajectories, separated by trial phase.
    """
    # Use the provided dataframe directly
    df_sub = df.copy()
    if df_sub.empty:
        print("No data found to plot average trajectories.")
        return

    # Merge to ensure we only use valid trials
    plot_data = pd.merge(
        aligned_mouse_df,
        df_sub[['subject_id', 'block', 'trial_nr']],
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'
    )

    if plot_data.empty:
        print("No trajectory data available after merging.")
        return

    # Define heatmap parameters
    bounds = (-4, 4)  # DVA range
    bin_size = 0.1     # DVA per bin
    sigma = 2.0         # Smoothing factor

    phases = [0, 1, 2]
    phase_labels = {0: 'Stimulus', 1: 'Response', 2: 'ITI'}
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    
    # Calculate canonical locations for overlay
    canonical_locs = df_sub[['target_x_aligned', 'target_y_aligned', 'distractor_x_aligned', 'distractor_y_aligned', 'control_x_aligned', 'control_y_aligned']].mean()

    for i, phase_val in enumerate(phases):
        ax = axes[i]
        label = phase_labels[phase_val]
        group = plot_data[plot_data['phase'] == phase_val]
        
        if group.empty:
            ax.set_title(f'{label} (No Data)')
            continue

        # Generate heatmap
        hist = calculate_dva_heatmap(group, bounds, bin_size, sigma)
        
        # Plot heatmap
        extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
        im = ax.imshow(hist, origin='lower', extent=extent, cmap='magma', vmin=0)
        
        # Overlay canonical locations
        ax.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, label='Target', markeredgecolor='white')
        ax.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, label='Distractor', markeredgecolor='white')
        ax.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, label='Control', markeredgecolor='white')
        
        # Formatting
        ax.set_title(f'Phase: {label}')
        ax.set_xlabel('X (dva)')
        if i == 0:
            ax.set_ylabel('Y (dva)')
        ax.grid(False)
        ax.axhline(0, color='white', lw=0.5, alpha=0.5)
        ax.axvline(0, color='white', lw=0.5, alpha=0.5)
        
        # Add start point
        ax.plot(0, 0, '+', color='white', markersize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Average Aligned Trajectory Density by Phase')
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_average_aligned_trajectories_correctness(tt_df, df_mouse_aligned)
plot_average_aligned_trajectories_phase(tt_df, df_mouse_aligned)

# --- Plot Towardness Over Trials ---
def plot_towardness_over_trials(df):
    """
    Plots the towardness scores across trials (cumulative).
    """
    # Ensure data is sorted by block and trial to get correct temporal order
    df_sorted = df.sort_values(['block', 'trial_nr']).reset_index(drop=True)
    df_sorted['cumulative_trial'] = df_sorted.index + 1
    
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Define metrics to plot
    metrics = {
        'target_towardness': ('Target', 'green'),
        'distractor_towardness': ('Distractor', 'red'),
        'control_towardness': ('Control', 'grey')
    }

    # Plot rolling average for better visibility of trends
    window_size = 23

    for col, (label, color) in metrics.items():
        # Plot raw data faintly
        ax1.plot(df_sorted['cumulative_trial'], df_sorted[col], color=color, alpha=0.15, linewidth=0.5)

        # Plot rolling mean
        rolling_mean = df_sorted[col].rolling(window=window_size, min_periods=1).mean()
        ax1.plot(df_sorted['cumulative_trial'], rolling_mean, color=color, label=f'{label} (Moving Avg {window_size})', linewidth=2)

    # Set symmetric y-limits for the primary axis to ensure 0 aligns with the secondary axis 0
    # and the step function appears symmetric.
    data_cols = list(metrics.keys())
    max_val = df_sorted[data_cols].max().max()
    min_val = df_sorted[data_cols].min().min()
    # Use a default if data is missing, otherwise calculate symmetric limit with padding
    abs_limit = 100 * 1.5
    abs_limit = max(105, abs_limit)  # Ensure the +/- 100 steps are visible
    ax1.set_ylim(-abs_limit, abs_limit)

    ax1.set_title('Trajectory Towardness Over Trials (Subject 905)')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Towardness (%)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Add HP Distractor Location on a secondary y-axis ---
    ax2 = ax1.twinx()
    direction = df_sorted['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1})
    hp_plot_vals = direction * df_sorted['HP_Distractor_Prob'] * 100
    ax2.plot(df_sorted['cumulative_trial'], hp_plot_vals, color='black', linestyle='--', alpha=0.6, label='HP Distractor')
    ax2.set_ylabel('HP Distractor Location & Probability')
    ax2.set_yticks([-80, -60, 60, 80])
    ax2.set_yticklabels(['Left (0.8)', 'Left (0.6)', 'Right (0.6)', 'Right (0.8)'])
    ax2.set_ylim(-abs_limit, abs_limit)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    fig.tight_layout()
    plt.show()

plot_towardness_over_trials(tt_df)

# --- Cross-Correlation Analysis ---
print("Analyzing cross-correlation...")

def analyze_towardness_cross_correlation(df):
    # Filter for HP blocks (Left/Right only) to match the analysis logic
    df_corr = df[df['HP_Distractor_Loc'].isin(['Left', 'Right'])].copy()
    
    if df_corr.empty:
        print("No data available for Left/Right HP blocks for cross-correlation.")
        return

    # Sort to ensure temporal order
    df_corr = df_corr.sort_values(['subject_id', 'block', 'trial_nr'])
    
    # Define parameters
    WINDOW = 23
    lags = np.arange(-45, 45)
    metrics = ['target_towardness', 'distractor_towardness', 'control_towardness']
    colors = {'target_towardness': 'green', 'distractor_towardness': 'red', 'control_towardness': 'grey'}
    titles = {'target_towardness': 'Target', 'distractor_towardness': 'Distractor', 'control_towardness': 'Control'}
    
    cc_results = {m: {0.8: [], 0.6: []} for m in metrics}

    # Calculate rolling means and correlations per subject AND probability condition
    for (sub_id, prob), sub_df in df_corr.groupby(['subject_id', 'HP_Distractor_Prob']):
        if prob not in [0.6, 0.8]:
            continue
            
        sub_df = sub_df.sort_values(['block', 'trial_nr'])
        
        # HP Signal: Incorporate both location and probability
        direction = sub_df['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1})
        hp_numeric = direction * prob
        
        # Skip if constant (no switch) or empty
        if hp_numeric.std() == 0 or len(sub_df) < WINDOW:
            continue
            
        sig1 = hp_numeric.reset_index(drop=True)

        # Map distractor locations for this subject to ensure we can split by Left/Right
        # loc_col is defined globally in the script (e.g. 'SingletonLoc')
        # We expect numeric 1=Left, 3=Right based on experiment code
        sub_df = sub_df.copy()
        sub_df['DistLoc_Mapped'] = sub_df[loc_col].replace({1: 'Left', 2: 'Front', 3: 'Right'})

        for m in metrics:
            # Split metric by Distractor Location (Left vs Right)
            # We only care about trials where the distractor was actually at these locations
            m_left = sub_df[m].where(sub_df['DistLoc_Mapped'] == 'Left')
            m_right = sub_df[m].where(sub_df['DistLoc_Mapped'] == 'Right')

            # Calculate rolling means for each spatial condition
            roll_left = m_left.rolling(window=WINDOW, center=True, min_periods=1).mean()
            roll_right = m_right.rolling(window=WINDOW, center=True, min_periods=1).mean()
            
            # Calculate Robust Spatial Bias Index: (Left - Right) / (|Left| + |Right|)
            # This bounds the metric between -1 and 1 and handles negative values correctly
            diff_sig = (roll_left - roll_right) / (roll_left.abs() + roll_right.abs() + 1e-8)

            if diff_sig.isnull().all() or diff_sig.std() == 0:
                continue

            sig2 = diff_sig.reset_index(drop=True)
            
            # Compute Correlation
            # shift(-lag): if lag > 0, we shift sig2 backward (future becomes present), 
            # effectively looking for correlation where sig1 (HP) leads sig2 (Beh).
            cc = [sig1.corr(sig2.shift(-lag)) for lag in lags]
            cc_results[m][prob].append(cc)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    styles = {
        0.8: {'ls': '-', 'alpha': 0.8, 'label': 'Prob 0.8'},
        0.6: {'ls': '--', 'alpha': 0.8, 'label': 'Prob 0.6'}
    }
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        has_data = False
        
        for prob in [0.8, 0.6]:
            metric_res = cc_results[m][prob]
            
            if not metric_res:
                continue
            
            has_data = True
            cc_df = pd.DataFrame(metric_res, columns=lags)
            cc_mean = cc_df.mean(axis=0)
            cc_sem = cc_df.sem(axis=0) if len(metric_res) > 1 else pd.Series(0, index=lags)
            
            ax.plot(lags, cc_mean, color=colors[m], linestyle=styles[prob]['ls'], 
                    alpha=styles[prob]['alpha'], label=f"Mean CC {styles[prob]['label']}")
            ax.fill_between(lags, cc_mean - cc_sem, cc_mean + cc_sem, color=colors[m], alpha=0.15)
            
        if not has_data:
            ax.set_title(f"{titles[m]} (No Data)")
            continue

        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f"Cross-Corr: HP Loc vs ({titles[m]} L-R)")
        ax.set_xlabel("Lag (Trials)")
        if i == 0:
            ax.set_ylabel("Correlation Coefficient")
        ax.legend(loc='upper right')
            
    plt.suptitle("Cross-Correlation: HP Distractor Location vs. Towardness (0.8 vs 0.6)")
    plt.tight_layout()
    plt.show()

analyze_towardness_cross_correlation(tt_df)
