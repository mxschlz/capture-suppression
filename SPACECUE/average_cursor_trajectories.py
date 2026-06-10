import seaborn as sns
import pandas as pd
import os
import SPACECUE
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from utils import calculate_trajectory_projections
import pingouin as pg
sns.set_theme(context="talk", style="ticks")
plt.ion()


SUBJECT_IDS = [1,2,3,4,6,7,8,9,10,11]

# --- Analysis & Plotting Parameters ---
EXPERIMENT_FOLDER = "/derivatives/preprocessing"

# Heatmap Parameters
HEATMAP_BOUNDS = (-4, 4)  # DVA range
HEATMAP_BIN_SIZE = 0.1    # DVA per bin
HEATMAP_SIGMA = 1.0       # Smoothing factor
EXCLUDE_FRONTAL_TARGETS = True

# Cross-Correlation Parameters
CC_LAGS = np.arange(-45, 45)
AC_LAGS = np.arange(-500, 500)
CHANCE_LEVEL = 1/3

# Rolling Average Parameters for Towardness Metrics
USE_ROLLING_AVERAGE = True
ROLLING_WINDOW = 45

# --- Data Loading Logic (from implicit_learning_effect.py) ---
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

if EXCLUDE_FRONTAL_TARGETS:
    df = df[df["TargetLoc"] != "Front"]

# Ensure key columns are of integer type for merging with mouse data
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

# Define probability condition based on Subject ID and Location
df['Probability'] = np.where(
    df[loc_col] == 'Absent',
    'Absent',
    np.where(df[loc_col] == df['HP_Distractor_Loc'], 'High', 'Low')
)
df['DistractorProb'] = df['Probability']

# Divide age into cohorts
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    conditions = [
        (df['Age'] >= 18) & (df['Age'] <= 35),
        (df['Age'] >= 60) & (df['Age'] <= 77)
    ]
    choices = ['Young', 'Old']
    df['age_cohort'] = np.select(conditions, choices, default=np.nan)


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

print("Aligning trajectories to a canonical reference frame...")
# Merge location data from the main dataframe into the mouse data
location_cols = ['subject_id', 'block', 'trial_nr', 'Target_pos_x', 'Target_pos_y', 'Distractor_pos_x', 'Distractor_pos_y', 'Control2_pos_x', 'Control2_pos_y']
df_mouse_with_locs = pd.merge(df_mouse, df[location_cols], on=['subject_id', 'block', 'trial_nr'], how='left')
df_mouse_with_locs.dropna(subset=['Target_pos_x'], inplace=True)

aligned_trials = []
# Rotates and flips a single trial's trajectory and its item locations.
for (sub_id, block, trial_nr), group in df_mouse_with_locs.groupby(['subject_id', 'block', 'trial_nr']):
    trial_group = group.copy()
    
    # Get locations for this trial (they are constant within the group)
    try:
        target_pos = np.array([trial_group['Target_pos_x'].iloc[0], trial_group['Target_pos_y'].iloc[0]])
        distractor_pos = np.array([trial_group['Distractor_pos_x'].iloc[0], trial_group['Distractor_pos_y'].iloc[0]])
        control_pos = np.array([trial_group['Control2_pos_x'].iloc[0], trial_group['Control2_pos_y'].iloc[0]])
    except IndexError:
        continue # Should not happen if data is clean

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

    aligned_trials.append(trial_group)

df_mouse_aligned = pd.concat(aligned_trials, ignore_index=True) if aligned_trials else pd.DataFrame()

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
towardness_scores_list = []
for idx, row in df.iterrows():
    # Build the locations_map from the ALIGNED coordinates.
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

# --- Apply Rolling Average ---
if USE_ROLLING_AVERAGE:
    print(f"Applying rolling average (window={ROLLING_WINDOW}) to towardness metrics...")
    metrics_to_smooth = ['target_towardness', 'distractor_towardness', 'control_towardness']
    tt_df = tt_df.sort_values(['subject_id', 'block', 'trial_nr'])
    for m in metrics_to_smooth:
        # Save to a new column to prevent overwriting single-trial raw data needed downstream
        tt_df[f'{m}_smooth'] = tt_df.groupby('subject_id')[m].transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean())

def draw_significance(ax, x1, x2, y, h, p):
    """Helper function to draw statistical significance brackets and asterisks."""
    star = 'ns'
    if p < 0.001: star = '***'
    elif p < 0.01: star = '**'
    elif p < 0.05: star = '*'
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold' if star != 'ns' else 'normal')


# --- Display average target towardness by probability condition and age cohort ---
if 'age_cohort' in tt_df.columns:
    print("Calculating average target towardness by distractor probability condition and age cohort...")
    # Filter for known age cohorts and valid probability conditions
    valid_data = tt_df[
        tt_df['age_cohort'].isin(['Young', 'Old']) & 
        tt_df['Probability'].isin(['Low', 'High'])
    ]
    
    if not valid_data.empty:
        # Group by age cohort and probability condition
        age_prob_means = valid_data.groupby(['age_cohort', 'Probability'])['target_towardness'].mean().reset_index()
        
        print("\n--- Average Target Towardness by Age Cohort and Distractor Probability ---")
        print(age_prob_means.to_string(index=False))
        print("--------------------------------------------------------------------------\n")
        
        # Plotting the age cohort breakdown
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        sns.barplot(data=valid_data, x='age_cohort', y='target_towardness', hue='Probability', hue_order=['Low', 'High'], order=['Young', 'Old'], errorbar='se', ax=ax)
        plt.title('Target Towardness by Age Cohort and Probability')
        plt.xlabel('Age Cohort')
        plt.ylabel('Mean Target Towardness')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid data found matching age cohorts 'Young' (18-35), 'Old' (60-77) and Probabilities 'Low'/'High'.")
else:
    print("No Age column was found or age_cohort could not be created to run the age-based breakdown.")


# Display the average target towardness in a barplot
# Filter to 'Low' and 'High' for the spaghetti plot
tt_df_mean = tt_df[tt_df['Probability'].isin(['Low', 'High'])].groupby(['subject_id', 'Probability'])['target_towardness'].mean().reset_index()

plt.figure(figsize=(10, 8))
ax = plt.gca()
err_bar = 'se' if len(SUBJECT_IDS) > 1 else None
plot_order = ['Low', 'High']

# Base barplot (slightly transparent so lines show up well)
sns.barplot(data=tt_df_mean, x='Probability', y='target_towardness', order=plot_order, errorbar=err_bar, alpha=0.7, ax=ax)

# Spaghetti lines connecting single-subject data
for sub_id, sub_data in tt_df_mean.groupby('subject_id'):
    sub_data = sub_data.set_index('Probability').reindex(plot_order).reset_index()
    if sub_data['target_towardness'].notna().all():
        plt.plot([0, 1], sub_data['target_towardness'], marker='o', color='black', alpha=0.5, linewidth=1.5)

# Calculate and plot paired t-test results using Pingouin
pivot_df = tt_df_mean.pivot(index='subject_id', columns='Probability', values='target_towardness').dropna()
if len(pivot_df) > 1 and 'Low' in pivot_df.columns and 'High' in pivot_df.columns:
    res = pg.ttest(pivot_df['Low'], pivot_df['High'], paired=True)
    y_max = tt_df_mean['target_towardness'].max()
    y_range = y_max - tt_df_mean['target_towardness'].min() if y_max != tt_df_mean['target_towardness'].min() else 1
    h = y_range * 0.03
    draw_significance(ax, 0, 1, y_max + h*1.5, h, res['p-val'].iloc[0])
    ax.set_ylim(top=y_max + h*7)

plt.title('Average Target Towardness by Probability')
plt.tight_layout()
plt.show()

# --- Subject-Mean Target Towardness (Split by HP %) ---
print("Plotting subject-mean target towardness split by HP block probability...")
split_df = tt_df[tt_df['Probability'].isin(['Low', 'High'])].copy()

# Create a new column specifically for this split breakdown using numpy.select
conditions = [
    split_df['Probability'] == 'Low',
    (split_df['Probability'] == 'High') & (split_df['HP_Distractor_Prob'] == 0.6),
    (split_df['Probability'] == 'High') & (split_df['HP_Distractor_Prob'] == 0.8)
]
choices = ['Low', 'High (60%)', 'High (80%)']
split_df['Prob_Split'] = np.select(conditions, choices, default='Unknown')

# Calculate mean per subject first so standard errors reflect between-subject variance
sub_mean_prob_df = split_df.groupby(['subject_id', 'Prob_Split'])['target_towardness'].mean().reset_index()

plt.figure(figsize=(10, 8))
ax = plt.gca()
sns.barplot(data=sub_mean_prob_df, x='Prob_Split', y='target_towardness', order=['Low', 'High (60%)', 'High (80%)'], errorbar=err_bar, palette='viridis', ax=ax)

# Calculate and plot paired t-test results using Pingouin
pivot_df2 = sub_mean_prob_df.pivot(index='subject_id', columns='Prob_Split', values='target_towardness').dropna()
if len(pivot_df2) > 1:
    y_max = sub_mean_prob_df['target_towardness'].max()
    y_range = y_max - sub_mean_prob_df['target_towardness'].min() if y_max != sub_mean_prob_df['target_towardness'].min() else 1
    h = y_range * 0.03
    current_y = y_max + h * 1.5
    
    if 'Low' in pivot_df2.columns and 'High (60%)' in pivot_df2.columns:
        res1 = pg.ttest(pivot_df2['Low'], pivot_df2['High (60%)'], paired=True)
        draw_significance(ax, 0, 1, current_y, h, res1['p-val'].iloc[0])
        
    if 'High (60%)' in pivot_df2.columns and 'High (80%)' in pivot_df2.columns:
        res2 = pg.ttest(pivot_df2['High (60%)'], pivot_df2['High (80%)'], paired=True)
        draw_significance(ax, 1, 2, current_y, h, res2['p-val'].iloc[0])
        
    current_y += h * 5
    if 'Low' in pivot_df2.columns and 'High (80%)' in pivot_df2.columns:
        res3 = pg.ttest(pivot_df2['Low'], pivot_df2['High (80%)'], paired=True)
        draw_significance(ax, 0, 2, current_y, h, res3['p-val'].iloc[0])
        
    ax.set_ylim(top=current_y + h*4)

plt.title('Subject-Mean Target Towardness\n(Low vs. High Probability Split)')
plt.xlabel('Distractor Probability Condition')
plt.ylabel('Mean Target Towardness')
plt.tight_layout()
plt.show()

# --- Variance / Noise Analysis ---
print("Plotting mean for all target locations...")
# 1. Calculate the standard deviation of target_towardness for each subject and target location
std_df = tt_df.groupby(['subject_id', 'TargetLoc'])[['target_towardness', 'distractor_towardness', 'control_towardness']].mean().reset_index()

# 2. Plot the mean of these standard deviations across subjects
plt.figure(figsize=(8, 6))
order_list = [loc for loc in ['Left', 'Front', 'Right'] if loc in std_df['TargetLoc'].unique()]
sns.barplot(data=std_df, x='TargetLoc', y='control_towardness', errorbar=err_bar, order=order_list)
plt.xlabel('Target Location')
plt.ylabel('Mean Towardness')
plt.tight_layout()
plt.show()

# --- Plot Average Aligned Trajectories ---
print("Plotting average aligned trajectories separated by correctness...")
df_sub_corr = tt_df.copy()

if df_sub_corr.empty:
    print("No data found to plot average trajectories.")
else:
    plot_data_corr = pd.merge(
        df_mouse_aligned,
        df_sub_corr[['subject_id', 'block', 'trial_nr', 'IsCorrect']],
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'
    )

    if plot_data_corr.empty:
        print("No trajectory data available after merging.")
    else:
        plot_data_corr = plot_data_corr[plot_data_corr['phase'] == 1].copy()
        print("Plotting heatmap data only for response interval (Phase 1)")

        if plot_data_corr.empty:
            print("Warning: No data remains after filtering for the response interval. The plot will be empty.")

        bounds = HEATMAP_BOUNDS
        bin_size = HEATMAP_BIN_SIZE
        sigma = HEATMAP_SIGMA
        min_dva, max_dva = bounds
        n_bins = int((max_dva - min_dva) / bin_size)

        conditions = [1, 0]
        condition_labels = {1: 'Correct', 0: 'Incorrect'}
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        
        canonical_locs = df_sub_corr[['target_x_aligned', 'target_y_aligned', 'distractor_x_aligned', 'distractor_y_aligned', 'control_x_aligned', 'control_y_aligned']].mean()

        histograms = {}

        for i, cond_val in enumerate(conditions):
            ax = axes[i]
            label = condition_labels[cond_val]
            group = plot_data_corr[plot_data_corr['IsCorrect'] == cond_val]
            
            if group.empty:
                ax.set_title(f'{label} (No Data)')
                histograms[label] = None
                continue

            hist, _, _ = np.histogram2d(
                group['y_aligned'], group['x_aligned'],
                bins=n_bins, range=[[min_dva, max_dva], [min_dva, max_dva]]
            )
            n_trials = group[['subject_id', 'block', 'trial_nr']].drop_duplicates().shape[0]
            if n_trials > 0: hist = hist / n_trials
            hist = gaussian_filter(hist, sigma=sigma)
            histograms[label] = hist
            
            extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
            im = ax.imshow(hist, origin='lower', extent=extent, cmap='magma', vmin=0)
            
            ax.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, label='Target', markeredgecolor='white')
            ax.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, label='Distractor', markeredgecolor='white')
            ax.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, label='Control', markeredgecolor='white')
            
            ax.set_title(f'Condition: {label}')
            ax.set_xlabel('X (dva)')
            if i == 0:
                ax.set_ylabel('Y (dva)')
            ax.grid(False)
            ax.axhline(0, color='white', lw=0.5, alpha=0.5)
            ax.axvline(0, color='white', lw=0.5, alpha=0.5)
            
            ax.plot(0, 0, '+', color='white', markersize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Difference Plot ---
        ax_diff = axes[2]
        if histograms.get('Correct') is not None and histograms.get('Incorrect') is not None:
            diff_hist = histograms['Correct'] - histograms['Incorrect']
            
            max_abs = np.max(np.abs(diff_hist))
            if max_abs == 0: max_abs = 1
            
            extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
            im_diff = ax_diff.imshow(diff_hist, origin='lower', extent=extent, cmap='seismic', vmin=-max_abs, vmax=max_abs)
            
            ax_diff.set_title('Difference (Correct - Incorrect)')
            ax_diff.set_xlabel('X (dva)')
            ax_diff.grid(False)
            ax_diff.axhline(0, color='black', lw=0.5, alpha=0.5)
            ax_diff.axvline(0, color='black', lw=0.5, alpha=0.5)
            
            ax_diff.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, markeredgecolor='black')
            ax_diff.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, markeredgecolor='black')
            ax_diff.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, markeredgecolor='black')
        else:
            ax_diff.set_title('Difference (Insufficient Data)')
            ax_diff.axis('off')

        plt.tight_layout()
        plt.show()

print("Plotting average aligned trajectories separated by trial phase...")
df_sub_phase = tt_df.copy()
if df_sub_phase.empty:
    print("No data found to plot average trajectories.")
else:
    plot_data_phase = pd.merge(
        df_mouse_aligned,
        df_sub_phase[['subject_id', 'block', 'trial_nr']],
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'
    )

    if plot_data_phase.empty:
        print("No trajectory data available after merging.")
    else:
        bounds = HEATMAP_BOUNDS
        bin_size = HEATMAP_BIN_SIZE
        sigma = HEATMAP_SIGMA
        min_dva, max_dva = bounds
        n_bins = int((max_dva - min_dva) / bin_size)

        phases = [0, 1, 2]
        phase_labels = {0: 'Stimulus', 1: 'Response', 2: 'ITI'}
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        
        canonical_locs = df_sub_phase[['target_x_aligned', 'target_y_aligned', 'distractor_x_aligned', 'distractor_y_aligned', 'control_x_aligned', 'control_y_aligned']].mean()

        for i, phase_val in enumerate(phases):
            ax = axes[i]
            label = phase_labels[phase_val]
            group = plot_data_phase[plot_data_phase['phase'] == phase_val]
            
            if group.empty:
                ax.set_title(f'{label} (No Data)')
                continue

            hist, _, _ = np.histogram2d(
                group['y_aligned'], group['x_aligned'],
                bins=n_bins, range=[[min_dva, max_dva], [min_dva, max_dva]]
            )
            n_trials = group[['subject_id', 'block', 'trial_nr']].drop_duplicates().shape[0]
            if n_trials > 0: hist = hist / n_trials
            hist = gaussian_filter(hist, sigma=sigma)
            
            extent = [bounds[0], bounds[1], bounds[0], bounds[1]]
            im = ax.imshow(hist, origin='lower', extent=extent, cmap='magma', vmin=0)
            
            ax.plot(canonical_locs['target_x_aligned'], canonical_locs['target_y_aligned'], 'o', color='green', markersize=10, label='Target', markeredgecolor='white')
            ax.plot(canonical_locs['distractor_x_aligned'], canonical_locs['distractor_y_aligned'], 'o', color='red', markersize=10, label='Distractor', markeredgecolor='white')
            ax.plot(canonical_locs['control_x_aligned'], canonical_locs['control_y_aligned'], 'o', color='grey', markersize=10, label='Control', markeredgecolor='white')
            
            ax.set_title(f'Phase: {label}')
            ax.set_xlabel('X (dva)')
            if i == 0:
                ax.set_ylabel('Y (dva)')
            ax.grid(False)
            ax.axhline(0, color='white', lw=0.5, alpha=0.5)
            ax.axvline(0, color='white', lw=0.5, alpha=0.5)
            
            ax.plot(0, 0, '+', color='white', markersize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('Average Aligned Trajectory Density by Phase')
        plt.tight_layout()
        plt.show()

# --- Cross-Correlation Analysis ---
print("Analyzing cross-correlation...")

# Filter for HP blocks (Left/Right only) to match the analysis logic
df_corr = tt_df[tt_df['HP_Distractor_Loc'].isin(['Left', 'Right'])].copy()

if df_corr.empty:
    print("No data available for Left/Right HP blocks for cross-correlation.")
else:
    # Sort to ensure temporal order
    df_corr = df_corr.sort_values(['subject_id', 'block', 'trial_nr'])

    # Define parameters
    cc_lags = CC_LAGS
    ac_lags = AC_LAGS
    metrics = ['target_towardness_smooth', 'distractor_towardness_smooth', 'control_towardness_smooth']
    colors = {'target_towardness_smooth': 'green', 'distractor_towardness_smooth': 'red', 'control_towardness_smooth': 'grey'}
    titles = {'target_towardness_smooth': 'Target', 'distractor_towardness_smooth': 'Distractor', 'control_towardness_smooth': 'Control'}

    cc_results = {m: [] for m in metrics}
    ac_hp_results = []
    ac_beh_results = {m: [] for m in metrics}
    cc_simulated_delayed_results = []


    for sub_id, subject_full_df in df_corr.groupby('subject_id'):
        fig_full, ax_full = plt.subplots(figsize=(18, 7))

        # Sort by trial order and create a continuous index
        subject_full_df = subject_full_df.sort_values(['block', 'trial_nr']).reset_index(drop=True)

        # Plot the full behavioral signals (raw)
        for m in metrics:
            ax_full.plot(subject_full_df.index, subject_full_df[m], color=colors[m], label=titles[m], alpha=0.7, lw=1.5)

        # Create the full HP signal for the secondary axis
        direction = subject_full_df['HP_Distractor_Loc'].map({'Left': 1, 'Right': -1})
        full_hp_signal = (direction * subject_full_df['HP_Distractor_Prob']) / CHANCE_LEVEL

        ax_twin_full = ax_full.twinx()
        ax_twin_full.step(subject_full_df.index, full_hp_signal, where='mid', color='black', linestyle='--', label='Full HP Signal')

        # Highlight the probability blocks using background color
        # Create a unique ID for each contiguous block of the same probability
        subject_full_df['prob_block_id'] = (subject_full_df['HP_Distractor_Prob'].diff() != 0).cumsum()
        added_labels = set()
        for _, block_df in subject_full_df.groupby('prob_block_id'):
            prob_val = block_df['HP_Distractor_Prob'].iloc[0]
            color = 'lightblue' if prob_val == 0.8 else 'lightcoral'
            label = f'Prob {prob_val} Blocks' if prob_val not in added_labels else None
            ax_full.axvspan(block_df.index.min(), block_df.index.max(), facecolor=color, alpha=0.3, zorder=-1)
            if label:
                ax_full.fill_between([], [], color=color, alpha=0.3, label=label) # Dummy for legend
                added_labels.add(prob_val)

        ax_full.set_title(f"Full Trial Series for Subject {sub_id}")
        ax_full.set_xlabel("Trial Index (within HP blocks)")
        ax_full.set_ylabel("Towardness")
        ax_twin_full.set_ylabel("HP Signal (Scaled)")

        hp_max_full = full_hp_signal.abs().max()
        if pd.isna(hp_max_full) or hp_max_full == 0:
            hp_max_full = 1.0
        ax_twin_full.set_ylim(-hp_max_full * 1.05, hp_max_full * 1.05)

        lines, labels = ax_full.get_legend_handles_labels()
        lines2, labels2 = ax_twin_full.get_legend_handles_labels()
        ax_full.legend(lines + lines2, labels + labels2, loc='best')
        plt.show()

    # Calculate correlations per subject
    for sub_id, sub_df in df_corr.groupby('subject_id'):
        sub_df = sub_df.sort_values(['block', 'trial_nr']).reset_index(drop=True)
        # Map distractor locations for this subject to ensure we can split by Left/Right
        # loc_col is defined globally in the script (e.g. 'SingletonLoc')
        # We expect numeric 1=Left, 3=Right based on experiment code
        sub_df = sub_df.copy()
        sub_df['DistLoc_Mapped'] = sub_df[loc_col].replace({1: 'Left', 2: 'Front', 3: 'Right'})

        # HP Signal: Weighted by probability relative to chance
        direction = sub_df['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1})
        probability = sub_df['HP_Distractor_Prob']
        hp_signal = (direction * probability) / CHANCE_LEVEL

        if hp_signal.std() == 0:
            continue

        sig1 = hp_signal.reset_index(drop=True)

        # Auto-correlation for HP Signal
        ac1 = [sig1.corr(sig1.shift(-lag)) for lag in ac_lags]
        ac_hp_results.append(ac1)

        # Cross-correlation for Simulated Delayed HP Signal (20 trials lag)
        sig_sim = sig1.shift(20)
        cc_sim = [sig1.corr(sig_sim.shift(-lag)) for lag in ac_lags]
        cc_simulated_delayed_results.append(cc_sim)

        # --- Create figure to plot ingredients ---
        fig_ing, axes_ing = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig_ing.suptitle(f"Cross-Correlation Ingredients (Subject {sub_id})")

        for m in metrics:
            # Split metric by Distractor Location (Left vs Right)
            # We only care about trials where the distractor was actually at these locations
            m_left = sub_df[m].where(sub_df['DistLoc_Mapped'] == 'Left')
            m_right = sub_df[m].where(sub_df['DistLoc_Mapped'] == 'Right')

            # Calculate rolling means for each spatial condition
            roll_left = m_left.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
            roll_right = m_right.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

            # Calculate Difference (Left - Right)
            # This represents the spatial bias in towardness
            diff_sig = roll_left - roll_right

            if diff_sig.isnull().all() or diff_sig.std() == 0:
                continue

            # Metric signal is the difference
            sig2 = diff_sig

            # --- Plot Ingredients to visualize what is being correlated ---
            ax = axes_ing[metrics.index(m)]
            trials = np.arange(len(sig1))

            # Plot Towardness (sig2)
            ax.plot(trials, sig2, color=colors[m], label=f'{titles[m]} Bias (L-R)', alpha=0.8)
            ax.set_ylabel('Towardness Bias (Left - Right)')
            ax.set_xlabel('Trial Index')
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

            # Plot HP Signal (sig1) on a secondary y-axis
            ax_twin = ax.twinx()
            ax_twin.step(trials, sig1, color='black', linestyle='--', alpha=0.6, where='mid', label='HP Signal')
            ax_twin.set_ylabel('HP Signal (Scaled)')

            hp_limit = sig1.abs().max()
            if pd.isna(hp_limit) or hp_limit == 0:
                hp_limit = 1.0
            ax_twin.set_ylim(-hp_limit * 1.05, hp_limit * 1.05)

            ax.set_title(f"{titles[m]}")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

            # Auto-correlation for Behavioral Signal
            ac2 = [sig2.corr(sig2.shift(-lag)) for lag in ac_lags]
            ac_beh_results[m].append(ac2)

            # Compute Correlation
            # shift(-lag): if lag > 0, we shift sig2 backward (future becomes present),
            # effectively looking for correlation where sig1 (HP) leads sig2 (Beh).
            cc = [sig1.corr(sig2.shift(-lag)) for lag in cc_lags]
            cc_results[m].append(cc)

        plt.tight_layout()
        plt.show()

    # --- Plot Auto-Correlation: HP Signal ---
    fig_hp, ax_hp = plt.subplots(figsize=(8, 6))
    has_hp_data = False

    if ac_hp_results:
        has_hp_data = True
        ac_df = pd.DataFrame(ac_hp_results, columns=ac_lags)
        ac_mean = ac_df.mean(axis=0)
        ac_sem = ac_df.sem(axis=0) if len(SUBJECT_IDS) > 1 and len(ac_hp_results) > 1 else pd.Series(0, index=ac_lags)

        ax_hp.plot(ac_lags, ac_mean, color='black', label="Mean AC")
        ax_hp.fill_between(ac_lags, ac_mean - ac_sem, ac_mean + ac_sem, color='black', alpha=0.15)

    if has_hp_data:
        ax_hp.axhline(0, color='k', linestyle='--')
        ax_hp.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax_hp.set_title("Auto-Correlation: HP Distractor Location")
        ax_hp.set_xlabel("Lag (Trials)")
        ax_hp.set_ylabel("Correlation Coefficient")
        ax_hp.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig_hp)

    # --- Plot Auto-Correlation: Behavioral Signal ---
    fig_beh, axes_beh = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, m in enumerate(metrics):
        ax = axes_beh[i]
        has_data = False

        metric_res = ac_beh_results[m]

        if metric_res:
            has_data = True
            ac_df = pd.DataFrame(metric_res, columns=ac_lags)
            ac_mean = ac_df.mean(axis=0)
            ac_sem = ac_df.sem(axis=0) if len(SUBJECT_IDS) > 1 and len(metric_res) > 1 else pd.Series(0, index=ac_lags)

            ax.plot(ac_lags, ac_mean, color=colors[m], label="Mean AC")
            ax.fill_between(ac_lags, ac_mean - ac_sem, ac_mean + ac_sem, color=colors[m], alpha=0.15)

        if not has_data:
            ax.set_title(f"{titles[m]} (No Data)")
            continue

        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f"Auto-Corr: {titles[m]} Towardness (Single-Trial)")
        ax.set_xlabel("Lag (Trials)")
        if i == 0:
            ax.set_ylabel("Correlation Coefficient")
        ax.legend(loc='upper right')

    plt.suptitle("Auto-Correlation: Towardness (Single-Trial)")
    plt.tight_layout()
    plt.show()

    # --- Plot Cross-Correlation ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, m in enumerate(metrics):
        ax = axes[i]
        has_data = False

        metric_res = cc_results[m]

        if metric_res:
            has_data = True
            cc_df = pd.DataFrame(metric_res, columns=cc_lags)
            cc_mean = cc_df.mean(axis=0)
            cc_sem = cc_df.sem(axis=0) if len(SUBJECT_IDS) > 1 and len(metric_res) > 1 else pd.Series(0, index=cc_lags)

            ax.plot(cc_lags, cc_mean, color=colors[m], label="Mean CC")
            ax.fill_between(cc_lags, cc_mean - cc_sem, cc_mean + cc_sem, color=colors[m], alpha=0.15)

        if not has_data:
            ax.set_title(f"{titles[m]} (No Data)")
            continue

        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f"Cross-Corr: HP Loc vs {titles[m]} Towardness")
        ax.set_xlabel("Lag (Trials)")
        if i == 0:
            ax.set_ylabel("Correlation Coefficient")
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- Plot Cross-Correlation: Split by Age Cohort ---
    if 'age_cohort' in df_corr.columns:
        print("Plotting Cross-Correlation split by Age Cohort...")
        fig_cc_age, axes_cc_age = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        for i, m in enumerate(metrics):
            ax = axes_cc_age[i]
            
            # Map subjects to their cohorts
            sub_to_cohort = df_corr.groupby('subject_id')['age_cohort'].first().to_dict()
            
            for cohort_name, cohort_color in zip(['Young', 'Old'], ['blue', 'orange']):
                cohort_subjects = [sub for sub, cohort in sub_to_cohort.items() if cohort == cohort_name]

                # Get the indices of the subjects in the cc_results
                # cc_results is built in the order of df_corr.groupby('subject_id')
                cohort_indices = [idx for idx, sub_id in enumerate(df_corr['subject_id'].unique()) if sub_id in cohort_subjects]

                cohort_res = [cc_results[m][idx] for idx in cohort_indices if idx < len(cc_results[m])]
                
                if cohort_res:
                    cc_df_cohort = pd.DataFrame(cohort_res, columns=cc_lags)
                    cc_mean_cohort = cc_df_cohort.mean(axis=0)
                    cc_sem_cohort = cc_df_cohort.sem(axis=0) if len(cohort_subjects) > 1 else pd.Series(0, index=cc_lags)
                    
                    ax.plot(cc_lags, cc_mean_cohort, color=cohort_color, label=f"{cohort_name} CC")
                    ax.fill_between(cc_lags, cc_mean_cohort - cc_sem_cohort, cc_mean_cohort + cc_sem_cohort, color=cohort_color, alpha=0.15)
            
            ax.axhline(0, color='k', linestyle='--')
            ax.axvline(0, color='k', linestyle=':', alpha=0.5)
            ax.set_title(f"Cross-Corr (Age Split): HP vs {titles[m]}")
            ax.set_xlabel("Lag (Trials)")
            if i == 0:
                ax.set_ylabel("Correlation Coefficient")
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    # --- Plot Cross-Correlation: Simulated Delayed HP Signal ---
    fig_sim, ax_sim = plt.subplots(figsize=(8, 6))
    if cc_simulated_delayed_results:
        cc_sim_df = pd.DataFrame(cc_simulated_delayed_results, columns=ac_lags)
        cc_sim_mean = cc_sim_df.mean(axis=0)
        cc_sim_sem = cc_sim_df.sem(axis=0) if len(SUBJECT_IDS) > 1 and len(cc_simulated_delayed_results) > 1 else pd.Series(0, index=cc_lags)

        ax_sim.plot(ac_lags, cc_sim_mean, color='purple', label="Mean CC (Simulated 20-trial lag)")
        ax_sim.fill_between(ac_lags, cc_sim_mean - cc_sim_sem, cc_sim_mean + cc_sim_sem, color='purple', alpha=0.15)

        ax_sim.axhline(0, color='k', linestyle='--')
        ax_sim.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax_sim.axvline(20, color='purple', linestyle=':', alpha=0.8, label='Expected Peak (Lag 20)')
        ax_sim.set_title("Cross-Corr: HP Signal vs HP Signal Lagged by 20 Trials")
        ax_sim.set_xlabel("Lag (Trials)")
        ax_sim.set_ylabel("Correlation Coefficient")
        ax_sim.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
