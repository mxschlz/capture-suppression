import os
from scipy.ndimage import gaussian_filter
from utils import *
from SPACEPRIME.subjects import subject_ids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Set plot to be interactive
plt.ion()

def infer_sampling_frequency(df):
    """
    Infer sampling frequency from the raw trajectory data frame.
    """
    if 'time' not in df.columns:
        raise ValueError("Timestamp column 'time' not found in the dataframe.")
    
    # Calculate the time difference between consecutive samples within each trial
    # and subject
    time_diffs = df.groupby(['subject_id', 'trial_nr'])['time'].diff()
    
    # Calculate the median time difference, ignoring NaNs from the first sample
    # of each trial
    median_time_diff = time_diffs.median()
    
    # The sampling frequency is the reciprocal of the median time difference
    if median_time_diff > 0:
        sampling_frequency = 1 / median_time_diff
    else:
        sampling_frequency = 0 # Avoid division by zero
        
    return sampling_frequency

# ===================================================================
# Script configuration
# ===================================================================
WIDTH = 1920
HEIGHT = 1080
DG_VA = 2
SCREEN_SIZE_CM_Y = 30
SCREEN_SIZE_CM_X = 40
VIEWING_DISTANCE_CM = 70
DWELL_TIME_FILTER_RADIUS = 0.4
SIGMA = 25
FILTER_PHASE = 2
OUTLIER_THRESHOLD = 2

# ===================================================================
# Data Loading and Preparation
# ===================================================================
# Define data root directory
try:
    data_root = get_data_path()
except NameError:
    print("get_data_path() not found. Please ensure it's defined in utils.py or define data_root manually.")
    data_root = "/path/to/your/data/" # PLEASE-ADJUST

# --- Subject Inclusion/Exclusion ---
all_subject_ids = subject_ids
subjects_to_exclude = []
sub_ids = [s for s in all_subject_ids if s not in subjects_to_exclude]
print(f"Analyzing {len(sub_ids)} subjects.")

# --- Load Trajectory Data ---
print("Loading trajectory data...")
df_list = []
raw_data_path = os.path.join(data_root, 'sourcedata', 'raw')
subjects_in_folder = os.listdir(raw_data_path)
for subject_folder in subjects_in_folder:
    try:
        current_sub_id = int(subject_folder.split("-")[1])
    except (IndexError, ValueError):
        continue

    if current_sub_id in sub_ids:
        files = glob.glob(os.path.join(raw_data_path, subject_folder, 'beh', f"{subject_folder}*mouse_data.csv"))
        if not files:
            print(f"Warning: No mouse data file found for sub-{current_sub_id}")
            continue
        filepath = files[0]
        temp_df = pd.read_csv(filepath)
        temp_df['subject_id'] = current_sub_id
        df_list.append(temp_df)

if not df_list:
    raise ValueError("No trajectory data loaded. Please check paths and subject IDs.")

df = pd.concat(df_list, ignore_index=True)

# --- Preprocess Trajectories ---
df['block'] = df.groupby('subject_id')['trial_nr'].transform(
    lambda x: ((x == 0) & (x.shift(1) != 0)).cumsum() - 1
)

# --- Infer Sampling Frequency ---
sampling_frequency = infer_sampling_frequency(df)
print(f"Inferred sampling frequency: {sampling_frequency:.2f} Hz")

# Calculate pixel coordinates
df['x_pixels'] = df["x"] * degrees_va_to_pixels(
    degrees=DG_VA, screen_pixels=WIDTH, screen_size_cm=SCREEN_SIZE_CM_X, viewing_distance_cm=VIEWING_DISTANCE_CM
)
df['y_pixels'] = df["y"] * degrees_va_to_pixels(
    degrees=DG_VA, screen_pixels=HEIGHT, screen_size_cm=SCREEN_SIZE_CM_Y, viewing_distance_cm=VIEWING_DISTANCE_CM
)

# --- Load and Merge Behavioral Data ---
print("Loading and merging behavioral data...")
behavior_files = []
preproc_data_path = os.path.join(data_root, 'derivatives', 'preprocessing')
for subject in sub_ids:
    files = glob.glob(os.path.join(preproc_data_path, f"sub-{subject}", 'beh', f"sub-{subject}_clean*.csv"))
    if files:
        behavior_files.append(files[0])
df_behavior = pd.concat([pd.read_csv(f) for f in behavior_files])

# FIX: Remove rows with missing identifiers before type conversion to prevent IntCastingNaNError
df_behavior.dropna(subset=['subject_id', 'block', 'trial_nr'], inplace=True)

# Ensure key columns are of the same type for merging
df['subject_id'] = df['subject_id'].astype(int)
df['block'] = df['block'].astype(int)
df['trial_nr'] = df['trial_nr'].astype(int)
df_behavior['subject_id'] = df_behavior['subject_id'].astype(int)
df_behavior['block'] = df_behavior['block'].astype(int)
df_behavior['trial_nr'] = df_behavior['trial_nr'].astype(int)

# Merge trajectory data with behavioral data
merged_df = pd.merge(
    df,
    df_behavior[['subject_id', 'block', 'trial_nr', 'SingletonPresent', 'Priming']],
    on=['subject_id', 'block', 'trial_nr'],
    how='left'
)
merged_df.dropna(subset=['SingletonPresent', 'Priming'], inplace=True)

# Add sample index within each trial
merged_df['sample_in_trial'] = merged_df.groupby(['subject_id', 'block', 'trial_nr']).cumcount()

# Define sample bins for the first plot based on time intervals.
# Panels will show 0-0.25s, the next 1.75s, and the remainder of the trial.
max_samples = merged_df['sample_in_trial'].max()
s_end_1 = int(0.25 * sampling_frequency)
s_end_2 = s_end_1 + int(1.75 * sampling_frequency)
SAMPLE_BINS = [(0, s_end_1), (s_end_1, s_end_2), (s_end_2, int(max_samples) + 1)]


# Create readable condition labels
priming_map = {1: 'Positive', 0: 'No', -1: 'Negative'}
merged_df['PrimingCondition'] = merged_df['Priming'].map(priming_map)
merged_df['Condition'] = np.where(merged_df['SingletonPresent'] == 1, 'Distractor Present', 'Distractor Absent')

# Filter out central starting point data
outside_center_mask = (merged_df['x']**2 + merged_df['y']**2) > (DWELL_TIME_FILTER_RADIUS**2)
merged_df = merged_df[outside_center_mask]

print("Data preparation complete.")

# ===================================================================
# Heatmap Generation
# ===================================================================

def calculate_smoothed_histogram(data, n_trials, sampling_frequency):
    """Calculates a smoothed, normalized histogram for the given data."""
    if data.empty or n_trials == 0:
        return np.zeros((HEIGHT, WIDTH))

    center_x = WIDTH / 2
    center_y = HEIGHT / 2
    x_shifted = data["x_pixels"] + center_x
    y_shifted = data["y_pixels"] + center_y

    hist, _, _ = np.histogram2d(y_shifted, x_shifted, bins=(HEIGHT, WIDTH), range=[[0, HEIGHT], [0, WIDTH]])

    # Normalize by number of trials and sampling frequency
    if sampling_frequency > 0:
        hist = hist / (n_trials * sampling_frequency)
    else:
        hist = hist / n_trials # Fallback if frequency is 0

    return gaussian_filter(hist, sigma=SIGMA)

def get_total_trials(df_behav, condition_col=None, condition_val=None, s_start=None, s_end=None):
    """Gets the total number of unique trials from the behavioral dataframe for a given condition."""

    if s_start is not None and s_end is not None:
        df_subset = df_behav[(df_behav['sample_in_trial'] >= s_start) & (df_behav['sample_in_trial'] < s_end)]
    else:
        df_subset = df_behav.copy()

    if condition_col and condition_val is not None:
        if condition_col == 'Condition':
            is_present = 1 if 'Present' in condition_val else 0
            df_subset = df_subset[df_subset['SingletonPresent'] == is_present]
        elif condition_col == 'PrimingCondition':
            priming_map_rev = {'Positive': 1, 'No': 0, 'Negative': -1}
            priming_val = priming_map_rev[condition_val]
            df_subset = df_subset[df_subset['Priming'] == priming_val]

    return df_subset[['subject_id', 'block', 'trial_nr']].drop_duplicates().shape[0]

# --- Plot 1: Sample-Resolved Dwell Time ---
print("--- Generating Plot 1: Sample-Resolved Dwell Time ---")
n_bins = len(SAMPLE_BINS)
fig1, axes1 = plt.subplots(1, n_bins, figsize=(5 * n_bins, 7), sharey=True)
if n_bins == 1:
    axes1 = [axes1]
fig1.suptitle('Total Dwell Time, Sample-Resolved', fontsize=20)
sample_hists = []

for i, (s_start, s_end) in enumerate(SAMPLE_BINS):
    subset_df = merged_df[(merged_df['sample_in_trial'] >= s_start) & (merged_df['sample_in_trial'] < s_end)]
    n_trials_in_bin = get_total_trials(merged_df, s_start=s_start, s_end=s_end)
    hist = calculate_smoothed_histogram(subset_df, n_trials_in_bin, sampling_frequency)
    sample_hists.append(hist)

if sample_hists:
    vmax1 = np.percentile(np.concatenate([h.ravel() for h in sample_hists]), 99.9)
    im = None # Initialize im to be accessible for colorbar
    for i, (s_start, s_end) in enumerate(SAMPLE_BINS):
        ax = axes1[i]
        im = ax.imshow(sample_hists[i], extent=[0, WIDTH, 0, HEIGHT], origin='lower', aspect='auto', cmap='inferno', vmax=vmax1, vmin=0)
        ax.set_title(f'Samples: {s_start} to {s_end}')
        ax.set_xlabel("X Position (pixels)")
        if i == 0:
            ax.set_ylabel("Y Position (pixels)")

    # Adjust subplot layout to make room for the colorbar
    fig1.subplots_adjust(right=0.9, top=0.85)
    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig1.colorbar(im, cax=cbar_ax, label='Average Dwell Time per Trial (s)')
    plt.show()


# --- Plot 2: Dwell Time by Distractor Presence ---
print("--- Generating Plot 2: Dwell Time by Distractor Presence ---")
distractor_conditions = ['Distractor Present', 'Distractor Absent']
fig2, axes2 = plt.subplots(1, len(distractor_conditions), figsize=(15, 7), sharey=True)
fig2.suptitle('Total Dwell Time by Distractor Condition', fontsize=20)
distractor_hists = []

for cond in distractor_conditions:
    subset_df = merged_df[merged_df['Condition'] == cond]
    n_trials = get_total_trials(df_behavior, 'Condition', cond)
    hist = calculate_smoothed_histogram(subset_df, n_trials, sampling_frequency)
    distractor_hists.append(hist)

if distractor_hists:
    vmax2 = np.percentile(np.concatenate([h.ravel() for h in distractor_hists]), 99.9)
    im = None # Initialize im
    for i, cond in enumerate(distractor_conditions):
        ax = axes2[i]
        im = ax.imshow(distractor_hists[i], extent=[0, WIDTH, 0, HEIGHT], origin='lower', aspect='auto', cmap='inferno', vmax=vmax2, vmin=0)
        ax.set_title(cond)
        ax.set_xlabel("X Position (pixels)")
        if i == 0:
            ax.set_ylabel("Y Position (pixels)")

    # Adjust subplot layout and add a dedicated colorbar axis
    fig2.subplots_adjust(right=0.88, top=0.85)
    cbar_ax = fig2.add_axes([0.9, 0.15, 0.03, 0.7])
    fig2.colorbar(im, cax=cbar_ax, label='Average Dwell Time per Trial (s)')
    plt.show()


# --- Plot 3: Dwell Time by Priming Condition ---
print("--- Generating Plot 3: Dwell Time by Priming Condition ---")
priming_conditions = ['Positive', 'No', 'Negative']
fig3, axes3 = plt.subplots(1, len(priming_conditions), figsize=(20, 7), sharey=True)
fig3.suptitle('Total Dwell Time by Priming Condition', fontsize=20)
priming_hists = []

for cond in priming_conditions:
    subset_df = merged_df[merged_df['PrimingCondition'] == cond]
    n_trials = get_total_trials(df_behavior, 'PrimingCondition', cond)
    hist = calculate_smoothed_histogram(subset_df, n_trials, sampling_frequency)
    priming_hists.append(hist)

if priming_hists:
    vmax3 = np.percentile(np.concatenate([h.ravel() for h in priming_hists]), 99.9)
    im = None # Initialize im
    for i, cond in enumerate(priming_conditions):
        ax = axes3[i]
        im = ax.imshow(priming_hists[i], extent=[0, WIDTH, 0, HEIGHT], origin='lower', aspect='auto', cmap='inferno', vmax=vmax3, vmin=0)
        ax.set_title(f'{cond} Priming')
        ax.set_xlabel("X Position (pixels)")
        if i == 0:
            ax.set_ylabel("Y Position (pixels)")

    # Adjust subplot layout and add a dedicated colorbar axis
    fig3.subplots_adjust(right=0.9, top=0.85)
    cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    fig3.colorbar(im, cax=cbar_ax, label='Average Dwell Time per Trial (s)')
    plt.show()
