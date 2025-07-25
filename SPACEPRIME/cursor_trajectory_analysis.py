import os
from scipy.ndimage import gaussian_filter
from utils import *
import seaborn as sns
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
from scipy.stats import linregress
import pingouin as pg

plt.ion()


# Script configuration parameters
# ===================================================================
WIDTH = 1920
HEIGHT = 1080
DG_VA = 2
SCREEN_SIZE_CM_Y = 30
SCREEN_SIZE_CM_X = 40
VIEWING_DISTANCE_CM = 70
DWELL_TIME_FILTER_RADIUS = 0.4
MOVEMENT_THRESHOLD = 0.05
SIGMA = 25
FILTER_PHASE = None
OUTLIER_THRESHOLD = 2
WINDOW_SIZE = 31
RESAMP_FREQ = 60
SUB_BLOCKS_PER_BLOCK = 2

# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"

# --- Subject Inclusion/Exclusion ---
# Get the full list of subjects from the project file
all_subject_ids = subject_ids

# Define a list of subjects to exclude from all analyses in this script
subjects_to_exclude = [108]
print(f"--- Subject Exclusion ---")
print(f"Excluding subjects: {subjects_to_exclude}")

# Create the final list of subjects to be included in the analysis
sub_ids = [s for s in all_subject_ids if s not in subjects_to_exclude]
print(f"Total subjects for analysis: {len(sub_ids)}")
print("-" * 25)

# --- Load Data ---
# Load data only for the included subjects
subjects_in_folder = os.listdir(data_root)
df_list = list()
for subject_folder in subjects_in_folder:
    # Extract subject ID from folder name like "sub-101"
    try:
        current_sub_id = int(subject_folder.split("-")[1])
    except (IndexError, ValueError):
        continue # Skip folders that don't match the format

    if current_sub_id in sub_ids:
        filepath = glob.glob(f"{get_data_path()}sourcedata/raw/{subject_folder}/beh/{subject_folder}*mouse_data.csv")[0]
        temp_df = pd.read_csv(filepath)
        temp_df['subject_id'] = current_sub_id  # Add subject ID as an integer
        df_list.append(temp_df)
df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames
# Corrected line:
df['block'] = df.groupby('subject_id')['trial_nr'].transform(
    lambda x: ((x == 0) & (x.shift(1) != 0)).cumsum() - 1)
rows_per_trial = df.groupby(['trial_nr', "block", 'subject_id']).size().reset_index(name='n_rows')

# Calculate scaled pixel coordinates and add them to the DataFrame
# This uses the same calculation you have for x and y, but stores them in df.
df['x_pixels'] = df["x"] * degrees_va_to_pixels(
    degrees=DG_VA,
    screen_pixels=WIDTH,
    screen_size_cm=SCREEN_SIZE_CM_X,  # screen_width_cm for x coordinates
    viewing_distance_cm=VIEWING_DISTANCE_CM
)
df['y_pixels'] = df["y"] * degrees_va_to_pixels(
    degrees=DG_VA,
    screen_pixels=HEIGHT,
    screen_size_cm=SCREEN_SIZE_CM_Y,  # screen_height_cm for y coordinates
    viewing_distance_cm=VIEWING_DISTANCE_CM
)
# ===================================================================
#                      DWELL TIME HEATMAP
# ===================================================================
print("\n--- Generating Dwell Time Heatmap ---")

# --- 1. Filter out data from the central starting point ---
# To better visualize dwell time on peripheral digits, we exclude the
# dense data from the central starting area (around digit 5).
print(f"Excluding data within a {DWELL_TIME_FILTER_RADIUS} dva radius of the center to improve visibility.")

# Create a boolean mask for points outside the central radius.
# We use squared values to avoid a costly square root operation on the entire column.
outside_center_mask = (df['x']**2 + df['y']**2) > (DWELL_TIME_FILTER_RADIUS**2)
df_for_heatmap = df[outside_center_mask]
print(f"Original data points: {len(df)}. Data points for heatmap: {len(df_for_heatmap)}.")

# --- 2. Prepare data for histogram ---
# Calculate the center of the screen in pixels
center_x = WIDTH / 2
center_y = HEIGHT / 2

# Shift the filtered data so that the data's origin (0,0) is at the screen's center
x_shifted = df_for_heatmap["x_pixels"] + center_x
y_shifted = df_for_heatmap["y_pixels"] + center_y

# --- 3. Calculate and normalize the 2D histogram ---
# The histogram counts the number of samples in each pixel bin.
hist, _, _ = np.histogram2d(y_shifted, x_shifted, bins=(HEIGHT, WIDTH), range=[[0, HEIGHT], [0, WIDTH]])

# To get dwell time in seconds, we need to divide the counts by the sampling frequency.
# The original script calculated sfreq but didn't apply it to the final plot; this is corrected here.
sfreq = rows_per_trial["n_rows"].mean() / 3.0  # 3 seconds per trial
print(f"Average sampling frequency: {sfreq:.2f} Hz")

# Normalize the histogram counts to get seconds.
# We add a small epsilon to avoid division by zero if sfreq is 0.
hist /= (sfreq + 1e-9)

# --- 4. Smooth and plot the heatmap ---
sigma = 25  # Sigma for the Gaussian smoothing filter
hist_smoothed = gaussian_filter(hist, sigma=sigma)

extent = [0, WIDTH, 0, HEIGHT] # Use 0, height for y-axis to match a standard Cartesian plot
plt.figure(figsize=(10, 8))
plt.imshow(hist_smoothed, extent=extent, origin='lower', aspect='auto', cmap='inferno')
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.title("Cursor Dwell Time (seconds, center excluded)")
cbar = plt.colorbar()
cbar.set_label("Dwell Time (s)")
plt.show()


# ===================================================================
#                      PATH LENGTH CALCULATION
# ===================================================================
# Group by subject and trial number, then apply the path length calculation
# This assumes your DataFrame 'df' has 'subject_id' and 'trial_nr' columns.
df_path_lengths = df.groupby(['subject_id', 'block', 'trial_nr']).apply(calculate_trial_path_length).reset_index(name='path_length_pixels')
# Ensure data types are consistent for merge keys in df_path_lengths as well
df_path_lengths['subject_id'] = df_path_lengths['subject_id'].astype(float) # Or int, if appropriate
df_path_lengths['block'] = df_path_lengths['block'].astype(float)
df_path_lengths['trial_nr'] = df_path_lengths['trial_nr'].astype(float)
# load other dataframe which has all the experiment trial info
df_behavior = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
# Perform the left merge
# We'll use the df_behavioral_example for this demonstration.
# Replace df_behavioral_example with your actual behavioral DataFrame.
merged_df = pd.merge(
    df_behavior,  # Or your actual df_behavioral
    df_path_lengths,
    on=['subject_id', 'block', 'trial_nr'],
    how='left'  # Keeps all rows from df_behavioral_example and adds path_length_pixels
)
df_clean = merged_df[merged_df["phase"]!=FILTER_PHASE]

df_clean = remove_outliers(df_clean, column_name="rt", threshold=OUTLIER_THRESHOLD)

# ===================================================================
#                  ANALYSIS OF FIRST MOUSE MOVEMENT (REVISED)
# ===================================================================
print("\n--- Analyzing Initial Mouse Movement (with filtering) ---")

# Find all data points where the cursor has moved away from the center.
moved_rows_df = df[(df['x'].abs() > MOVEMENT_THRESHOLD) | (df['y'].abs() > MOVEMENT_THRESHOLD)].copy()

# For each trial, find the timestamp of the *first* recorded movement.
first_movement_times_df = moved_rows_df.groupby(['subject_id', 'block', 'trial_nr'])['time'].min().reset_index()
first_movement_times_df.rename(columns={'time': 'first_movement_s'}, inplace=True)

# --- Diagnostic Step: Filter out trials where the response was '5' ---
print("Testing hypothesis: Removing trials with response '5' to check the late peak.")

# To filter by response, we need to merge with the behavioral data `df_clean`.
# Ensure data types are consistent for a reliable merge.
# The `df_clean` DataFrame has float types for these columns from a previous step.
first_movement_times_df['subject_id'] = first_movement_times_df['subject_id'].astype(float)
first_movement_times_df['block'] = first_movement_times_df['block'].astype(float)
first_movement_times_df['trial_nr'] = first_movement_times_df['trial_nr'].astype(float)

# Merge to get the 'response' column from the behavioral data.
merged_movement_df = pd.merge(
    first_movement_times_df,
    df_clean[['subject_id', 'block', 'trial_nr', 'response']],  # Only need the response column
    on=['subject_id', 'block', 'trial_nr'],
    how='left'  # Use left merge to keep all movement trials
)

# Create the filtered DataFrame, excluding trials where the response was 5.
initial_count = len(merged_movement_df)
# We also drop rows where the merge failed to find a response (just in case)
filtered_movement_df = merged_movement_df.dropna(subset=['response'])
filtered_movement_df = filtered_movement_df[filtered_movement_df['response'] != 5].copy()
removed_count = initial_count - len(filtered_movement_df)
print(f"Removed {removed_count} trials where the response was 5 (or response was missing).")

# --- Visualize the distribution of these filtered first movement times ---
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_movement_df, x='first_movement_s', bins=50, kde=True)
plt.title("Distribution of Initial Movement Time (Trials with response '5' excluded)", fontsize=16)
plt.xlabel('Time of First Movement (seconds prior to trial end)', fontsize=12)
plt.ylabel('Number of Trials', fontsize=12)
sns.despine()
plt.show()


# ===================================================================
#                  INITIAL MOVEMENT DIRECTION ANALYSIS
# ===================================================================
print("\n--- Starting Initial Movement Direction Analysis ---")

# --- Step 1a: Calculate Data-Driven ROI Coordinates ---
print("Calculating data-driven ROI centers...")

# To find the click location, we must use the raw mouse data `df` which contains the x/y coordinates.
# We find the index of the last recorded data point for each trial, which corresponds to the click.
last_point_indices = df.loc[df.groupby(['subject_id', 'block', 'trial_nr'])['time'].idxmax()]

# Create a new DataFrame with just the final click coordinates for each trial
final_click_coords_df = last_point_indices[['subject_id', 'block', 'trial_nr', 'x', 'y']]

# Ensure data types are consistent for a reliable merge. We'll use integers.
# The raw `df` has subject_id as a string, so we convert it here.
final_click_coords_df['subject_id'] = final_click_coords_df['subject_id'].astype(int)
final_click_coords_df['block'] = final_click_coords_df['block'].astype(int)
final_click_coords_df['trial_nr'] = final_click_coords_df['trial_nr'].astype(int)

# Also ensure types in the behavioral DataFrame are correct before merging.
df_clean['subject_id'] = df_clean['subject_id'].astype(int)
df_clean['block'] = df_clean['block'].astype(int)
df_clean['trial_nr'] = df_clean['trial_nr'].astype(int)

# Merge the click coordinates with the trial-level behavioral data
click_analysis_df = pd.merge(
    df_clean,
    final_click_coords_df,
    on=['subject_id', 'block', 'trial_nr'],
    how='inner'  # Use inner merge to keep only trials with both behavioral and click data
)

# Filter for correct trials where the response matches the target digit
correct_clicks_df = click_analysis_df[click_analysis_df['response'] == click_analysis_df['TargetDigit']].copy()

# Now, calculate the mean x and y position for each clicked digit.
# This gives us our data-driven centers.
data_driven_rois = correct_clicks_df.groupby('TargetDigit').agg(
    x_dva=('x', 'mean'),
    y_dva=('y', 'mean')
).to_dict('index')

# --- Create a robust, final ROI map ---
# Start with a default grid as a fallback for any digits that were never correctly clicked.
numpad_locations_dva = {
    7: (-1, 1), 8: (0, 1), 9: (1, 1),
    4: (-1, 0), 5: (0, 0), 6: (1, 0),
    1: (-1, -1), 2: (0, -1), 3: (1, -1),
}

# Update the default map with the data-driven values. This overwrites the defaults
# for any digit we have data for, ensuring our map is as accurate as possible.
data_driven_map = {digit: (coords['x_dva'], coords['y_dva']) for digit, coords in data_driven_rois.items()}
numpad_locations_dva.update(data_driven_map)

print("Using data-driven ROI locations with default fallbacks.")
print("Final ROI map:", {k: tuple(round(vi, 2) for vi in v) for k, v in sorted(numpad_locations_dva.items())})

# The start point is always digit 5, which is at (0,0) in our map
start_point_vec = np.array(numpad_locations_dva[5])

# --- Step 2: Find the First Point of Movement for Each Trial ---
# The section "Analysis of First Movement Time" already found the rows where
# movement occurred. We'll build on that to get the coordinates.
# `moved_rows_df` is already defined in your script.

# Use idxmin() to find the index of the row with the minimum time for each trial group.
# This is the first recorded movement for that trial.
first_move_indices = moved_rows_df.loc[moved_rows_df.groupby(['subject_id', 'block', 'trial_nr'])['time'].idxmin()]

# Create a new DataFrame with just the initial movement coordinates.
initial_move_points_df = first_move_indices[['subject_id', 'block', 'trial_nr', 'x', 'y']].rename(
    columns={'x': 'initial_x_dva', 'y': 'initial_y_dva'})

# Merge these initial movement points back into the main behavioral DataFrame.
# We'll use `merged_df` as it contains all trial information before filtering.
# First, transform subject_id column to equal data types in both DataFrames.
df_clean['subject_id'] = df_clean['subject_id'].astype(float)
initial_move_points_df['subject_id'] = initial_move_points_df['subject_id'].astype(float)
# Choose only singleton present trials in merged_df
df_clean = df_clean.query("SingletonPresent==1")
analysis_df = pd.merge(
    df_clean,
    initial_move_points_df,
    on=['subject_id', 'block', 'trial_nr'],
    how='left'  # Use left merge to keep all trials
)

# Drop trials where no movement was detected (initial coordinates will be NaN)
analysis_df.dropna(subset=['initial_x_dva', 'initial_y_dva'], inplace=True)
print(f"Found initial movement data for {len(analysis_df)} trials.")

# --- Exclude trials where any primary ROI is at the center (digit 5) ---
initial_trial_count = len(analysis_df)
print(f"Filtering out trials with a primary ROI at the center...")

# Define the columns that hold the digit locations for the primary ROIs.
# Make sure these column names match your DataFrame exactly.
roi_digit_columns = ['TargetDigit', 'SingletonDigit', 'Non-Singleton2Digit']

# Create a boolean mask to identify rows where any of these columns have the value 5.
# The .any(axis=1) checks across the row for any True value.
is_center_roi_mask = (analysis_df[roi_digit_columns] == 5).any(axis=1)

# Keep only the trials where the mask is False (i.e., no ROI was at the center).
# Using .copy() is good practice here to avoid SettingWithCopyWarning later.
analysis_df = analysis_df[~is_center_roi_mask].copy()

excluded_count = initial_trial_count - len(analysis_df)
print(f"Excluded {excluded_count} trials.")
print(f"Remaining trials for analysis: {len(analysis_df)}")

# --- Step 3 & 4: Calculate Vectors, Angles, and Classify Movement ---
# Apply the classification function to each row of the DataFrame
analysis_df['initial_movement_direction'] = analysis_df.apply(
    classify_initial_movement, axis=1, locations_map=numpad_locations_dva, start_vec=start_point_vec
)

# --- Diagnostic Step: Investigate the peak near the response time ---
print("\n--- Investigating the sharp peak in initial movement times ---")

# Based on your observation, the peak occurs around -0.2s before the response.
# Let's define a narrow window around this peak to isolate these specific trials.
peak_window_start = -0.4
print(f"Isolating trials where the first movement was detected between {peak_window_start}s (relative to response).")

# Get the trials that contribute to this peak
peak_trials_df = first_movement_times_df[
    (first_movement_times_df['first_movement_s'] > peak_window_start)
]

print(f"\nFound {len(peak_trials_df)} trials within this peak window.")

if not peak_trials_df.empty:
    # To understand these trials, let's look at their properties from the main behavioral dataframe.
    # We need to merge them back with `df_clean` which has the RT and response info.
    # Ensure dtypes match for merging.
    peak_trials_df['subject_id'] = peak_trials_df['subject_id'].astype(int)
    peak_trials_df['block'] = peak_trials_df['block'].astype(int)
    peak_trials_df['trial_nr'] = peak_trials_df['trial_nr'].astype(int)

    # Merge to get full trial info
    peak_details_df = pd.merge(
        df_clean,
        peak_trials_df,
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'  # We only want the trials that are in both dataframes
    )

    print("\n--- Characteristics of trials in the peak ---")
    print("Distribution of their Reaction Times (rt):")
    print(peak_details_df['rt'].describe())

    print("\nDistribution of their Response Correctness (select_target):")
    print(peak_details_df['select_target'].value_counts(normalize=True, dropna=False))

    print("\n(For comparison, the RT distribution of ALL trials in df_clean):")
    print(df_clean['rt'].describe())

    # --- Visualize a few example trajectories from the peak ---
    print("\n--- Visualizing a few example trajectories from the peak ---")
    # We will use the `analysis_df` as it has all the necessary columns for plotting
    # and has already been merged with initial movement data.
    peak_viz_df = pd.merge(
        analysis_df,
        peak_trials_df[['subject_id', 'block', 'trial_nr']], # Just use the identifiers to filter
        on=['subject_id', 'block', 'trial_nr'],
        how='inner'
    )

    if not peak_viz_df.empty:
        # Take up to 3 random trials from the peak to visualize
        num_to_plot = min(len(peak_viz_df), 3)
        print(f"Plotting {num_to_plot} random example trajectories...")
        for i, trial_to_plot in peak_viz_df.sample(n=num_to_plot).iterrows():
            print(f"Plotting sub-{int(trial_to_plot['subject_id'])}, block-{int(trial_to_plot['block'])}, trial-{int(trial_to_plot['trial_nr'])}")
            visualize_full_trajectory(trial_to_plot, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)
    else:
        print("Could not find matching trials in `analysis_df` to visualize.")

else:
    print("No trials found in the specified peak window.")


# ===================================================================
#       SAVE CLASSIFICATION RESULTS FOR EXTERNAL USE
# ===================================================================
print("\n--- Saving Initial Movement Classifications for external use ---")
# Define a dedicated output directory for this script's results
output_dir_cursor = f"{get_data_path()}concatenated\\"

# Select only the essential columns for saving
output_df = analysis_df[['subject_id', 'block', 'trial_nr', 'initial_movement_direction']].copy()

# Ensure data types are standard integers for compatibility
output_df['subject_id'] = output_df['subject_id'].astype(int)
output_df['block'] = output_df['block'].astype(int)
output_df['trial_nr'] = output_df['trial_nr'].astype(int)

# Define the output path and save the file
output_filepath = f"{output_dir_cursor}initial_movement_classifications.csv"
output_df.to_csv(output_filepath, index=False)
print(f"Saved classifications for {len(output_df)} trials to: {output_filepath}")

# ===================================================================
#       STEP 5: VISUALIZE INITIAL MOVEMENT DIRECTION BY BLOCK
# ===================================================================
print("\n--- Visualizing Initial Movement Direction by Block ---")

# --- Step 5a: Calculate counts per subject, per block ---
# We need to get the number of trials classified into each direction for each subject and each block.
# This gives us the subject-level data needed for plotting with error bars.
subject_block_counts = analysis_df.groupby(
    ['subject_id', 'block', 'initial_movement_direction']
).size().reset_index(name='count')

# --- Step 5b: Normalize the 'other' category ---
# As requested, we divide the count for the 'other' category by 6.
# This gives an "average count per available location" to make it comparable.
is_other_mask = subject_block_counts['initial_movement_direction'] == 'other'
# Transform 'count' column to int
subject_block_counts["count"] = subject_block_counts["count"].astype(float)
subject_block_counts.loc[is_other_mask, 'count'] /= 6.0
# Calculate total trials per subject/block to use as a denominator
total_counts = subject_block_counts.groupby(['subject_id', 'block'])['count'].transform('sum')
subject_block_counts['percentage'] = (subject_block_counts['count'] / total_counts) * 100

# --- Step 5c: Create the Bar Plot with Hue for Blocks ---
# Now we can use seaborn's barplot, which will automatically aggregate the data
# (calculating the mean count across subjects) and show the 95% confidence interval as error bars.

plt.figure(figsize=(14, 8))
plot_order = ['target', 'distractor', 'control', 'other']

# The `hue_order` can be sorted to ensure the legend is in a logical sequence.
hue_order = sorted(analysis_df['block'].unique().astype(int))

sns.barplot(
    data=subject_block_counts,
    x='initial_movement_direction',
    y='percentage',
    hue='block',
    order=plot_order,
    hue_order=hue_order,
    palette='viridis',  # A sequential palette is great for ordered data like blocks
    errorbar='ci'    # 'ci' for 95% confidence interval is the default
)

plt.title('Initial Movement Direction Across Blocks', fontsize=16)
plt.xlabel('Classified Initial Movement Direction', fontsize=12)
plt.ylabel('Mean Percentage of Trials per Block', fontsize=12)
plt.legend(title='Block', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
sns.despine()
plt.show()

# ===================================================================
#       ANALYSIS: RUNNING AVERAGE OF SUPPRESSION EFFECT OVER TRIALS
# ===================================================================
print("\n--- Starting Running Average Analysis (Revised Logic) ---")

# --- Step 1: Calculate a Per-Trial Suppression Score ---

# First, ensure the DataFrame is sorted correctly. This is crucial for any time-series analysis.
analysis_df.sort_values(['subject_id', 'block', 'trial_nr'], inplace=True)

# Create a single score for each trial based on the initial movement direction:
#  +1 if the movement was toward the control item (suppression)
#  -1 if the movement was toward the distractor item (capture)
#   0 otherwise (e.g., toward target or other)
conditions = [
    analysis_df['initial_movement_direction'] == 'control',
    analysis_df['initial_movement_direction'] == 'distractor'
]
choices = [1, -1]
analysis_df['per_trial_effect'] = np.select(conditions, choices, default=0)

# --- Step 2: Apply a Rolling Average to the Per-Trial Score ---

# For each subject, calculate the rolling mean of the per-trial effect score.
# The result is a smooth curve representing the suppression effect over time.
# We multiply by 100 to convert the proportion into a percentage.
#
# CRITICAL FIX: We use `min_periods=1`. This ensures that we get a value even
# at the very beginning and end of the experiment where the window is not full.
# Without this, we would lose the data from the crucial initial trials.
analysis_df['suppression_effect'] = analysis_df.groupby('subject_id')['per_trial_effect'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=False, min_periods=1).mean()
) * 100

# Create a sequential trial index across the entire experiment for each subject.
analysis_df['singleton_trial_idx'] = analysis_df.groupby('subject_id').cumcount()

# --- Create the Final Plot ---

plt.figure(figsize=(12, 7))
# Plot the mean suppression effect
sns.lineplot(data=analysis_df, x=analysis_df["singleton_trial_idx"], y=analysis_df["suppression_effect"])

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No Effect (Capture = Suppression)')

# Formatting the plot
plt.title(f'Running Average of Motor Suppression Effect, Window Size = {WINDOW_SIZE}', fontsize=16)
plt.xlabel('Trial Number (Singleton Present)', fontsize=12)
plt.ylabel('Suppression Effect (% Control − % Distractor)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine()

# Limit the x-axis if there are very few subjects in the last trials, which can make the CI explode.
trial_counts = analysis_df['singleton_trial_idx'].value_counts()
last_reliable_trial = trial_counts[trial_counts >= 3].index.max()
if pd.notna(last_reliable_trial):
    plt.xlim(0, last_reliable_trial)

plt.show()

# ===================================================================
#       STATISTICAL ANALYSIS: MODELING THE LEARNING EFFECT
# ===================================================================
print("\n--- Modeling the Learning Effect Over Trials ---")

# We will fit a logarithmic model to each subject's data to quantify the learning rate.
# Model: suppression_effect = intercept + slope * log(trial_number)
# A positive slope indicates that suppression increases over time.

# --- Step 1: Fit a logarithmic model for each subject ---
subject_fits = []
# Use the 'per_trial_effect' which is the raw score (-1, 0, or 1) for robust fitting.
# Add 1 to trial index to avoid log(0).
analysis_df['log_trial_idx'] = np.log(analysis_df['singleton_trial_idx'] + 1)

for subject, group in analysis_df.groupby('subject_id'):
    # Ensure there are enough data points to fit a line
    if len(group) < 3:
        continue

    # Perform linear regression on the log-transformed trial index
    slope, intercept, r_value, p_value, std_err = linregress(
        group['log_trial_idx'],
        group['per_trial_effect']
    )
    subject_fits.append({'subject_id': subject, 'slope': slope, 'intercept': intercept})

fit_results_df = pd.DataFrame(subject_fits)

print(f"Successfully fitted logarithmic models for {len(fit_results_df)} subjects.")

# --- Step 2: Perform a one-sample t-test on the slopes ---
# We test if the mean of the subjects' slopes is significantly different from zero.

print("\n--- One-Sample T-test on Learning Rate (Slopes) ---")
ttest_result = pg.ttest(fit_results_df['slope'], 0, confidence=0.95)
print(ttest_result.round(4))

# --- Step 3: Visualize the model fit ---
# We plot the running average data and overlay the curve from our model.
avg_intercept = fit_results_df['intercept'].mean()
avg_slope = fit_results_df['slope'].mean()

# Define the x-axis range for the fitted curve. We use the `last_reliable_trial`
# calculated in the previous plot to ensure the range is consistent.
if pd.notna(last_reliable_trial):
    x_trials = np.arange(0, last_reliable_trial + 1)
else:
    # Fallback if last_reliable_trial is not available for any reason
    x_trials = np.arange(0, analysis_df['singleton_trial_idx'].max() + 1)

log_x_trials = np.log(x_trials + 1)

# Calculate the predicted y-values from the average model parameters
# Multiply by 100 to match the scale of the running average plot
predicted_y = (avg_intercept + avg_slope * log_x_trials) * 100

# Create a new plot showing both the running average and the logarithmic fit
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the running average with its confidence interval using seaborn
sns.lineplot(
    data=analysis_df,
    x='singleton_trial_idx',
    y='suppression_effect',
    ax=ax,
    color='gray',
    label='Mean Suppression Effect (Running Avg)'
)

# Overlay the fitted logarithmic curve
ax.plot(x_trials, predicted_y, color='blue', linestyle='--', linewidth=2.5, label=f'Logarithmic Fit (Mean Slope={avg_slope:.3f})')

# Formatting
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_title('Running Average and Logarithmic Model of Suppression Effect', fontsize=16)
ax.set_xlabel('Trial Number (Singleton Present)', fontsize=12)
ax.set_ylabel('Suppression Effect (% Control − % Distractor)', fontsize=12)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine(ax=ax)

# Apply the same x-axis limit as the previous plot
if pd.notna(last_reliable_trial):
    ax.set_xlim(0, last_reliable_trial)

plt.show()

# ===================================================================
#       ANALYSIS: BLOCK-LEVEL SUPPRESSION EFFECT BAR PLOT
# ===================================================================
print("\n--- Creating Block-Level Suppression Effect Bar Plot ---")

# --- Step 1: Calculate a Per-Trial Suppression Score ---
# This score is +1 for a movement to control (suppression), -1 for a movement
# to the distractor (capture), and 0 otherwise. This logic is robust and clear.
conditions = [
    analysis_df['initial_movement_direction'] == 'control',
    analysis_df['initial_movement_direction'] == 'distractor'
]
choices = [1, -1]
analysis_df['per_trial_effect'] = np.select(conditions, choices, default=0)

# --- Step 2: Calculate Mean Suppression Effect per Subject, per Block ---
# We group by subject and block and take the mean of our per-trial score.
# This gives us a single, stable suppression value for each subject in each block.
block_level_effect = analysis_df.groupby(['subject_id', 'block'])['per_trial_effect'].mean().reset_index()

# Multiply by 100 to get a percentage-like value for the y-axis.
block_level_effect['suppression_effect'] = block_level_effect['per_trial_effect'] * 100

# --- Step 3: Create the Final Bar Plot ---
# Seaborn's barplot will automatically calculate the mean suppression effect across
# all subjects for each block and display the 95% confidence interval as error bars.

plt.figure(figsize=(12, 7))

sns.barplot(
    data=block_level_effect,
    x='block',
    y='suppression_effect',
    palette='magma',
    errorbar='ci'  # 'ci' for 95% confidence interval is the default
)

# Add a horizontal line at y=0 for reference, making capture vs. suppression easy to see
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No Effect (Capture = Suppression)')

# Formatting the plot
plt.title('Suppression Effect Across Blocks', fontsize=16)
plt.xlabel('Block Number', fontsize=12)
plt.ylabel('Suppression Effect (% Control − % Distractor)', fontsize=12)
plt.legend()
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
sns.despine()

plt.show()

# ===================================================================
#           EXAMPLE USAGE OF THE TRIAL PLOTTING FUNCTION
# ===================================================================
example = 7
# --- Find a specific trial to plot ---
distractor_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'distractor'
].iloc[example]
# Call the function with the selected trial data
plot_trial_vectors(distractor_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the target ---
target_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'target'
].iloc[example]
plot_trial_vectors(target_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the other ---
control_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'control'
    ].iloc[example]  # Using index 5 for variety
plot_trial_vectors(control_trial_to_plot, numpad_locations_dva)

other_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'other'
    ].iloc[example]  # Using index 5 for variety
plot_trial_vectors(other_trial_to_plot, numpad_locations_dva)


# ===================================================================
#           VISUALIZE FULL TRAJECTORIES WITH RESPONSE TIME
# ===================================================================
distractor_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'distractor'].iloc[example]
visualize_full_trajectory(distractor_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

# Find a trial where the initial movement was towards the target
target_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'target'].iloc[example]
visualize_full_trajectory(target_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

control_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'control'].iloc[example]
visualize_full_trajectory(control_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

other_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'other'].iloc[example]
visualize_full_trajectory(other_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

# Call the new diagnostic function
#visualize_absolute_trajectory(control_trial_for_viz, df, numpad_locations_dva, target_hz=RESAMP_FREQ)

# Use the exact same trial that shows the doubling effect
#distractor_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'distractor'].iloc[10]

# Run the absolute plot and carefully check the coordinates printed in the legend
#print("Checking the absolute start and end points for the trial...")
#visualize_absolute_trajectory(
#    distractor_trial_for_viz,
#    df,
#    numpad_locations_dva,
#    target_hz=RESAMP_FREQ)