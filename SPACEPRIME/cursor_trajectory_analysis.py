import os
from scipy.ndimage import gaussian_filter
from utils import *
import seaborn as sns
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers

plt.ion()




# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
sub_ids = subject_ids
# load data from children
# Load data from children and add subject_id column
df_list = list()
for subject in subjects:
    if int(subject.split("-")[1]) in sub_ids:
        filepath = glob.glob(f"{get_data_path()}sourcedata/raw/{subject}/beh/{subject}*mouse_data.csv")[0]
        temp_df = pd.read_csv(filepath)
        temp_df['subject_id'] = subject.split("-")[1]  # Extract subject ID and add as a column
        df_list.append(temp_df)# get rows per trial
df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames
# Corrected line:
df['block'] = df.groupby('subject_id')['trial_nr'].transform(
    lambda x: ((x == 0) & (x.shift(1) != 0)).cumsum() - 1)
rows_per_trial = df.groupby(['trial_nr', "block", 'subject_id']).size().reset_index(name='n_rows')
# define some setup params
width = 1920
height = 1080
dg_va = 2
viewing_distance_cm = 70
# Calculate scaled pixel coordinates and add them to the DataFrame
# This uses the same calculation you have for x and y, but stores them in df.
df['x_pixels'] = df["x"] * degrees_va_to_pixels(
    degrees=dg_va,
    screen_pixels=width,
    screen_size_cm=40,  # screen_width_cm for x coordinates
    viewing_distance_cm=viewing_distance_cm
)
df['y_pixels'] = df["y"] * degrees_va_to_pixels(
    degrees=dg_va,
    screen_pixels=height,
    screen_size_cm=30,  # screen_height_cm for y coordinates
    viewing_distance_cm=viewing_distance_cm
)
data = np.array([df["x_pixels"], df["y_pixels"]]).transpose()
# define height and width of the screen
# get the canvas
canvas = np.vstack((data[:, 0], data[:, 1]))  # shape (2, n_samples)
_range = [[0, height], [0, width]]
bins_x, bins_y = width, height
extent = [0, width, height, 0]
hist, _, _ = np.histogram2d(canvas[1, :], canvas[0, :], bins=(bins_y, bins_x), range=_range)
sfreq = rows_per_trial["n_rows"].mean()/3  # divide by 3 because 1 trial is 3 seconds long
sns.displot(x=rows_per_trial["n_rows"]/3)
sigma = 25
alpha = 1.0
hist /= sfreq
# Calculate the center of the screen
center_x = width / 2
center_y = height / 2
# Calculate the center of your data (you might want to adjust this based on your specific needs)
data_center_x = 0  # Or use a fixed value if you know the center
data_center_y = 0
# Shift the data so that the center of the data is at the center of the screen
x_shifted = df["x_pixels"] - data_center_x + center_x
y_shifted = df["y_pixels"] - data_center_y + center_y
# Recalculate the histogram with the shifted data
hist, _, _ = np.histogram2d(y_shifted, x_shifted, bins=(height, width), range=[[0, height], [0, width]]) # Note the change here as well
hist = gaussian_filter(hist, sigma=sigma)
extent = [0, width, height, 0]
plt.figure()
plt.imshow(hist, extent=extent, origin='upper', aspect='auto', alpha=alpha)
plt.gca().invert_yaxis()  # Invert the y-axis
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.title("Cursor dwell time [s]")
# Path length analysis

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
df_clean = merged_df[merged_df["phase"]!=2]

df_clean = remove_outliers(df_clean, column_name="rt", threshold=2)

# --- Analysis of First Movement Time ---

# To make the detection robust, we'll define a small threshold. A movement is
# registered when the cursor's distance from the center (0,0) exceeds this.
# This avoids detecting tiny jitters from the mouse hardware as a real movement.
# The coordinates are normalized, so the threshold should be small.
movement_threshold = 0.05

# Find all data points where the cursor has moved away from the center.
# We check if the absolute value of x OR y is greater than our threshold.
moved_rows_df = df[(df['x'].abs() > movement_threshold) | (df['y'].abs() > movement_threshold)].copy()

# For each trial, find the timestamp of the *first* recorded movement.
# We group by trial identifiers and find the minimum timestamp within each group.
# Using .min() is efficient for this task.
first_movement_times_df = moved_rows_df.groupby(['subject_id', 'block', 'trial_nr'])['time'].min().reset_index()

# Rename the column for clarity
first_movement_times_df.rename(columns={'time': 'first_movement_s'}, inplace=True)

# Now, let's visualize the distribution of these first movement times.
plt.figure(figsize=(10, 6))
sns.histplot(data=first_movement_times_df, x='first_movement_s', bins=50, kde=True)
plt.title('Distribution of Initial Movement Time', fontsize=16)
plt.xlabel('Time of First Movement (seconds from trial start)', fontsize=12)
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
plot_order = ['target', 'distractor', 'neutral', 'other']

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
#  +1 if the movement was toward the neutral item (suppression)
#  -1 if the movement was toward the distractor item (capture)
#   0 otherwise (e.g., toward target or other)
conditions = [
    analysis_df['initial_movement_direction'] == 'neutral',
    analysis_df['initial_movement_direction'] == 'distractor'
]
choices = [1, -1]
analysis_df['per_trial_effect'] = np.select(conditions, choices, default=0)

# --- Step 2: Apply a Rolling Average to the Per-Trial Score ---

# Define the size of the rolling window (11 trials, as per the paper).
window_size = 51

# For each subject, calculate the rolling mean of the per-trial effect score.
# The result is a smooth curve representing the suppression effect over time.
# We multiply by 100 to convert the proportion into a percentage.
#
# CRITICAL FIX: We use `min_periods=1`. This ensures that we get a value even
# at the very beginning and end of the experiment where the window is not full.
# Without this, we would lose the data from the crucial initial trials.
analysis_df['suppression_effect'] = analysis_df.groupby('subject_id')['per_trial_effect'].transform(
    lambda x: x.rolling(window_size, center=False, min_periods=1).mean()
) * 100

# --- Step 3: Aggregate Data for Plotting (Using Within-Subject CI Method) ---

# Create a sequential trial index across the entire experiment for each subject.
analysis_df['singleton_trial_idx'] = analysis_df.groupby('subject_id').cumcount()

# To calculate a within-subject CI, we first need to remove the stable between-subject variance.
# We'll follow the Cousineau-Morey method.

# 1. Pivot the data so each row is a subject and each column is a trial.
pivoted_data = analysis_df.pivot_table(
    index='subject_id',
    columns='singleton_trial_idx',
    values='suppression_effect'
)

# 2. Calculate the mean score for each subject (their average suppression effect).
subject_means = pivoted_data.mean(axis=1)

# 3. Calculate the grand mean across all subjects and all trials.
grand_mean = pivoted_data.stack().mean()

# 4. Normalize each data point: original_value - subject_mean + grand_mean
# This removes the subject's overall bias but preserves the group average.
normalized_data = pivoted_data.subtract(subject_means, axis=0).add(grand_mean)

# 5. Calculate the mean and SEM on this *normalized* data.
# The mean of the normalized data is identical to the mean of the original data.
plot_data_mean = normalized_data.mean(axis=0)
plot_data_sem = normalized_data.sem(axis=0)

# 6. Apply the Morey (2008) correction factor for bias in repeated-measures variance.
M = len(pivoted_data.columns) # Number of conditions (time points)
correction_factor = np.sqrt(M / (M - 1))
plot_data_sem_corrected = plot_data_sem * correction_factor

# 7. Create the final DataFrame for plotting.
plot_data = pd.DataFrame({
    'mean': plot_data_mean,
    'sem': plot_data_sem_corrected
}).reset_index()

# Calculate the 95% confidence interval from the corrected SEM.
plot_data['ci_lower'] = plot_data['mean'] - 1.96 * plot_data['sem']
plot_data['ci_upper'] = plot_data['mean'] + 1.96 * plot_data['sem']

# --- Step 4: Create the Final Plot ---

plt.figure(figsize=(12, 7))

# Plot the mean suppression effect
plt.plot(plot_data['singleton_trial_idx'], plot_data['mean'], label='Mean Suppression Effect', color='black', linewidth=2)

# Add the shaded confidence interval region
plt.fill_between(
    plot_data['singleton_trial_idx'],
    plot_data['ci_lower'],
    plot_data['ci_upper'],
    color='gray',
    alpha=0.3,
    label='95% Confidence Interval'
)

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No Effect (Capture = Suppression)')

# Formatting the plot
plt.title('Running Average of Oculomotor Suppression Effect', fontsize=16)
plt.xlabel('Trial Number (Singleton Present)', fontsize=12)
plt.ylabel('Suppression Effect (% Neutral − % Distractor)', fontsize=12)
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
#       ANALYSIS: BLOCK-LEVEL SUPPRESSION EFFECT BAR PLOT
# ===================================================================
print("\n--- Creating Block-Level Suppression Effect Bar Plot ---")

# --- Step 1: Calculate a Per-Trial Suppression Score ---
# This score is +1 for a movement to neutral (suppression), -1 for a movement
# to the distractor (capture), and 0 otherwise. This logic is robust and clear.
conditions = [
    analysis_df['initial_movement_direction'] == 'neutral',
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
plt.ylabel('Suppression Effect (% Neutral − % Distractor)', fontsize=12)
plt.legend()
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
sns.despine()

plt.show()

# ===================================================================
#           EXAMPLE USAGE OF THE TRIAL PLOTTING FUNCTION
# ===================================================================

# --- Find a specific trial to plot ---
distractor_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'distractor'
].iloc[10]
# Call the function with the selected trial data
plot_trial_vectors(distractor_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the target ---
target_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'target'
].iloc[10]
plot_trial_vectors(target_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the other ---
neutral_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'neutral'
    ].iloc[10]  # Using index 5 for variety
plot_trial_vectors(neutral_trial_to_plot, numpad_locations_dva)

other_trial_to_plot = analysis_df[
    analysis_df['initial_movement_direction'] == 'other'
    ].iloc[10]  # Using index 5 for variety
plot_trial_vectors(other_trial_to_plot, numpad_locations_dva)


# ===================================================================
#           VISUALIZE FULL TRAJECTORIES WITH RESPONSE TIME
# ===================================================================
resamp_freq = 60
distractor_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'distractor'].iloc[10]
visualize_full_trajectory(distractor_trial_for_viz, df, movement_threshold, target_hz=resamp_freq)

# Find a trial where the initial movement was towards the target
target_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'target'].iloc[10]
visualize_full_trajectory(target_trial_for_viz, df, movement_threshold, target_hz=resamp_freq)

target_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'neutral'].iloc[10]
visualize_full_trajectory(target_trial_for_viz, df, movement_threshold, target_hz=resamp_freq)

target_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'other'].iloc[10]
visualize_full_trajectory(target_trial_for_viz, df, movement_threshold, target_hz=resamp_freq)

