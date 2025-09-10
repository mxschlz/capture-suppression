import os
from scipy.ndimage import gaussian_filter
from utils import *
import seaborn as sns
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
from scipy.stats import ttest_rel, spearmanr
import pingouin as pg
import SPACEPRIME

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
FILTER_PHASE = 2
OUTLIER_THRESHOLD = 2
WINDOW_SIZE = 31
RESAMP_FREQ = 100
SUB_BLOCKS_PER_BLOCK = 2

# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"

# --- Subject Inclusion/Exclusion ---
# Get the full list of subjects from the project file
all_subject_ids = subject_ids

# Define a list of subjects to exclude from all analyses in this script
subjects_to_exclude = []
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

# resample data
df = resample_all_trajectories(df, RESAMP_FREQ)

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
sfreq = rows_per_trial["n_rows"].mean() / 1.75 # 1.75 seconds per response duration on average
print(f"Average sampling frequency: {sfreq:.2f} Hz")

# Normalize the histogram counts to get seconds.
# We add a small epsilon to avoid division by zero if sfreq is 0.
hist /= (sfreq + 1e-9)

# --- 4. Smooth and plot the heatmap ---
hist_smoothed = gaussian_filter(hist, sigma=SIGMA)

extent = [0, WIDTH, 0, HEIGHT] # Use 0, height for y-axis to match a standard Cartesian plot
plt.figure(figsize=(10, 8))
plt.imshow(hist_smoothed, extent=extent, origin='lower', aspect='auto', cmap='inferno')
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.title("Cursor Dwell Time")
#cbar = plt.colorbar()
#cbar.set_label("Dwell Time (s)")
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
    7: (-0.6, -0.6), 8: (0, -0.6), 9: (0.6, -0.6),
    4: (-0.6, 0), 5: (0, 0), 6: (0.6, 0),
    1: (-0.6, 0.6), 2: (0, 0.6), 3: (0.6, 0.6),
}

# Update the default map with the data-driven values. This overwrites the defaults
# for any digit we have data for, ensuring our map is as accurate as possible.
#data_driven_map = {digit: (coords['x_dva'], coords['y_dva']) for digit, coords in data_driven_rois.items()}
#numpad_locations_dva.update(data_driven_map)

print("Using data-driven ROI locations with default fallbacks.")
print("Final ROI map:", {k: tuple(round(vi, 2) for vi in v) for k, v in sorted(numpad_locations_dva.items())})

# ===================================================================
#      CONTINUOUS MOVEMENT ANALYSIS (CAPTURE SCORE)
# ===================================================================
print("\n--- Starting Continuous Movement Analysis (Full Trajectory) ---")
print("This method calculates a capture score for both singleton-present and singleton-absent trials.")

# --- Step 1: Calculate the Average Vector for Each Trial's Full Trajectory ---
print("Calculating the average vector for each trial's full trajectory...")
avg_full_trajectory_vectors_df = df.groupby(['subject_id', 'block', 'trial_nr']).agg(
    avg_x_dva=('x', 'mean'),
    avg_y_dva=('y', 'mean')
).reset_index()

# --- Step 2: Merge Average Vectors into the Main Analysis DataFrame ---
# We use `df_clean` as our base, but this time we DO NOT filter by SingletonPresent.
analysis_df_temp = df_clean.copy()  # Keep all trials
# TODO: this is temporary for coherence with the rest of the plots
# analysis_df_temp = analysis_df_temp.query("SingletonPresent==1")


# Ensure data types are consistent for merging.
analysis_df_temp['subject_id'] = analysis_df_temp['subject_id'].astype(int)
analysis_df_temp['block'] = analysis_df_temp['block'].astype(int)
analysis_df_temp['trial_nr'] = analysis_df_temp['trial_nr'].astype(int)
avg_full_trajectory_vectors_df['subject_id'] = avg_full_trajectory_vectors_df['subject_id'].astype(int)
avg_full_trajectory_vectors_df['block'] = avg_full_trajectory_vectors_df['block'].astype(int)
avg_full_trajectory_vectors_df['trial_nr'] = avg_full_trajectory_vectors_df['trial_nr'].astype(int)

# Use a left merge to add the average trajectory vectors to our trial data.
analysis_df_temp = pd.merge(
    analysis_df_temp,
    avg_full_trajectory_vectors_df,
    on=['subject_id', 'block', 'trial_nr'],
    how='left'
)
analysis_df_temp.dropna(subset=['avg_x_dva', 'avg_y_dva'], inplace=True)
print(f"Found trajectory data for {len(analysis_df_temp)} trials.")

# --- Add a readable column for Priming condition ---
# We map the numeric Priming values to meaningful labels for plotting.
priming_map = {1: 'Positive', 0: 'No', -1: 'Negative'}
analysis_df_temp['PrimingCondition'] = analysis_df_temp['Priming'].map(priming_map)

# Drop any trials that don't have a valid priming condition, just in case.
analysis_df_temp.dropna(subset=['PrimingCondition'], inplace=True)

print(f"Successfully calculated capture score for {len(analysis_df_temp)} trials.")
print("Analysis will be divided by the following priming conditions:")
print(analysis_df_temp['PrimingCondition'].value_counts())
analysis_df_temp['Condition'] = np.where(analysis_df_temp['SingletonPresent'] == 1, 'Distractor Present', 'Distractor Absent')
print(f"Successfully calculated capture score for {len(analysis_df_temp)} trials.")

# --- Step 3: Apply the new, comprehensive projection calculation ---
print("Applying comprehensive trajectory projection calculation...")
# This will add new columns like 'proj_target', 'proj_distractor', etc. to the DataFrame
projection_scores_df = analysis_df_temp.apply(
    calculate_trajectory_projections, axis=1, locations_map=numpad_locations_dva
)

analysis_df = pd.concat([analysis_df_temp, projection_scores_df], axis=1)
analysis_df.dropna(subset=['proj_target'], inplace=True) # Drop trials where score couldn't be computed

# --- Step 4: Calculate derived, meaningful scores from the projections ---
print("Calculating new, improved scores from projections...")

# A. The new "Target Focus Score"
# This score measures how strongly movement was directed to the target relative
# to the strongest competing non-target item on that trial.
# A high positive score means excellent focus on the target.
control_proj = analysis_df['proj_control_max']
analysis_df['target_towardness'] = analysis_df['proj_target']
analysis_df['distractor_towardness'] = analysis_df['proj_distractor']
analysis_df['target_distractor_towardness_diff'] = analysis_df['target_towardness'] - analysis_df['distractor_towardness']

# Calculate the length of the ideal vector to the target and distractor for each trial
analysis_df['target_vec_length'] = analysis_df['TargetDigit'].apply(get_vector_length, locations_map=numpad_locations_dva)
analysis_df['distractor_vec_length'] = analysis_df['SingletonDigit'].apply(get_vector_length, locations_map=numpad_locations_dva)

# --- Create the Standardized Scores ---
# The standardized score is the projection value divided by the length of the ideal vector.
# This converts the score from an absolute distance (in dva) to a relative proportion.
analysis_df['target_towardness_std'] = analysis_df['target_towardness'] / (analysis_df['target_vec_length'])
analysis_df['distractor_towardness_std'] = analysis_df['distractor_towardness'] / (analysis_df['distractor_vec_length'])

# The difference score is now a comparison of these relative proportions
analysis_df['target_distractor_towardness_diff_std'] = analysis_df['target_towardness_std'] - analysis_df['distractor_towardness_std']

# ===================================================================
#      NEW: FILTER OUT NOISY TRAJECTORIES
# ===================================================================
print("\n--- Filtering out noisy trajectories that go beyond the response box ---")

# --- Step 1: Define the boundary ---
# The numpad corners are at (+/-1, +/-1), so the max distance from center is sqrt(2) ~= 1.41 dva.
# We'll set a generous boundary to only catch truly erratic movements.
TRAJECTORY_BOUNDARY_DVA = 2
print(f"Defining boundary for valid trajectories at {TRAJECTORY_BOUNDARY_DVA} dva from the center.")

# --- Step 2: Identify trials that exceed the boundary ---
# We check the maximum absolute x and y coordinate for each trial in the raw trajectory data `df`.
max_coords_per_trial = df.groupby(['subject_id', 'block', 'trial_nr']).agg(
    max_abs_x=('x', lambda s: s.abs().max()),
    max_abs_y=('y', lambda s: s.abs().max())
).reset_index()

# Create a mask to find trials where any point went out of bounds.
noisy_mask = (max_coords_per_trial['max_abs_x'] > TRAJECTORY_BOUNDARY_DVA) | \
             (max_coords_per_trial['max_abs_y'] > TRAJECTORY_BOUNDARY_DVA)
noisy_trials_df = max_coords_per_trial[noisy_mask]

# --- Step 3: Create a set of identifiers for efficient filtering ---
# Ensure data types match the `analysis_df` for comparison.
noisy_trials_df['subject_id'] = noisy_trials_df['subject_id'].astype(int)
noisy_trials_df['block'] = noisy_trials_df['block'].astype(int)
noisy_trials_df['trial_nr'] = noisy_trials_df['trial_nr'].astype(int)

# A set of tuples is very fast for checking membership.
noisy_trial_identifiers = set(zip(
    noisy_trials_df['subject_id'],
    noisy_trials_df['block'],
    noisy_trials_df['trial_nr']
))

# --- Step 4: Filter the main analysis DataFrame ---
initial_count = len(analysis_df)

# Create a boolean mask by checking if each trial in analysis_df is in our noisy set.
# We keep the trials that are *not* in the set.
is_not_noisy_mask = [
    (row.subject_id, row.block, row.trial_nr) not in noisy_trial_identifiers
    for row in analysis_df.itertuples()
]

analysis_df = analysis_df[is_not_noisy_mask].copy()
removed_count = initial_count - len(analysis_df)
print(f"Removed {removed_count} noisy trials ({removed_count / initial_count:.2%}).")
print(f"Remaining trials for analysis: {len(analysis_df)}")


# ===================================================================
#       VISUALIZE DERIVED CAPTURE SCORES BY PRIMING & BLOCK
# ===================================================================
print("\n--- Visualizing Derived Capture Scores by Priming Condition and Block ---")

# Define the scores to plot and their user-friendly titles
scores_to_plot = {
    'target_towardness': 'Target towardness',
    'distractor_towardness': 'Distractor towardness',
    'target_distractor_towardness_diff': 'Target vs. Distractor Difference'
}

# Create a figure with 3 subplots in a row, sharing the y-axis for easy comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

# Define common plotting parameters for a consistent look
hue_order = ['Negative', 'No', 'Positive']
palette = {'Positive': 'mediumseagreen', 'No': 'gray', 'Negative': 'salmon'}

for i, (col_name, title) in enumerate(scores_to_plot.items()):
    ax = axes[i]
    sns.pointplot(
        data=analysis_df,
        x='block',
        y=col_name,
        hue='PrimingCondition',
        hue_order=hue_order,
        palette=palette,
        errorbar=('se', 2),
        join=True,
        ax=ax
    )
    # Add a reference line at 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Block Number', fontsize=12)

    # Only set the y-label for the first (leftmost) plot
    if i == 0:
        ax.set_ylabel('Capture Score', fontsize=12)
    else:
        ax.set_ylabel('')  # Hide y-label for other plots to avoid clutter

    ax.grid(True, which='major', axis='y', linestyle=':', linewidth=0.5)
    sns.despine(ax=ax)

    # Only show the legend on the last (rightmost) plot to avoid redundancy
    if i < len(scores_to_plot) - 1:
        ax.get_legend().remove()
    else:
        ax.legend(title='Priming Condition', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout() # Adjust layout to prevent title overlap

# Compare capture score effect sizes versus response time and accuracy effect sizes in Priming conditions
analysis_df_mean = analysis_df.groupby(["subject_id", "Priming"])[["rt", "select_target", "target_towardness_std",
                                                                   "target_towardness", "distractor_towardness",
                                                                   "target_distractor_towardness_diff",
                                                      "distractor_towardness_std",
                                                      "target_distractor_towardness_diff_std"]].mean().reset_index()

# ===================================================================
#       EFFECT SIZE COMPARISON: CAPTURE SCORE vs. RT vs. ACCURACY
# ===================================================================
print("\n--- Comparing Effect Sizes: Capture Score vs. RT & Accuracy ---")
print("Objective: Determine if the capture score is a more sensitive metric for priming effects.")

# Use the subject-level mean data calculated earlier
df_effects = analysis_df_mean.copy()

# Define the metrics we want to compare and their user-friendly names
metrics_to_compare = {
    'Response Time (s)': 'rt',
    'Accuracy (%)': 'select_target',
    'Towardness (Target) (dva)': 'target_towardness',
    'Towardness (Distractor) (dva)': 'distractor_towardness',
    'Towardness (Target-Distractor) (dva)': 'target_distractor_towardness_diff'
}

# Prepare a list to store the results
effect_size_results = []

# Isolate the data for each priming condition for easier comparison
data_pos = df_effects[df_effects['Priming'] == 1].set_index('subject_id')
data_neg = df_effects[df_effects['Priming'] == -1].set_index('subject_id')
data_no = df_effects[df_effects['Priming'] == 0].set_index('subject_id')

# Ensure all subjects are present in all conditions for a true paired test
common_subjects = data_pos.index.intersection(data_neg.index).intersection(data_no.index)
data_pos = data_pos.loc[common_subjects]
data_neg = data_neg.loc[common_subjects]
data_no = data_no.loc[common_subjects]

print(f"Found {len(common_subjects)} subjects with data in all three priming conditions for comparison.")

# Loop through each metric to calculate and store its effect sizes
for metric_name, col_name in metrics_to_compare.items():
    # --- (FIX) Convert data to numeric right before the test ---
    # This prevents the TypeError by ensuring the data is in a numeric format.
    # `errors='coerce'` will turn any non-numeric values into NaN.
    x_pos_numeric = pd.to_numeric(data_pos[col_name], errors='coerce')
    y_no_numeric_for_pos = pd.to_numeric(data_no[col_name], errors='coerce')

    x_neg_numeric = pd.to_numeric(data_neg[col_name], errors='coerce')
    y_no_numeric_for_neg = pd.to_numeric(data_no[col_name], errors='coerce')
    # -------------------------------------------------------------

    # --- Comparison 1: Positive Priming vs. No Priming ---
    ttest_pos_vs_no = pg.ttest(
        x=x_pos_numeric,
        y=y_no_numeric_for_pos,
        paired=True,
        correction=False
    )
    cohens_d_pos = ttest_pos_vs_no['cohen-d'].iloc[0]
    p_val_pos = ttest_pos_vs_no['p-val'].iloc[0]  # Extract the p-value
    effect_size_results.append({
        'Metric': metric_name,
        'Comparison': 'Positive vs. No Priming',
        "Cohen's d": cohens_d_pos,
        "p-val": p_val_pos  # Store the p-value
    })

    # --- Comparison 2: Negative Priming vs. No Priming ---
    ttest_neg_vs_no = pg.ttest(
        x=x_neg_numeric,
        y=y_no_numeric_for_neg,
        paired=True,
        correction=False
    )
    cohens_d_neg = ttest_neg_vs_no['cohen-d'].iloc[0]
    p_val_neg = ttest_neg_vs_no['p-val'].iloc[0] # Extract the p-value
    effect_size_results.append({
        'Metric': metric_name,
        'Comparison': 'Negative vs. No Priming',
        "Cohen's d": cohens_d_neg,
        "p-val": p_val_neg # Store the p-value
    })

    # --- Comparison 3: Positive Priming vs. Negative Priming ---
    ttest_pos_vs_neg = pg.ttest(
        x=x_pos_numeric,
        y=x_neg_numeric,  # Compare positive against negative
        paired=True,
        correction=False
    )
    cohens_d_pos_neg = ttest_pos_vs_neg['cohen-d'].iloc[0]
    p_val_pos_neg = ttest_pos_vs_neg['p-val'].iloc[0]
    effect_size_results.append({
        'Metric': metric_name,
        'Comparison': 'Positive vs. Negative Priming',
        "Cohen's d": cohens_d_pos_neg,
        "p-val": p_val_pos_neg
    })


# Convert results to a DataFrame for easy viewing
effect_size_df = pd.DataFrame(effect_size_results)

print("\n--- Calculated Effect Sizes (Cohen's d) and p-values ---")
print(effect_size_df)

# --- Visualize Performance Metrics with Effect Size Annotations ---
print("\n--- Visualizing Performance with Effect Size Annotations ---")

# Helper function to convert p-values to significance stars
def p_to_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'

fig, axes = plt.subplot_mosaic(mosaic="""
ab.
ecd
""", figsize=(18, 8))
axes = [ax for ax in list(axes.values())] # Flatten the 2D array of axes for easy iteration

# Define the order, labels, and a safe color palette for the x-axis
priming_order = [-1, 0, 1]
priming_labels = ['Negative', 'No', 'Positive']
palette_list = ['salmon', 'gray', 'mediumseagreen']

for i, (metric_name, col_name) in enumerate(metrics_to_compare.items()):
    ax = axes[i]

    # --- 1. Plot the main bar plot showing the performance metric ---
    sns.barplot(
        data=df_effects,
        x='Priming',
        y=col_name,
        order=priming_order,
        ax=ax,
        palette=palette_list,
        errorbar=('se', 2)
    )
    ax.set_title(metric_name, fontsize=14, pad=50) # Increased padding for annotations
    ax.set_xlabel("Priming Condition", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xticklabels(priming_labels)
    sns.despine(ax=ax)

    # --- 2. Add Effect Size and Significance Annotations ---
    # Get the heights of the bars for annotation placement
    bar_heights = df_effects.groupby('Priming')[col_name].mean()
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = y_range * 0.05  # 5% of the y-axis range for spacing

    # Comparison 1: Positive (1) vs. No (0)
    x1_pos, x2_pos = 2, 1  # Positions for Positive and No
    y1 = bar_heights.get(1, 0)
    y2 = bar_heights.get(0, 0)
    y_bracket_pos = max(y1, y2) + offset

    # Retrieve the stats
    pos_stats = effect_size_df[
        (effect_size_df['Metric'] == metric_name) &
        (effect_size_df['Comparison'] == 'Positive vs. No Priming')
    ].iloc[0]
    d_pos_val = pos_stats["Cohen's d"]
    p_pos_val = pos_stats["p-val"]
    p_pos_str = p_to_stars(p_pos_val)

    # Draw the annotation bracket and text
    ax.plot([x1_pos, x1_pos, x2_pos, x2_pos], [y_bracket_pos, y_bracket_pos + offset, y_bracket_pos + offset, y_bracket_pos], lw=1.5, c='k')
    ax.text((x1_pos + x2_pos) / 2, y_bracket_pos + offset * 1.5, f"d = {d_pos_val:.2f}\n{p_pos_str}", ha='center', va='bottom', color='k', fontsize=10)

    # Comparison 2: Negative (-1) vs. No (0)
    x1_neg, x2_neg = 0, 1  # Positions for Negative and No
    y1 = bar_heights.get(-1, 0)
    y2 = bar_heights.get(0, 0)
    y_bracket_neg = max(y1, y2) + offset

    # Check if the two inner brackets would overlap and adjust if necessary
    if abs(y_bracket_pos - y_bracket_neg) < y_range * 0.15:
        y_bracket_neg = y_bracket_pos + y_range * 0.2  # Push the second bracket up more

    # Retrieve the stats
    neg_stats = effect_size_df[
        (effect_size_df['Metric'] == metric_name) &
        (effect_size_df['Comparison'] == 'Negative vs. No Priming')
    ].iloc[0]
    d_neg_val = neg_stats["Cohen's d"]
    p_neg_val = neg_stats["p-val"]
    p_neg_str = p_to_stars(p_neg_val)

    # Draw the annotation bracket and text
    ax.plot([x1_neg, x1_neg, x2_neg, x2_neg], [y_bracket_neg, y_bracket_neg + offset, y_bracket_neg + offset, y_bracket_neg], lw=1.5, c='k')
    ax.text((x1_neg + x2_neg) / 2, y_bracket_neg + offset * 1.5, f"d = {d_neg_val:.2f}\n{p_neg_str}", ha='center', va='bottom', color='k', fontsize=10)

    # Comparison 3: Positive (1) vs. Negative (-1)
    x1_pvn, x2_pvn = 2, 0  # Positions for Positive and Negative
    # This bracket should be the highest one
    y_bracket_pvn = max(y_bracket_pos, y_bracket_neg) + y_range * 0.2

    # Retrieve the stats
    pvn_stats = effect_size_df[
        (effect_size_df['Metric'] == metric_name) &
        (effect_size_df['Comparison'] == 'Positive vs. Negative Priming')
    ].iloc[0]
    d_pvn_val = pvn_stats["Cohen's d"]
    p_pvn_val = pvn_stats["p-val"]
    p_pvn_str = p_to_stars(p_pvn_val)

    # Draw the annotation bracket and text
    ax.plot([x1_pvn, x1_pvn, x2_pvn, x2_pvn], [y_bracket_pvn, y_bracket_pvn + offset, y_bracket_pvn + offset, y_bracket_pvn], lw=1.5, c='k')
    ax.text((x1_pvn + x2_pvn) / 2, y_bracket_pvn + offset * 1.5, f"d = {d_pvn_val:.2f}\n{p_pvn_str}", ha='center', va='bottom', color='k', fontsize=10)

    # Adjust y-limit to make space for ALL annotations
    ax.set_ylim(top=y_bracket_pvn + y_range * 0.3)


# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# ===================================================================
#       CORRELATION MATRIX OF PERFORMANCE METRICS
# ===================================================================
# Compare capture score effect sizes versus response time and accuracy effect sizes in Priming conditions
corr_df = analysis_df.groupby(["subject_id"])[["rt", "select_target", "target_towardness_std",
                                                                   "target_towardness", "distractor_towardness",
                                                                   "target_distractor_towardness_diff",
                                                      "distractor_towardness_std",
                                                      "target_distractor_towardness_diff_std"]].mean().reset_index()

# Select the columns for the correlation matrix from the df_effects DataFrame.
# These are the subject-level average scores for each metric, across all conditions.
metric_cols = list(metrics_to_compare.values())
correlation_df = corr_df[metric_cols]

# For a more readable plot, let's rename the columns to their user-friendly names.
correlation_df = correlation_df.rename(columns={v: k for k, v in metrics_to_compare.items()})


# Visualize the correlation matrix as a heatmap for better interpretation.
sns.pairplot(
    correlation_df
)
plt.xticks(rotation=45, ha="right") # Rotate labels for better readability
plt.yticks(rotation=0)
plt.tight_layout() # Adjust layout to prevent labels from being cut off


# Calculate the Pearson correlation matrix.
corr_matrix = correlation_df.corr(method="spearman")
p_values = df_effects[metric_cols].corr(method=lambda x, y: spearmanr(x, y)[1])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))


fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, cmap='vlag', center=0, annot=False, linewidths=.5, vmin=-1, vmax=1, ax=ax)

# Add custom annotations with significance
for i in range(len(corr_matrix)):
    for j in range(i):  # Only iterate through lower triangle
        r, p = corr_matrix.iloc[i, j], p_values.iloc[i, j]
        text = f"{r:.2f}"
        if p < 0.001:
            text += "***"
        elif p < 0.01:
            text += "**"
        elif p < 0.05:
            text += "*"
        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="white" if abs(r) > 0.6 else "black")

ax.set_title('Spearman correlation matrix', fontsize=15, fontweight='bold')
ax.text(0.98, 0.98, "* p<0.05\n** p<0.01\n*** p<0.001", transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()

# save target towardness
towardness_df = analysis_df[['subject_id', 'block', 'trial_nr', 'target_towardness']]
output_path = f'{SPACEPRIME.get_data_path()}concatenated\\target_towardness.csv'
towardness_df.to_csv(output_path, index=True)

# ===================================================================
#       STATISTICAL ANALYSIS: CAPTURE SCORE BY PRIMING
# ===================================================================
print("\n--- Statistical Analysis of Capture Score (Priming x Block) ---")

# We will run a 3xN repeated measures ANOVA to test for:
# 1. A main effect of PrimingCondition (does priming affect capture?)
# 2. A main effect of Block (is there learning?)
# 3. An interaction effect (does the priming effect change across blocks?)
print("\nRunning 3xN Repeated Measures ANOVA (PrimingCondition x Block)...")

analysis_df_anova = analysis_df.copy()
rm_anova_results = pg.rm_anova(
    data=analysis_df_anova,
    dv='target_towardness',  # Use the standardized score
    within=['PrimingCondition', 'block'],
    subject='subject_id',
    detailed=True
)

print("ANOVA Results:")
print(rm_anova_results.round(4))

# ===================================================================
#      ANALYSIS: RUNNING AVERAGE OF capture SCORE OVER TRIALS
# ===================================================================
print("\n--- Analyzing Running Average of capture Score Over Trials ---")

# To get a more granular view of learning, we'll plot the capture score
# over a continuous trial index using a running average to smooth the curve.

# --- Step 1: Prepare the DataFrame ---
# Ensure the DataFrame is sorted correctly for time-series analysis.
analysis_df.sort_values(['subject_id', 'block', 'trial_nr'], inplace=True)

# Create a sequential trial index for each subject (only for singleton-present trials).
analysis_df['singleton_trial_idx'] = analysis_df.groupby('subject_id').cumcount()

# --- Step 2: Calculate the Running Average ---
# For each subject, calculate the rolling mean of the capture score.
# This smooths out trial-to-trial noise and reveals the underlying trend.
# We use the same WINDOW_SIZE defined earlier in the script for consistency.
analysis_df['towardness_running_avg'] = analysis_df.groupby('subject_id')['target_towardness'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=False, min_periods=1).mean()
)

print(f"Calculated running average with a window size of {WINDOW_SIZE} trials.")

# --- Step 3: Create the Line Plot ---
plt.figure(figsize=(12, 7))

# Use seaborn's lineplot to show the mean running average across all subjects.
# The confidence interval will be plotted automatically.
sns.lineplot(
    data=analysis_df,
    x='singleton_trial_idx',
    y='towardness_running_avg'
)

# Add a horizontal line at y=0 for a clear reference point
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Capture = Target capture')

# Formatting the plot
plt.title(f'Running Average of towardness (Window Size = {WINDOW_SIZE})', fontsize=16)
plt.xlabel('Trial Number (Singleton Present)', fontsize=12)
plt.ylabel('capture Score (Negative = Capture, Positive = Target)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine()

plt.show()

# ===================================================================
#       SAVE RESULTS FOR EXTERNAL USE
# ===================================================================
print("\n--- Saving Initial Movement Classifications for external use ---")
# Define a dedicated output directory for this script's results
output_dir_capture = f"{get_data_path()}concatenated\\"

# Define the output path and save the file
output_filepath = f"{output_dir_capture}capture_scores.csv"
analysis_df.to_csv(output_filepath, index=False)
print(f"Saved capture scores for {len(analysis_df)} trials to: {output_filepath}")

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
df_clean_sp = df_clean.copy().query("SingletonPresent==1")
analysis_df_classification = pd.merge(
    df_clean_sp,
    initial_move_points_df,
    on=['subject_id', 'block', 'trial_nr'],
    how='left'  # Use left merge to keep all trials
)

# Drop trials where no movement was detected (initial coordinates will be NaN)
analysis_df_classification.dropna(subset=['initial_x_dva', 'initial_y_dva'], inplace=True)
print(f"Found initial movement data for {len(analysis_df_classification)} trials.")

# --- Exclude trials where any primary ROI is at the center (digit 5) ---
initial_trial_count = len(analysis_df_classification)
print(f"Filtering out trials with a primary ROI at the center...")

# Define the columns that hold the digit locations for the primary ROIs.
# Make sure these column names match your DataFrame exactly.
roi_digit_columns = ['TargetDigit', 'SingletonDigit', 'Non-Singleton2Digit']

# Create a boolean mask to identify rows where any of these columns have the value 5.
# The .any(axis=1) checks across the row for any True value.
is_center_roi_mask = (analysis_df_classification[roi_digit_columns] == 5).any(axis=1)

# Keep only the trials where the mask is False (i.e., no ROI was at the center).
# Using .copy() is good practice here to avoid SettingWithCopyWarning later.
#analysis_df = analysis_df[~is_center_roi_mask].copy()

excluded_count = initial_trial_count - len(analysis_df_classification)
print(f"Excluded {excluded_count} trials.")
print(f"Remaining trials for analysis: {len(analysis_df_classification)}")

# --- Step 3 & 4: Calculate Vectors, Angles, and Classify Movement ---
# Apply the classification function to each row of the DataFrame
analysis_df_classification['initial_movement_direction'] = analysis_df_classification.apply(
    classify_initial_movement, axis=1, locations_map=numpad_locations_dva, start_vec=start_point_vec
)

# ===================================================================
#       SAVE CLASSIFICATION RESULTS FOR EXTERNAL USE
# ===================================================================
print("\n--- Saving Initial Movement Classifications for external use ---")
# Define a dedicated output directory for this script's results
output_dir_cursor = f"{get_data_path()}concatenated\\"

# Select only the essential columns for saving
output_df = analysis_df_classification[['subject_id', 'block', 'trial_nr', 'initial_movement_direction']].copy()

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
subject_block_counts = analysis_df_classification.groupby(
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
hue_order = sorted(analysis_df_classification['block'].unique().astype(int))

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
analysis_df_classification.sort_values(['subject_id', 'block', 'trial_nr'], inplace=True)

# Create a single score for each trial based on the initial movement direction:
#  +1 if the movement was toward the control item (suppression)
#  -1 if the movement was toward the distractor item (capture)
#   0 otherwise (e.g., toward target or other)
conditions = [
    analysis_df_classification['initial_movement_direction'] == 'control',
    analysis_df_classification['initial_movement_direction'] == 'distractor'
]
choices = [1, -1]
analysis_df_classification['per_trial_effect'] = np.select(conditions, choices, default=0)

# --- Step 2: Apply a Rolling Average to the Per-Trial Score ---

# For each subject, calculate the rolling mean of the per-trial effect score.
# The result is a smooth curve representing the suppression effect over time.
# We multiply by 100 to convert the proportion into a percentage.
#
# CRITICAL FIX: We use `min_periods=1`. This ensures that we get a value even
# at the very beginning and end of the experiment where the window is not full.
# Without this, we would lose the data from the crucial initial trials.
analysis_df_classification['suppression_effect'] = analysis_df_classification.groupby('subject_id')['per_trial_effect'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=False, min_periods=1).mean()
) * 100

# Create a sequential trial index across the entire experiment for each subject.
analysis_df_classification['singleton_trial_idx'] = analysis_df_classification.groupby('subject_id').cumcount()

# Use the 'per_trial_effect' which is the raw score (-1, 0, or 1) for robust fitting.
# Add 1 to trial index to avoid log(0).
analysis_df_classification['log_trial_idx'] = np.log(analysis_df_classification['singleton_trial_idx'] + 1)

# --- Create the Final Plot ---
plt.figure(figsize=(12, 7))
# Plot the mean suppression effect
sns.lineplot(data=analysis_df_classification, x=analysis_df_classification["singleton_trial_idx"], y=analysis_df_classification["suppression_effect"])

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
trial_counts = analysis_df_classification['singleton_trial_idx'].value_counts()
last_reliable_trial = trial_counts[trial_counts >= 3].index.max()
if pd.notna(last_reliable_trial):
    plt.xlim(0, last_reliable_trial)

plt.show()

# ===================================================================
#       ANALYSIS: BLOCK-LEVEL SUPPRESSION EFFECT BAR PLOT
# ===================================================================
print("\n--- Creating Block-Level Suppression Effect Bar Plot ---")

# --- Step 1: Calculate a Per-Trial Suppression Score ---
# This score is +1 for a movement to control (suppression), -1 for a movement
# to the distractor (capture), and 0 otherwise. This logic is robust and clear.
conditions = [
    analysis_df_classification['initial_movement_direction'] == 'control',
    analysis_df_classification['initial_movement_direction'] == 'distractor'
]
choices = [1, -1]
analysis_df_classification['per_trial_effect'] = np.select(conditions, choices, default=0)

# --- Step 2: Calculate Mean Suppression Effect per Subject, per Block ---
# We group by subject and block and take the mean of our per-trial score.
# This gives us a single, stable suppression value for each subject in each block.
block_level_effect = analysis_df_classification.groupby(['subject_id', 'block'])['per_trial_effect'].mean().reset_index()

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

print("\n--- One-Sample T-test on Block-level suppression effects ---")
ttest_result = ttest_rel(block_level_effect.query("block==0")["suppression_effect"], block_level_effect.query("block==1")["suppression_effect"])
print(ttest_result)



# THE CODE BELOW IS FOR DIAGNOSTICS ONLY
"""
# ===================================================================
#           EXAMPLE USAGE OF THE TRIAL PLOTTING FUNCTION
# ===================================================================
example = 7
# --- Find a specific trial to plot ---
distractor_trial_to_plot = analysis_df_classification[
    analysis_df_classification['initial_movement_direction'] == 'distractor'
].iloc[example]
# Call the function with the selected trial data
plot_trial_vectors(distractor_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the target ---
target_trial_to_plot = analysis_df_classification[
    analysis_df_classification['initial_movement_direction'] == 'target'
].iloc[example]
plot_trial_vectors(target_trial_to_plot, numpad_locations_dva)

# --- Or, plot a trial that went towards the other ---
control_trial_to_plot = analysis_df_classification[
    analysis_df_classification['initial_movement_direction'] == 'control'
    ].iloc[example]  # Using index 5 for variety
plot_trial_vectors(control_trial_to_plot, numpad_locations_dva)

other_trial_to_plot = analysis_df_classification[
    analysis_df_classification['initial_movement_direction'] == 'other'
    ].iloc[example]  # Using index 5 for variety
plot_trial_vectors(other_trial_to_plot, numpad_locations_dva)


# ===================================================================
#           VISUALIZE FULL TRAJECTORIES WITH RESPONSE TIME
# ===================================================================
distractor_trial_for_viz = analysis_df_classification[analysis_df_classification['initial_movement_direction'] == 'distractor'].iloc[example]
visualize_full_trajectory(distractor_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

# Find a trial where the initial movement was towards the target
target_trial_for_viz = analysis_df_classification[analysis_df_classification['initial_movement_direction'] == 'target'].iloc[example]
visualize_full_trajectory(target_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

control_trial_for_viz = analysis_df_classification[analysis_df_classification['initial_movement_direction'] == 'control'].iloc[example]
visualize_full_trajectory(control_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

other_trial_for_viz = analysis_df_classification[analysis_df_classification['initial_movement_direction'] == 'other'].iloc[example]
visualize_full_trajectory(other_trial_for_viz, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)

# Call the new diagnostic function
visualize_absolute_trajectory(control_trial_for_viz, df, numpad_locations_dva, target_hz=RESAMP_FREQ)

# Use the exact same trial that shows the doubling effect
distractor_trial_for_viz = analysis_df[analysis_df['initial_movement_direction'] == 'distractor'].iloc[10]

# Run the absolute plot and carefully check the coordinates printed in the legend
print("Checking the absolute start and end points for the trial...")
visualize_absolute_trajectory(
    distractor_trial_for_viz,
    df,
    numpad_locations_dva,
    target_hz=RESAMP_FREQ)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from matplotlib.collections import LineCollection # This is no longer needed


def plot_poster_trajectory_with_projection(trial_info, full_trajectory_df, locations_map):
    """
    Generates a minimalist, poster-quality visualization with a square
    coordinate system, ending at the button click.

    This version explicitly visualizes the 'target_towardness' by projecting
    the average trajectory vector onto the ideal target vector, using a
    colorblind-friendly and highly distinguishable color palette.

    Args:
        trial_info (pd.Series): A row from the analysis DataFrame containing trial metadata.
        full_trajectory_df (pd.DataFrame): The raw DataFrame with all cursor data points.
        locations_map (dict): A dictionary mapping digits to their (x, y) coordinates in dva.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects (fig, ax).
    """
    # --- 1. Extract Trial Information ---
    sub = int(trial_info['subject_id'])
    blk = int(trial_info['block'])
    trn = int(trial_info['trial_nr'])
    target_digit = int(trial_info['TargetDigit'])
    distractor_digit = int(trial_info['SingletonDigit'])
    target_towardness = trial_info['target_towardness']  # Get the pre-calculated score

    # --- Get the trajectory for this specific trial, UP TO THE CLICK (time <= 0) ---
    trial_trajectory = full_trajectory_df.query(
        'subject_id == @sub and block == @blk and trial_nr == @trn and time <= 0'
    ).copy()

    # --- 2. Set up the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    sns.set_style("white")  # A clean white background

    # --- 3. Plot the Cursor Trajectory and Calculate its Average Vector ---
    avg_vec = np.array([0, 0])
    if not trial_trajectory.empty and len(trial_trajectory) > 1:
        x = trial_trajectory['x'].values
        y = trial_trajectory['y'].values

        # --- MODIFIED PART: Plot trajectory as fewer, uniform black arrows ---
        # To reduce visual clutter and create uniform arrows, we'll subsample the trajectory
        # and normalize the direction vectors.
        subsample_rate = 5  # Plot an arrow every N points. Increase to reduce arrow density.
        x_sub = x[::subsample_rate]
        y_sub = y[::subsample_rate]

        # We need at least two points to draw arrows between them.
        if len(x_sub) > 1:
            # Calculate the change in x and y for each segment of the subsampled trajectory.
            u = np.diff(x_sub)
            v = np.diff(y_sub)

            # Normalize the (u, v) vectors to make all arrows the same length.
            # This makes the visualization focus on direction, not speed.
            magnitudes = np.sqrt(u**2 + v**2)
            # Create a mask to handle segments with no movement (to avoid division by zero).
            has_moved_mask = magnitudes > 1e-6

            # Only calculate normalized vectors for segments where there was movement.
            u_norm = np.zeros_like(u)
            v_norm = np.zeros_like(v)
            u_norm[has_moved_mask] = u[has_moved_mask] / magnitudes[has_moved_mask]
            v_norm[has_moved_mask] = v[has_moved_mask] / magnitudes[has_moved_mask]

            # Plot using quiver.
            # The 'scale' parameter now controls the uniform length of all arrows.
            # A larger scale value makes the arrows shorter.
            ax.quiver(
                x_sub[:-1][has_moved_mask], y_sub[:-1][has_moved_mask],
                u_norm[has_moved_mask], v_norm[has_moved_mask],
                color='black',
                angles='xy',
                scale_units='xy',
                scale=30,      # Increase this value to make arrows shorter.
                width=0.004,   # Arrow body thickness.
                headwidth=4,   # Arrow head width.
                headlength=5,  # Arrow head length.
                zorder=1
            )
        # The color bar is removed as it's no longer needed.
        # --- END OF MODIFICATION ---

        # Use new, consistent colors
        ax.plot(x[0], y[0], 'o', color='#2ca02c', markersize=12, label='Start', zorder=3) # Tableau Green
        ax.plot(x[-1], y[-1], 'X', color='black', markersize=12, label='End (Click)', zorder=3)
        avg_vec = trial_trajectory[['x', 'y']].mean().to_numpy()

    # --- 4. Plot Directional & Projection Vectors ---
    start_vec = np.array([0, 0])
    target_vec = np.array(locations_map[target_digit])
    distractor_vec = np.array(locations_map[distractor_digit])

    # --- Visualize Target Towardness ---
    target_vec_norm = np.linalg.norm(target_vec)
    if target_vec_norm > 1e-6:
        unit_target_vec = target_vec / target_vec_norm
        projection_vec = unit_target_vec * target_towardness

        # Helper line from average vector tip to its projection on the target vector
        ax.plot([avg_vec[0], projection_vec[0]], [avg_vec[1], projection_vec[1]],
                color='gray', linestyle='--', linewidth=1.5, zorder=2, label='Projection Line')

        # Arrow for the "Target Towardness" score itself
        ax.arrow(start_vec[0], start_vec[1], projection_vec[0], projection_vec[1],
                 color='#17becf', label=f'Target Towardness ({target_towardness:.2f})', # Tableau Cyan
                 width=0.02, head_width=0.07, length_includes_head=True, zorder=4,
                 edgecolor='black', linewidth=0.5)

    # --- Plot Ideal and Average Vectors with new color palette ---
    arrow_props = dict(width=0.015, head_width=0.06, length_includes_head=True, zorder=3, linestyle="-")
    ax.arrow(start_vec[0], start_vec[1], target_vec[0], target_vec[1],
             color='#1f77b4', label=f'To Target ({target_digit})', **arrow_props) # Tableau Blue
    ax.arrow(start_vec[0], start_vec[1], distractor_vec[0], distractor_vec[1],
             color='#d62728', label=f'To Distractor ({distractor_digit})', **arrow_props) # Tableau Red
    ax.arrow(start_vec[0], start_vec[1], avg_vec[0], avg_vec[1],
             color='#9467bd', label='Average Trajectory', **arrow_props) # Tableau Purple

    # --- 5. Final Touches & Axis Scaling ---
    ax.set_title(f'Trajectory Example: Sub-{sub}, Trial-{trn}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position (dva)', fontsize=12)
    ax.set_ylabel('Y Position (dva)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    all_points = np.array([start_vec, target_vec, distractor_vec, avg_vec, projection_vec])
    if not trial_trajectory.empty:
        all_points = np.vstack([all_points, trial_trajectory[['x', 'y']].values])

    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    max_range = max(x_max - x_min, y_max - y_min, 0.1)
    padding = max_range * 0.15
    half_range_with_pad = (max_range / 2) + padding
    ax.set_xlim(x_center - half_range_with_pad, x_center + half_range_with_pad)
    ax.set_ylim(y_center - half_range_with_pad, y_center + half_range_with_pad)

    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    sns.despine(trim=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax


# ===================================================================
#       GENERATE THE PLOT FOR A PERFECT EXAMPLE TRIAL
# ===================================================================
print("\n--- Searching for a poster-worthy example trial... ---")

# --- Merge analysis dataframes to get all metrics in one place ---
# This gives us access to both 'initial_movement_direction' and 'target_towardness'
full_analysis_df = pd.merge(
    analysis_df,
    analysis_df_classification[['subject_id', 'block', 'trial_nr', 'initial_movement_direction']],
    on=['subject_id', 'block', 'trial_nr'],
    how='inner'
)
print(f"Created a combined analysis dataframe with {len(full_analysis_df)} trials.")

# --- Define pairs of digits that are opposite each other on the numpad ---
# These pairs have the maximal angular distance.
opposite_pairs = [
    (1, 9), (9, 1),
    (3, 7), (7, 3),
    (2, 8), (8, 2),
    (4, 6), (6, 4)
]

# --- Create a boolean mask for trials where the target and distractor are opposites ---
is_opposite_mask = pd.Series(False, index=full_analysis_df.index)
for t, d in opposite_pairs:
    is_opposite_mask |= (
        (full_analysis_df['TargetDigit'] == t) &
        (full_analysis_df['SingletonDigit'] == d)
    )
print(f"Found {is_opposite_mask.sum()} trials with maximally distant target/distractor pairs.")


# --- Find a trial that meets all our criteria for a great example ---
# 1. Initial movement is towards the target.
# 2. Final response is to the distractor (a "change of mind").
# 3. Target and distractor are on opposite sides of the screen.
candidate_trials = full_analysis_df[
    (full_analysis_df['initial_movement_direction'] == 'distractor') &
    (full_analysis_df['response'] == full_analysis_df['TargetDigit']) &
    (is_opposite_mask)
]

if not candidate_trials.empty:
    # Select one of these interesting trials at random
    #example_trial_for_poster = candidate_trials.sample(n=1).iloc[0]
    example_trial_for_poster = candidate_trials.query("subject_id==136&block==7.0&trial_nr==171").iloc[0]

    print(f"Found a great example! Plotting trial {example_trial_for_poster['trial_nr']}, block {example_trial_for_poster['block']}, from subject {example_trial_for_poster['subject_id']}.")
    print(f"  -> Target: {int(example_trial_for_poster['TargetDigit'])}, Distractor: {int(example_trial_for_poster['SingletonDigit'])}")

    # Call the updated plotting function, which now returns the figure
    fig, ax = plot_poster_trajectory_with_projection(
        trial_info=example_trial_for_poster,
        full_trajectory_df=df,
        locations_map=numpad_locations_dva
    )

    # --- Save and Show the Figure ---
    output_path = "G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\cursor_trajectory_with_projection.svg"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    plt.show()

else:
    print("Could not find a trial that matched the specified criteria (initial target move, final distractor response, and maximal distance).")


# ===================================================================
#       POSTER PLOT: TARGET TOWARDNESS VS. CONTROL PROJECTION
# ===================================================================
print("\n--- Generating Poster Plot: Target Towardness vs. Control Projection ---")

# --- Step 1: Prepare data for plotting ---
# We need to "melt" the dataframe to plot two different columns on the same y-axis.
# This converts 'target_towardness' and 'proj_control_max' from wide to long format.

analysis_df["target_towardness_contrast"] = analysis_df["target_towardness"] - control_proj
analysis_df_mean = analysis_df.groupby(["subject_id", "block", "Priming"])["target_towardness_contrast"].mean().reset_index()
plot_data = analysis_df_mean.melt(
    id_vars=['subject_id', 'block', 'Priming'],
    value_vars=['target_towardness_contrast'],
    var_name='Projection Type',
    value_name='Projection Score (dva)'
)

# Improve the legend labels for clarity
plot_data['Projection Type'] = plot_data['Projection Type'].map({
    'target_towardness': 'Toward Target'
})


# --- Step 2: Create the pointplot ---
plt.figure(figsize=(10, 7))

ax = sns.lineplot(
    data=plot_data,
    x='block',
    y='Projection Score (dva)',
    hue="Priming",
    color='darkblue',
    errorbar=("se", 2),  # Show standard error (x2 for ~95% CI)
    linewidth=3.5,  # Thicker line for better visibility from a distance
    marker='o',  # Add markers to emphasize each block's data point
    markersize=9,  # Make markers larger and clearer
    legend=True,  # Explicitly disable the legend
)

# --- Step 3: Formatting for poster quality ---
# Add a reference line at y=0
ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Set clear titles and labels with appropriate font sizes
ax.set_title('Trajectory Projection Toward Target vs. Control Items', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Block Number', fontsize=14)
ax.set_ylabel('Projection Score (dva)', fontsize=14)

# Customize ticks and legend
ax.tick_params(axis='both', which='major', labelsize=12)

# Clean up the plot aesthetics
sns.despine()
plt.grid(True, which='major', axis='y', linestyle=':', linewidth=0.7)
plt.tight_layout()

# Show the final plot
plt.show()

# --- Save and Show the Figure ---
output_path = "G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\target_towardness_over_blocks.svg"
plt.savefig(output_path, bbox_inches='tight')

from scipy.stats import page_trend_test

print("\n--- Page's Trend Test for Monotonic Increase ---")

# The data needs to be in a "wide" format: (subjects x conditions)
# You already created this for the Cronbach's Alpha calculation.
# We'll reuse df_pivot.
df_pivot = plot_data.pivot(index="subject_id", columns="block", values='Projection Score (dva)')

# The test can't handle missing values, so we'll drop any subjects
# who might be missing a block for any reason.
df_pivot_clean = df_pivot.dropna()

print(f"Running test on {len(df_pivot_clean)} complete subjects out of {len(df_pivot)} total.")

# Perform the test.
# We use alternative='greater' because we hypothesize an INCREASING trend
# in rt_diff (from negative capture to positive suppression).
# The default predicted_ranks=(1, 2, 3...) is exactly what we want.
L_statistic, p_value = page_trend_test(df_pivot_clean)

print(f"\nPage's L statistic: {L_statistic:.2f}")
print(f"P-value: {p_value:.5f}")

# --- Interpretation ---
if p_value < 0.05:
    print("\n> Conclusion: The result is significant. We can reject the null hypothesis.")
    print("> There is strong evidence for a monotonically increasing trend in the RT difference across blocks.")
    print("> This supports the interpretation of a systematic shift from attentional capture to suppression.")
else:
    print("\n> Conclusion: The result is not significant.")
    print("> We cannot conclude there is a monotonic trend in the RT difference across blocks.")
