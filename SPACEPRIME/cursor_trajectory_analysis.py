import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from SPACEPRIME import get_data_path
import seaborn as sns
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import load_concatenated_csv
from stats import remove_outliers, r_squared_mixed_model
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

plt.ion()


# Define some functions
def degrees_va_to_pixels(degrees, screen_pixels, screen_size_cm, viewing_distance_cm):
    """
    Converts degrees visual angle to pixels.

    Args:
        degrees: The visual angle in degrees.
        screen_pixels: The number of pixels on the screen (horizontal or vertical).
        screen_size_cm: The physical size of the screen in centimeters (width or height).
        viewing_distance_cm: The viewing distance in centimeters.

    Returns:
        The number of pixels corresponding to the given visual angle.
    """

    pixels = degrees * (screen_pixels / screen_size_cm) * (viewing_distance_cm * np.tan(np.radians(1)))
    return pixels

def calculate_trial_path_length(trial_group):
    """
    Calculates the total Euclidean path length for a single trial's trajectory.
    Assumes trial_group is a DataFrame with 'x_pixels' and 'y_pixels' columns
    sorted chronologically.
    """
    if len(trial_group) < 2:
        return 0.0  # Path length is 0 if less than 2 points

    # Get x and y coordinates for the current trial
    x_coords = trial_group['x_pixels'].to_numpy()
    y_coords = trial_group['y_pixels'].to_numpy()

    # Calculate the differences between consecutive points (dx, dy)
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    # Calculate the Euclidean distance for each segment: sqrt(dx^2 + dy^2)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    # Total path length is the sum of segment lengths
    total_path_length = np.sum(segment_lengths)
    return total_path_length

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

# do pearson correlation
corr_df_st = df_clean.dropna(subset=["rt", "path_length_pixels"])
corr_df_avrg = corr_df_st.groupby(["subject_id"])[["rt", "path_length_pixels"]].mean().reset_index()
pearsonr(corr_df_st.path_length_pixels, corr_df_st.rt)

# Load up ERP data
df_pd = load_concatenated_csv("df_pd_model.csv", index_col=0)
df_n2ac = load_concatenated_csv("df_n2ac_model.csv", index_col=0)
# Combine df_pd and df_n2ac
# This assumes df_pd and df_n2ac share the same index and common column structure.
df_erp_combined = df_pd.copy()
df_erp_combined['N2ac'] = df_n2ac['N2ac']
# Now df_erp_combined contains all common columns, 'Pd', and 'N2ac'

# Prepare df_clean_for_merge
# Identify columns in df_clean that are also in df_pd
cols_in_df_pd = df_pd.columns
overlapping_cols_with_pd = cols_in_df_pd.intersection(df_clean.columns)
df_clean_for_merge = df_clean.drop(columns=overlapping_cols_with_pd, errors='ignore')

# Merge the combined ERP data with the prepared behavioral data
lmm_data_for_model = pd.merge(
    df_erp_combined,
    df_clean_for_merge,
    left_index=True,
    right_index=True,
    how='left'  # Keeps all rows from df_erp_combined and adds matching from df_clean_for_merge
)

# Prepare the final input for the LMM
lmm_input = lmm_data_for_model.drop(columns=["duration"], errors='ignore').dropna()

formula = "path_length_pixels ~ Pd + N2ac + block + rt + C(TargetLoc, Treatment(reference='mid')) + C(SingletonLoc, Treatment(reference='mid')) + C(Priming, Treatment(reference='no-p')) + C(select_target_int) + C(SingletonLoc, Treatment(reference='mid')):block"
pd_lmm = smf.mixedlm(formula=formula, data=lmm_input, groups="subject_id")
pd_lmm_results = pd_lmm.fit(reml=True)
print("\n--- Path length LMM Summary (Full Model) ---")
print(pd_lmm_results.summary())
# Calculate and print R-squared
r2_pd = r_squared_mixed_model(pd_lmm_results)
if r2_pd:
    print(f"LMM Marginal R-squared (fixed effects): {r2_pd['marginal_r2']:.3f}")
    print(f"LMM Conditional R-squared (fixed + random effects): {r2_pd['conditional_r2']:.3f}")
