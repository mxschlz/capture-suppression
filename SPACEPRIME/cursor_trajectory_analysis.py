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
corr_df_st = remove_outliers(df_clean, column_name="rt", threshold=2)
corr_df_st = remove_outliers(corr_df_st, column_name="path_length_pixels", threshold=2)
corr_df_avrg = corr_df_st.groupby(["subject_id"])[["rt", "path_length_pixels"]].mean().reset_index()
pearsonr(corr_df_st.path_length_pixels, corr_df_st.rt)
corr_df_st = corr_df_st.dropna(subset=["rt", "path_length_pixels"])
sns.lmplot(x="path_length_pixels", y="rt", data=corr_df_st,
            scatter=True)
plt.ylabel("Reaction time [s]")
plt.xlabel("Path length [px]")
plt.title("Pearson r = 0.18, p < 0.001")
sns.despine()

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

# Create SingletonPresent column
# Assuming 'mid' in SingletonLoc means singleton absent, and other values mean present
lmm_input['SingletonPresent'] = lmm_input['SingletonLoc'].apply(lambda x: 'absent' if x == 'absent' else 'present')

# Define formulas (as in your script)
formula_rt = "rt ~ block + C(SingletonLoc, Treatment('absent')) + C(Priming, Treatment(reference='no-p')) + C(select_target_int) + C(SingletonLoc):block"
formula_path_length = "path_length_pixels ~ block + C(SingletonLoc, Treatment('absent')) + C(Priming, Treatment(reference='no-p')) + C(select_target_int) + C(SingletonLoc):block"

# --- Fit Model 1: Reaction Time (RT) ---
print("\n--- Fitting RT LMM ---")
model_rt = smf.mixedlm(formula=formula_rt, data=lmm_input, groups="subject_id")
results_rt = model_rt.fit(reml=True)
print("\n--- RT LMM Summary ---")
print(results_rt.summary())
# r2_rt = r_squared_mixed_model(results_rt) # Uncomment if available
# if r2_rt:
#     print(f"RT LMM Marginal R-squared: {r2_rt['marginal_r2']:.3f}")
#     print(f"RT LMM Conditional R-squared: {r2_rt['conditional_r2']:.3f}")

# --- Fit Model 2: Path Length ---
print("\n--- Fitting Path Length LMM ---")
model_path_length = smf.mixedlm(formula=formula_path_length, data=lmm_input, groups="subject_id")
results_path_length = model_path_length.fit(reml=True)
print("\n--- Path Length LMM Summary ---")
print(results_path_length.summary())
# r2_path_length = r_squared_mixed_model(results_path_length) # Uncomment if available
# if r2_path_length:
#     print(f"Path Length LMM Marginal R-squared: {r2_path_length['marginal_r2']:.3f}")
#     print(f"Path Length LMM Conditional R-squared: {r2_path_length['conditional_r2']:.3f}")


# --- Helper function to prepare DataFrame for plotting ---
def get_plot_df_for_model(results):
    params = results.fe_params
    conf_int_df = results.conf_int()

    plot_df = pd.DataFrame({
        'coef': params,
        'conf_lower': conf_int_df.iloc[:, 0],
        'conf_upper': conf_int_df.iloc[:, 1]
    })
    # Calculate error lengths for asymmetric error bars
    plot_df['error_lower_len'] = plot_df['coef'] - plot_df['conf_lower']
    plot_df['error_upper_len'] = plot_df['conf_upper'] - plot_df['coef']

    plot_df['param_names'] = plot_df.index
    # Exclude the intercept for plotting (which is a fixed effect)
    # Random effects parameters (like variances) are not part of fe_params
    plot_df = plot_df[~plot_df['param_names'].str.contains("Intercept", case=False)]

    # Determine significance: CI does not include 0
    plot_df['significant'] = ~((plot_df['conf_lower'] < 0) & (plot_df['conf_upper'] > 0))
    return plot_df


# Prepare DataFrames for plotting
plot_df_rt = get_plot_df_for_model(results_rt)
plot_df_path = get_plot_df_for_model(results_path_length)

# --- Plotting with Subplots ---

# Get unique parameter names for y-axis ordering.
all_param_names_rt = plot_df_rt['param_names'].unique()
all_param_names_path = plot_df_path['param_names'].unique()
union_param_names = sorted(list(set(all_param_names_rt) | set(all_param_names_path)))
param_to_y_index = {name: i for i, name in enumerate(union_param_names)}

num_total_params = len(union_param_names)
fig_height = max(6, num_total_params * 0.5)
fig_width = 16

fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)


# --- Helper function to plot coefficients on a given axis ---
def plot_coeffs_on_ax(ax, plot_data, model_title_suffix, param_y_map,
                      color_sig, color_nonsig, ecolor_sig, ecolor_nonsig,
                      mec_sig, mec_nonsig, label_prefix):
    plot_data['y_pos'] = plot_data['param_names'].map(param_y_map)
    plot_data_to_draw = plot_data.dropna(subset=['y_pos'])

    ns_df = plot_data_to_draw[~plot_data_to_draw['significant']]
    ax.errorbar(x=ns_df['coef'], y=ns_df['y_pos'],
                xerr=[ns_df['error_lower_len'], ns_df['error_upper_len']],
                fmt='o', color=color_nonsig, ecolor=ecolor_nonsig, elinewidth=1.5,
                capsize=3, markersize=6, markeredgecolor=mec_nonsig,
                label=f'{label_prefix}: Not Significant (p > 0.05)')
    s_df = plot_data_to_draw[plot_data_to_draw['significant']]
    ax.errorbar(x=s_df['coef'], y=s_df['y_pos'],
                xerr=[s_df['error_lower_len'], s_df['error_upper_len']],
                fmt='o', color=color_sig, ecolor=ecolor_sig, elinewidth=2,
                capsize=4, markersize=7, markeredgecolor=mec_sig,
                label=f'{label_prefix}: Significant (p < 0.05)')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Coefficient Estimate")
    ax.set_title(f"{model_title_suffix}")
    ax.legend(loc='best')
    sns.despine(ax=ax, left=True, bottom=False)

# Define common colors
common_color_sig = 'black'
common_color_nonsig = 'lightgrey'
common_ecolor_sig = 'black'
common_ecolor_nonsig = 'lightgrey'
common_mec_sig = "black" # Markeredgecolor for significant
common_mec_nonsig = "lightgrey" # Markeredgecolor for non-significant


# Plot RT model on the first subplot (axes[0])
plot_coeffs_on_ax(axes[0], plot_df_rt, 'Reaction Time', param_to_y_index,
                  color_sig=common_color_sig, color_nonsig=common_color_nonsig,
                  ecolor_sig=common_ecolor_sig, ecolor_nonsig=common_ecolor_nonsig,
                  mec_sig=common_mec_sig, mec_nonsig=common_mec_nonsig, label_prefix='RT')

# Plot Path Length model on the second subplot (axes[1])
plot_coeffs_on_ax(axes[1], plot_df_path, 'Path Length', param_to_y_index,
                  color_sig=common_color_sig, color_nonsig=common_color_nonsig,
                  ecolor_sig=common_ecolor_sig, ecolor_nonsig=common_ecolor_nonsig,
                  mec_sig=common_mec_sig, mec_nonsig=common_mec_nonsig, label_prefix='Path Length')

# Set y-axis ticks and labels for the first subplot
axes[0].set_yticks(list(param_to_y_index.values()))
axes[0].set_yticklabels(union_param_names)

plt.tight_layout()
