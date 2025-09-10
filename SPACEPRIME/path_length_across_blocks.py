import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from SPACEPRIME import get_data_path
import seaborn as sns
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
from utils import degrees_va_to_pixels, calculate_trial_path_length

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
# Insert this code after the line: df_clean = merged_df[merged_df["phase"]!=2]

# Define the number of trials per block
TRIALS_PER_BLOCK = 180

# Calculate total_trial_nr to range from 1 to 1800.
# This assumes:
# - df_clean['block'] is 0-indexed (e.g., 0 for the first block, 1 for the second, etc.).
# - df_clean['trial_nr'] is 0-indexed within each block (e.g., 0, 1, ..., 179).
# This indexing is consistent with how 'block' is calculated earlier in your script.
df_clean['total_trial_nr'] = (df_clean['block'] * TRIALS_PER_BLOCK) + df_clean['trial_nr'] + 1

# divide into subblocks (optional)
df_clean['sub_block'] = df_clean.total_trial_nr // 180  # choose division arbitrarily
df_singleton_absent = df_clean[df_clean['SingletonPresent'] == 0]
df_singleton_present = df_clean[df_clean['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['sub_block', "subject_id"])['path_length_pixels']
                       .mean().reset_index(name='path_length_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['sub_block', "subject_id"])['path_length_pixels']
                       .mean().reset_index(name='path_length_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['sub_block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['path_length_diff'] = df_merged['path_length_singleton_absent'] - df_merged['path_length_singleton_present']

# Add labels and title
fig, ax = plt.subplots()
sns.lineplot(x='sub_block', y='path_length_diff', data=df_merged, errorbar=("ci", 95), color="black")
ax.set_xlabel('Block')
ax.set_ylabel('Path length (distractor absent - distractor present) [px]', color="black")
ax.tick_params(axis='y', labelcolor="black", color="black")
ax.set_xlim([0, 9])
ax.set_title("")
ax.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], linestyles='--', color="grey")
ax.legend("")  # Place legend outside the plot
ax.set_title("Transition from distractor attentional capture to suppression")

# load df
df_rt = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df_rt = df_rt[df_rt["phase"]!=2]
df_rt = remove_outliers(df_rt, column_name="rt", threshold=2)

# divide into subblocks (optional)
df_rt['sub_block'] = df_rt.index // 180  # choose division arbitrarily
df_rt_singleton_absent = df_rt[df_rt['SingletonPresent'] == 0]
df_rt_singleton_present = df_rt[df_rt['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_rt_singleton_absent_mean = (df_rt_singleton_absent.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_rt_singleton_present_mean = (df_rt_singleton_present.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged_rt = pd.merge(df_rt_singleton_absent_mean, df_rt_singleton_present_mean, on=['sub_block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged_rt['rt_diff'] = df_merged_rt['rt_singleton_absent'] - df_merged_rt['rt_singleton_present']
twin = ax.twinx()
sns.lineplot(x='sub_block', y='rt_diff', data=df_merged_rt, errorbar=("ci", 95), ax=twin, color="purple")
twin.set_ylabel("Reaction time (distractor absent - distractor present) [s]", color="purple", alpha=0.7)
twin.tick_params(axis='y', labelcolor="purple", color="purple")
sns.despine(right=False)

plt.tight_layout()