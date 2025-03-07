import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers, split_dataframe_by_blocks_balanced_subjects
from scipy.stats import pearsonr
plt.ion()


# load df
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

# divide into absent and present trials
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['rt_diff'] = df_merged['rt_singleton_absent'] - df_merged['rt_singleton_present']

# get reliability
half1, half2 = split_dataframe_by_blocks_balanced_subjects(df_merged, "block", "subject_id")
# calculate pearson r correlation
r, p = pearsonr(half1['rt_diff'], half2['rt_diff'])
# apply spearman-brown prediction formula
observation = (2 * r) / (1 + r)
# now, calculate reliability
# random split-half at the block-level
# do this approach 10000 times
n_permutations = 10000
reliabilities = list()
for n in range(n_permutations):
    half1, half2 = split_dataframe_by_blocks_balanced_subjects(df_merged, "block", "subject_id")
    # calculate pearson r correlation
    r, p = pearsonr(half1['rt_diff'], half2['rt_diff'])
    # apply spearman-brown prediction formula
    reliability = (2 * r) / (1 + r)
    reliabilities.append(reliability)
