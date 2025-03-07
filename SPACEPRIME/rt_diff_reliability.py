import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from stats import remove_outliers
from stats import cronbach_alpha
from scipy.stats import pearsonr
plt.ion()


# load df
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

# divide into subblocks (optional)
df['sub_block'] = df.index // 180  # choose division arbitrarily
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['sub_block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['rt_diff'] = df_merged['rt_singleton_absent'] - df_merged['rt_singleton_present']
# running average
window_size = 1
# Apply running average *per subject*
df_merged['rt_diff_running_avg'] = df_merged.groupby('subject_id')['rt_diff'].transform(
    lambda x: x.rolling(window=window_size, min_periods=None, center=True).mean())
# now, calculate reliability
# split-half at the block-level
even_blocks = df_merged[df_merged["sub_block"] % 2 == 0].dropna()
uneven_blocks = df_merged[df_merged["sub_block"] % 2 != 0].dropna()
# calculate pearson r correlation
r, p = pearsonr(even_blocks['rt_diff_running_avg'], uneven_blocks['rt_diff_running_avg'])
# apply spearman-brown prediction formula
reliability = (2 * r) / (1 + r)
