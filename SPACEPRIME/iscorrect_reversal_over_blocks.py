import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import get_subpalette
from scipy.stats import ttest_ind
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([14, 84, 44]).values()))

# load up dataframe
df = pd.read_excel("/home/max/data/SPACEPRIME/sub-101/beh/results_October_10_2024_16_56_41.xlsx", index_col=0)
# some cleaning
df = df[(df['event_type'] == 'mouse_click')]

# Filter the dataframe to only include rows where SingletonPresent is 0
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['block', "subject_id"])['iscorrect']
                       .mean().reset_index(name='iscorrect_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['block', "subject_id"])['iscorrect']
                       .mean().reset_index(name='iscorrect_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['iscorrect_diff'] = df_merged['iscorrect_singleton_absent'] - df_merged['iscorrect_singleton_present']

# Add labels and title
fig, ax = plt.subplots(figsize=(6, 4))
barplot = sns.barplot(x='block', y='iscorrect_diff', data=df_merged, errorbar=("se", 1))
ax.set_ylabel('Proportion correct (Distractor absent - Distractor pesent)')
ax.set_xlabel('Block')
ax.set_ylim(-0.1, 0.1)
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
plt.savefig("/home/max/figures/SPACEPRIME/iscorrect_reversal_over_blocks.svg")
