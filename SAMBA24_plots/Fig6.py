import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import get_subpalette
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([14, 84, 44]).values()))

# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx")
# some cleaning
df_filt = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# Create two new dataframes
df_singleton_present = (df_filt[df_filt['SingletonPresent'] == 1].groupby(['block'])['iscorrect']
                        .mean().reset_index(name='iscorrect_singleton_present'))
df_singleton_absent = (df_filt[df_filt['SingletonPresent'] == 0].groupby(['block'])['iscorrect']
                       .mean().reset_index(name='iscorrect_singleton_absent'))
# Merge the two dataframes
df_merged = pd.merge(df_singleton_present, df_singleton_absent, on='block', how='outer')
# Calculate the difference
df_merged['difference'] = df_merged['iscorrect_singleton_absent'] - df_merged['iscorrect_singleton_present']
# plot
fig, ax = plt.subplots(figsize=(6, 4))
barplot = sns.barplot(data=df_merged, x='block', y='difference')
ax.set_ylabel('Proportion correct (Distractor absent - Distractor pesent)')
ax.set_xlabel('Block')
ax.set_ylim(-0.1, 0.1)
ax.set_xticklabels([1, 2, 3, 4, 5, 6])

plt.savefig("/home/max/temp/SAMBA24/Fig6.svg")



from scipy.stats import ttest_ind
import pandas as pd

# Filter the dataframe to only include rows where SingletonPresent is 0
df = df_filt
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['block', "subject_id"])['iscorrect']
                       .mean().reset_index(name='iscorrect_singleton_absent'))

# Filter the dataframe to only include rows where SingletonPresent is 1
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['block', "subject_id"])['iscorrect']
                       .mean().reset_index(name='iscorrect_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['iscorrect_diff'] = df_merged['iscorrect_singleton_absent'] - df_merged['iscorrect_singleton_present']

# Group df_merged by block and calculate the mean and standard error of iscorrect_diff
df_summary = (df_merged.groupby('block')['iscorrect_diff'].agg(iscorrect_diff_mean='mean', iscorrect_diff_sem='sem').
              reset_index())

# Reshape the DataFrame from wide to long format
df_plot = df_summary.melt(id_vars='block', var_name='Metric', value_name='Score')

# Create the bar plot using seaborn
plt.figure(figsize=(10, 6))
to_plot = df_plot[df_plot['Metric'] == 'iscorrect_diff_mean']
sns.barplot(x='block', y='Score', data=to_plot)

# Add error bars to the 'Mean Difference' bars
mean_diff_sem = df_plot[df_plot['Metric'] == 'iscorrect_diff_sem']
plt.errorbar(x=mean_diff_sem['block'], y=df_plot.loc[df_plot['Metric'] == 'iscorrect_diff_mean', 'Score'],
             yerr=mean_diff_sem['Score'], fmt='none', color='black', capsize=5)

# Add labels and title
plt.xlabel('Block', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Mean Difference and SEM by Block\n(Singleton Absent vs. Singleton Present)', fontsize=14)
plt.legend(title='Metric', labels=['Mean Difference', 'SEM'])

# Filter the dataframe to create df_block_0
df_block_0 = df_merged[df_merged['block'] == 0]['iscorrect_diff']

# Filter the dataframe to create df_other_blocks
df_other_blocks = df_merged[df_merged['block'] != 0]['iscorrect_diff']

ttest_ind(df_block_0, df_other_blocks)
