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
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx")
# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]

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
ax.set_xticklabels([1, 2, 3, 4, 5, 6])
plt.savefig("/home/max/temp/SAMBA24/Fig6.svg")


# Filter the dataframe to create df_block_0
df_merged = pd.merge(df_singleton_absent, df_singleton_present, on=['block', 'subject_id', 'trial_nr'], how='outer')
# Convert iscorrect_x and iscorrect_y to numeric, coercing errors to NaN
df_merged['iscorrect_x'] = pd.to_numeric(df_merged['iscorrect_x'], errors='coerce')
df_merged['iscorrect_y'] = pd.to_numeric(df_merged['iscorrect_y'], errors='coerce')
# Fill missing values with 0
df_merged['iscorrect_x'] = df_merged['iscorrect_x'].fillna(0)
df_merged['iscorrect_y'] = df_merged['iscorrect_y'].fillna(0)
# Recalculate the difference between iscorrect_x and iscorrect_y
df_merged['iscorrect_diff'] = df_merged['iscorrect_x'] - df_merged['iscorrect_y']
# Filter to only block 0 and assign to block_0
block_0 = df_merged[df_merged['block'] == 0]['iscorrect_diff']
# Filter to all other blocks and assign to other_blocks
other_blocks = df_merged[df_merged['block'] != 0]['iscorrect_diff']
# Conduct an independent t-test
t_statistic, p_value = ttest_ind(block_0.dropna(), other_blocks.dropna())
# Print the results
print(f'T-statistic: {t_statistic}, p-value: {p_value}')
