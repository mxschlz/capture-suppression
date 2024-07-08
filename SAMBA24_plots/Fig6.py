import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import get_subpalette
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([]).values()))

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
