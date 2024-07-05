import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import palette as palette
from color_palette import get_subpalette
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([14, 84, 44]).values()))

# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_01_2024_14_30_30.xlsx")
# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# get data for dashed line
line_data = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
# get only singleton present trials
# sp = df[df["SingletonPresent"] == 1]
# Filter for 'positive_priming' and 'negative_priming'
spaceprime = df[df['SpatialPriming'].isin([-1, 1])]
# Calculate mean correct by subject and priming type
mean_correct_df = df.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df = mean_correct_df.pivot(index='subject_id',
                                 columns='SpatialPriming',
                                 values='mean_correct').fillna(0).reset_index()
# look for correlation between spatial and temporal priming
fig, ax = plt.subplots(figsize=(4, 4))
sns.lineplot(data=line_data, x="x", y="y", color="black", linestyle="solid")
sns.scatterplot(data=pivot_df, x=-1, y=1, s=300, markers="O", color=palette[71], alpha=0.5)
ax.set_xlabel("Percentage Correct (Negative Priming; NP)")
ax.set_ylabel("Percentage Correct (Positive Priming; PP)")
plt.savefig("/home/max/temp/SAMBA24/Fig5.svg")


# make subject-wise lineplots for average values in df
df_mean = df.groupby(['subject_id', 'SpatialPriming']).mean(numeric_only=True).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
# make barplot
barplot = sns.barplot(data=df, y="iscorrect", x="SpatialPriming")
# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
subjects = df['subject_id'].unique()
for subject in subjects:
    subject_data = df_mean[df_mean['subject_id'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['SpatialPriming'])]
    plt.plot(x_positions, subject_data['iscorrect'], marker='', linestyle='-', color='black', alpha=0.5)
ax.set_xlabel("NP                               X                               PP")
ax.set_ylabel("Percentage Correct")
plt.savefig("/home/max/temp/SAMBA24/figure5_subfigure.svg")


from stats import permutation_test

x = df.iscorrect[df.SpatialPriming==1]
y = df.iscorrect[df.SpatialPriming==-1]
permutation_test(x, y)