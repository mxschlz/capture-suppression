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

# load dataframe
fp = "/home/max/data/behavior/SPACEPRIME/all_subjects_additional_metrics_and_priming.csv"
df = pd.read_csv(fp)
# get data for dashed line
line_data = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
# get only singleton present trials
sp = df[df["Singletonpres"] == 1]
# Filter for 'positive_priming' and 'negative_priming'
spaceprime = sp[sp['spatial_priming'].isin(['positive_priming', "negative_priming"])]
# Calculate mean correct by subject and priming type
mean_correct_df = spaceprime.groupby(['subject', 'spatial_priming'])['correct'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df = mean_correct_df.pivot(index='subject',
                                 columns='spatial_priming',
                                 values='mean_correct').fillna(0).reset_index()
# look for correlation between spatial and temporal priming
fig, ax = plt.subplots(figsize=(4, 4))
sns.lineplot(data=line_data, x="x", y="y", color="black", linestyle="solid")
sns.scatterplot(x=pivot_df.negative_priming,
                y=pivot_df.positive_priming, s=300, markers="O", color=palette[74])
ax.set_xlabel("Percentage Correct (Negative Priming; NP)")
ax.set_ylabel("Percentage Correct (Positive Priming; PP)")
plt.savefig("/home/max/obsolete/SAMBA24/figure3.svg")

# control for distractor presence
no_priming_trials = df[(df['spatial_priming'] == 'no_priming') & (df['Singletonpres'] == 1)]
positive_priming_trials = df[(df['spatial_priming'] == 'positive_priming') & (df['Singletonpres'] == 1)]
negative_priming_trials = df[(df['spatial_priming'] == 'negative_priming') & (df['Singletonpres'] == 1)]
# Concatenate data
df = pd.concat([negative_priming_trials, no_priming_trials, positive_priming_trials])
# Plot individual subject data as lines
subjects = df['subject'].unique()

# make subject-wise lineplots for average values in df
df_mean = df.groupby(['subject', 'spatial_priming']).mean(numeric_only=True).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
# make barplot
barplot = sns.barplot(data=df, y=df.correct, x=df.spatial_priming)
# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['spatial_priming'])]
    plt.plot(x_positions, subject_data['correct'], marker='', linestyle='-', color='black', alpha=0.5)
ax.set_xlabel("NP                               X                               PP")
ax.set_ylabel("Percentage Correct")
plt.savefig("/home/max/figures/SAMBA24/figure3_subfigure.svg")
