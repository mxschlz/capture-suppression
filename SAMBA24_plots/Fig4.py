import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import palette as palette
from color_palette import get_subpalette
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([104, 22]).values()))

# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx", index_col=0)
# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]

# make subject-wise lineplots for average values in df
df_mean = df.groupby(['subject_id', 'SingletonPresent']).mean(numeric_only=True).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
# make barplot
barplot = sns.barplot(data=df, y="iscorrect", x="SingletonPresent", errorbar=("se", 2))
# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
subjects = df['subject_id'].unique()
for subject in subjects:
    subject_data = df_mean[df_mean['subject_id'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['SingletonPresent'])]
    plt.plot(x_positions, subject_data['iscorrect'], marker='', linestyle='-', color='black', alpha=0.5)
ax.set_xticklabels(["absent", "present"])
ax.set_xlabel("Singleton Distractor")
ax.set_ylabel("Proportion Correct")
plt.savefig("/home/max/obsolete/SAMBA24/Fig4.svg")


from stats import permutation_test

x = df.iscorrect[df.SingletonPresent==1]
y = df.iscorrect[df.SingletonPresent==0]
permutation_test(x, y)
plt.savefig("/home/max/obsolete/SAMBA24/permutation_fig4.svg")
