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
df = pd.read_excel("/home/max/data/SPACEPRIME/sub-101/beh/sub-101_task-spaceprime.xlsx", index_col=0)
# some cleaning
df = df[(df['event_type'] == 'mouse_click')]
df = df[df["SingletonPresent"] == 1]

barplot = sns.barplot(data=df, x="Priming", y="iscorrect")
df_mean = df.groupby(['subject_id', 'Priming']).mean(numeric_only=True).reset_index()
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
subjects = df['subject_id'].unique()
for subject in subjects:
    subject_data = df_mean[df_mean['subject_id'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['Priming'])]
    plt.plot(x_positions, subject_data['iscorrect'], marker='', linestyle='-', color='black', alpha=0.5)
plt.xlabel("Priming")
plt.ylabel("Proportion correct")
barplot.set_xticklabels([-1, 0, 1])
# save figure
plt.savefig("/home/max/figures/SPACEPRIME/target_iscorrect.svg")
