import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
plt.ion()


# load up dataframe
df = pd.read_csv(glob.glob("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-103/beh/sub-103_clean*.csv")[0])
# some cleaning
#df = df[(df['event_type'] == 'mouse_click') & (df["phase"] != 2)]
df = df[df["SingletonPresent"] == 1]

barplot = sns.barplot(data=df, x="Priming", y="select_target", errorbar=None)
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
