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
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/sub-99_clean.xlsx")
# some cleaning
df = df[(df['event_type'] == 'mouse_click') & (df['phase'] != 2)]
sp = df[df["SingletonPresent"] == 1]
barplot = sns.barplot(data=sp, x="SingletonLoc", y="iscorrect")
df_mean = sp.groupby(['subject_id', 'SingletonLoc']).mean(numeric_only=True).reset_index()
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
subjects = df['subject_id'].unique()
for subject in subjects:
    subject_data = df_mean[df_mean['subject_id'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['SingletonLoc'])]
    plt.plot(x_positions, subject_data['iscorrect'], marker='', linestyle='-', color='black', alpha=0.5)
plt.xlabel("Singleton Position")
plt.ylabel("Proportion correct")
barplot.set_xticklabels([-90, 0, 90])
plt.savefig("/home/max/obsolete/SAMBA24/Fig10.svg")

