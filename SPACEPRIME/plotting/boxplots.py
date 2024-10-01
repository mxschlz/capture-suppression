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
# control for distractor presence
no_priming_trials = df[(df['Priming'] == 0)]
positive_priming_trials = df[(df['Priming'] == 1)]
negative_priming_trials = df[(df['Priming'] == -1)]
# Concatenate data
df = pd.concat([negative_priming_trials, no_priming_trials, positive_priming_trials])
# Plot individual subject data as lines
subjects = df['subject_id'].unique()

# make subject-wise lineplots for average values in df
df_mean = df.groupby(['subject_id', 'Priming']).mean(numeric_only=True).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
# make barplot
barplot = sns.barplot(data=df, y="iscorrect", x="Priming")
# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
for subject in subjects:
    subject_data = df_mean[df_mean['subject_id'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['Priming'])]
    plt.plot(x_positions, subject_data['iscorrect'], marker='', linestyle='-', color='black', alpha=0.5)
ax.set_ylabel("Percentage Correct")
