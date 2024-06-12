import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

df_mean = df.groupby(['subject', 'Singletonpres']).mean().reset_index()

df.RT[df.RT > 2 * np.std(df.RT)] = np.nan  # drop values over 2 standard deviations of the mean

subjects = df['subject'].unique()

plot = sns.barplot(data=df, x=df.congruent, y=df.RT)
# Get the positions of the bars for aligning lines correctly
bar_positions = []
for patch in plot.patches:
    bar_positions.append(patch.get_x() + patch.get_width() / 2)


for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[1] if cond == 1 else bar_positions[0] for cond in subject_data['Singletonpres']]
    plt.plot(x_positions, subject_data['RT'], marker='o', linestyle='-', color='grey', alpha=0.5)

plt.savefig("/home/max/figures/congruency_effect_rt.png", dpi=400)
plt.close()

plot = sns.barplot(data=df, x=df.congruent, y=df.correct)
# Get the positions of the bars for aligning lines correctly
bar_positions = []
for patch in plot.patches:
    bar_positions.append(patch.get_x() + patch.get_width() / 2)


for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[1] if cond == 1 else bar_positions[0] for cond in subject_data['Singletonpres']]
    plt.plot(x_positions, subject_data['RT'], marker='o', linestyle='-', color='grey', alpha=0.5)

plt.savefig("/home/max/figures/congruency_effect_correct_responses.png", dpi=400)
plt.close()
