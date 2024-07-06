import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/CAPSUP/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

# Plot individual subject data as lines
subjects = df['subject'].unique()

# make subject-wise lineplots for average values in df
df_mean = df.groupby(['subject', 'Singletonpres']).mean().reset_index()

# remove outliers
df.RT[df.RT > 2 * np.std(df.RT)] = np.nan  # drop values which deviate 2 standard deviations from the mean

# make barplot
barplot = sns.barplot(data=df, y=df.RT, hue="Singletonpres")

# Get the positions of the bars for aligning lines correctly
bar_positions = []
for patch in barplot.patches:
    bar_positions.append(patch.get_x() + patch.get_width() / 2)


for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[1] if cond == 1 else bar_positions[0] for cond in subject_data['Singletonpres']]
    plt.plot(x_positions, subject_data['RT'], marker='o', linestyle='-', color='grey', alpha=0.5)

plt.savefig("/home/max/figures/rt_singleton_abs_vs_pres.png", dpi=400)
plt.close()

barplot = sns.barplot(data=df, y="correct", x="Singletonpres")

bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
# Plot individual subject data as lines
subjects = df['subject'].unique()
for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['Singletonpres'])]
    plt.plot(x_positions, subject_data['correct'], marker='', linestyle='-', color='black', alpha=0.5)
plt.savefig("/home/max/temp/SAMBA24/pilot_correct.svg")
plt.close()


from stats import permutation_test

x = df.correct[df.Singletonpres==1]
y = df.correct[df.Singletonpres==0]
permutation_test(x, y)
plt.savefig("/home/max/temp/SAMBA24/permutation_pliot.svg")