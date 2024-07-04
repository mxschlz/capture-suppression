import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from stats import cohen_d
plt.ion()


fp = "/home/max/data/behavior/CAPSUP/all_subjects_additional_metrics_and_priming.csv"
df = pd.read_csv(fp)

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

# make barplot
barplot = sns.barplot(data=df, y=df.correct, x=df.spatial_priming)

# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]

# Plot individual subject data as lines
for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['spatial_priming'])]
    plt.plot(x_positions, subject_data['correct'], marker='o', linestyle='-', color='grey', alpha=0.5)

plt.savefig("/home/max/figures/spatial_priming_correct.png", dpi=400)
plt.close()

# drop unrealistically high RT values
df.loc[df.RT > 2 * np.std(df.RT), df.RT] = np.nan  # drop values over 2 standard deviations of the mean

# make barplot
barplot = sns.barplot(data=df, y="RT", x="spatial_priming")

# Get the positions of the bars for aligning lines correctly
bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]

# Plot individual subject data as lines
for subject in subjects:
    subject_data = df_mean[df_mean['subject'] == subject]
    # Aligning subject data with bar positions
    x_positions = [bar_positions[i] for i, _ in enumerate(subject_data['spatial_priming'])]
    plt.plot(x_positions, subject_data['RT'], marker='o', linestyle='-', color='grey', alpha=0.5)

plt.savefig("/home/max/figures/spatial_priming_rt.png", dpi=400)
plt.close()

# a little bit of stats
from statsmodels.stats.anova import AnovaRM
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


# Perform repeated measures ANOVA for 'correct'
anova_correct = AnovaRM(df, depvar='correct', subject='subject', within=['spatial_priming'], aggregate_func="mean").fit()
print(anova_correct.summary())

# ad hoc stats
# Example for the 'correct' variable
pvals = []
for priming1 in df['spatial_priming'].unique():
    for priming2 in df['spatial_priming'].unique():
        if priming1 != priming2:
            subset = df[(df['spatial_priming'] == priming1) | (df['spatial_priming'] == priming2)]
            t_stat, p_val = stats.ttest_ind(subset[subset['spatial_priming'] == priming1]['correct'],
                                            subset[subset['spatial_priming'] == priming2]['correct'])
            d = cohen_d(subset[subset['spatial_priming'] == priming1]['correct'], subset[subset['spatial_priming'] == priming2]['correct'])
            print(f"Cohen's d between {priming1} and {priming2}: {d:.2f}")
            pvals.append(p_val)
pvals = np.unique(pvals)

# Apply Bonferroni correction
reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

print("Corrected p-values (Bonferroni):", pvals_corrected)
print("Reject null hypothesis:", reject)

# Perform repeated measures ANOVA for 'RT'
anova_rt = AnovaRM(df, depvar='RT', subject='subject', within=['spatial_priming'], aggregate_func="mean").fit()
print(anova_rt.summary())

# RT is not significant, therefore no ad hoc statistics needed
