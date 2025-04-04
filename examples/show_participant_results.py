import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from stats import remove_outliers
from SPACEPRIME.plotting import plot_individual_lines
plt.ion()


# load df
subject_id = 152
df = pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/beh/sub-{subject_id}_clean*.csv")[0])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

# incorrect trials
plt.figure()
df_long = pd.melt(df, id_vars=['block', "subject_id"], value_vars=['select_target', 'select_distractor', 'select_control', 'select_other'])
sns.barplot(x='block', y='value', hue="variable", data=df_long, errorbar=("se", 1),
            palette=['forestgreen', 'red', 'grey', 'purple'])

# divide into subblocks (optional)
df['sub_block'] = df.index // 180  # choose division arbitrarily
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['sub_block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['rt_diff'] = df_merged['rt_singleton_absent'] - df_merged['rt_singleton_present']

# Add labels and title
plt.figure()
sns.lineplot(x='sub_block', y='rt_diff', data=df_merged)
plt.ylabel('RT (Distractor absent - Distractor pesent)')
plt.xlabel('Sub-Block / Time Window')
# lineplot
sns.lineplot(x='sub_block', y='rt_diff', hue='subject_id', data=df_merged,
             palette="tab20")
plt.xlabel('Sub-Block / Time Window')
plt.ylabel('Reaction time (Distractor absent - Distractor present)')
plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], linestyles='solid', color="black")
plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# priming accuracy
plt.figure()
barplot = sns.barplot(data=df, x="Priming", y="select_target", errorbar=None)
plot_individual_lines(ax=barplot, data=df, y_col="select_target")
plt.ylabel("Response accuracy")
plt.xlabel("Priming")
barplot.set_xticklabels(["Negative", "No", "Positive"])

# priming reaction time
plt.figure()
barplot = sns.barplot(data=df, x="Priming", y="rt", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, y_col="rt")
plt.ylabel("Reaction time")
plt.xlabel("Priming")
barplot.set_xticklabels(["Negative", "No", "Positive"])

plt.figure()
barplot = sns.barplot(data=df, x="TargetLoc", y="select_target", errorbar=None)
