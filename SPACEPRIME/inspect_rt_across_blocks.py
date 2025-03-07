import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from stats import remove_outliers
from stats import cronbach_alpha
plt.ion()


# load df
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

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

# running average
window_size = 3
# Apply running average *per subject*
df_merged['rt_diff_running_avg'] = df_merged.groupby('subject_id')['rt_diff'].transform(
    lambda x: x.rolling(window=window_size, min_periods=None, center=True).mean())

# Add labels and title
plt.figure()
sns.boxplot(x='sub_block', y='rt_diff', data=df_merged)
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

# stats
anova_correct = AnovaRM(df_merged, depvar='rt_diff', subject='subject_id', within=['sub_block'], aggregate_func="mean").fit()
print(anova_correct.summary())

# --- Plotting ---
plt.figure()
# Lineplot with running average
sns.lineplot(x='sub_block', y='rt_diff_running_avg', hue='subject_id', data=df_merged,
             palette="tab20", legend=True, alpha=0.7)  # Added alpha for better visibility of overlapping lines

# Mean running average across subjects (bold line)
mean_running_avg = df_merged.groupby('sub_block')['rt_diff_running_avg'].mean()
plt.plot(mean_running_avg.index, mean_running_avg.values, color='black', linewidth=3, label='Mean Running Avg')

# Baseline at 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Labels and title
plt.xlabel('Sub-Block')
plt.ylabel('Reaction Time Difference (Absent - Present)')
plt.title(f'Reaction Time Difference with Running Average (Window = {window_size})')
plt.legend("")

# regression plot
sns.lmplot(data=df_merged, x="sub_block", y="rt_diff_running_avg", hue="subject_id", palette="tab20", scatter=False,
           ci=None)
# run linear mixed model
df["trial_nr_abs"] = list(range(len(df)))
df.drop("duration", axis=1, inplace=True)  # drop duration because it is always NaN
df.dropna(subset="rt", inplace=True)  # drop NaN in reaction time
model = smf.mixedlm("rt_diff ~ sub_block", data=df_merged, groups="subject_id", re_formula="~sub_block")
result = model.fit()
print(result.summary())

# Cronbach Alpha
df_pivot = df_merged.pivot(index="subject_id", columns="sub_block", values='rt_diff')
cronbach_alpha(data=df_pivot)
