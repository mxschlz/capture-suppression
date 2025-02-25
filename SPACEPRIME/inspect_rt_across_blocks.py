import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
import numpy as np
from SPACEPRIME.subjects import subject_ids
from statsmodels.stats.anova import AnovaRM
plt.ion()


# load df
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
# divide into subblocks (optional)
df['sub_block'] = np.floor(df.index / 180)  # choose division arbitrarily
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
