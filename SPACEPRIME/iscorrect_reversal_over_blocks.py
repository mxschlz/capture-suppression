import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from SPACEPRIME import get_data_path
import numpy as np
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
sub_ids = [108, 110, 112, 114]
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in sub_ids])
df['sub_block'] = np.floor(df.index / 180)  # Assuming df.index represents trial number (0-indexed)
#df = df[df["phase"]==1]
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['sub_block', "subject_id", "target_modulation"])['select_target']
                       .mean().reset_index(name='iscorrect_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['sub_block', "subject_id", "target_modulation"])['select_target']
                       .mean().reset_index(name='iscorrect_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['sub_block', 'subject_id', "target_modulation"])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['iscorrect_diff'] = df_merged['iscorrect_singleton_absent'] - df_merged['iscorrect_singleton_present']

# Add labels and title
fig, ax = plt.subplots(1, 1)
barplot = sns.barplot(x='sub_block', y='iscorrect_diff', data=df_merged.query("subject_id==114"), errorbar=("se", 1))
ax.set_ylabel('Proportion correct (Distractor absent - Distractor pesent)')
ax.set_xlabel('Sub-Block / Time Window')
# ax.set_ylim(-0.1, 0.1)
#ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
