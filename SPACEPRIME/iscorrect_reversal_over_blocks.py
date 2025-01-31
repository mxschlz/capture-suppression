import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
plt.ion()


# load up the data
# load data from children
#df = pd.read_excel("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME_behavioral_pilot_n-5/results_July_06_2024_14_16_40.xlsx")
# some cleaning
#df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]# Filter the dataframe to only include rows where SingletonPresent is 0
# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects])
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['block', "subject_id"])['select_target']
                       .mean().reset_index(name='iscorrect_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['block', "subject_id"])['select_target']
                       .mean().reset_index(name='iscorrect_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['iscorrect_diff'] = df_merged['iscorrect_singleton_absent'] - df_merged['iscorrect_singleton_present']

# Add labels and title
fig, ax = plt.subplots(figsize=(6, 4))
barplot = sns.barplot(x='block', y='iscorrect_diff', data=df_merged[df_merged["subject_id"]==105], errorbar=("se", 1))
ax.set_ylabel('Proportion correct (Distractor absent - Distractor pesent)')
ax.set_xlabel('Block')
# ax.set_ylim(-0.1, 0.1)
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
