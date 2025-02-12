import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 108, 110]])
df = df[df["SingletonPresent"] == True]
# Reshape the data
df_long = pd.melt(df, id_vars=['block', "subject_id"], value_vars=['select_target', 'select_distractor', 'select_control', 'select_other'])
fig, ax = plt.subplots(2, 1)
sns.barplot(x='block', y='value', hue="variable", data=df_long.query("subject_id%2==1"), errorbar=("se", 1),
            palette=['forestgreen', 'red', 'grey', 'purple'], ax=ax[0])
sns.barplot(x='block', y='value', hue="variable", data=df_long.query("subject_id%2==0"), errorbar=("se", 1),
            palette=['forestgreen', 'red', 'grey', 'purple'], ax=ax[1])

# get only the incorrect trials
incorrects = df[df["select_target"] == False]
# plot a heatmap to see which response is given in incorrect trials
crosstab = pd.crosstab(index=incorrects["response"], columns=incorrects["TargetDigit"])
heatmap = sns.heatmap(crosstab)
heatmap.invert_yaxis()
