import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects])
sub_id = 103
df = df[df["subject_id"]==sub_id]
df = df[df["SingletonPresent"] == True]
# Reshape the data
df_long = pd.melt(df, id_vars=['block', "subject_id"], value_vars=['select_target', 'select_distractor', 'select_control', 'select_other'])
barplot = sns.barplot(x='block', y='value', hue="variable", data=df_long, errorbar=("se", 1), palette=['forestgreen', 'red', 'grey', 'purple'])
# get only the incorrect trials
incorrects = df[df["select_target"] == False]
# plot a heatmap to see which response is given in incorrect trials
crosstab = pd.crosstab(index=incorrects["response"], columns=incorrects["TargetDigit"])
heatmap = sns.heatmap(crosstab)
heatmap.invert_yaxis()
# plot some stuff
barplot = sns.barplot(data=df, x="SingletonPresent", y="select_target", errorbar=None)
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="select_target")