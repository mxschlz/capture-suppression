import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean.csv") for subject in subjects])
# because the original dataframe does not contain these, make columns for accuracy to distractors and controls
df["disconf"] = df["response"] == df["SingletonDigit"]
df["controlconf"] = (df["response"] == df["Non-Singleton2Digit"]) | (df["response"] == df["Non-Singleton1Digit"])
df["otherconf"] = (df["response"] != df["Non-Singleton2Digit"]) & (df["response"] != df["Non-Singleton1Digit"]) & (df["response"] != df["TargetDigit"]) & (df["response"] != df["SingletonDigit"])
# get only the incorrect trials
incorrects = df[df["iscorrect"] == False]
# plot a heatmap to see which response is given in incorrect trials
crosstab = pd.crosstab(index=incorrects["response"], columns=incorrects["TargetDigit"])
heatmap = sns.heatmap(crosstab)
heatmap.invert_yaxis()
# plot some stuff
barplot = sns.barplot(data=df, x="SingletonPresent", y="iscorrect", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="iscorrect")
# Reshape the data
df_long = pd.melt(df[df["block"]<5], id_vars=['block', "subject_id"], value_vars=['iscorrect', 'disconf', 'controlconf', 'otherconf'])
barplot = sns.barplot(x='block', y='value', hue="variable", data=df_long, errorbar=("se", 1), palette=['forestgreen', 'red', 'grey', 'purple'])
