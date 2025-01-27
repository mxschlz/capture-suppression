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
#df = pd.concat([pd.read_csv(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean.csv") for subject in subjects])
all_data = []
sub_id = 104
fp = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{sub_id}/beh/"
for file in os.listdir(fp):
    if sub_id != 101:
        if "csv" in file:
            if "flanker" not in file:
                all_data.append(pd.read_csv(os.path.join(fp, file)))
    else:
        if "xlsx" in file:
            all_data.append(pd.read_excel(os.path.join(fp, file)))
df = pd.concat(all_data)
df = df[df["event_type"]=="mouse_click"]

df["iscorrect"] = df["response"].astype(int) == df["TargetDigit"].astype(int)
# get only singleton present trials
df = df[df["SingletonPresent"] == True]
# because the original dataframe does not contain these, make columns for accuracy to distractors and controls
df["disconf"] = df["response"].astype(int) == df["SingletonDigit"].astype(int)
df["controlconf"] = (df["response"].astype(int) == df["Non-Singleton2Digit"].astype(int)) | (df["response"].astype(int) == df["Non-Singleton1Digit"].astype(int))
df["otherconf"] = (df["response"].astype(int) != df["Non-Singleton2Digit"].astype(int)) & (df["response"].astype(int) != df["Non-Singleton1Digit"].astype(int)) & (df["response"].astype(int) != df["TargetDigit"].astype(int)) & (df["response"].astype(int) != df["SingletonDigit"].astype(int))
# Reshape the data
df_long = pd.melt(df, id_vars=['block', "subject_id"], value_vars=['iscorrect', 'disconf', 'controlconf', 'otherconf'])
barplot = sns.barplot(x='block', y='value', hue="variable", data=df_long, errorbar=("se", 1), palette=['forestgreen', 'red', 'grey', 'purple'])
# get only the incorrect trials
incorrects = df[df["iscorrect"] == False]
# plot a heatmap to see which response is given in incorrect trials
crosstab = pd.crosstab(index=incorrects["response"], columns=incorrects["TargetDigit"])
heatmap = sns.heatmap(crosstab)
heatmap.invert_yaxis()
# plot some stuff
barplot = sns.barplot(data=df, x="SingletonPresent", y="iscorrect", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="iscorrect")