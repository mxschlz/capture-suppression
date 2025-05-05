import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers
from scipy import stats
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)
# some cleaning
df_mean = df.groupby(["subject_id", "TargetLoc"])[["select_target", "rt"]].mean().reset_index()
barplot = sns.barplot(data=df_mean, x="TargetLoc", y="select_target")
plot_individual_lines(barplot, data=df, x_col="TargetLoc", y_col="select_target")
plt.xlabel("Target Position")
plt.ylabel("Proportion correct")
barplot.set_xticklabels([-90, 0, 90])
# do stats
stats.ttest_rel(df_mean.query("TargetLoc==2")["rt"], df_mean.query("TargetLoc==1")["rt"])
