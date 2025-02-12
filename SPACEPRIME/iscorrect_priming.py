import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
from scipy.stats import ttest_ind


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 108, 110]])
df = df[df["SingletonPresent"] == 1]
# plot
barplot = sns.barplot(data=df, x="Priming", y="select_target", errorbar=("se", 1), hue="target_modulation")
plot_individual_lines(ax=barplot, data=df, y_col="select_target")
plt.xlabel("Priming")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Negative", "No", "Positive"])
# ttest
t, pval = ttest_ind(df.query("Priming==-1")["rt"], df.query("Priming==1")["rt"], nan_policy="omit")
