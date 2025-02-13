import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
from SPACEPRIME import get_data_path
from scipy.stats import ttest_ind


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
sub_ids = [103, 104, 105, 106, 108, 110]
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in sub_ids])
df = df[df["SingletonPresent"] == 1]
# plot
barplot = sns.barplot(data=df, x="Priming", y="select_target", errorbar=("se", 1), hue="target_modulation")
plot_individual_lines(ax=barplot, data=df, y_col="select_target")
plt.xlabel("Priming")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Negative", "No", "Positive"])
# ttest
t, pval = ttest_ind(df.query("Priming==-1")["rt"], df.query("Priming==1")["rt"], nan_policy="omit")
