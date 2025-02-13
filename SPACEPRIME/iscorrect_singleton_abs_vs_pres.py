import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from SPACEPRIME import get_data_path
from SPACEPRIME.plotting import plot_individual_lines
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
sub_ids = [103, 104, 105, 106, 107, 108, 110]
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in sub_ids])
# plot
barplot = sns.barplot(data=df, x="SingletonPresent", y="select_target", errorbar=("se", 1), hue="target_modulation")
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="select_target")
plt.xlabel("Singleton Distractor")
plt.ylabel("Proportion Correct")


from scipy.stats import ttest_ind
x = df.rt[df.SingletonPresent==1]
y = df.rt[df.SingletonPresent==0]
ttest_ind(x, y)
