import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
from SPACEPRIME import get_data_path
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
sub_ids = [104, 106, 108, 110, 112, 114]
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in sub_ids])
# some cleaning
barplot = sns.barplot(data=df, x="TargetLoc", y="select_target")
plot_individual_lines(barplot, data=df, x_col="TargetLoc", y_col="select_target")
plt.xlabel("Target Position")
plt.ylabel("Proportion correct")
barplot.set_xticklabels([-90, 0, 90])
