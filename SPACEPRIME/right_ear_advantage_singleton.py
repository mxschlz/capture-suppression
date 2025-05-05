import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from scipy import stats
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df.query("SingletonPresent==1")
# some cleaning
barplot = sns.barplot(data=df, x="SingletonLoc", y="select_target")
plot_individual_lines(barplot, data=df, x_col="SingletonLoc", y_col="select_target")
plt.xlabel("Distractor Position")
plt.ylabel("Proportion correct")
barplot.set_xticklabels([-90, 0, 90])
# do stats
