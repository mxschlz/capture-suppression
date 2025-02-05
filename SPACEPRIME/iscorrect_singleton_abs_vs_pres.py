import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from SPACEPRIME.plotting import plot_individual_lines
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 107]])
# plot
barplot = sns.barplot(data=df, x="SingletonPresent", y="rt", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="rt")
plt.xlabel("Singleton Distractor")
#plt.ylabel("Proportion Correct")


from scipy.stats import ttest_ind
x = df.rt[df.SingletonPresent==1]
y = df.rt[df.SingletonPresent==0]
ttest_ind(x, y)
