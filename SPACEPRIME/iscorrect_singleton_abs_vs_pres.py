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
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects])
# plot
barplot = sns.barplot(data=df, x="SingletonPresent", y="select_target", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="select_target")
plt.xlabel("Singleton Distractor")
plt.ylabel("Proportion Correct")


from stats import permutation_test

x = df.iscorrect[df.SingletonPresent==1]
y = df.iscorrect[df.SingletonPresent==0]
permutation_test(x, y)
