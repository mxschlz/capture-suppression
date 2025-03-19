import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.plotting import plot_individual_lines
from SPACEPRIME.subjects import subject_ids
plt.ion()


# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
# plot
barplot = sns.barplot(data=df, x="SingletonPresent", y="select_target", errorbar=("se", 1), hue="target_modulation")
plot_individual_lines(ax=barplot, data=df, x_col="SingletonPresent", y_col="select_target")
plt.xlabel("Singleton Distractor")
plt.ylabel("Proportion Correct")


from scipy.stats import ttest_rel
x = df.query("SingletonPresent==0").groupby(["subject_id"])["select_target"].mean().astype(float)
y = df.query("SingletonPresent==1").groupby(["subject_id"])["select_target"].mean().astype(float)
ttest_rel(x, y)
