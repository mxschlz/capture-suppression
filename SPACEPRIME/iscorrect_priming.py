import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
from SPACEPRIME import get_data_path
from scipy.stats import ttest_rel
from stats import remove_outliers
from SPACEPRIME.subjects import subject_ids
from stats import compute_effsize_from_t


df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)
#df = df[df["SingletonPresent"] == 0]
df_mean = df.groupby(["subject_id", "Priming"])[["select_target", "rt"]].mean().reset_index()
# plot
plt.figure()
barplot = sns.barplot(data=df_mean, x="Priming", y="rt", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df_mean, y_col="rt")
plt.xlabel("Priming")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Negative", "No", "Positive"])
# ttest
n = df.subject_id.unique().__len__()
t, pval = ttest_rel(df_mean.query("Priming==1")["select_target"].astype(float), df_mean.query("Priming==0")["select_target"].astype(float),
                    nan_policy="omit")
# compute effect size
compute_effsize_from_t(t, N=n)
#df_pivot = df.groupby(["subject_id", "Priming"])["select_target"].mean().reset_index().pivot(index="subject_id",
#                                                                                             columns="Priming",
#                                                                                             values="select_target").astype(float)
#cronbach_alpha(data=df_pivot)
