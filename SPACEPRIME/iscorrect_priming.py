import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
from SPACEPRIME import get_data_path
from scipy.stats import ttest_ind
from stats import cronbach_alpha
from SPACEPRIME.subjects import subject_ids


df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
#df = df[df["SingletonPresent"] == 0]
# plot
barplot = sns.barplot(data=df, x="Priming", y="select_target", errorbar=("se", 1))
plot_individual_lines(ax=barplot, data=df, y_col="select_target")
plt.xlabel("Priming")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Negative", "No", "Positive"])
# ttest
t, pval = ttest_ind(df.query("Priming==-1")["select_target"].astype(int),
                    df.query("Priming==0")["select_target"].astype(int), nan_policy="omit")

df_pivot = df.groupby(["subject_id", "Priming"])["select_target"].mean().reset_index().pivot(index="subject_id",
                                                                                             columns="Priming",
                                                                                             values="select_target").astype(float)
cronbach_alpha(data=df_pivot)
