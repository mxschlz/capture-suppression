import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from scipy.stats import ttest_rel
from stats import remove_outliers
from SPACEPRIME.subjects import subject_ids
from stats import compute_effsize_from_t
from statannotations.Annotator import Annotator
from SPACEPRIME.plotting import plot_individual_lines


df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)
#df = df[df["SingletonPresent"] == 0]
df_mean = df.groupby(["subject_id", "Priming"])[["select_target", "rt"]].mean().reset_index()
# Create plots
fig, ax = plt.subplots(1, 2) # Slightly larger figure
# Define colors
color_acc = 'tab:blue'
color_rt = 'tab:red'
# --- Plot Accuracy Boxplot on ax1 (Left Y-axis) ---
sns.barplot(data=df_mean, x="Priming", y="select_target",
            order=[-1, 0, 1], # Use numbers matching your data
            color=color_acc, # Use a single color for clarity or a palette
            ax=ax[0],
            width=0.8) # Adjust box width if desired
plot_individual_lines(ax=ax[0], data=df_mean, x_col="Priming", y_col="select_target")
ax[0].set_xlabel("Priming Condition")
ax[0].set_ylabel("Proportion Correct")
ax[0].tick_params(axis='y')
# Adjust ylim based on data + potential annotation space
acc_max = df_mean['select_target'].max()
ax[0].set_ylim(0.2, 0.7) # Add extra space above
ax[0].set_xticklabels(["Negative", "No", "Positive"])# The colored axes labels serve as the legend
# plot reaction time
sns.barplot(data=df_mean, x="Priming", y="rt",
            order=[-1, 0, 1], # Use numbers matching your data
            color=color_rt,
            ax=ax[1],
            width=0.8)
plot_individual_lines(ax=ax[1], data=df_mean, x_col="Priming", y_col="rt")
ax[1].set_ylabel("Reaction Time (s)")
ax[1].tick_params(axis='y')
# Adjust ylim based on data + potential annotation space
rt_max = df_mean['rt'].max()
ax[1].set_ylim(1.2, 1.6) # Add extra space above
# --- Final Touches ---
fig.suptitle("Accuracy and Reaction Time by Priming Condition")
# Set specific tick labels for the x-axis
ax[1].set_xticklabels(["Negative", "No", "Positive"])# The colored axes labels serve as the legend
fig.tight_layout() # Adjust layout to prevent overlap
# add stats annotations
pairs = [[0, -1], [0, 1], [1, -1]]
# annotate ax1 --> accuracy
annotator1 = Annotator(ax=ax[0], plot="barplot", pairs=pairs, data=df_mean, x="Priming", y="select_target")
annotator1.configure(test="t-test_paired", text_format="star", comparisons_correction="bonferroni")
annotator1.apply_and_annotate()
# annotate ax2 --> reaction time
annotator2 = Annotator(ax=ax[1], plot="barplot", pairs=pairs, data=df_mean, x="Priming", y="rt")
annotator2.configure(test="t-test_paired", text_format="star", comparisons_correction="bonferroni")
annotator2.apply_and_annotate()
# further stats
n = df.subject_id.unique().__len__()
t, pval = ttest_rel(df_mean.query("Priming==-1")["rt"].astype(float), df_mean.query("Priming==0")["rt"].astype(float),
                    nan_policy="omit")
# compute effect size
compute_effsize_from_t(t, N=n)
