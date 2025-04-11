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
import numpy as np


df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)
#df = df[df["SingletonPresent"] == 0]
df_mean = df.groupby(["subject_id", "Priming", "block"])[["select_target", "rt"]].mean().reset_index()
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
ax[0].set_xticklabels(["Negative", "No", "Positive"])  # The colored axes labels serve as the legend
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
# Look at priming as a function of time
# Since we are interested in how priming develops over time, we want to look at reaction times/accuracies as a function
# of blocks. We set the hue parameter to "Priming", so that we isolate barplots according to negative and positive priming
# conditions. Because doing so with the current df_mean would yield a very busy plot, we are going to compute the difference
# between priming conditions (negative/positive priming - control).
# Your existing code to create subsets:
df_np = df_mean.query("Priming==-1").copy()
df_no_p = df_mean.query("Priming==0").copy()
df_pp = df_mean.query("Priming==1").copy()
# Define columns that we want to subtract
cols_to_subtract = ["rt", "select_target"]
# Capture the original indices from df_mean for the -1 and 1 priming conditions
np_original_indices = df_np.index
pp_original_indices = df_pp.index
# Calculate differences (this part is correct under your assumption)
# We reset index here for the subtraction calculation only
diff_np = df_np[cols_to_subtract].reset_index(drop=True).subtract(df_no_p[cols_to_subtract].reset_index(drop=True))
diff_pp = df_pp[cols_to_subtract].reset_index(drop=True).subtract(df_no_p[cols_to_subtract].reset_index(drop=True))
# Create the new column names for the differences
np_diff_cols = {col: f"np_diff_{col}" for col in cols_to_subtract}
pp_diff_cols = {col: f"pp_diff_{col}" for col in cols_to_subtract}
# Rename the columns in the difference dataframes
diff_np.rename(columns=np_diff_cols, inplace=True)
diff_pp.rename(columns=pp_diff_cols, inplace=True)
# --- Map the differences back to the original df_mean ---
# Initialize the new columns in df_mean with NaN
for col_name in list(np_diff_cols.values()) + list(pp_diff_cols.values()):
    df_mean[col_name] = np.nan
# Assign the calculated differences to the correct rows in df_mean
# using the original indices captured earlier.
# Important: We need to align the index of the diff dataframes
# with the original indices before assignment using .loc.
diff_np.index = np_original_indices
diff_pp.index = pp_original_indices
# Now use .loc for index-based assignment
df_mean.loc[np_original_indices, diff_np.columns] = diff_np
df_mean.loc[pp_original_indices, diff_pp.columns] = diff_pp

# Plot the stuff
fig, ax = plt.subplots(1, 2)
sns.barplot(data=df_mean, x="block", y="select_target", hue="Priming", ax=ax[0])
sns.barplot(data=df_mean, x="block", y="rt", hue="Priming", ax=ax[1])
plt.tight_layout()
# Plot the difference between positive and negative priming compared to baseline directly
# Plot the stuff
fig, ax = plt.subplots(1, 2)
sns.barplot(data=df_mean, x="block", y="np_diff_select_target", ax=ax[0])
sns.barplot(data=df_mean, x="block", y="pp_diff_rt", ax=ax[1])
plt.tight_layout()

# compute effect size
compute_effsize_from_t(t, N=n)

# compute balanced integration score (BIS)
# **Important Assumption**: 'select_target' is the accuracy/PC (Proportion Correct).
# If not, you need to calculate PC first before this step.
# Example: If you had a 'correct' column (1/0), you'd use .agg(..., mean_pc=('correct', 'mean'))
grouping_cols = ["subject_id", "Priming"]
df_agg = df.groupby(grouping_cols).agg(
    mean_rt=('rt', 'mean'),
    mean_pc=('select_target', 'mean') # Calculate mean accuracy per cell
).reset_index() # Use reset_index to turn grouping keys back into columns
print(f"--- Aggregated Data (by {', '.join(grouping_cols)}) ---")
print(df_agg)
print("\n")
# 2. Standardize (Z-transform) Aggregated Measures ACROSS Design Cells
# Calculate the overall mean and standard deviation for median_rt and mean_pc
# Use ddof=1 for sample standard deviation (unbiased estimate)
overall_mean_rt = df_agg['mean_rt'].mean()
overall_std_rt = df_agg['mean_rt'].std(ddof=1)
overall_mean_pc = df_agg['mean_pc'].mean()
overall_std_pc = df_agg['mean_pc'].std(ddof=1)
print(f"Overall Mean Mean RT: {overall_mean_rt:.4f}, Std Dev: {overall_std_rt:.4f}")
print(f"Overall Mean Mean PC: {overall_mean_pc:.4f}, Std Dev: {overall_std_pc:.4f}")
print("\n")
df_agg['zAcc'] = (df_agg['mean_pc'] - overall_mean_pc) / overall_std_pc
# 3. Calculate Balanced Integration Score (BIS)
df_agg['BIS'] = df_agg['zAcc'] - df_agg['zRT']
# Plot BIS as a function of Priming
sns.barplot(data=df_agg, x="Priming", y="BIS")
# stats
ttest_rel(df_agg.query("Priming==1")["BIS"].astype(float), df_agg.query("Priming==0")["BIS"].astype(float), nan_policy="omit")
