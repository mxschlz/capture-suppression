import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from stats import remove_outliers
from stats import cronbach_alpha
from statannotations.Annotator import Annotator
plt.ion()


# load df
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

# divide into subblocks (optional)
df['sub_block'] = df.index // 180  # choose division arbitrarily
df_singleton_absent = df[df['SingletonPresent'] == 0]
df_singleton_present = df[df['SingletonPresent'] == 1]

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_absent_mean = (df_singleton_absent.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_absent'))

# Calculate the mean of iscorrect for each block and subject_id
df_singleton_present_mean = (df_singleton_present.groupby(['sub_block', "subject_id"])['rt']
                       .mean().reset_index(name='rt_singleton_present'))

# Merge df_singleton_absent_mean and df_singleton_present_mean on block and subject_id
df_merged = pd.merge(df_singleton_absent_mean, df_singleton_present_mean, on=['sub_block', 'subject_id'])

# Calculate the difference between iscorrect_singleton_absent and iscorrect_singleton_present
df_merged['rt_diff'] = df_merged['rt_singleton_absent'] - df_merged['rt_singleton_present']

# running average
window_size = 3
# Apply running average *per subject*
df_merged['rt_diff_running_avg'] = df_merged.groupby('subject_id')['rt_diff'].transform(
    lambda x: x.rolling(window=window_size, min_periods=None, center=True).mean())

# --- Poster-Ready Line Plot ---
print("\n--- Generating Poster-Ready Plot ---")

# Set a clean and professional style for the plot
sns.set_style("ticks")
# Use a context that scales fonts and lines, perfect for presentations or posters
sns.set_context("talk")

# Create the figure and axes with a good aspect ratio for clarity
fig, ax = plt.subplots(figsize=(10, 7))

# Define a clear, strong color for the plot line
plot_color = sns.color_palette("deep")[0]  # A nice, deep blue

# Plot the lineplot with enhanced aesthetics for visibility
sns.lineplot(
    x='sub_block',
    y='rt_diff',
    data=df_merged,
    color=plot_color,
    errorbar=("se", 2),  # Show standard error (x2 for ~95% CI)
    linewidth=3.5,       # Thicker line for better visibility from a distance
    marker='o',          # Add markers to emphasize each block's data point
    markersize=9,        # Make markers larger and clearer
    legend=False,        # Explicitly disable the legend
    ax=ax
)

# Add a horizontal line at y=0 to indicate the baseline of no difference
ax.axhline(y=0, linestyle='--', color='black', linewidth=2, alpha=0.7)

# --- Set Labels and Title with Larger, Clearer Fonts ---
ax.set_title(
    "Transition from Distractor Attentional Capture to Suppression",
    fontsize=22,
    fontweight='bold', # Make the title stand out
    pad=25             # Add padding between title and plot
)
ax.set_xlabel(
    "Experimental Block",
    fontsize=18,
    labelpad=15        # Add padding below the x-axis label
)
# Assuming RT is in seconds. If it's milliseconds, change to "[ms]".
# A two-line label can improve readability.
ax.set_ylabel(
    "RT Difference (s)\n(Distractor Absent − Present)",
    fontsize=18,
    labelpad=15        # Add padding to the left of the y-axis label
)

# Customize tick parameters for clarity
ax.tick_params(axis='both', which='major', labelsize=16)

# Ensure x-axis ticks are integers, as they represent discrete blocks
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Remove top and right plot spines for a cleaner, modern look
sns.despine(ax=ax, trim=True)

# Adjust layout to ensure everything fits perfectly without overlapping
plt.tight_layout()

# Pro-tip: Save the figure with high resolution for your poster
# fig.savefig("poster_plot_rt_transition.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# stats
anova_correct = AnovaRM(df_merged, depvar='rt_diff', subject='subject_id', within=['sub_block'], aggregate_func="mean").fit()
print(anova_correct.summary())

from scipy.stats import page_trend_test

print("\n--- Page's Trend Test for Monotonic Increase ---")

# The data needs to be in a "wide" format: (subjects x conditions)
# You already created this for the Cronbach's Alpha calculation.
# We'll reuse df_pivot.
df_pivot = df_merged.pivot(index="subject_id", columns="sub_block", values='rt_diff')

# The test can't handle missing values, so we'll drop any subjects
# who might be missing a block for any reason.
df_pivot_clean = df_pivot.dropna()

print(f"Running test on {len(df_pivot_clean)} complete subjects out of {len(df_pivot)} total.")

# Perform the test.
# We use alternative='greater' because we hypothesize an INCREASING trend
# in rt_diff (from negative capture to positive suppression).
# The default predicted_ranks=(1, 2, 3...) is exactly what we want.
L_statistic, p_value = page_trend_test(df_pivot_clean)

print(f"\nPage's L statistic: {L_statistic:.2f}")
print(f"P-value: {p_value:.5f}")

# --- Interpretation ---
if p_value < 0.05:
    print("\n> Conclusion: The result is significant. We can reject the null hypothesis.")
    print("> There is strong evidence for a monotonically increasing trend in the RT difference across blocks.")
    print("> This supports the interpretation of a systematic shift from attentional capture to suppression.")
else:
    print("\n> Conclusion: The result is not significant.")
    print("> We cannot conclude there is a monotonic trend in the RT difference across blocks.")


# --- Plotting ---
plt, ax = plt.subplots()
# Lineplot with running average
sns.lineplot(x='sub_block', y='rt_diff_running_avg', hue='subject_id', data=df_merged,
                    palette="tab20", legend=True, alpha=0.7, ax=ax)  # Added alpha for better visibility of overlapping lines

# Mean running average across subjects (bold line)
mean_running_avg = df_merged.groupby('sub_block')['rt_diff_running_avg'].mean()
ax.plot(mean_running_avg.index, mean_running_avg.values, color='black', linewidth=3, label='Mean Running Avg')

# Baseline at 0
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Labels and title
ax.set_xlabel('Sub-Block')
ax.set_ylabel('Reaction Time Difference (Absent - Present)')
ax.set_title(f'Reaction Time Difference with Running Average (Window = {window_size})')
ax.legend("")
plt.tight_layout()

# regression plot
sns.lmplot(data=df_merged, x="sub_block", y="rt_diff_running_avg", hue="subject_id", palette="tab20", scatter=False,
           ci=None, legend=False)
# run linear mixed model
df["trial_nr_abs"] = list(range(len(df)))
df.drop("duration", axis=1, inplace=True)  # drop duration because it is always NaN
df.dropna(subset="rt", inplace=True)  # drop NaN in reaction time
model = smf.mixedlm("rt_diff ~ sub_block", data=df_merged.dropna(), groups="subject_id", re_formula="~sub_block")
result = model.fit()
print(result.summary())

# Cronbach Alpha
df_pivot = df_merged.pivot(index="subject_id", columns="sub_block", values='rt_diff')
cronbach_alpha(data=df_pivot)
