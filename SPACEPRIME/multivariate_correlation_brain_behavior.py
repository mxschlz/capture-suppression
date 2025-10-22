import matplotlib.pyplot as plt
import numpy as np
import SPACEPRIME
import pandas as pd
import seaborn as sns
from scipy import stats

plt.ion()

# --- Script Configuration Parameters ---

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
SINGLETON_PRESENT_COL = 'SingletonPresent'
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'
ACCURACY_COL = 'select_target'
ACCURACY_INT_COL = 'select_target_int'

df = SPACEPRIME.load_concatenated_csv("target_towardness.csv", index_col=0)
n2ac_df = SPACEPRIME.load_concatenated_csv("spaceprime_n2ac_subject_averages.csv")[["subject_id", "st_latency_50", "st_mean_amp_50", "Priming"]]
pd_df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_subject_averages.csv")[["subject_id", "st_latency_50", "st_mean_amp_50", "Priming"]]

# Map priming values
PRIMING_MAP = {"np": "negative", "no-p": "no", "pp": "positive"}
n2ac_df["Priming"] = n2ac_df["Priming"].map(PRIMING_MAP)
pd_df["Priming"] = pd_df["Priming"].map(PRIMING_MAP)

# Average over priming conditions
n2ac_df = n2ac_df.groupby(SUBJECT_ID_COL)[["st_latency_50", "st_mean_amp_50"]].mean().reset_index()
pd_df = pd_df.groupby(SUBJECT_ID_COL)[["st_latency_50", "st_mean_amp_50"]].mean().reset_index()

# Flanker task configuration
FLANKER_EXPERIMENT_NAME = 'flanker_data.csv'
FLANKER_CONGRUENCY_COL = 'congruency'
FLANKER_ACC_COL = 'correct'

# --- 1. Calculate Flanker Effect ---
print("\nLoading and processing flanker task data...")
flanker_df = SPACEPRIME.load_concatenated_csv(FLANKER_EXPERIMENT_NAME)
flanker_df[SUBJECT_ID_COL] = flanker_df[SUBJECT_ID_COL].astype(int)
flanker_rt_by_subject = flanker_df.groupby(
    [SUBJECT_ID_COL, FLANKER_CONGRUENCY_COL]
)[REACTION_TIME_COL].mean().unstack()
flanker_rt_by_subject['flanker_effect'] = (flanker_rt_by_subject['incongruent'] - flanker_rt_by_subject['congruent']) * 1000
flanker_subject_df = flanker_rt_by_subject.reset_index()[[SUBJECT_ID_COL, 'flanker_effect']]
print("Flanker effect calculated per subject.")

# --- Calculate Singleton Presence Effect (RT and Accuracy) ---
print("\nCalculating singleton presence effect (RT & Accuracy) per subject...")

# The main 'df' contains the trial-by-trial data from the spaceprime task
# Ensure SingletonPresent is numeric for calculations
df[SINGLETON_PRESENT_COL] = pd.to_numeric(df[SINGLETON_PRESENT_COL], errors='coerce')

# Group by subject and singleton presence, then calculate mean RT and Accuracy
singleton_perf = df.groupby([SUBJECT_ID_COL, SINGLETON_PRESENT_COL])[[REACTION_TIME_COL, ACCURACY_COL]].mean()

# Unstack to get singleton present (1) / absent (0) as columns
singleton_perf_unstacked = singleton_perf.unstack(level=SINGLETON_PRESENT_COL)

# Flatten the multi-level column index (e.g., from ('rt', 0) to 'rt_0')
singleton_perf_unstacked.columns = [f'{col}_{int(level)}' for col, level in singleton_perf_unstacked.columns]

# Calculate the difference (Present - Absent).
# We multiply RT by 1000 to get ms, and accuracy by 100 to get percentage points.
singleton_perf_unstacked['singleton_rt_effect'] = (singleton_perf_unstacked[f'{REACTION_TIME_COL}_1'] - singleton_perf_unstacked[f'{REACTION_TIME_COL}_0']) * 1000
singleton_perf_unstacked['singleton_acc_effect'] = (singleton_perf_unstacked[f'{ACCURACY_COL}_1'] - singleton_perf_unstacked[f'{ACCURACY_COL}_0']).astype(float) * 100

# Reset index to make 'subject_id' a column for merging
singleton_effect_df = singleton_perf_unstacked.reset_index()

print("Singleton presence effect calculated.")

# --- 3. Load Questionnaire Data ---
print("Loading questionnaire data...")
questionnaire_data = SPACEPRIME.load_concatenated_csv("combined_questionnaire_results.csv")
questionnaire_data[SUBJECT_ID_COL] = questionnaire_data[SUBJECT_ID_COL].astype(int)

# 4. Load fooof data
print("Loading FOOOF data ...")
fooof_data = SPACEPRIME.load_concatenated_csv("fooof_exponents.csv")
fooof_data[SUBJECT_ID_COL] = fooof_data[SUBJECT_ID_COL].astype(int)

### MERGED ### --- Merge All Data for Correlation ---
print("\nMerging all EEG and behavioral data sources...")
# Start with a list of all dataframes to merge
data_frames = [flanker_subject_df, singleton_effect_df, questionnaire_data, fooof_data, n2ac_df, pd_df]

# Merge all dataframes on 'subject_id'
# Start with the first dataframe and iteratively merge the rest
merged_df = data_frames[0]
for i in range(1, len(data_frames)):
    merged_df = pd.merge(merged_df, data_frames[i], on=SUBJECT_ID_COL, how='inner')

correlation_df = merged_df.copy()
correlation_df.dropna(inplace=True)
print(f"Found {len(correlation_df)} subjects with complete data across all measures.")

# Merge the new wide-format ERP data with the main subject dataframe.
# An 'inner' merge ensures we only analyze subjects present in both datasets.
final_df = correlation_df.copy() # Now correlation_df already contains n2ac_df and pd_df

# This hardcoded outlier removal was in the original script.
# A more robust approach would be to identify outliers and remove by subject_id.
# TODO: to remove or not to remove ...
if 10 in final_df.index:
    print("Warning: Removing subject at hardcoded index 10 from correlation analysis.")
    #final_df = final_df.drop(index=10).reset_index(drop=True)

### MERGED ### --- Multivariate Correlation Analysis ---
print("\n--- Multivariate Correlation Analysis ---")
# --- 2. Perform Correlation Analysis ---
cols_to_correlate = final_df.columns

existing_cols = [col for col in cols_to_correlate if col in final_df.columns]
print(f"Analyzing correlations between: {existing_cols}")

# --- 1. Visualization: Focused Subplots for Significant Correlations ---
print("Generating focused plots for significant correlations (p < 0.05)...")

# --- Step 1: Identify all significant pairs ---
# Use the pre-calculated correlation and p-value matrices
corr_matrix = final_df[existing_cols].corr(method='spearman')
p_values = final_df[existing_cols].corr(method=lambda x, y: stats.spearmanr(x, y)[1])
SIGNIFICANCE_THRESHOLD = 0.05

# Get the lower triangle of the p-value matrix to avoid duplicate pairs
p_lower = p_values.where(np.tril(np.ones(p_values.shape), k=-1).astype(bool))

# Find row, col indices where the p-value is below the threshold
sig_rows, sig_cols = np.where(p_lower < SIGNIFICANCE_THRESHOLD)

# Create a list of (var1, var2) tuples for the significant pairs
significant_pairs = [
    (p_lower.index[r], p_lower.columns[c]) for r, c in zip(sig_rows, sig_cols)
]

n_significant = len(significant_pairs)

# --- Step 2: Create a dynamic subplot grid for the results ---
if n_significant == 0:
    print("No significant correlations found to plot.")
    # Create a plot that explicitly states this finding
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.text(0.5, 0.5, "No significant correlations found\n(Spearman, p < 0.05)",
            ha='center', va='center', fontsize=14, style='italic', color='grey')
    ax.set_axis_off()
    plt.show()
else:
    print(f"Found {n_significant} significant correlations. Generating plots...")
    # Determine grid size to be as square as possible
    cols = int(np.ceil(np.sqrt(n_significant)))
    rows = int(np.ceil(n_significant / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5), squeeze=False)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # --- Step 3: Plot each significant pair in its own subplot ---
    for i, (var1, var2) in enumerate(significant_pairs):
        ax = axes[i]

        # Draw the regression plot showing the distribution of data points
        sns.regplot(
            data=final_df, x=var1, y=var2, ax=ax,
            scatter_kws={'alpha': 0.7, 'edgecolor': 'w', 's': 50, 'color': 'steelblue'},
            line_kws={'color': 'crimson', 'lw': 2.5}
        )

        # Add clear annotations for r and p values
        r = corr_matrix.loc[var1, var2]
        p = p_values.loc[var1, var2]
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        annotation = f"Spearman's r = {r:.2f}\n{p_text}"
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.6))

        # Clean up titles and labels for readability
        ax.set_title(f"{var1.replace('_', ' ').title()}\nvs\n{var2.replace('_', ' ').title()}",
                     fontweight='bold', fontsize=12)
        ax.set_xlabel(var1.replace('_', ' '), fontsize=10)
        ax.set_ylabel(var2.replace('_', ' '), fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- Step 4: Clean up the overall figure ---
    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_significant, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Significant Brain-Behavior Correlations', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()

# --- 2. Visualization: Correlation Heatmap ---
print("Generating correlation matrix heatmap...")
corr_matrix = final_df[existing_cols].corr()
p_values = final_df[existing_cols].corr(method=lambda x, y: stats.spearmanr(x, y)[1])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, cmap='vlag', center=0, annot=False, linewidths=.5, vmin=-1, vmax=1, ax=ax)

# Add custom annotations with significance
for i in range(len(corr_matrix)):
    for j in range(i):  # Only iterate through lower triangle
        r, p = corr_matrix.iloc[i, j], p_values.iloc[i, j]
        text = f"{r:.2f}"
        if p < 0.001:
            text += "***"
        elif p < 0.01:
            text += "**"
        elif p < 0.05:
            text += "*"
        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="white" if abs(r) > 0.6 else "black")

ax.set_title('Spearman correlation matrix', fontsize=15, fontweight='bold')
ax.text(0.98, 0.98, "* p<0.05\n** p<0.01\n*** p<0.001", transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()

print("\n--- Correlation Matrix (r-values) ---")
print(corr_matrix.round(3))
print("\n--- P-Values ---")
print(p_values.round(3))
