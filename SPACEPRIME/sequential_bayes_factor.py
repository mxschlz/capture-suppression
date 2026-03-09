import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(context="talk", style="ticks")
from SPACEPRIME import load_concatenated_csv
import pingouin as pg


df = load_concatenated_csv("target_towardness_all_variables.csv")

# 1. Calculate behavioral metrics on a subject level
subject_metrics = df.groupby(['subject_id', 'PrimingCondition'])[["target_towardness"]].mean().reset_index()

# 2. Prepare data for Sequential Bayes Factor Analysis
# Pivot to wide format to facilitate paired comparisons
df_wide = subject_metrics.pivot(index='subject_id', columns='PrimingCondition', values='target_towardness')

# Define condition labels (Update these strings to match the exact values in your CSV)
cond_control = 'No'
cond_pos = 'Positive'
cond_neg = 'Negative'

# Filter for subjects that have data in all relevant conditions to ensure valid paired tests
df_wide = df_wide.dropna(subset=[cond_control, cond_pos, cond_neg])
subjects = df_wide.index.tolist()

# 3. Calculate Sequential Bayes Factors
results = []

# Iterate through subjects, adding one at a time (starting from N=2 to allow variance calculation)
for i in range(2, len(subjects) + 1):
    current_subset = df_wide.iloc[:i]

    # Calculate BF for Positive vs Control (Paired t-test)
    bf_pos = pg.ttest(current_subset[cond_pos], current_subset[cond_control], paired=True)['BF10'].values[0]

    # Calculate BF for Negative vs Control (Paired t-test)
    bf_neg = pg.ttest(current_subset[cond_neg], current_subset[cond_control], paired=True)['BF10'].values[0]

    results.append({
        'N': i,
        'BF_Positive': float(bf_pos),
        'BF_Negative': float(bf_neg)
    })

bf_df = pd.DataFrame(results)

# 4. Plotting
plot_log_space = True  # Set to True for log scale, False for raw Bayes Factors
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot Positive
axes[0].plot(bf_df['N'], bf_df['BF_Positive'], label='Positive vs. No Priming', marker='o', color='tab:blue')
axes[0].set_title('Positive vs. No Priming')

# Add vertical line where BF > 3
if (bf_df['BF_Positive'] > 3).any():
    n_cross = bf_df.loc[bf_df['BF_Positive'] > 3, 'N'].iloc[0]
    axes[0].axvline(x=n_cross, color='grey', linestyle='--', label=f'BF > 3 at N={n_cross}')

# Plot Negative
axes[1].plot(bf_df['N'], bf_df['BF_Negative'], label='Negative vs. No Priming', marker='o', color='tab:orange')
axes[1].set_title('Negative vs. No Priming')

# Add vertical line where BF > 3
if (bf_df['BF_Negative'] > 3).any():
    n_cross = bf_df.loc[bf_df['BF_Negative'] > 3, 'N'].iloc[0]
    axes[1].axvline(x=n_cross, color='grey', linestyle='--', label=f'BF > 3 at N={n_cross}')

for ax in axes:
    if plot_log_space:
        ax.set_yscale('log')
        ax.set_ylabel('Bayes Factor (Log Scale)')
    else:
        ax.set_ylabel('Bayes Factor')

    # Add evidence threshold lines (Jeffreys' scale)
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (BF=1)')
    ax.axhline(y=3, color='green', linestyle='--', linewidth=1, label='Evidence for H1 (BF=3)')
    ax.axhline(y=1/3, color='red', linestyle='--', linewidth=1, label='Evidence for H0 (BF=1/3)')

    # Ensure limits include the thresholds
    bottom, top = ax.get_ylim()
    if plot_log_space:
        ax.set_ylim(min(bottom, 0.2), max(top, 10))
    else:
        ax.set_ylim(0, max(top, 4.0))
    ax.legend()

plt.xlabel('Number of Participants')
sns.despine(fig=fig)
plt.tight_layout()
plt.show()
