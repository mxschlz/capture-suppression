import matplotlib.pyplot as plt
import seaborn as sns
import SPACEPRIME
import pingouin as pg

sns.set_theme("talk", "ticks")

analysis_df = SPACEPRIME.load_concatenated_csv("target_towardness_all_variables.csv")

# ===================================================================
#       ANALYSIS: CORRELATION OF PRIMING EFFECT STRENGTHS
# ===================================================================
print("\n--- Correlating Positive vs. Negative Priming Effect Strengths ---")
print("Objective: Test if subjects with a strong negative priming effect also show a strong positive priming effect.")

# --- Step 1: Get subject-level mean towardness for each priming condition ---
# We use 'target_towardness' as the metric for the priming effect.
subject_means_df = analysis_df.groupby(["subject_id", "Priming"])["target_towardness"].mean().reset_index()

# --- Step 2: Pivot the data to a wide format ---
# This creates a DataFrame where each row is a subject and columns are the priming conditions.
# This structure is essential for calculating the per-subject effect.
priming_effects_df = subject_means_df.pivot(
    index='subject_id',
    columns='Priming',
    values='target_towardness'
).rename(columns={-1: 'negative', 0: 'no_priming', 1: 'positive'})

# Drop any subjects who might be missing data for one of the conditions
priming_effects_df.dropna(inplace=True)
print(f"Analyzing priming effects for {len(priming_effects_df)} subjects with complete data.")

# --- Step 3: Calculate the priming effect sizes for each subject ---
# The "effect" is the change in performance relative to the 'no priming' baseline.
# A positive value for both effects indicates the expected behavior.
# Positive Effect: Performance gain from positive priming.
# Negative Effect: Performance cost from negative priming (calculated as a positive value).
priming_effects_df['positive_effect'] = priming_effects_df['positive'] - priming_effects_df['no_priming']
priming_effects_df['negative_effect'] = priming_effects_df['negative'] - priming_effects_df['no_priming']

# --- Step 4: Visualize the correlation and calculate statistics ---
plt.figure(figsize=(10, 8))

# Create the regression plot
ax = sns.regplot(
    data=priming_effects_df,
    x='negative_effect',
    y='positive_effect',
    scatter_kws={'alpha': 0.6, 's': 50}, # Make points slightly transparent
    line_kws={'color': 'red', 'linewidth': 2}
)

# Calculate Spearman's rank correlation using Pingouin to include Bayes Factor
# Since Pingouin only returns BF10 for Pearson, we run Pearson on the ranks, which is equivalent to Spearman.
stats = pg.corr(priming_effects_df['negative_effect'].rank(), priming_effects_df['positive_effect'].rank(), method='pearson')
rho = stats['r'].iloc[0]
p_value = stats['p-val'].iloc[0]
bf10 = stats['BF10'].iloc[0]

print("\n--- Correlation Statistics with Bayes Factor ---")
print(stats)

# Using newlines in the labels is still good practice
ax.set_xlabel('Negative Priming Effect')
ax.set_ylabel('Positive Priming Effect')

# Filter ticks to show every other tick, but ensure at least 3 ticks remain
xticks = ax.get_xticks()
if len(xticks[::2]) >= 3:
    ax.set_xticks(xticks[::2])

yticks = ax.get_yticks()
if len(yticks[::2]) >= 3:
    ax.set_yticks(yticks[::2])

sns.despine()

# Determine significance asterisks
if p_value < 0.001:
    stars = '***'
elif p_value < 0.01:
    stars = '**'
elif p_value < 0.05:
    stars = '*'
else:
    stars = 'ns'

# Add a text box with the correlation results
stats_text = f"ρ = {rho:.3f} {stars}"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig("plots/priming_effect_correlation.svg")
plt.show()
