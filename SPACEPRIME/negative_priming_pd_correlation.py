import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
import SPACEPRIME
sns.set_theme("talk", "ticks")

df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_subject_averages.csv")

# 1. Calculate Subject-Averaged Target Towardness Difference (NP - No-P)
# Filter for relevant conditions
subset = df[df['Priming'].isin(['np', 'no-p'])]

# Pivot to wide format to calculate difference per subject
wide_df = subset.pivot(index='subject_id', columns='Priming', values='target_towardness')

# Calculate difference: Negative Priming - No Priming
wide_df['np_diff'] = wide_df['no-p'] - wide_df['np']

# 2. Prepare Pd Magnitude Data
# Averaging Pd metrics per subject (in case they vary by condition rows or are repeated)
pd_metrics = ['st_latency_50', 'st_amp_at_lat_50']
pd_df = df.groupby('subject_id')[pd_metrics].mean()

# 3. Merge Data
corr_df = wide_df.join(pd_df).dropna()

# 4. Compute Correlations and Plot
print("--- Correlation Results ---")
for metric in pd_metrics:
    r, p = pearsonr(corr_df['np_diff'], corr_df[metric])
    print(f"NP Effect vs {metric}: r = {r:.3f}, p = {p:.3f}")

    plt.figure(figsize=(6, 6))
    sns.regplot(x='np_diff', y=metric, data=corr_df)
    plt.title(f"Correlation: NP Effect vs {metric}\nr={r:.3f}, p={p:.3f}")
    plt.xlabel("Target Towardness Difference (No-P - NP)")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


mixed_model_df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_erp_behavioral_lmm_long_data_between-within.csv")

# 5. Mixed Model Analysis (Single-Trial)
print("\n--- Mixed Model Analysis (Single-Trial) ---")
# Filter for relevant conditions
lmm_subset = mixed_model_df[mixed_model_df['Priming'].isin(['np', 'no-p'])].copy()

for metric in pd_metrics:
    # Prepare data: drop NaNs for the current metric and relevant columns
    data_clean = lmm_subset.dropna(subset=['target_towardness', 'Priming', metric])

    # Interaction term (Priming * metric) tests if the relationship between Pd and behavior differs by condition
    formula = f"target_towardness ~ Priming * {metric}"

    # Fit LMM with random intercepts for subjects
    model = smf.mixedlm(formula, data_clean, groups=data_clean["subject_id"])
    result = model.fit()

    print(f"\nModel: {formula}")
    print(result.summary())

    # Calculate simple slopes to aid interpretation
    # The main effect of 'metric' is the slope for the reference group (No-P)
    slope_nop = result.params[metric]
    # The slope for NP is the reference slope + the interaction coefficient
    slope_np = slope_nop + result.params[f"Priming[T.np]:{metric}"]
    
    print(f"\n--- Interpretation for {metric} ---")
    print(f"Slope (No-P): {slope_nop:.4f}")
    print(f"Slope (NP):   {slope_np:.4f}")
    print(f"Difference:   {slope_np - slope_nop:.4f} (Interaction Coef)")

    # Plotting the interaction
    sns.lmplot(x=metric, y='target_towardness', hue='Priming', data=data_clean,
               scatter_kws={'alpha': 0.05, 's': 10}, height=6)
    plt.title(f"LMM Interaction: {metric} x Priming")
    plt.show()