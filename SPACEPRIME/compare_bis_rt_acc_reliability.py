import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import SPACEPRIME
from stats import remove_outliers, calculate_bis

plt.ion()

# --- Script Configuration ---

# --- 1. Analysis Parameters ---
N_PERMUTATIONS = 1000  # Number of random splits for reliability calculation

# --- 2. Data Loading & Preprocessing Config (from SAT_BIS.py) ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2

# --- 3. Column Names ---
SUBJECT_ID_COL = 'subject_id'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
ACCURACY_INT_COL = 'select_target_int'


# PASTE THE NEW, UPDATED split_dataframe_by_trials_within_subject FUNCTION HERE
def split_dataframe_by_trials_within_subject(df, subject_col, method='random', seed=None):
    """
    Splits a DataFrame into two halves of trials for each subject using one of
    several methods...
    """
    # ... (full function code as shown above) ...
    supported_methods = ['random', 'sequential', 'odd_even']
    if method not in supported_methods:
        raise ValueError(f"Method '{method}' is not supported. Choose from {supported_methods}.")
    rng = np.random.default_rng(seed)
    list_df1, list_df2 = [], []
    for _, subject_df in df.groupby(subject_col):
        indices = subject_df.index.to_numpy()
        if method == 'random':
            rng.shuffle(indices)
            midpoint = len(indices) // 2
            indices1 = indices[:midpoint]
            indices2 = indices[midpoint:]
        elif method == 'sequential':
            midpoint = len(indices) // 2
            indices1 = indices[:midpoint]
            indices2 = indices[midpoint:]
        elif method == 'odd_even':
            indices1 = indices[::2]
            indices2 = indices[1::2]
        list_df1.append(df.loc[indices1])
        list_df2.append(df.loc[indices2])
    if not list_df1 or not list_df2:
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    df1 = pd.concat(list_df1, ignore_index=True)
    df2 = pd.concat(list_df2, ignore_index=True)
    return df1, df2


def spearman_brown_correction(r):
    """
    Applies the Spearman-Brown prophecy formula to a correlation coefficient.

    Args:
        r (float): The Pearson correlation between the two halves of the test.

    Returns:
        float: The estimated reliability of the full-length test.
    """
    # Handle potential NaN or perfect correlation cases
    if pd.isna(r):
        return np.nan
    # Avoid division by zero if r is -1
    if r == -1:
        return -1.0
    return (2 * r) / (1 + r)


# Main function to load data, run split-half reliability analysis,
# and compare the reliability of RT, Accuracy, and BIS.
# --- 1. Data Loading and Preprocessing ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")
df = epochs.metadata.copy()
print(f"Original number of trials: {len(df)}")

# Standard preprocessing steps
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)
print("Preprocessing complete.")

# --- 2. Split-Half Reliability Analysis Loop ---
print(f"\nRunning split-half reliability with {N_PERMUTATIONS} permutations...")

reliability_results = []
for i in range(N_PERMUTATIONS):
    print(f"Run permutation {i} ...")
    # Use a different seed for each split to ensure randomness
    df1, df2 = split_dataframe_by_trials_within_subject(
        df, subject_col=SUBJECT_ID_COL, seed=i, method="random"
    )

    # Aggregate metrics for each half
    agg1 = df1.groupby(SUBJECT_ID_COL).agg(
        rt=(REACTION_TIME_COL, 'mean'),
        pc=(ACCURACY_INT_COL, 'mean')
    ).reset_index()
    agg1['split'] = 'half1'

    agg2 = df2.groupby(SUBJECT_ID_COL).agg(
        rt=(REACTION_TIME_COL, 'mean'),
        pc=(ACCURACY_INT_COL, 'mean')
    ).reset_index()
    agg2['split'] = 'half2'

    # Combine to calculate BIS with consistent z-scoring across both halves
    agg_combined = pd.concat([agg1, agg2], ignore_index=True)

    # Important: Ensure we only include subjects present in BOTH splits for correlation
    subject_counts = agg_combined[SUBJECT_ID_COL].value_counts()
    complete_subjects = subject_counts[subject_counts == 2].index
    agg_combined = agg_combined[agg_combined[SUBJECT_ID_COL].isin(complete_subjects)]

    if len(complete_subjects) < 2:
        print(f"Skipping permutation {i}: Not enough subjects with data in both splits.")
        continue

    # Calculate BIS on the combined dataset for this permutation
    agg_with_bis = calculate_bis(agg_combined, rt_col='rt', pc_col='pc')

    # Separate back into two halves
    metrics_half1 = agg_with_bis[agg_with_bis['split'] == 'half1']
    metrics_half2 = agg_with_bis[agg_with_bis['split'] == 'half2']

    # Merge to align subjects for correlation
    merged_metrics = pd.merge(
        metrics_half1,
        metrics_half2,
        on=SUBJECT_ID_COL,
        suffixes=('_1', '_2')
    )

    # Calculate Pearson correlation for each measure
    r_rt = merged_metrics['rt_1'].corr(merged_metrics['rt_2'])
    r_pc = merged_metrics['pc_1'].corr(merged_metrics['pc_2'])
    r_bis = merged_metrics['bis_1'].corr(merged_metrics['bis_2'])

    # Apply Spearman-Brown correction and store results
    reliability_results.append({
        'permutation': i,
        'rt_reliability': spearman_brown_correction(r_rt),
        'pc_reliability': spearman_brown_correction(r_pc),
        'bis_reliability': spearman_brown_correction(r_bis)
    })

reliability_df = pd.DataFrame(reliability_results).dropna()

# --- 3. Statistical Comparison ---
print("\n--- Mean Reliability Results ---")
print(reliability_df[['rt_reliability', 'pc_reliability', 'bis_reliability']].mean())

# Compare BIS reliability to RT and PC reliability using a paired t-test
# This tests if the difference in reliability is consistent across permutations
t_bis_vs_rt, p_bis_vs_rt = ttest_rel(
    reliability_df['bis_reliability'], reliability_df['rt_reliability']
)
t_bis_vs_pc, p_bis_vs_pc = ttest_rel(
    reliability_df['bis_reliability'], reliability_df['pc_reliability']
)

print("\n--- Statistical Comparison of Reliabilities ---")
print(f"BIS vs. RT: t-statistic = {t_bis_vs_rt:.3f}, p-value = {p_bis_vs_rt:.5f}")
print(f"BIS vs. PC: t-statistic = {t_bis_vs_pc:.3f}, p-value = {p_bis_vs_pc:.5f}")

# --- 4. Visualization ---
print("\nGenerating plot...")
# Melt the dataframe for easier plotting with seaborn
plot_df = reliability_df.melt(
    id_vars=['permutation'],
    value_vars=['rt_reliability', 'pc_reliability', 'bis_reliability'],
    var_name='measure',
    value_name='reliability'
)
plot_df['measure'] = plot_df['measure'].map({
    'rt_reliability': 'RT',
    'pc_reliability': 'Accuracy',
    'bis_reliability': 'BIS'
})

plt.figure(figsize=(10, 7))
sns.violinplot(data=plot_df, x='measure', y='reliability', order=['RT', 'Accuracy', 'BIS'], cut=0)
plt.title(f'Split-Half Reliability Distributions ({len(reliability_df)} Permutations)')
plt.xlabel('Performance Measure')
plt.ylabel('Spearman-Brown Corrected Reliability (r)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
