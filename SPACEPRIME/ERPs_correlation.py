import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import SPACEPRIME
import pingouin as pg
sns.set_theme("talk", "ticks")


SUBJECT_ID_COL = 'subject_id'

# --- NEW: Correlate Metrics Between N2ac and Pd ---
print("\n--- Correlating N2ac and Pd Metrics at the Subject Level ---")

def plot_component_correlation(df1, df2, metric_col, ax, df1_name='N2ac', df2_name='Pd'):
    """
    Calculates subject-level averages for a given metric from two dataframes,
    merges them, and creates a regression plot on a given axes object.

    Args:
        df1 (pd.DataFrame): DataFrame for the first component (e.g., n2ac_final_df).
        df2 (pd.DataFrame): DataFrame for the second component (e.g., pd_final_df).
        metric_col (str): The name of the column to correlate (e.g., 'st_latency_50').
        ax (matplotlib.axes.Axes): The axes object to plot on.
        df1_name (str): Name of the first component for plot labels.
        df2_name (str): Name of the second component for plot labels.
    """
    if df1.empty or df2.empty or metric_col not in df1.columns or metric_col not in df2.columns:
        print(f"Skipping correlation plot for '{metric_col}': Data or column is missing.")
        ax.text(0.5, 0.5, "Data or column missing", ha='center', va='center', fontsize=12)
        ax.set_title(f"Correlation for {metric_col}")
        return

    # 1. Calculate subject-level averages for the specified metric
    avg1 = df1.groupby(SUBJECT_ID_COL)[metric_col].mean().rename(f"{df1_name}_{metric_col}")
    avg2 = df2.groupby(SUBJECT_ID_COL)[metric_col].mean().rename(f"{df2_name}_{metric_col}")

    # 2. Merge the two series into a single dataframe on subject_id.
    merged_avg_df = pd.merge(avg1, avg2, on=SUBJECT_ID_COL, how='inner')

    if merged_avg_df.empty:
        print(f"Skipping correlation plot for '{metric_col}': No common subjects found after averaging.")
        ax.text(0.5, 0.5, "No common subjects", ha='center', va='center', fontsize=12)
        ax.set_title(f"Correlation for {metric_col}")
        return

    # 3. Calculate Pearson correlation and p-value
    try:
        clean_df = merged_avg_df.dropna(subset=[f"{df1_name}_{metric_col}", f"{df2_name}_{metric_col}"])
        if len(clean_df) < 2:
            raise ValueError("Not enough data points to compute correlation.")

        stats = pg.corr(clean_df[f"{df1_name}_{metric_col}"], clean_df[f"{df2_name}_{metric_col}"], method='pearson')
        r = stats['r'].iloc[0]
        p = stats['p-val'].iloc[0]
        bf10 = stats['BF10'].iloc[0]
        stat_text = f'r = {r:.2f}, p = {p:.3f}\nBF10 = {bf10}\nn = {len(clean_df)}'
    except ValueError as e:
        print(f"Could not compute correlation for {metric_col}: {e}")
        stat_text = 'Cannot compute correlation'

    # 4. Create the regression plot on the provided axes
    sns.regplot(
        data=merged_avg_df,
        x=f"{df1_name}_{metric_col}",
        y=f"{df2_name}_{metric_col}",
        scatter_kws={'alpha': 0.6},
        ax=ax
    )

    # Add the correlation stats to the plot
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Add titles and labels
    if 'latency' in metric_col:
        label_suffix = 'mean latency'
    elif 'amp' in metric_col:
        label_suffix = 'mean amplitude'
    else:
        label_suffix = metric_col.replace('_', ' ')  # Fallback
    ax.set_xlabel(f'{df1_name} {label_suffix}')
    ax.set_ylabel(f'{df2_name} {label_suffix}')
    sns.despine()

# --- Example Usage ---
# Define which latency/amplitude percentage to use for the correlation
METRIC_PERCENTAGE = 50
latency_metric = f'st_latency_{METRIC_PERCENTAGE}'
amplitude_metric = f'st_mean_amp_{METRIC_PERCENTAGE}'

# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f'Subject-Level Correlation: N2ac vs. Pd Metrics ({METRIC_PERCENTAGE}% Threshold)', fontsize=18)

n2ac_final_df = SPACEPRIME.load_concatenated_csv("spaceprime_n2ac_erp_behavioral_lmm_long_data_between-within.csv")
pd_final_df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_erp_behavioral_lmm_long_data_between-within.csv")

# Plot 1: Latency Correlation
print(f"\nPlotting correlation for Single-Trial Latency ({METRIC_PERCENTAGE}%)...")
plot_component_correlation(
    df1=n2ac_final_df,
    df2=pd_final_df,
    metric_col=latency_metric,
    ax=axes[0],  # Pass the first axes object
    df1_name='N2ac',
    df2_name='Pd'
)

# Plot 2: Amplitude Correlation
print(f"\nPlotting correlation for Single-Trial Mean Amplitude ({METRIC_PERCENTAGE}%)...")
plot_component_correlation(
    df1=n2ac_final_df,
    df2=pd_final_df,
    metric_col=amplitude_metric,
    ax=axes[1],  # Pass the second axes object
    df1_name='N2ac',
    df2_name='Pd'
)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout()  # Adjust for suptitle

# Save the figure as an SVG file
correlation_plot_path = f'plots/n2ac_pd_correlation.svg'
plt.savefig(correlation_plot_path, format='svg', bbox_inches='tight')
print(f"Correlation plot saved to:\n{correlation_plot_path}")
