import seaborn as sns
import SPACEPRIME
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot_latency_amplitude_by_towardness(df, component_name, ax, subject_col='subject_id',
                                         latency_col='st_latency_50', amplitude_col='st_mean_amp_50',
                                         add_legend=True, center_data=False):
    """
    Generates a within-subject plot of ERP amplitude vs. latency on a given axis.

    This plot shows:
    1. Individual subject means for high/low towardness (dots).
    2. Lines connecting the within-subject data points.
    3. Summary crosses representing the group mean +/- standard error.

    Args:
        df (pd.DataFrame): The input dataframe with single-trial data.
        component_name (str): The name of the ERP component for plot titles.
        ax (matplotlib.axes.Axes): The subplot axis to draw the plot on.
        subject_col (str): The name of the column identifying subjects.
        latency_col (str): The name of the latency data column.
        amplitude_col (str): The name of the amplitude data column.
        add_legend (bool): If True, a legend is added to the plot.
        center_data (bool): If True, performs within-subject centering to reduce spread.
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, f'No data for {component_name}', ha='center', va='center')
        ax.set_title(component_name)
        return

    # --- 1. Data Preparation: Within-subject median split and aggregation ---
    subject_means = []
    for subject_id, subject_df in df.groupby(subject_col):
        if subject_df.empty:
            continue

        median_towardness = subject_df['target_towardness'].median()
        low_df = subject_df[subject_df['target_towardness'] <= median_towardness]
        high_df = subject_df[subject_df['target_towardness'] > median_towardness]

        if high_df.empty and not low_df.empty:
            high_df = low_df  # Handle cases with no variability in towardness

        if not low_df.empty:
            subject_means.append({
                subject_col: subject_id,
                'towardness_level': 'Low',
                'latency': low_df[latency_col].mean(),
                'amplitude': low_df[amplitude_col].mean()
            })
        if not high_df.empty:
            subject_means.append({
                subject_col: subject_id,
                'towardness_level': 'High',
                'latency': high_df[latency_col].mean(),
                'amplitude': high_df[amplitude_col].mean()
            })

    agg_df = pd.DataFrame(subject_means)
    if agg_df.empty:
        ax.text(0.5, 0.5, f'No data for {component_name}', ha='center', va='center')
        ax.set_title(component_name)
        return

    # --- 2. Optional Within-Subject Centering ---
    plot_df = agg_df.copy()
    if center_data:
        # Use transform to get each subject's grand mean and subtract it
        subject_grand_means = plot_df.groupby(subject_col)[['latency', 'amplitude']].transform('mean')
        plot_df['latency'] -= subject_grand_means['latency']
        plot_df['amplitude'] -= subject_grand_means['amplitude']
        x_label, y_label = 'Latency (Centered, ms)', 'Amplitude (Centered, µV)'
    else:
        x_label, y_label = 'Latency (ms)', 'Amplitude (µV)'

    # --- 3. Plotting (using plot_df) ---
    sns.lineplot(
        data=plot_df, x='latency', y='amplitude', hue='towardness_level',
        units=subject_col, estimator=None, color='grey', alpha=0.4,
        ax=ax, legend=False
    )
    sns.scatterplot(
        data=plot_df, x='latency', y='amplitude', hue='towardness_level',
        palette={'Low': 'royalblue', 'High': 'firebrick'}, s=80, alpha=0.8,
        ax=ax, legend=False
    )

    # --- 4. Summary Crosses (Mean +/- SEM) ---
    summary_stats = plot_df.groupby('towardness_level').agg(
        mean_latency=('latency', 'mean'),
        sem_latency=('latency', 'sem'),
        mean_amplitude=('amplitude', 'mean'),
        sem_amplitude=('amplitude', 'sem')
    ).reset_index()

    for _, row in summary_stats.iterrows():
        color = 'royalblue' if row['towardness_level'] == 'Low' else 'firebrick'
        ax.errorbar(
            x=row['mean_latency'], y=row['mean_amplitude'],
            xerr=row['sem_latency'], yerr=row['sem_amplitude'],
            fmt='X', markersize=0, markeredgecolor='white', markerfacecolor=color,
            color=color, elinewidth=2.5, capsize=5, zorder=10
        )

    # --- 5. Final Touches ---
    ax.set_title(f'{component_name} Component', fontsize=18)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if add_legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor='royalblue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor='firebrick', markersize=10)
        ]
        ax.legend(handles=legend_handles, labels=['Low Towardness', 'High Towardness'],
                  title='Condition', title_fontsize='14', fontsize='12', loc='best')


# Load data
n2ac_df = SPACEPRIME.load_concatenated_csv("spaceprime_n2ac_erp_behavioral_lmm_long_data_between-within.csv")
pd_df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_erp_behavioral_lmm_long_data_between-within.csv")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=False) # sharey=False because N2ac is inverted

# --- Generate the plot for the N2ac component on the first axis ---
plot_latency_amplitude_by_towardness(
    df=n2ac_df,
    component_name="N2ac",
    ax=axes[0],
    subject_col='subject_id',
    add_legend=True,
    center_data=True  # <-- Center the data
)

# --- Generate the plot for the Pd component on the second axis ---
plot_latency_amplitude_by_towardness(
    df=pd_df,
    component_name="Pd",
    ax=axes[1],
    subject_col='subject_id',
    add_legend=False,
    center_data=True  # <-- Center the data
)

# Clean up the layout and display the plot
sns.despine(fig=fig)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()
