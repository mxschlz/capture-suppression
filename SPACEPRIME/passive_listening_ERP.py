import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import get_passive_listening_ERPs_grand_average
from mne.stats import permutation_cluster_1samp_test
import seaborn as sns

# Set plot style
plt.style.use('seaborn-v0_8-talk')
plt.ion()

# 1. Get the data using the new function
# This dictionary contains grand-averages, subject-level data, and times
results = get_passive_listening_ERPs_grand_average()
times = results['times']
grand_average = results['grand_average']
subject_data = results['subject_data']

# --- 2. Perform Statistical Analysis (the correct way) ---
# We will run a cluster-based permutation test on the difference waves (Contra - Ipsi)
# for each subject. This is a robust way to test for significance across time.

# Define conditions to loop through
conditions = {
    "target": "Target",
    "singleton": "Singleton",  # Note: 'singleton' corresponds to 'distractor' in the old script
    "control": "Control"
}

stats_results = {}
for key in conditions.keys():
    # Create the subject-level difference wave: (n_subjects, n_times)
    contra = np.array(subject_data[f'contra_{key}'])
    ipsi = np.array(subject_data[f'ipsi_{key}'])

    # Ensure we have data to process
    if contra.size == 0 or ipsi.size == 0:
        print(f"Skipping stats for '{key}' due to missing data.")
        continue

    subject_diff = contra - ipsi

    # Run the cluster permutation test
    # We test the difference wave against 0.
    t_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        subject_diff,
        n_permutations=10000,  # For a real analysis, 5000+ is recommended
        threshold=None,  # Use default t-threshold
        tail=0,  # Two-tailed test
        n_jobs=-1,  # Use all available CPU cores
        verbose=False
    )

    # Find significant clusters (p < 0.05)
    significant_clusters = [clusters[i] for i, p_val in enumerate(cluster_p_values) if p_val < 0.05]

    stats_results[key] = {
        't_obs': t_obs,
        'significant_clusters': significant_clusters
    }

# --- 3. Plot the Results ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
axes = axes.flatten()

# Conversion factor from V to µV
to_microvolts = 1e6

for ax, (key, title) in zip(axes, conditions.items()):
    # Plot grand-average contralateral and ipsilateral waveforms
    ax.plot(times, grand_average[f'contra_{key}'][0] * to_microvolts, color="crimson", label="Contralateral")
    ax.plot(times, grand_average[f'ipsi_{key}'][0] * to_microvolts, color="steelblue", label="Ipsilateral")

    # Plot the grand-average difference wave
    diff_wave = grand_average[f'contra_{key}'][0] - grand_average[f'ipsi_{key}'][0]
    ax.plot(times, diff_wave * to_microvolts, color="black", linestyle='--', label="Difference (Contra-Ipsi)")

    # Add a horizontal line at 0
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    # Shade significant time windows identified by the cluster test
    if key in stats_results:
        for cluster in stats_results[key]['significant_clusters']:
            ax.fill_between(
                times[cluster[1]],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color='orange',
                alpha=0.3,
                label='p < 0.05'
            )

    ax.set_title(f"{title} Lateralization")
    ax.set_xlabel("Time [s]")

# Set labels and legend for the figure
axes[0].set_ylabel("Amplitude [µV]")
handles, labels = axes[0].get_legend_handles_labels()
# Remove duplicate labels for the fill_between
unique_labels = dict(zip(labels, handles))
fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

sns.despine(fig=fig)
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for legend

# --- 4. Compare Difference Waves Across Conditions ---

print("\n--- Comparing Difference Waves Across Conditions ---")

# We need the subject-level difference waves (Contra - Ipsi) calculated earlier.
# Let's store them in a dictionary for the new comparison.
subject_diffs = {}
for key in conditions.keys():
    contra = np.array(subject_data[f'contra_{key}'])
    ipsi = np.array(subject_data[f'ipsi_{key}'])
    if contra.size > 0 and ipsi.size > 0:
        subject_diffs[key] = contra - ipsi

# --- 4a. Statistical Comparison between Difference Waves ---
# We'll run a paired cluster-based permutation test for each pair of conditions.
# This is done by taking the difference of the difference waves for each subject
# and testing if that new value is significantly different from zero.

# Define the pairs of conditions to compare
comparisons = [
    ("target", "singleton"),
    ("target", "control"),
    ("singleton", "control")
]

pairwise_stats_results = {}
for cond1, cond2 in comparisons:
    print(f"Running stats for {cond1} vs. {cond2}...")
    # Ensure we have data for both conditions to compare
    if cond1 not in subject_diffs or cond2 not in subject_diffs:
        print(f"Skipping {cond1} vs. {cond2} due to missing data.")
        continue

    # Create the paired difference for each subject: (cond1_diff - cond2_diff)
    X = subject_diffs[cond1] - subject_diffs[cond2]

    # Test this new difference wave against 0.
    t_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        X,
        n_permutations=10000,
        threshold=None,  # Use MNE's default
        tail=0,          # Two-tailed test
        n_jobs=-1,
        verbose=False
    )

    # Store significant clusters
    significant_clusters = [clusters[i] for i, p_val in enumerate(cluster_p_values) if p_val < 0.05]
    pairwise_stats_results[f"{cond1}_vs_{cond2}"] = significant_clusters
    print(f"Found {len(significant_clusters)} significant cluster(s) for {cond1} vs. {cond2}.")


# --- 4b. Plot the Comparison of Difference Waves ---
fig_comp, ax_comp = plt.subplots(1, 1, figsize=(12, 7))

# Define colors for clarity
comp_colors = {
    "target": "darkorange",
    "singleton": "dodgerblue",
    "control": "darkgrey"
}

# Plot each grand-average difference wave
for key, title in conditions.items():
    if key not in subject_diffs:  # Check if data exists
        continue
    diff_wave = grand_average[f'contra_{key}'][0] - grand_average[f'ipsi_{key}'][0]
    ax_comp.plot(times, diff_wave * to_microvolts, color=comp_colors[key], label=f"{title} Difference")

# Add reference lines
ax_comp.axhline(0, color='black', linestyle='-', linewidth=0.7)
ax_comp.axvline(0, color='black', linestyle=':', linewidth=0.7)

# --- Visualize the statistical results ---
# We'll add horizontal bars at the bottom of the plot to show significant differences.

# Add some space at the bottom for the significance bars
ax_comp.set_ylim(bottom=ax_comp.get_ylim()[0] * 1.25)
ylim = ax_comp.get_ylim()
y_range = ylim[1] - ylim[0]

# Define vertical positions and colors for the bars
sig_bar_config = {
    "target_vs_singleton": (ylim[0] + 0.18 * y_range, "purple"),
    "target_vs_control": (ylim[0] + 0.10 * y_range, "green"),
    "singleton_vs_control": (ylim[0] + 0.02 * y_range, "saddlebrown")
}

legend_handles = []

for comp_key, (y_val, color) in sig_bar_config.items():
    if comp_key in pairwise_stats_results and pairwise_stats_results[comp_key]:
        # Create a handle for the legend
        label = f"{comp_key.replace('_', ' ').replace('vs', 'vs.')} p < 0.05"
        legend_handles.append(plt.Line2D([], [], color=color, linewidth=5, label=label))

        # Plot all significant clusters for this comparison
        for cluster in pairwise_stats_results[comp_key]:
            time_slice = cluster[1]
            start_time = times[time_slice.start]
            end_time = times[time_slice.stop - 1]
            ax_comp.plot([start_time, end_time], [y_val, y_val], color=color, linewidth=5, solid_capstyle='butt')

# --- Configure plot aesthetics ---
ax_comp.set_title("Comparison of Difference Waves (Contralateral - Ipsilateral)")
ax_comp.set_xlabel("Time [s]")
ax_comp.set_ylabel("Amplitude [µV]")

# Create a combined legend
line_handles, line_labels = ax_comp.get_legend_handles_labels()
ax_comp.legend(handles=line_handles + legend_handles, loc='upper left', fontsize='small')

sns.despine(fig=fig_comp)
fig_comp.tight_layout()
plt.show()
