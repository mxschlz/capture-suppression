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
        n_permutations=1024,  # For a real analysis, 5000+ is recommended
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
