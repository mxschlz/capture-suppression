import sys
import os
import glob
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE

def main():
    print("=========================================================")
    # Visual and Auditory stim are both present in spacecue task,
    # and vertex channels (Fz, FCz, Cz) capture clear N1-P2 complexes.
    print("   SPACECUE AUDITORY/VERTEX EVOKED POTENTIALS (AEP)   ")
    print("=========================================================\n")

    # 1. Configuration & Grouping
    data_path = SPACECUE.get_data_path()
    plots_dir = os.path.join(project_root, "spacecue", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    from SPACECUE.subjects import subject_ids
    all_subjects = subject_ids
    cohorts = {'Young': [], 'Old': []}

    # Fronto-central channel ROI for AEP / Vertex potentials
    roi_channels = ["Fz", "FCz", "Cz"]
    tmin, tmax = -0.2, 0.8  # Focus on the 0-500ms post-stimulus window
    baseline_desc = "None (aggressively high-pass filtered with 1 Hz)"

    # 2. Load and average EEG data per subject
    subject_waveforms = {}
    subject_evokeds = {'Young': [], 'Old': []}
    times = None

    print("Loading subject epochs and calculating evoked responses...")
    for sub_id in all_subjects:
        sub_str = f"sci-{sub_id}"
        # Locate preprocessed epochs file
        epoch_files = glob.glob(os.path.join(data_path, "derivatives", "epoching", sub_str, "eeg", f"{sub_str}_task-spacecue-epo.fif"))
        if not epoch_files:
            print(f"  Warning: Epochs file not found for subject {sub_id}. Skipping.")
            continue
        
        print(f"  Processing Subject {sub_id:2d}...")
        try:
            epochs = mne.read_epochs(epoch_files[0], preload=True, verbose=False)
            
            # Infer cohort based on Age in metadata
            age = epochs.metadata['Age'].dropna().iloc[0]
            cohort = 'Young' if age < 36 else 'Old'
            
            if sub_id not in cohorts[cohort]:
                cohorts[cohort].append(sub_id)
            
            # Crop to window of interest
            epochs.crop(tmin, tmax).apply_baseline()
            
            # Get full evoked average over all trials for topographies
            evoked = epochs.average()
            subject_evokeds[cohort].append(evoked)
            
            # Pick our central ROI for waveforms
            evoked_roi = evoked.copy().pick(roi_channels)
            
            # Mean across the ROI channels to get a single vertex waveform
            evoked_data = evoked_roi.data.mean(axis=0) * 1e6  # Convert Volts to Microvolts (uV)
            
            if times is None:
                times = evoked_roi.times
                
            subject_waveforms[sub_id] = evoked_data
        except Exception as e:
            print(f"    Error processing subject {sub_id}: {e}")

    # 3. Aggregate waveforms by cohort
    cohort_data = {'Young': [], 'Old': []}
    for cohort_name, sub_list in cohorts.items():
        for sub_id in sub_list:
            if sub_id in subject_waveforms:
                cohort_data[cohort_name].append(subject_waveforms[sub_id])

    # Convert to numpy arrays: shape = (n_subjects_in_cohort, n_times)
    y_data = np.array(cohort_data['Young'])
    o_data = np.array(cohort_data['Old'])

    print(f"\nGroup Sizes: Young (N={len(y_data)}), Old (N={len(o_data)})")

    # 4. Statistical Analysis: Cluster-based permutation test (independent groups)
    from mne.stats import permutation_cluster_test
    
    print("Running cluster-based permutation test (Young vs. Old)...")
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [y_data, o_data], n_permutations=1000, tail=0, threshold=None, out_type='indices', verbose=False
    )
    
    # Create a boolean mask of significant time points
    is_sig = np.zeros(len(times), dtype=bool)
    if clusters:
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < 0.05:
                is_sig[cluster[0]] = True

    # Calculate grand averages and SEMs
    mean_young = y_data.mean(axis=0)
    sem_young = y_data.std(axis=0) / np.sqrt(len(y_data))

    mean_old = o_data.mean(axis=0)
    sem_old = o_data.std(axis=0) / np.sqrt(len(o_data))

    # 5. Plotting
    print("\nGenerating AEP age comparison plot...")
    plt.figure(figsize=(10, 6.5))

    # Plot individual subject lines beside the group mean
    for sub_data in y_data:
        plt.plot(times * 1000, sub_data, color='#1F77B4', alpha=0.15, lw=1)
    for sub_data in o_data:
        plt.plot(times * 1000, sub_data, color='#FF7F0E', alpha=0.15, lw=1)

    # Plot waveforms with shaded SEM bands (using alpha=0.3 for SEM as per guidelines)
    plt.plot(times * 1000, mean_young, color='#1F77B4', lw=3, label=f'Young Cohort (N={len(y_data)})')
    plt.fill_between(times * 1000, mean_young - sem_young, mean_young + sem_young, color='#1F77B4', alpha=0.3)

    plt.plot(times * 1000, mean_old, color='#FF7F0E', lw=3, label=f'Old Cohort (N={len(o_data)})')
    plt.fill_between(times * 1000, mean_old - sem_old, mean_old + sem_old, color='#FF7F0E', alpha=0.3)

    # Shade significant difference windows (p < 0.05)
    for i in range(len(times)):
        if is_sig[i]:
            plt.axvspan(times[i] * 1000, times[i] * 1000 + (times[1]-times[0])*1000, color='purple', alpha=0.08, lw=0)

    # Label key ERP components
    # Find local minimum/maximum for N1 / P2
    n1_win = (times >= 0.07) & (times <= 0.15)
    p2_win = (times >= 0.17) & (times <= 0.28)
    
    # N1 peak typically negative, P2 positive
    n1_time_ms = times[n1_win][np.argmin(mean_old[n1_win])] * 1000
    p2_time_ms = times[p2_win][np.argmax(mean_old[p2_win])] * 1000

    # Add reference lines
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, lw=1)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5, lw=1)

    # Styling
    plt.xlabel("Time (ms)", fontsize=13, labelpad=8)
    plt.ylabel("Amplitude (μV)", fontsize=13, labelpad=8)
    plt.xlim(tmin * 1000, tmax * 1000)
    
    # Create custom legend entries to include significance shading
    handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(color='purple', alpha=0.15))
    labels.append('Significant Difference (p < 0.05)')
    
    plt.legend(handles, labels, loc='lower right', frameon=False, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine()

    # Save figure
    plot_path = os.path.join(plots_dir, "auditory_evoked_potentials_age_split.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

    # 6. Topographies
    print("\nGenerating Topographies...")
    grand_avg_young = mne.grand_average(subject_evokeds['Young'])
    grand_avg_old = mne.grand_average(subject_evokeds['Old'])
    
    n1_time_s = n1_time_ms / 1000.0
    p2_time_s = p2_time_ms / 1000.0

    fig_topo, axes_topo = plt.subplots(2, 3, figsize=(12, 8))
    
    def plot_topo(data, info, ax, title):
        # We manually plot topomaps to handle MNE version differences gracefully
        res = mne.viz.plot_topomap(data, info, axes=ax, show=False)
        im = res[0] if isinstance(res, tuple) else res
        ax.set_title(title, pad=10, fontsize=12)
        return im

    # N1 Data
    n1_idx_y = grand_avg_young.time_as_index(n1_time_s)[0]
    n1_idx_o = grand_avg_old.time_as_index(n1_time_s)[0]
    n1_data_y = grand_avg_young.data[:, n1_idx_y] * 1e6
    n1_data_o = grand_avg_old.data[:, n1_idx_o] * 1e6

    # P2 Data
    p2_idx_y = grand_avg_young.time_as_index(p2_time_s)[0]
    p2_idx_o = grand_avg_old.time_as_index(p2_time_s)[0]
    p2_data_y = grand_avg_young.data[:, p2_idx_y] * 1e6
    p2_data_o = grand_avg_old.data[:, p2_idx_o] * 1e6

    # 400-600ms Data
    late_idx_y = (grand_avg_young.times >= 0.4) & (grand_avg_young.times <= 0.6)
    late_idx_o = (grand_avg_old.times >= 0.4) & (grand_avg_old.times <= 0.6)
    late_data_y = grand_avg_young.data[:, late_idx_y].mean(axis=1) * 1e6
    late_data_o = grand_avg_old.data[:, late_idx_o].mean(axis=1) * 1e6

    # Young Cohort Topographies
    im1 = plot_topo(n1_data_y, grand_avg_young.info, axes_topo[0, 0], f"Young N1 ({n1_time_ms:.0f} ms)")
    im2 = plot_topo(p2_data_y, grand_avg_young.info, axes_topo[0, 1], f"Young P2 ({p2_time_ms:.0f} ms)")
    im3 = plot_topo(late_data_y, grand_avg_young.info, axes_topo[0, 2], "Young 400-600 ms")

    # Old Cohort Topographies
    im4 = plot_topo(n1_data_o, grand_avg_old.info, axes_topo[1, 0], f"Old N1 ({n1_time_ms:.0f} ms)")
    im5 = plot_topo(p2_data_o, grand_avg_old.info, axes_topo[1, 1], f"Old P2 ({p2_time_ms:.0f} ms)")
    im6 = plot_topo(late_data_o, grand_avg_old.info, axes_topo[1, 2], "Old 400-600 ms")

    # Add colorbars for each row
    plt.colorbar(im3, ax=axes_topo[0, 2], label="Amplitude (μV)")
    plt.colorbar(im6, ax=axes_topo[1, 2], label="Amplitude (μV)")

    plt.tight_layout()
    topo_plot_path = os.path.join(plots_dir, "auditory_evoked_topographies.png")
    plt.savefig(topo_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved topographies plot to {topo_plot_path}")
    
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close(fig_topo)

    # 7. Save statistical report
    report_lines = [
        "=========================================================",
        "     AEP AGE SPLIT STATISTICAL COMPARISON REPORT         ",
        "=========================================================\n",
        f"Fronto-Central ROI: {roi_channels}",
        f"Baseline Correction: {baseline_desc}",
        f"Waveform Window: {tmin*1000}ms to {tmax*1000}ms\n",
        f"Subjects Included:",
        f"  Young Cohort (N={len(y_data)}): {cohorts['Young']}",
        f"  Old Cohort (N={len(o_data)}): {cohorts['Old']}\n",
        "Key Peak Comparisons (Old Cohort):",
        f"  N1 peak found at: {n1_time_ms:.1f} ms",
        f"  P2 peak found at: {p2_time_ms:.1f} ms\n",
        "Statistical Differences (Young vs. Old, Cluster-Based Permutation Test):"
    ]

    if clusters:
        has_sig = False
        for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            if p_val < 0.05:
                has_sig = True
                indices = cluster[0]
                t_start_ms = times[indices[0]] * 1000
                t_end_ms = times[indices[-1]] * 1000
                report_lines.append(f"  Significant Cluster {c_idx+1:d}: {t_start_ms:5.1f} ms to {t_end_ms:5.1f} ms | Cluster p-value = {p_val:.4f}")
        if not has_sig:
            report_lines.append("  No significant clusters found.")
    else:
        report_lines.append("  No clusters formed.")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(plots_dir, "auditory_evoked_potentials_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved plot to {plot_path}")
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
