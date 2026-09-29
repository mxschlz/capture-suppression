import sys
import os
import glob
import pickle
import numpy as np
import pandas as pd
import mne
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from SPACECUE.subjects import subject_ids

# Add project path to sys.path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)
import SPACECUE


def run_computation():
    print("No cache found. Running Time-Frequency analysis for all subjects...")

    # Analysis parameters matching alpha_lateralization.py
    freqs = np.arange(5, 20, 1)
    n_cycles = freqs / 2
    method = "morlet"
    decim = 5
    
    subject_data = {}
    montage = mne.channels.make_standard_montage("easycap-M1")

    for subject in subject_ids:
        print(f"Processing Subject: {subject}...")
        epoch_file_pattern = os.path.join(SPACECUE.get_data_path(), "derivatives", "epoching", f"sci-{subject}", "eeg",
                                          f"sci-{subject}_task-spacecue-epo.fif")
        epoch_files = glob.glob(epoch_file_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for subject {subject}. Skipping.")
            continue
            
        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        epochs_sub.set_montage(montage)
        
        # Determine cohort based on Age
        age = float(epochs_sub.metadata['Age'].iloc[0])
        cohort = 'Young' if age <= 35 else 'Old'
        
        # Mask for expected distractor location
        hp_loc_numeric = pd.to_numeric(epochs_sub.metadata['HP_Distractor_Loc'], errors='coerce')
        left_expected_epochs = epochs_sub[hp_loc_numeric == 1.0]
        right_expected_epochs = epochs_sub[hp_loc_numeric == 3.0]
        
        if len(left_expected_epochs) == 0 or len(right_expected_epochs) == 0:
            print(f"  Skipping subject {subject} due to zero trials in one condition.")
            continue
            
        # Compute TFR for Left expected distractor
        tfr_left = left_expected_epochs.compute_tfr(
            method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
            return_itc=False, average=False, n_jobs=4
        )
        alpha_left_avg = tfr_left.average()
        
        # Compute TFR for Right expected distractor
        tfr_right = right_expected_epochs.compute_tfr(
            method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
            return_itc=False, average=False, n_jobs=4
        )
        alpha_right_avg = tfr_right.average()
        
        # Calculate LI: (Left - Right) / (Left + Right)
        li_data = (alpha_left_avg.data - alpha_right_avg.data) / (alpha_left_avg.data + alpha_right_avg.data)
        
        # Calculate mean power: (Left + Right) / 2
        mean_power_data = (alpha_left_avg.data + alpha_right_avg.data) / 2
        
        # Store essential data
        subject_data[subject] = {
            'li_data': li_data,
            'mean_power_data': mean_power_data,
            'times': alpha_left_avg.times,
            'freqs': alpha_left_avg.freqs,
            'ch_names': alpha_left_avg.ch_names,
            'cohort': cohort,
            'age': age
        }

    print("Saved calculated LI and Power TFRs to cache.")
    return subject_data

def main():
    # Always recompute (no cache)
    subject_data = run_computation()

    # Left and Right ROI definitions
    left_roi_hemi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
    right_roi_hemi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]

    subject_ids = list(subject_data.keys())
    epoch_file_pattern = os.path.join(SPACECUE.get_data_path(), "derivatives", "epoching", f"sci-{subject_ids[0]}", "eeg",
                                      f"sci-{subject_ids[0]}_task-spacecue-epo.fif")
    first_subject_epochs = mne.read_epochs(glob.glob(epoch_file_pattern)[0], preload=False)
    montage = mne.channels.make_standard_montage("easycap-M1")
    first_subject_epochs.set_montage(montage)
    info = first_subject_epochs.info

    cohorts = ['Young', 'Old']
    times_vector = subject_data[subject_ids[0]]['times']
    freqs = subject_data[subject_ids[0]]['freqs']
    
    # 8-12 Hz Alpha band index
    alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
    
    # Pre-stimulus time window [-0.75, 0.0] index
    pre_stim_idx = np.where((times_vector >= -0.75) & (times_vector <= 0.0))[0]

    # Data structures for Time Courses (tc) and Topographies (topo)
    li_tcs = {c: [] for c in cohorts}
    li_topos = {c: [] for c in cohorts}
    
    power_tcs = {c: [] for c in cohorts}
    power_topos = {c: [] for c in cohorts}

    for sub_id, sub_info in subject_data.items():
        li_data = sub_info['li_data']
        power_raw = sub_info['mean_power_data']
        ch_names = sub_info['ch_names']
        cohort = sub_info['cohort']
        
        # Normalize subject absolute power by their overall mean power to scale across subjects
        sub_overall_mean = power_raw.mean()
        power_scaled = power_raw / sub_overall_mean
        
        # 1. Topography: Mean pre-stimulus alpha value per channel
        li_topo = li_data[:, alpha_idx, :][:, :, pre_stim_idx].mean(axis=(1, 2))
        power_topo = power_scaled[:, alpha_idx, :][:, :, pre_stim_idx].mean(axis=(1, 2))
        
        li_topos[cohort].append(li_topo)
        power_topos[cohort].append(power_topo)
        
        # 2. Time Courses: Average over ROI channels & alpha band
        valid_left_chans = [ch_names.index(ch) for ch in left_roi_hemi if ch in ch_names]
        valid_right_chans = [ch_names.index(ch) for ch in right_roi_hemi if ch in ch_names]
        
        # For LI, hemispheric difference: Left Hemi - Right Hemi
        li_left_tc = li_data[valid_left_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        li_right_tc = li_data[valid_right_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        li_tcs[cohort].append(li_left_tc - li_right_tc)
        
        # For Power, mean over all ROI channels
        roi_power_left = power_scaled[valid_left_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        roi_power_right = power_scaled[valid_right_chans, :, :][:, alpha_idx, :].mean(axis=(0, 1))
        power_tcs[cohort].append((roi_power_left + roi_power_right) / 2)

    # --- Calculations of Means and SEMs ---
    avg_li_tc = {}; sem_li_tc = {}; avg_li_topo = {}
    avg_power_tc = {}; sem_power_tc = {}; avg_power_topo = {}
    
    for c in cohorts:
        avg_li_tc[c] = np.mean(li_tcs[c], axis=0)
        sem_li_tc[c] = sem(li_tcs[c], axis=0)
        avg_li_topo[c] = np.mean(li_topos[c], axis=0)
        
        avg_power_tc[c] = np.mean(power_tcs[c], axis=0)
        sem_power_tc[c] = sem(power_tcs[c], axis=0)
        avg_power_topo[c] = np.mean(power_topos[c], axis=0)

    # --- PLOTTING (2 Rows, 3 Columns) ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    
    colors = {'Young': '#1F77B4', 'Old': '#FF7F0E'}
    
    # ------------------ ROW 1: ALPHA LI ------------------
    # Time course (-1.0 to 1.0 s)
    ax_li_tc = axes[0, 0]
    for c in cohorts:
        ax_li_tc.plot(times_vector, avg_li_tc[c], color=colors[c], lw=3, label=c)
        ax_li_tc.fill_between(times_vector, avg_li_tc[c] - sem_li_tc[c], avg_li_tc[c] + sem_li_tc[c], color=colors[c], alpha=0.15)
    
    ax_li_tc.axvspan(-0.75, 0.0, color='gray', alpha=0.08, label='Pre-stimulus')
    ax_li_tc.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax_li_tc.axvline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    ax_li_tc.set_xlim(-1.0, 1.0)
    ax_li_tc.set_ylabel("Alpha LI (Left - Right Hemi)", fontsize=13)
    ax_li_tc.legend(frameon=False, fontsize=11, loc='upper left')
    ax_li_tc.set_title("Time Course", fontsize=14, fontweight='semibold')
    sns.despine(ax=ax_li_tc)

    # Consistent LI topo scaling
    li_vmax = max(np.abs(avg_li_topo['Young']).max(), np.abs(avg_li_topo['Old']).max())
    li_vmax = min(li_vmax, 0.15)
    li_vmin = -li_vmax
    
    # Younger LI Topomap
    ax_li_topo_y = axes[0, 1]
    im_li_y, _ = mne.viz.plot_topomap(
        avg_li_topo['Young'], info, axes=ax_li_topo_y, cmap='RdBu_r',
        vlim=(li_vmin, li_vmax), show=False, outlines='head', sensors=False
    )
    ax_li_topo_y.set_title("Younger Participants\n(-0.75 to 0.0 s)", fontsize=13, fontweight='semibold')
    
    # Older LI Topomap
    ax_li_topo_o = axes[0, 2]
    im_li_o, _ = mne.viz.plot_topomap(
        avg_li_topo['Old'], info, axes=ax_li_topo_o, cmap='RdBu_r',
        vlim=(li_vmin, li_vmax), show=False, outlines='head', sensors=False
    )
    ax_li_topo_o.set_title("Older Participants\n(-0.75 to 0.0 s)", fontsize=13, fontweight='semibold')
    
    # Shared LI colorbar
    cbar_li_ax = fig.add_axes([0.92, 0.58, 0.015, 0.3])
    cbar_li = plt.colorbar(im_li_y, cax=cbar_li_ax)
    cbar_li.set_label('Alpha LI', fontsize=11)
    cbar_li.ax.tick_params(labelsize=10)

    # ------------------ ROW 2: TOTAL ALPHA POWER ------------------
    # Time course (-1.0 to 1.0 s)
    ax_pwr_tc = axes[1, 0]
    for c in cohorts:
        ax_pwr_tc.plot(times_vector, avg_power_tc[c], color=colors[c], lw=3, label=c)
        ax_pwr_tc.fill_between(times_vector, avg_power_tc[c] - sem_power_tc[c], avg_power_tc[c] + sem_power_tc[c], color=colors[c], alpha=0.15)
        
    ax_pwr_tc.axvspan(-0.75, 0.0, color='gray', alpha=0.08)
    ax_pwr_tc.axhline(1.0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax_pwr_tc.axvline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    ax_pwr_tc.set_xlim(-1.0, 1.0)
    ax_pwr_tc.set_xlabel("Time (s)", fontsize=13)
    ax_pwr_tc.set_ylabel("Relative Alpha Power (Scaled)", fontsize=13)
    ax_pwr_tc.set_title("Time Course", fontsize=14, fontweight='semibold')
    sns.despine(ax=ax_pwr_tc)

    # Consistent Power topo scaling (Power is positive, use standard 'viridis' or 'Reds' for absolute activation)
    pwr_vmax = max(avg_power_topo['Young'].max(), avg_power_topo['Old'].max())
    pwr_vmin = min(avg_power_topo['Young'].min(), avg_power_topo['Old'].min())
    
    # Younger Power Topomap
    ax_pwr_topo_y = axes[1, 1]
    im_pwr_y, _ = mne.viz.plot_topomap(
        avg_power_topo['Young'], info, axes=ax_pwr_topo_y, cmap='Reds',
        vlim=(pwr_vmin, pwr_vmax), show=False, outlines='head', sensors=False
    )
    ax_pwr_topo_y.set_title("Younger Participants\n(-0.75 to 0.0 s)", fontsize=13, fontweight='semibold')
    
    # Older Power Topomap
    ax_pwr_topo_o = axes[1, 2]
    im_pwr_o, _ = mne.viz.plot_topomap(
        avg_power_topo['Old'], info, axes=ax_pwr_topo_o, cmap='Reds',
        vlim=(pwr_vmin, pwr_vmax), show=False, outlines='head', sensors=False
    )
    ax_pwr_topo_o.set_title("Older Participants\n(-0.75 to 0.0 s)", fontsize=13, fontweight='semibold')
    
    # Shared Power colorbar
    cbar_pwr_ax = fig.add_axes([0.92, 0.12, 0.015, 0.3])
    cbar_pwr = plt.colorbar(im_pwr_y, cax=cbar_pwr_ax)
    cbar_pwr.set_label('Rel. Alpha Power', fontsize=11)
    cbar_pwr.ax.tick_params(labelsize=10)

    # Labels and layout adjustments
    fig.suptitle(f"Pre- and Post-Stimulus Alpha Dynamics (LI & Total Power), N = {len(subject_ids)}", fontsize=17, fontweight='semibold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the output plot
    try:
        current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        current_file_dir = os.path.abspath(os.getcwd())
    output_path = os.path.join(current_file_dir, "spacecue/plots/pre_stim_alpha_lateralization.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {output_path}")
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
