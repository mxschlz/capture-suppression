import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os # Added for path joining
import pandas as pd # Added for potential metadata handling

# --- Import necessary components from DvM ---
# Adjust the import path based on your project structure
# Assuming DvM is installable or in the Python path
try:
    # If DvM is installed or properly structured in PYTHONPATH
    from DvM.eeg_analyses.ERP import ERP
    # If FolderStructure is needed and adapted for BIDS (optional)
    # from DvM.FolderStructure import FolderStructureBIDS
except ImportError:
    # If DvM is accessed via relative paths (adjust as needed)
    import sys
    # Example: Add the parent directory of DvM if needed
    # sys.path.append('/path/to/parent/of/DvM')
    from DvM.eeg_analyses.ERP import ERP
    # from DvM.FolderStructure import FolderStructureBIDS
    print("Imported DvM components via sys.path modification.")
# --- End DvM Import ---

from SPACEPRIME import get_data_path
from mne.stats import permutation_cluster_test
from scipy.stats import ttest_ind # Can likely be replaced by MNE stats
from scipy.stats import t
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos # Keep for now, may need adaptation
import seaborn as sns

plt.ion()

# --- Configuration ---
bids_root = get_data_path() # Assuming this returns the root like /.../SPACEPRIME/
pipeline_name = 'epoching' # Where the input epochs are located
output_pipeline_name = 'DvM_ERP_lateral' # Name for saving ERP derivatives
subjects_to_process = subject_ids[:20] # Or your full list

# ERP Class Parameters
baseline = None # Define your baseline period
l_filter = None # Optional low-pass filter frequency
h_filter = None # Optional high-pass filter frequency
downsample = None # Optional downsampling frequency
crop_tmin = 0     # Crop time for analysis
crop_tmax = 0.7

# Define conditions using pandas query strings on metadata
# These queries select trials based on the NON-lateralized feature
pos_labels_target = {
    'TargetLoc': [1, 3] # Select trials where singleton (distractor) was midline
}
cnds_distractor = {
    'SingletonLoc': [1, 3] # Select trials where target was midline
}
# Which metadata column indicates the lateral position for flipping?
# This column MUST contain the values defined in pos_labels ('1', '3')
lateral_target_header = 'target_pos'
lateral_distractor_header = 'singleton_pos'

midline = {"SingletonLoc": [2]} # Value indicating midline position (used for exclusion if needed)

# Electrodes of interest for contra/ipsi difference (used in group analysis)
elec_oi_contra = ['C3']
elec_oi_ipsi = ['C4']


# --- Subject Processing Loop ---
print(f"--- Starting ERP processing for {len(subjects_to_process)} subjects ---")
for sj_id in subjects_to_process:
    subject_label = f"sub-{sj_id}"
    print(f"\nProcessing {subject_label}...")

    # --- Load Data for Subject ---
    # Construct the BIDS-like path manually (or use FolderStructureBIDS if adapted)
    epoch_fname_pattern = f"{bids_root}derivatives/{pipeline_name}/{subject_label}/eeg/{subject_label}_task-spaceprime-epo.fif"
    epoch_files = glob.glob(epoch_fname_pattern)

    if not epoch_files:
        print(f"  WARNING: Epoch file not found for {subject_label}. Skipping.")
        continue

    epoch_file = epoch_files[0]
    try:
        epochs_sj = mne.read_epochs(epoch_file, preload=True)
        # --- CRUCIAL: Ensure Metadata Exists ---
        if epochs_sj.metadata is None:
             print(f"  ERROR: Epochs for {subject_label} lack metadata. Cannot proceed with condition selection. Skipping.")
             # Add logic here to load metadata from a separate .tsv file if necessary
             # beh_file = f"{bids_root}derivatives/{pipeline_name}/{subject_label}/beh/{subject_label}_task-spaceprime_events.tsv"
             # metadata = pd.read_csv(beh_file, sep='\t')
             # # Ensure metadata length matches epochs and add it:
             # if len(metadata) == len(epochs_sj):
             #    epochs_sj.metadata = metadata
             # else:
             #    print(f"  ERROR: Metadata length mismatch for {subject_label}. Skipping.")
             #    continue
             continue # Skip subject if no metadata

        # Apply cropping if needed (can also be done in ERP class via time_oi)
        epochs_sj.crop(crop_tmin, crop_tmax)

        # Apply baseline if not already done during preprocessing
        # epochs_sj.apply_baseline(baseline) # Or let ERP class handle it

    except Exception as e:
        print(f"  ERROR: Failed to load or preprocess epochs for {subject_label}: {e}")
        continue

    # --- Instantiate ERP Analyzer ---
    # The 'header' argument here is used by FolderStructure for subdirs,
    # less critical if using BIDS paths directly, but provide something descriptive.
    # We will save outputs to a specific derivatives pipeline.
    erp_analyzer = ERP(sj=sj_id, epochs=epochs_sj, beh=epochs_sj.metadata,
                       header=output_pipeline_name, # Used for organizing output if not BIDS
                       baseline=baseline,
                       l_filter=l_filter, h_filter=h_filter, downsample=downsample,
                       report=False) # Disable MNE report generation for now

    # --- Calculate and Save Lateralized ERPs ---
    print(f"  Calculating lateralized ERPs for Target...")
    try:
        # This calculates the contra-ipsi difference wave and saves it
        erp_analyzer.lateralized_erp(pos_labels=pos_labels_target,
                                     cnds=None,
                                     midline=midline,
                                     topo_flip=None, # Essential for lateralized
                                     RT_split=False,
                                     time_oi=(crop_tmin, crop_tmax)) # Optional time window crop
    except Exception as e:
        print(f"  ERROR: Failed lateralized ERP calculation (Target) for {subject_label}: {e}")
        # Consider more specific error handling based on expected DvM errors

    print(f"  Calculating lateralized ERPs for Distractor...")
    try:
        erp_analyzer.lateralized_erp(pos_labels=pos_labels,
                                     cnds=cnds_distractor,
                                     midline=midline_policy,
                                     midline_val=midline_val,
                                     header=lateral_distractor_header, # Column indicating singleton position
                                     topo_flip=True,
                                     save=True,
                                     report=False,
                                     RT_split=False,
                                     time_oi=(crop_tmin, crop_tmax))
    except Exception as e:
        print(f"  ERROR: Failed lateralized ERP calculation (Distractor) for {subject_label}: {e}")

print("--- Subject processing complete ---")


# --- Group Analysis ---
print("\n--- Starting Group Analysis ---")

# --- Load Saved Subject-Level Difference Waves ---
# Construct paths to the saved evoked files based on ERP class naming convention
# The ERP class saves files like: sj_id_pipeline_name_condition_name-ave.fif
# in the directory structure defined by FolderStructure (likely relative to cwd
# unless FolderStructure was adapted).

# *** IMPORTANT: Adapt this path based on where ERP.save actually saves files ***
# If ERP uses the original FolderStructure, it saves relative to os.getcwd().
# If you adapted FolderStructure/ERP to save in BIDS derivatives, adjust accordingly.
# Assuming original FolderStructure saving relative to script execution dir:
erp_output_dir = os.path.join(os.getcwd(), output_pipeline_name) # Base dir ERP class saves into

target_erp_files = {}
distractor_erp_files = {}

for sj_id in subjects_to_process:
    subject_label = f"sub-{sj_id}"
    # Find target ERP file (assuming condition name was 'target_lateral')
    target_fname = f"{sj_id}_{output_pipeline_name}_target_lateral-ave.fif"
    target_fpath = os.path.join(erp_output_dir, str(sj_id), target_fname) # ERP class creates sj subfolder
    if os.path.exists(target_fpath):
        target_erp_files[subject_label] = target_fpath
    else:
        print(f"WARNING: Target ERP file not found for {subject_label} at {target_fpath}")

    # Find distractor ERP file (assuming condition name was 'distractor_lateral')
    distractor_fname = f"{sj_id}_{output_pipeline_name}_distractor_lateral-ave.fif"
    distractor_fpath = os.path.join(erp_output_dir, str(sj_id), distractor_fname)
    if os.path.exists(distractor_fpath):
        distractor_erp_files[subject_label] = distractor_fpath
    else:
        print(f"WARNING: Distractor ERP file not found for {subject_label} at {distractor_fpath}")

# Load the evoked objects
target_evokeds = {sj: mne.read_evokeds(fname, verbose=False)[0] for sj, fname in target_erp_files.items()}
distractor_evokeds = {sj: mne.read_evokeds(fname, verbose=False)[0] for sj, fname in distractor_erp_files.items()}

# Check if we loaded any ERPs
if not target_evokeds or not distractor_evokeds:
    print("\nERROR: No ERP files loaded for group analysis. Exiting.")
    exit()

print(f"Loaded target ERPs for {len(target_evokeds)} subjects.")
print(f"Loaded distractor ERPs for {len(distractor_evokeds)} subjects.")

# --- Calculate Group Averages using DvM.ERP static method ---
# This method returns stacked data (subjects x time) for selected electrodes
# and the grand average Evoked object.
print(f"Calculating group averages for electrodes: Contra={elec_oi_contra}, Ipsi={elec_oi_ipsi}")

# For Target
target_stacked_data, target_grand_average = ERP.group_lateralized_erp(
    list(target_evokeds.values()), # Pass list of Evoked objects
    elec_oi_c=elec_oi_contra,
    elec_oi_i=elec_oi_ipsi,
    set_mean=True # Average across electrodes if multiple are given
)

# For Distractor
distractor_stacked_data, distractor_grand_average = ERP.group_lateralized_erp(
    list(distractor_evokeds.values()),
    elec_oi_c=elec_oi_contra,
    elec_oi_i=elec_oi_ipsi,
    set_mean=True
)

# target_stacked_data / distractor_stacked_data are now (n_subjects, n_times) numpy arrays
# target_grand_average / distractor_grand_average are mne.Evoked objects

# Get times from one of the grand averages (should be the same)
times = target_grand_average.times

# --- Plotting Group Averages ---
print("Plotting group average difference waves...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6)) # Simplified plot for difference waves

# Plot Target Difference Wave (Grand Average)
ax.plot(times, target_grand_average.get_data(picks='diff').flatten() * 1e6, color="g", label="Target Contra-Ipsi")
# Add shaded error (e.g., SEM or 95% CI)
target_sem = np.std(target_stacked_data, axis=0) / np.sqrt(target_stacked_data.shape[0])
ax.fill_between(times,
                (target_grand_average.get_data(picks='diff').flatten() - target_sem) * 1e6,
                (target_grand_average.get_data(picks='diff').flatten() + target_sem) * 1e6,
                color="g", alpha=0.2)

# Plot Distractor Difference Wave (Grand Average)
ax.plot(times, distractor_grand_average.get_data(picks='diff').flatten() * 1e6, color="purple", label="Distractor Contra-Ipsi")
# Add shaded error
distractor_sem = np.std(distractor_stacked_data, axis=0) / np.sqrt(distractor_stacked_data.shape[0])
ax.fill_between(times,
                (distractor_grand_average.get_data(picks='diff').flatten() - distractor_sem) * 1e6,
                (distractor_grand_average.get_data(picks='diff').flatten() + distractor_sem) * 1e6,
                color="purple", alpha=0.2)


ax.hlines(y=0, xmin=times[0], xmax=times[-1], linestyle="--", color="k")
ax.legend()
ax.set_title(f"Group Average Lateralized ERPs (Contra-Ipsi)\nElectrodes: C={elec_oi_contra}, I={elec_oi_ipsi}")
ax.set_ylabel("Amplitude [µV]")
ax.set_xlabel("Time [s]")
sns.despine(fig=fig)
plt.tight_layout()


# --- Statistics (Cluster Permutation on Difference Waves) ---
print("Running cluster permutation tests...")
n_permutations = 5000 # Reduced for speed, increase for publication
alpha = 0.05
tail = 0 # Two-tailed test (testing for difference from zero)
n_jobs = -1

# Threshold for single observation (difference wave vs 0)
# Use a paired t-test threshold (df = n_subjects - 1)
df_target = target_stacked_data.shape[0] - 1
threshold_target = t.ppf(1 - alpha / 2, df_target) if df_target > 0 else 3.0 # Fallback threshold

df_distractor = distractor_stacked_data.shape[0] - 1
threshold_distractor = t.ppf(1 - alpha / 2, df_distractor) if df_distractor > 0 else 3.0

print(f"Target threshold (df={df_target}): {threshold_target}")
print(f"Distractor threshold (df={df_distractor}): {threshold_distractor}")

# Test Target difference wave against zero
if df_target > 0:
    t_obs_t, clusters_t, cluster_pv_t, h0_t = mne.stats.permutation_cluster_1samp_test(
        target_stacked_data, # Data is (n_subjects, n_times)
        threshold=threshold_target,
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
        out_type="mask"
    )
    # Add significance shading to plot
    for i_c, c in enumerate(clusters_t):
        c = c[0] # Cluster is over time dimension
        if cluster_pv_t[i_c] <= alpha:
            ax.axvspan(times[c.start], times[c.stop - 1], color="g", alpha=0.2, label=f'_nolegend_ Target Sig p={cluster_pv_t[i_c]:.3f}')
else:
    print("Not enough subjects to run target cluster test.")

# Test Distractor difference wave against zero
if df_distractor > 0:
    t_obs_d, clusters_d, cluster_pv_d, h0_d = mne.stats.permutation_cluster_1samp_test(
        distractor_stacked_data,
        threshold=threshold_distractor,
        n_permutations=n_permutations,
        tail=tail,
        n_jobs=n_jobs,
        out_type="mask"
    )
    # Add significance shading to plot
    for i_c, c in enumerate(clusters_d):
        c = c[0]
        if cluster_pv_d[i_c] <= alpha:
            ax.axvspan(times[c.start], times[c.stop - 1], color="purple", alpha=0.2, label=f'_nolegend_ Distractor Sig p={cluster_pv_d[i_c]:.3f}')
else:
    print("Not enough subjects to run distractor cluster test.")

# Refresh legend after adding significance spans potentially
handles, labels = ax.get_legend_handles_labels()
# Filter out _nolegend_ entries if necessary before calling legend again
# ax.legend(handles, labels)
plt.show() # Show updated plot

# --- TOPOGRAPHIES (Adaptation Needed) ---
# The original `difference_topos` function likely needs adaptation.
# It probably expects a single concatenated Epochs object.
# You might need to modify it to:
# 1. Take the dictionaries of subject evokeds (target_evokeds, distractor_evokeds)
# 2. Calculate the difference topo per subject.
# 3. Average the subject topographies.
# OR modify it to work directly on the grand average difference waves
# (target_grand_average, distractor_grand_average) which contain data for all channels.

print("\n--- Topography plotting requires adaptation ---")
# Example: Using the grand average difference wave directly
# Ensure your montage is loaded correctly
settings_path = f"{get_data_path()}settings/"
try:
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    # Make sure the grand average info has the montage
    target_grand_average.set_montage(montage, on_missing='ignore')
    distractor_grand_average.set_montage(montage, on_missing='ignore')

    # Plot topomap sequence for Target grand average difference
    plot_title_prefix = "Target Grand Average"
    diff_wave_evoked = target_grand_average # Use the grand average Evoked object

    start_time = 0.1
    end_time = 0.4
    time_step = 0.05
    times_to_plot = np.arange(start_time, end_time + time_step, time_step)

    fig_topo_t = diff_wave_evoked.plot_topomap(times=times_to_plot, average=0.05, # Average over 50ms windows centered on times
                                            title=f"{plot_title_prefix} Contra-Ipsi",
                                            cmap='RdBu_r', show=False) # Show False to display later if needed
    fig_topo_t.suptitle(f"{plot_title_prefix} Contra-Ipsi Topomaps", fontsize=16, y=1.02) # Adjust title position
    plt.show() # Show the target topomap figure

    # Plot topomap sequence for Distractor grand average difference
    plot_title_prefix = "Distractor Grand Average"
    diff_wave_evoked = distractor_grand_average

    fig_topo_d = diff_wave_evoked.plot_topomap(times=times_to_plot, average=0.05,
                                            title=f"{plot_title_prefix} Contra-Ipsi",
                                            cmap='RdBu_r', show=False)
    fig_topo_d.suptitle(f"{plot_title_prefix} Contra-Ipsi Topomaps", fontsize=16, y=1.02)
    plt.show() # Show the distractor topomap figure

except FileNotFoundError:
    print("WARNING: Montage file not found. Skipping topography plots.")
except Exception as e:
    print(f"ERROR during topography plotting: {e}")


# --- Optional: Export quantified ERP measures ---
# Define time windows and electrodes for quantification
quantify_windows = {
    'N2pc_time': (0.18, 0.28),
    'CDA_time': (0.3, 0.6)
}
# Use the same elec_oi_contra/ipsi as defined earlier

print("\nExporting quantified ERP measures...")
try:
    # Create a dictionary matching the structure expected by erp_to_csv
    erps_for_csv = {
        'Target_Lateral': list(target_evokeds.values()),
        'Distractor_Lateral': list(distractor_evokeds.values())
    }
    # Define the output CSV path
    csv_filename = os.path.join(os.getcwd(), f"{output_pipeline_name}_lateral_erp_measures.csv")

    ERP.erp_to_csv(
        erps=erps_for_csv,
        sj_id=[sj.replace('sub-', '') for sj in target_evokeds.keys()], # Pass list of subject IDs
        window_oi=quantify_windows,
        elec_oi_c=elec_oi_contra, # Pass contra electrodes
        elec_oi_i=elec_oi_ipsi,  # Pass ipsi electrodes
        method='mean', # or 'auc'
        fname=csv_filename,
        folder=None, # Let fname handle the full path
        name='diff', # Measure the 'diff' channel created by group_lateralized_erp
        individual_channels=False # We want the difference measure
    )
    print(f"Quantified measures saved to: {csv_filename}")
except Exception as e:
    print(f"ERROR: Failed to export ERP measures to CSV: {e}")


print("\n--- Analysis Script Finished ---")
