import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from SPACEPRIME import get_data_path, load_concatenated_epochs
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers # Added for preprocessing

plt.ion()

# --- Script Parameters ---

# --- 1. Preprocessing Parameters (adopted from LMM script) ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'

# --- 2. General Script Parameters --_

# Source Localization Specific Parameters
# get standard fMRI head model
SUBJECTS_DIR = mne.datasets.fetch_fsaverage(verbose=True)
# The fetcher might return a path ending in 'fsaverage', which can cause
# a double-folder issue (e.g., '.../fsaverage/fsaverage').
# We ensure we have the correct parent subjects directory.
if SUBJECTS_DIR.name == "fsaverage":
    SUBJECTS_DIR = SUBJECTS_DIR.parent
FSMRI_SUBJ = 'fsaverage' # FreeSurfer average subject for source space
INV_METHOD = 'dSPM' # Inverse method (e.g., 'dSPM', 'MNE', 'sLORETA')
LAMBDA2 = 1.0 / 9.0 # Regularization parameter for inverse solution
# PICK_ORI determines how to handle source orientation.
# "normal": Projects activity onto the direction perpendicular to the cortical surface.
#           This is the best choice for ERP component analysis (like N2ac/Pd)
#           because it provides a single, signed value that preserves the polarity
#           of the effect (e.g., negativity or positivity).
PICK_ORI = "normal"

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = -0.3, 0.7  # Seconds (should match sensor-level)

# --- End of Parameters ---

# --- Data Storage ---
# Store source estimates for each subject and condition
subject_source_estimates = {
    'left_target': [],
    'right_target': [],
    'left_distractor': [],
    'right_distractor': []
}
processed_subjects = []
times_vector = None # To be populated from the first subject's STC
epochs_info = None  # To be populated from the first subject's epochs

# --- Load and Preprocess Data (Sensor-level epochs) ---
print("--- Loading and Preprocessing Sensor-Level Data ---")
epochs = load_concatenated_epochs("spaceprime").crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Original number of trials: {len(epochs)}")

# Get metadata for preprocessing
df = epochs.metadata.copy().reset_index(drop=True)

# 1. Filter by phase
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    print(f"Filtering out trials from phase {FILTER_PHASE}...")
    df = df[df[PHASE_COL] != FILTER_PHASE]
    print(f"  Trials remaining after phase filter: {len(df)}")

# 2. Remove RT outliers
if REACTION_TIME_COL in df.columns:
    print(f"Removing RT outliers (threshold: {OUTLIER_RT_THRESHOLD} SD)...")
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"  Trials remaining after RT outlier removal: {len(df)}")

# 3. Apply the filter back to the epochs object
epochs = epochs[df.index]
epochs.set_eeg_reference('average', projection=True)
# use standard montage (the existing montage of the epochs is wrong)
montage = mne.channels.make_standard_montage("easycap-M1")
epochs.set_montage(montage)

print(f"Final number of trials after preprocessing: {len(epochs)}")
# --- End of Preprocessing ---

# --- Subject Loop for Source Estimation ---
for subject_id_num in subject_ids:
    subject_str = f"sub-{subject_id_num}"
    print(f"\\n--- Processing Subject: {subject_str} for Source Localization ---")
    try:
        epochs_sub = epochs[f"subject_id=={subject_id_num}"]
        print(f"  Loaded {len(epochs_sub)} epochs.")
    except Exception as e:
        print(f"  Error loading data for {subject_str}: {e}. Skipping.")
        continue

    if len(epochs_sub) == 0:
        print(f"  No epochs found for {subject_str}. Skipping.")
        continue

    if epochs_info is None:
        epochs_info = epochs_sub.info.copy()

    # --- Load Inverse Solution ---
    try:
        # Assuming inverse operators are stored per subject in a 'derivatives/source_localization_dSPM/sub-XX' folder
        inv_op_path = os.path.join(get_data_path(), 'derivatives', 'source_localization_dSPM', subject_str,
                                   f'{subject_str}_task-spaceprime-inv.fif')
        if not os.path.exists(inv_op_path):
            print(f"  Inverse operator not found for {subject_str} at {inv_op_path}. Skipping.")
            continue
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_op_path)
        print(f"  Loaded inverse operator for {subject_str}.")
    except Exception as e:
        print(f"  Could not load inverse operator for {subject_str}: {e}. Skipping.")
        continue

    all_conds_sub = list(epochs_sub.event_id.keys())

    try:
        # Define conditions for source estimation
        left_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
        right_target_epochs = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]].copy()
        left_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
        right_distractor_epochs = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]].copy()

        # --- Logic: Compute source estimates from epochs, then average ---

        def epochs_to_avg_stc(epochs_obj, inv_op, l2, meth, po):
            """Apply inverse solution to epochs and average the resulting STCs."""
            if len(epochs_obj) == 0:
                return None
            stcs = mne.minimum_norm.apply_inverse_epochs(
                epochs_obj, inv_op, lambda2=l2, method=meth, pick_ori=po, verbose=False
            )
            # Average the source estimates across trials, resulting in a single STC
            avg_stc = sum(stcs) / len(stcs)
            return avg_stc

        # --- Target Condition (N2ac) ---
        avg_stc_left_target = epochs_to_avg_stc(left_target_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)
        avg_stc_right_target = epochs_to_avg_stc(right_target_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)

        if avg_stc_left_target and avg_stc_right_target:
            subject_source_estimates['left_target'].append(avg_stc_left_target)
            subject_source_estimates['right_target'].append(avg_stc_right_target)

        # --- Distractor Condition (Pd) ---
        avg_stc_left_distractor = epochs_to_avg_stc(left_distractor_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)
        avg_stc_right_distractor = epochs_to_avg_stc(right_distractor_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)

        if avg_stc_left_distractor and avg_stc_right_distractor:
            subject_source_estimates['left_distractor'].append(avg_stc_left_distractor)
            subject_source_estimates['right_distractor'].append(avg_stc_right_distractor)

        # Set times_vector from the first valid STC
        if times_vector is None:
            if avg_stc_left_target:
                times_vector = avg_stc_left_target.times.copy()
            elif avg_stc_left_distractor:
                times_vector = avg_stc_left_distractor.times.copy()

        # Add subject to processed list if at least one condition was processed
        if (avg_stc_left_target and avg_stc_right_target) or \
           (avg_stc_left_distractor and avg_stc_right_distractor):
            processed_subjects.append(subject_id_num)

    except Exception as e_proc:
        print(f"  Error during source estimation for subject {subject_str}: {e_proc}. Skipping subject.")
        continue

print(f"\\n--- Successfully processed {len(processed_subjects)} subjects for source estimation: {processed_subjects} ---")

if not processed_subjects:
    raise RuntimeError("No subjects were successfully processed for source estimation. Cannot continue.")
if times_vector is None:
    raise RuntimeError("Essential data (times_vector) not populated. Check subject loop.")

# --- Group Level Analysis (Source Space) ---
# Average source estimates across subjects
print("\\n--- Performing Group-Level Source Analysis ---")

# Define the time window for ERP components
ERP_TMIN, ERP_TMAX = 0.200, 0.400  # 200 to 400 ms

# Helper function to compute grand average
def compute_ga(stc_list):
    if not stc_list:
        return None
    return sum(stc_list) / len(stc_list)

# Compute grand average for each condition
ga_left_target = compute_ga(subject_source_estimates['left_target'])
ga_right_target = compute_ga(subject_source_estimates['right_target'])
ga_left_distractor = compute_ga(subject_source_estimates['left_distractor'])
ga_right_distractor = compute_ga(subject_source_estimates['right_distractor'])

print("\n--- 1. Plotting Grand Average of Individual Conditions (Sanity Check) ---")

def plot_ga_condition(stc, condition_name):
    if stc is None:
        print(f"  Skipping plot for {condition_name}: No data.")
        return
    # Find the time of peak activity within the ERP window to set the plot time
    print(f"  Plotting {condition_name} at peak time: {0.1 * 1000:.1f} ms")
    stc.plot(
        subject=FSMRI_SUBJ,
        clim="auto",
        hemi='split',
        smoothing_steps=20,
        initial_time=0.1,
        time_label=f'GA {condition_name}\nPeak at {0.1 * 1000:.0f} ms',
        transparent=None,
        time_viewer=True,
        show_traces=True,
        surface="inflated",
        cortex="classic"
    )

plot_ga_condition(ga_left_target, "Left Target")
plot_ga_condition(ga_right_target, "Right Target")
plot_ga_condition(ga_left_distractor, "Left Distractor")
plot_ga_condition(ga_right_distractor, "Right Distractor")

print("\n--- 2. Computing and Plotting Paired T-tests (Contra vs. Ipsi) ---")

def compute_t_test_stc(contra_stcs, ipsi_stcs, component_name):
    """
    Averages STCs in the ERP window (200-400ms) and then performs a paired t-test
    between contralateral and ipsilateral conditions. The resulting t-values are
    then z-transformed across vertices. Returns an STC of z-scored t-values.
    """
    from scipy.stats import ttest_rel

    if not contra_stcs or not ipsi_stcs or len(contra_stcs) != len(ipsi_stcs):
        print(f"  Cannot compute t-test for {component_name}: Mismatched or empty data.")
        return None

    # 1. Average activity within the ERP time window for each subject and condition
    # The resulting data array will have shape (n_vertices, 1) for each subject
    X_contra_avg = [stc.copy().crop(ERP_TMIN, ERP_TMAX).mean().data for stc in contra_stcs]
    X_ipsi_avg = [stc.copy().crop(ERP_TMIN, ERP_TMAX).mean().data for stc in ipsi_stcs]

    # 2. Stack the averaged data from all subjects. Shape: (n_subjects, n_vertices, 1)
    X_contra = np.stack(X_contra_avg, axis=0)
    X_ipsi = np.stack(X_ipsi_avg, axis=0)

    # Perform paired t-test for each vertex and time point
    # We are interested in contra vs ipsi, so the test is on the difference
    # The result will have shape (n_vertices, 1)
    t_values, _ = ttest_rel(X_contra, X_ipsi, axis=0, nan_policy='omit')

    # Z-transform the t-values across all vertices
    mean_t = np.nanmean(t_values)
    std_t = np.nanstd(t_values)
    z_t_values = (t_values - mean_t) / std_t
    print(f"  T-values z-transformed (mean={mean_t:.2f}, std={std_t:.2f}).")

    # Create a new STC to store the t-values
    # This STC will have only one "time" point, representing the average over the window
    t_stc = mne.SourceEstimate(
        z_t_values, vertices=contra_stcs[0].vertices, tmin=0, tstep=1, subject=FSMRI_SUBJ
    )
    print(f"  Computed t-test for {component_name} based on average activity in {ERP_TMIN*1000:.0f}-{ERP_TMAX*1000:.0f} ms window.")
    return t_stc


def _get_contra_ipsi_stcs(subject_estimates, left_key, right_key):
    """
    Helper to create lists of contralateral and ipsilateral STCs for each subject.
    """
    contra_stcs = []
    ipsi_stcs = []
    n_subjects = len(subject_estimates[left_key])

    for i in range(n_subjects):
        stc_left = subject_estimates[left_key][i]
        stc_right = subject_estimates[right_key][i]

        # Contralateral: LH from right stim, RH from left stim
        contra_data = np.r_[stc_right.lh_data, stc_left.rh_data]
        contra_stc = mne.SourceEstimate(contra_data, vertices=stc_left.vertices,
                                        tmin=stc_left.tmin, tstep=stc_left.tstep, subject=FSMRI_SUBJ)
        contra_stcs.append(contra_stc)

        # Ipsilateral: LH from left stim, RH from right stim
        ipsi_data = np.r_[stc_left.lh_data, stc_right.rh_data]
        ipsi_stc = mne.SourceEstimate(ipsi_data, vertices=stc_left.vertices,
                                      tmin=stc_left.tmin, tstep=stc_left.tstep, subject=FSMRI_SUBJ)
        ipsi_stcs.append(ipsi_stc)

    return contra_stcs, ipsi_stcs


# --- Reorganize data for Contra vs. Ipsi comparison ---
print("\n--- Assembling Contralateral and Ipsilateral Datasets ---")
contra_target_stcs_full, ipsi_target_stcs_full = _get_contra_ipsi_stcs(subject_source_estimates, 'left_target', 'right_target')
contra_distractor_stcs_full, ipsi_distractor_stcs_full = _get_contra_ipsi_stcs(subject_source_estimates, 'left_distractor', 'right_distractor')

# --- N2ac (Target) T-test ---
t_stc_n2ac = compute_t_test_stc(contra_target_stcs_full, ipsi_target_stcs_full, 'N2ac')
if t_stc_n2ac:
    brain_n2ac = t_stc_n2ac.plot( # Plot both lateral and medial views
        subject=FSMRI_SUBJ, hemi='split', size=(800, 600),
        views=['lat', 'med'],
        smoothing_steps=20,
        # Show only negative values: from min_z to 0
        colormap='mne_analyze', # Good for single-sided data
        clim="auto",
        time_label=f'N2ac Z-scores (Contra vs Ipsi)\nAvg: {ERP_TMIN*1000:.0f}-{ERP_TMAX*1000:.0f} ms',
        surface="inflated",
        cortex="classic",
    )
    # Save the plot as an image file
    save_path_n2ac = 'group_N2ac_zmap.png'
    brain_n2ac.save_image(save_path_n2ac)
    print(f"  Saved N2ac z-map image to {save_path_n2ac}")
    brain_n2ac.close() # Close the plot window to avoid clutter

# --- Pd (Distractor) T-test ---
t_stc_pd = compute_t_test_stc(contra_distractor_stcs_full, ipsi_distractor_stcs_full, 'Pd')
if t_stc_pd:
    # For Pd (contra > ipsi), we expect positive t-values.
    # Find the max z-value to set the upper bound of the colormap.
    max_z = t_stc_pd.data.max()

    brain_pd = t_stc_pd.plot( # Plot both lateral and medial views
        subject=FSMRI_SUBJ, hemi='split', size=(800, 600),
        views=['lat', 'med'],
        smoothing_steps=20,
        # Show only positive values: from 0 to max_z
        colormap='mne_analyze',
        clim=dict(kind='value', lims=[0, max_z / 2, max_z]),
        time_label=f'Pd Z-scores (Contra vs Ipsi)\nAvg: {ERP_TMIN*1000:.0f}-{ERP_TMAX*1000:.0f} ms',
        surface="inflated",
        cortex="classic"
    )
    # Save the plot as an image file
    save_path_pd = 'group_Pd_zmap.png'
    brain_pd.save_image(save_path_pd)
    print(f"  Saved Pd z-map image to {save_path_pd}")
    brain_pd.close() # Close the plot window to avoid clutter

print("\\nSource localization script finished. Review the generated plots.")
