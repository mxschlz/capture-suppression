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

# Statistical Test Type Switch
# Set to 'one-sided' for directional hypotheses (e.g., N2ac < 0, Pd > 0)
# Set to 'two-sided' for non-directional hypotheses (e.g., N2ac != 0)
TEST_TYPE = 'two-sided' # <--- NEW PARAMETER

# Statistical Threshold for Plotting
if TEST_TYPE == 'one-sided':
    SIGNIFICANCE_Z_THRESHOLD = 1.645
elif TEST_TYPE == 'two-sided':
    SIGNIFICANCE_Z_THRESHOLD = 1.96
else:
    raise ValueError("TEST_TYPE must be 'one-sided' or 'two-sided'")
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

def compute_t_test_stc(contra_stcs, ipsi_stcs, component_name, test_type): # Added test_type
    """
    Averages STCs in the ERP window (200-400ms) and then performs a paired t-test
    between contralateral and ipsilateral conditions. The resulting t-values are
    then z-transformed across vertices. Returns an STC of z-scored t-values.
    """
    from scipy.stats import ttest_rel, norm

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

    # --- Determine the 'alternative' for ttest_rel based on component and test_type ---
    if test_type == 'one-sided':
        if component_name == 'Pd': # Pd: contra > ipsi (positive effect)
            alternative_ttest = 'greater'
        elif component_name == 'N2ac': # N2ac: contra < ipsi (negative effect)
            alternative_ttest = 'less'
        else:
            raise ValueError(f"Unknown component_name '{component_name}' for one-sided test.")
    elif test_type == 'two-sided':
        alternative_ttest = 'two-sided'
    else:
        raise ValueError("test_type must be 'one-sided' or 'two-sided'")

    print(f"  Performing {test_type} t-test for {component_name} with alternative='{alternative_ttest}'...")

    # Perform paired t-test for each vertex and time point
    t_values, p_values = ttest_rel(X_contra, X_ipsi, axis=0, nan_policy='omit', alternative=alternative_ttest)

    # Convert p-values to z-scores
    if test_type == 'one-sided':
        z_scores = norm.isf(p_values) # p_values are already one-sided
    elif test_type == 'two-sided':
        z_scores = norm.isf(p_values / 2) # Convert two-sided p to one-sided for norm.isf
    else:
        raise ValueError("test_type must be 'one-sided' or 'two-sided'")
    z_scores[np.isinf(z_scores)] = 0 # Handle cases where p=0 -> z=inf
    z_scores *= np.sign(t_values) # Re-apply the original sign of the effect

    # Create a new STC to store the z-scores
    # This STC will have only one "time" point, representing the average over the window
    t_stc = mne.SourceEstimate(
        z_scores, vertices=contra_stcs[0].vertices, tmin=0, tstep=1, subject=FSMRI_SUBJ
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


def report_significant_clusters(stc, threshold, component_name, subjects_dir):
    """
    Identifies and reports the anatomical labels corresponding to significant
    clusters in a source estimate, based on a z-score threshold.

    Args:
        stc (mne.SourceEstimate): The source estimate containing z-scores.
        threshold (float): The z-score significance threshold.
        component_name (str): The name of the component (e.g., 'N2ac', 'Pd').
        subjects_dir (str): Path to the FreeSurfer subjects directory.
    """
    print(f"\n--- Anatomical Report for {component_name} (Threshold: z > {abs(threshold):.2f}) ---")

    # Load the Desikan-Killiany atlas ('aparc')
    try:
        labels_aparc = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
    except Exception as e:
        print(f"  Could not load 'aparc' atlas. Error: {e}. Skipping report.")
        return

    # Determine which vertices are significant
    # For N2ac (negative effect), we look for z < -threshold. For Pd (positive), z > threshold.
    if threshold < 0:
        sig_mask = stc.data < threshold
    else:
        sig_mask = stc.data > threshold

    # Get the vertex numbers for each hemisphere
    sig_lh_verts = stc.vertices[0][np.where(sig_mask[:len(stc.vertices[0])])[0]]
    sig_rh_verts = stc.vertices[1][np.where(sig_mask[len(stc.vertices[0]):])[0]]

    # Find which labels contain these significant vertices
    report = {}
    for label in labels_aparc:
        label_verts = label.vertices
        hemi_verts = sig_lh_verts if label.hemi == 'lh' else sig_rh_verts
        
        # Find the intersection of significant vertices and vertices in the current label
        n_sig_in_label = len(np.intersect1d(hemi_verts, label_verts))
        
        if n_sig_in_label > 0:
            report[label.name] = n_sig_in_label

    if not report:
        print("  No significant clusters found in any atlas labels.")
    else:
        # Sort and print the report for clarity
        sorted_report = sorted(report.items(), key=lambda item: item[1], reverse=True)
        print("  Significant activity found in the following regions:")
        for label_name, count in sorted_report:
            print(f"    - {label_name}: {count} vertices")

# --- Reorganize data for Contra vs. Ipsi comparison ---
print("\n--- Assembling Contralateral and Ipsilateral Datasets ---")
contra_target_stcs_full, ipsi_target_stcs_full = _get_contra_ipsi_stcs(subject_source_estimates, 'left_target', 'right_target')
contra_distractor_stcs_full, ipsi_distractor_stcs_full = _get_contra_ipsi_stcs(subject_source_estimates, 'left_distractor', 'right_distractor')

# --- N2ac (Target) T-test ---
t_stc_n2ac = compute_t_test_stc(contra_target_stcs_full, ipsi_target_stcs_full, 'N2ac', TEST_TYPE) # Pass TEST_TYPE
t_stc_n2ac_plot = None
if t_stc_n2ac:
    min_val = t_stc_n2ac.data.min()
    # We are interested in negative z-scores (ipsi > contra for N2ac)
    if min_val < -SIGNIFICANCE_Z_THRESHOLD:
        # --- Plotting Logic for N2ac ---
        # 1. Create a copy for plotting and threshold it.
        t_stc_n2ac_plot = t_stc_n2ac.copy()
        t_stc_n2ac_plot.data[t_stc_n2ac_plot.data > -SIGNIFICANCE_Z_THRESHOLD] = 0

        # 2. Sign-flip the data to make it positive for easier visualization.
        t_stc_n2ac_plot.data *= -1
        max_val_flipped = t_stc_n2ac_plot.data.max()

        # 3. Define color limits for the positive, sign-flipped data.
        # Start color exactly at threshold to avoid vague boundaries.
        lims = [SIGNIFICANCE_Z_THRESHOLD, SIGNIFICANCE_Z_THRESHOLD, max_val_flipped]

        brain_n2ac = t_stc_n2ac_plot.plot(
            subject=FSMRI_SUBJ, hemi='split', size=(1000, 800), # Consistent size
            views=['lat', 'med'],
            smoothing_steps=20, # Increased smoothing for visualization
            clim=dict(kind='value', lims=lims), colormap='Greens',
            time_label=f'N2ac z-scores\nAvg: {ERP_TMIN*1000:.0f}-{ERP_TMAX*1000:.0f} ms',
            cortex="low_contrast",
            surface="white"
        )

        # 4. Modify colorbar labels to show original negative values
        # The scalar_bars attribute is a dictionary. We get the first colorbar actor
        # from its values and then set the label format.
        # The format string '-%g' adds a negative sign and uses a general format for the number.
        scalar_bar = list(brain_n2ac._renderer.plotter.scalar_bars.values())[0]
        scalar_bar.SetLabelFormat("-%.2f")
        
        # --- Generate and print the anatomical report for N2ac ---
        report_significant_clusters(t_stc_n2ac, -SIGNIFICANCE_Z_THRESHOLD, 'N2ac', SUBJECTS_DIR)

        # Save the plot as an image file
        save_path_n2ac = 'group_N2ac_zmap.png'
        brain_n2ac.save_image(save_path_n2ac)
        print(f"  Saved N2ac z-map image to {save_path_n2ac}")
    else:
        print(f"  Skipping N2ac plot: No significant negative z-scores found "
              f"(min_val: {min_val:.3f}, threshold: {-SIGNIFICANCE_Z_THRESHOLD}).")

# --- Pd (Distractor) T-test ---
t_stc_pd = compute_t_test_stc(contra_distractor_stcs_full, ipsi_distractor_stcs_full, 'Pd', TEST_TYPE) # Pass TEST_TYPE
t_stc_pd_plot = None
if t_stc_pd:
    # For Pd (contra > ipsi), we expect positive t-values.
    # Mask non-significant values (z < 1.96)
    t_stc_pd_plot = t_stc_pd.copy()
    t_stc_pd_plot.data[t_stc_pd_plot.data < SIGNIFICANCE_Z_THRESHOLD] = 0
    
    max_val = t_stc_pd_plot.data.max()
    
    if max_val > SIGNIFICANCE_Z_THRESHOLD:
        # Start color exactly at threshold to avoid vague boundaries.
        lims = [SIGNIFICANCE_Z_THRESHOLD, SIGNIFICANCE_Z_THRESHOLD, max_val]

        brain_pd = t_stc_pd_plot.plot( # Plot both lateral and medial views
            subject=FSMRI_SUBJ, hemi='split', size=(1000, 800), # Consistent size
            views=['lat', 'med'],
            smoothing_steps=20,
            # Show only positive values: from 0 to max_z
            clim=dict(kind='value', lims=lims), colormap='Reds',
            time_label=f'Pd z-scores\nAvg: {ERP_TMIN*1000:.0f}-{ERP_TMAX*1000:.0f} ms',
            cortex="low_contrast",
            surface="white"
        )
        
        # --- Generate and print the anatomical report for Pd ---
        report_significant_clusters(t_stc_pd, SIGNIFICANCE_Z_THRESHOLD, 'Pd', SUBJECTS_DIR)
        
        # Save the plot as an image file
        save_path_pd = 'group_Pd_zmap.png'
        brain_pd.save_image(save_path_pd)
        print(f"  Saved Pd z-map image to {save_path_pd}")

# --- Direct Comparison: Pd vs. N2ac ---
print("\n--- Performing Direct Comparison: Pd vs. N2ac ---")

if t_stc_n2ac is not None and t_stc_pd is not None:
    print("  Combining computed z-scores from N2ac and Pd...")

    # --- Prepare Data for Plotting ---
    # 1. N2ac (Green): Absolute values of significant negative z-scores
    n2ac_plot_data = t_stc_n2ac.data.copy()
    # Keep only negative values (Contra < Ipsi)
    n2ac_plot_data[n2ac_plot_data > 0] = 0
    n2ac_plot_data = np.abs(n2ac_plot_data)
    # Threshold
    n2ac_mask = n2ac_plot_data > SIGNIFICANCE_Z_THRESHOLD
    n2ac_plot_data[~n2ac_mask] = 0

    # 2. Pd (Red): Significant positive z-scores
    pd_plot_data = t_stc_pd.data.copy()
    # Keep only positive values (Contra > Ipsi)
    pd_plot_data[pd_plot_data < 0] = 0
    # Threshold
    pd_mask = pd_plot_data > SIGNIFICANCE_Z_THRESHOLD
    pd_plot_data[~pd_mask] = 0

    # 3. Overlap (Yellow): Where both are significant
    overlap_mask = n2ac_mask & pd_mask
    overlap_plot_data = np.zeros_like(n2ac_plot_data)
    if np.any(overlap_mask):
        # Use the maximum intensity of either signal for the overlap
        overlap_plot_data[overlap_mask] = np.maximum(n2ac_plot_data[overlap_mask], pd_plot_data[overlap_mask])

    max_n2ac = n2ac_plot_data.max()
    max_pd = pd_plot_data.max()
    max_overlap = overlap_plot_data.max()

    if max_n2ac > 0 or max_pd > 0:
        print(f"  Plotting Combined Map: N2ac (Green), Pd (Red), Overlap (Yellow)")

        # Initialize Brain with N2ac (Green)
        stc_combined = t_stc_n2ac.copy()
        stc_combined.data = n2ac_plot_data

        # Ensure upper limit is valid
        upper_lim_n2ac = max(max_n2ac, SIGNIFICANCE_Z_THRESHOLD + 0.1)
        clim_n2ac = dict(kind='value', pos_lims=[SIGNIFICANCE_Z_THRESHOLD, SIGNIFICANCE_Z_THRESHOLD, upper_lim_n2ac])

        brain_combined = stc_combined.plot(
            subject=FSMRI_SUBJ, hemi='split', size=(1000, 800),
            views=['lat', 'med'],
            smoothing_steps=20,
            colormap='Greens',
            clim=clim_n2ac,
            time_label=f'N2ac (Green) + Pd (Red) + Overlap (Yellow)\nThreshold: |z| > {SIGNIFICANCE_Z_THRESHOLD}',
            cortex="low_contrast",
            surface="white",
            transparent=True,
            background='black'
        )

        # Add Pd layer (Red)
        if max_pd > SIGNIFICANCE_Z_THRESHOLD:
            brain_combined.add_data(
                pd_plot_data,
                colormap='Reds',
                vertices=t_stc_pd.vertices,
                smoothing_steps=20,
                fmin=SIGNIFICANCE_Z_THRESHOLD,
                fmid=SIGNIFICANCE_Z_THRESHOLD,
                fmax=max_pd,
                transparent=True,
                colorbar=False
            )

        # Add Overlap layer (Yellow)
        if max_overlap > SIGNIFICANCE_Z_THRESHOLD:
            print(f"  Highlighting {np.sum(overlap_mask)} overlapping vertices in Yellow.")
            brain_combined.add_data(
                overlap_plot_data,
                colormap='Wistia', # Yellowish
                vertices=t_stc_pd.vertices,
                smoothing_steps=20,
                fmin=SIGNIFICANCE_Z_THRESHOLD,
                fmid=SIGNIFICANCE_Z_THRESHOLD,
                fmax=max_overlap,
                transparent=True,
                colorbar=False
            )

        # Fix scalar bar title if possible
        try:
            scalar_bar = list(brain_combined._renderer.plotter.scalar_bars.values())[0]
            scalar_bar.SetTitle("Z-score (N2ac)")
        except Exception:
            pass

        print("\n--- Anatomical Report for Combined Map ---")
        print("N2ac Clusters (Green):")
        report_significant_clusters(t_stc_n2ac, -SIGNIFICANCE_Z_THRESHOLD, 'N2ac', SUBJECTS_DIR)
        print("Pd Clusters (Red):")
        report_significant_clusters(t_stc_pd, SIGNIFICANCE_Z_THRESHOLD, 'Pd', SUBJECTS_DIR)

        if np.any(overlap_mask):
            stc_overlap = t_stc_n2ac.copy()
            stc_overlap.data = overlap_plot_data
            print("Overlap Clusters (Yellow):")
            report_significant_clusters(stc_overlap, SIGNIFICANCE_Z_THRESHOLD, 'Overlap', SUBJECTS_DIR)

        save_path_combined = 'group_combined_N2ac_Pd_zmap.png'
        brain_combined.save_image(save_path_combined)
        print(f"  Saved combined z-map image to {save_path_combined}")
    else:
        print(f"  Skipping combined plot: No significant values found.")
else:
    print("  Cannot perform comparison: Missing N2ac or Pd data.")

print("\\nSource localization script finished. Review the generated plots.")
