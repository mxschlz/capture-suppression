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

def compute_t_test_stc(contra_stcs, ipsi_stcs, component_name, test_type, time_resolved=False): # Added test_type
    """
    Averages STCs in the ERP window (200-400ms) and then performs a paired t-test
    between contralateral and ipsilateral conditions. The resulting t-values are
    then z-transformed across vertices. Returns an STC of z-scored t-values.
    """
    from scipy.stats import ttest_rel, norm

    if not contra_stcs or not ipsi_stcs or len(contra_stcs) != len(ipsi_stcs):
        print(f"  Cannot compute t-test for {component_name}: Mismatched or empty data.")
        return None

    # 1. Prepare data
    if time_resolved:
        X_contra_list = [stc.copy().crop(ERP_TMIN, ERP_TMAX).data for stc in contra_stcs]
        X_ipsi_list = [stc.copy().crop(ERP_TMIN, ERP_TMAX).data for stc in ipsi_stcs]
        ref_stc = contra_stcs[0].copy().crop(ERP_TMIN, ERP_TMAX)
        tmin, tstep = ref_stc.tmin, ref_stc.tstep
    else:
        # Average activity within the ERP time window
        X_contra_list = [stc.copy().crop(ERP_TMIN, ERP_TMAX).mean().data for stc in contra_stcs]
        X_ipsi_list = [stc.copy().crop(ERP_TMIN, ERP_TMAX).mean().data for stc in ipsi_stcs]
        tmin, tstep = 0, 1

    # 2. Stack the data. Shape: (n_subjects, n_vertices, n_times_or_1)
    X_contra = np.stack(X_contra_list, axis=0)
    X_ipsi = np.stack(X_ipsi_list, axis=0)

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
    t_stc = mne.SourceEstimate(
        z_scores, vertices=contra_stcs[0].vertices, tmin=tmin, tstep=tstep, subject=FSMRI_SUBJ
    )
    if time_resolved:
        print(f"  Computed time-resolved t-test for {component_name}.")
    else:
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


def rotate_brain_view(brain, azimuth_lh=0, azimuth_rh=0, elevation=0):
    """
    Rotates the camera view for all subplots in the brain object.
    """
    try:
        # Access the PyVista plotter from the MNE Brain object
        plotter = brain._renderer.plotter

        # Iterate over all renderers (subplots)
        for i, renderer in enumerate(plotter.renderers):
            camera = renderer.GetActiveCamera()
            # Even indices are Left Hemisphere, Odd indices are Right Hemisphere
            if i % 2 == 0:
                camera.Azimuth(azimuth_lh)
            else:
                camera.Azimuth(azimuth_rh)
            camera.Elevation(elevation)
            renderer.ResetCameraClippingRange()

        plotter.render()
    except Exception as e:
        print(f"  Warning: Failed to rotate view: {e}")


def adjust_colorbar(brain, title=None):
    """
    Moves the colorbar to the lower-right subplot and adjusts its style.
    """
    try:
        plotter = brain._renderer.plotter
        if not plotter.scalar_bars:
            return

        scalar_bar = list(plotter.scalar_bars.values())[0]

        # Move to the last renderer (usually Bottom-Right) to ensure it is on the far right bottom
        if len(plotter.renderers) > 1:
            target_renderer = plotter.renderers[-1]
            for renderer in plotter.renderers:
                renderer.RemoveActor(scalar_bar)
            target_renderer.AddActor(scalar_bar)

        if title is not None:
            scalar_bar.SetTitle(title)

        # Adjust colorbar position and size
        scalar_bar.SetOrientationToVertical()
        scalar_bar.SetHeight(1.0)  # Height relative to the subplot
        scalar_bar.SetWidth(0.1)  # Width
        scalar_bar.SetPosition(0.92, 0.05)  # x=0.92 (Far Right), y=0.05 (Bottom)
        scalar_bar.SetNumberOfLabels(0)
    except Exception as e:
        print(f"  Warning: Failed to adjust colorbar: {e}")


def save_transparent_screenshot(brain, filename, spacing=10):
    """
    Takes a screenshot of the brain instance, makes the white background transparent,
    crops the image to the content with some padding, and saves it.
    """
    # Get the screenshot as a numpy array
    img = brain.screenshot()

    # Ensure it's a writable numpy array
    img = np.array(img)

    # Check if RGB or RGBA
    if img.shape[2] == 3:
        # Add alpha channel
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)

    # Identify white background pixels (assuming [255, 255, 255])
    mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)

    # Set alpha to 0 for background
    img[mask, 3] = 0

    # Identify rows and columns that have at least one non-transparent pixel
    non_empty_rows = np.where(np.any(img[:, :, 3] > 0, axis=1))[0]
    non_empty_cols = np.where(np.any(img[:, :, 3] > 0, axis=0))[0]

    if non_empty_rows.size > 0 and non_empty_cols.size > 0:
        # Add spacing to the indices to preserve some whitespace around content
        n_rows, n_cols = img.shape[:2]

        # Expand rows
        expanded_rows = np.concatenate([non_empty_rows + i for i in range(-spacing, spacing + 1)])
        expanded_rows = np.unique(expanded_rows)
        expanded_rows = expanded_rows[(expanded_rows >= 0) & (expanded_rows < n_rows)]

        # Expand cols
        expanded_cols = np.concatenate([non_empty_cols + i for i in range(-spacing, spacing + 1)])
        expanded_cols = np.unique(expanded_cols)
        expanded_cols = expanded_cols[(expanded_cols >= 0) & (expanded_cols < n_cols)]

        # Use np.ix_ to select the intersection of expanded rows and cols.
        img_cropped = img[np.ix_(expanded_rows, expanded_cols)]
    else:
        img_cropped = img

    # Save using matplotlib
    plt.imsave(filename, img_cropped)
    print(f"  Saved transparent screenshot to {filename}")


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
            subject=FSMRI_SUBJ, hemi='split', size=(1500, 1000), # Consistent size
            views=['lat', 'med'],
            smoothing_steps=20, # Increased smoothing for visualization
            clim=dict(kind='value', lims=lims), colormap='Greens',
            cortex="low_contrast",
            surface="inflated",
            background="white"
        )
        # Fix scalar bar title if possible
        adjust_colorbar(brain_n2ac)

        # --- Generate and print the anatomical report for N2ac ---
        report_significant_clusters(t_stc_n2ac, -SIGNIFICANCE_Z_THRESHOLD, 'N2ac', SUBJECTS_DIR)

        rotate_brain_view(brain_n2ac, azimuth_lh=-30, azimuth_rh=30)
        # Save the plot as an image file
        save_path_n2ac = 'group_N2ac_zmap.png'
        save_transparent_screenshot(brain_n2ac, save_path_n2ac, spacing=20)
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
            subject=FSMRI_SUBJ, hemi='split', size=(1500, 1000), # Consistent size
            views=['lat', 'med'],
            smoothing_steps=20,
            # Show only positive values: from 0 to max_z
            clim=dict(kind='value', lims=lims), colormap='Reds',
            cortex="low_contrast",
            surface="inflated",
            background="white"
        )
        # Fix scalar bar title if possible
        adjust_colorbar(brain_pd)

        # --- Generate and print the anatomical report for Pd ---
        report_significant_clusters(t_stc_pd, SIGNIFICANCE_Z_THRESHOLD/2, 'Pd', SUBJECTS_DIR)
        
        rotate_brain_view(brain_pd, azimuth_lh=-30, azimuth_rh=30)
        # Save the plot as an image file
        save_path_pd = 'group_Pd_zmap.png'
        save_transparent_screenshot(brain_pd, save_path_pd, spacing=20)

# --- Direct Comparison: Pd vs. N2ac ---
print("\n--- Performing Direct Comparison: Pd vs. N2ac ---")

if t_stc_n2ac is not None and t_stc_pd is not None:
    print("  Combining computed z-scores from N2ac and Pd...")

    # Filter N2ac: Keep only negative z-scores (Contra < Ipsi)
    # Positive z-scores in N2ac condition are treated as noise/irrelevant for N2ac map
    n2ac_filtered = t_stc_n2ac.data.copy()
    n2ac_filtered[n2ac_filtered > 0] = 0

    # Filter Pd: Keep only positive z-scores (Contra > Ipsi)
    # Negative z-scores in Pd condition are treated as noise/irrelevant for Pd map
    pd_filtered = t_stc_pd.data.copy()
    pd_filtered[pd_filtered < 0] = 0

    # Combine preserving the stronger signal (avoid cancellation)
    # If a voxel has both negative (N2ac) and positive (Pd) values, we keep the one with larger magnitude.
    combined_data = np.where(np.abs(n2ac_filtered) > np.abs(pd_filtered), n2ac_filtered, pd_filtered)

    t_stc_combined = t_stc_n2ac.copy()
    t_stc_combined.data = combined_data

    # Option: Clamp the max limit to avoid one strong peak desaturating the other
    # For example, use the 99.9th percentile or a fixed cap if outliers are present
    #max_abs_z = np.percentile(np.abs(t_stc_combined.data), 99.9)
    max_abs_z = np.max(np.abs(t_stc_combined.data)) # Original line

    if max_abs_z > SIGNIFICANCE_Z_THRESHOLD:
        print(f"  Combined Z-score range: {t_stc_combined.data.min():.2f} to {t_stc_combined.data.max():.2f}")

        # Plotting
        brain_combined = t_stc_combined.plot(
            subject=FSMRI_SUBJ, hemi='split', size=(1000, 800),
            views=['lat', 'med'],
            smoothing_steps=20,
            colormap='RdYlGn_r', # Red for positive (Pd), Green for negative (N2ac)
            clim=dict(kind='value', pos_lims=[SIGNIFICANCE_Z_THRESHOLD, SIGNIFICANCE_Z_THRESHOLD, max_abs_z]),
            time_label=None,
            cortex="low_contrast",
            surface="inflated",
            transparent=True,
            background="white"
        )

        # Fix scalar bar title if possible
        adjust_colorbar(brain_combined, title="Z-score")

        print("\n--- Anatomical Report for Combined Map ---")
        print("Positive Clusters (Pd dominant):")
        report_significant_clusters(t_stc_combined, SIGNIFICANCE_Z_THRESHOLD, 'Pd (Combined)', SUBJECTS_DIR)
        print("Negative Clusters (N2ac dominant):")
        report_significant_clusters(t_stc_combined, -SIGNIFICANCE_Z_THRESHOLD, 'N2ac (Combined)', SUBJECTS_DIR)

        rotate_brain_view(brain_combined, azimuth_lh=-10, azimuth_rh=10)
        save_path_combined = 'group_combined_N2ac_Pd_zmap.png'
        save_transparent_screenshot(brain_combined, save_path_combined)
    else:
        print(f"  Skipping combined plot: No significant values found (max_abs_z: {max_abs_z:.3f}).")
else:
    print("  Cannot perform comparison: Missing N2ac or Pd data.")

print("\n--- Performing Time-Resolved Combined Analysis (200-400ms) ---")

t_stc_n2ac_time = compute_t_test_stc(contra_target_stcs_full, ipsi_target_stcs_full, 'N2ac', TEST_TYPE, time_resolved=True)
t_stc_pd_time = compute_t_test_stc(contra_distractor_stcs_full, ipsi_distractor_stcs_full, 'Pd', TEST_TYPE, time_resolved=True)

if t_stc_n2ac_time is not None and t_stc_pd_time is not None:
    print("  Combining time-resolved z-scores...")

    # Filter N2ac (keep negative)
    n2ac_data = t_stc_n2ac_time.data.copy()
    n2ac_data[n2ac_data > 0] = 0

    # Filter Pd (keep positive)
    pd_data = t_stc_pd_time.data.copy()
    pd_data[pd_data < 0] = 0

    # Combine preserving the stronger signal (avoid cancellation)
    combined_data_time = np.where(np.abs(n2ac_data) > np.abs(pd_data), n2ac_data, pd_data)

    t_stc_combined_time = t_stc_n2ac_time.copy()
    t_stc_combined_time.data = combined_data_time

    max_abs_z_time = np.max(np.abs(t_stc_combined_time.data))

    if max_abs_z_time > SIGNIFICANCE_Z_THRESHOLD:
        print(f"  Time-resolved Combined Z-score range: {t_stc_combined_time.data.min():.2f} to {t_stc_combined_time.data.max():.2f}")

        brain_combined_time = t_stc_combined_time.plot(
            subject=FSMRI_SUBJ, hemi='split', size=(1400, 900),
            views=['lat', 'med'],
            smoothing_steps=20,
            colormap='RdYlGn_r',
            clim=dict(kind='value', pos_lims=[SIGNIFICANCE_Z_THRESHOLD, SIGNIFICANCE_Z_THRESHOLD, max_abs_z_time]),
            time_label='Time: %0.3f s',
            cortex="0.5",
            surface="pial",
            transparent=True,
            background="white",
            time_viewer=True
        )
        try:
            scalar_bar = list(brain_combined_time._renderer.plotter.scalar_bars.values())[0]
            scalar_bar.SetTitle("Z-score")
            # Adjust colorbar position and size
            scalar_bar.SetOrientationToVertical()
            scalar_bar.SetHeight(0.6)  # Length
            scalar_bar.SetWidth(0.05)  # Thickness
            scalar_bar.SetPosition(0.475, 0.2)  # x, y position (0-1)
        except Exception:
            pass
    else:
        print("  No significant values found in time-resolved analysis.")

print("\\nSource localization script finished. Review the generated plots.")
