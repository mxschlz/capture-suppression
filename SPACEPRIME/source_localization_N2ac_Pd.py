import mne
import numpy as np
import matplotlib.pyplot as plt
import os
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
PICK_ORI = "normal" # Orientation of dipoles ('normal' or None)

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.7  # Seconds (should match sensor-level)

# Time windows for N2ac and Pd in source space (adjust as needed)
# These are example windows, you might need to refine them based on your data
N2AC_SOURCE_WINDOW = (None, None) # Example: 250-350ms
PD_SOURCE_WINDOW = (None, None)   # Example: 300-400ms

# --- End of Parameters ---

# --- Data Storage ---
# Store source estimates for each subject and condition
subject_source_estimates = {
    'target_diff_stc': [],
    'distractor_diff_stc': []
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

        # --- Corrected Logic: Compute source estimates from epochs, then average ---

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
            # Create the contra-ipsi difference STC for the target condition
            # For the Left Hemisphere (LH), contralateral is the right-sided stimulus.
            lh_contra_minus_ipsi = avg_stc_right_target.lh_data - avg_stc_left_target.lh_data

            # For the Right Hemisphere (RH), contralateral is the left-sided stimulus.
            rh_contra_minus_ipsi = avg_stc_left_target.rh_data - avg_stc_right_target.rh_data

            # Combine the hemisphere data into a new STC object
            target_diff_stc_sub = avg_stc_left_target.copy()  # Use as a template
            target_diff_stc_sub.data = np.vstack([lh_contra_minus_ipsi, rh_contra_minus_ipsi])
            subject_source_estimates['target_diff_stc'].append(target_diff_stc_sub)

        # --- Distractor Condition (Pd) ---
        avg_stc_left_distractor = epochs_to_avg_stc(left_distractor_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)
        avg_stc_right_distractor = epochs_to_avg_stc(right_distractor_epochs, inverse_operator, LAMBDA2, INV_METHOD, PICK_ORI)

        if avg_stc_left_distractor and avg_stc_right_distractor:
            # Create the contra-ipsi difference STC for the distractor condition
            # For the Left Hemisphere (LH), contralateral is the right-sided stimulus.
            lh_contra_minus_ipsi = avg_stc_right_distractor.lh_data - avg_stc_left_distractor.lh_data

            # For the Right Hemisphere (RH), contralateral is the left-sided stimulus.
            rh_contra_minus_ipsi = avg_stc_left_distractor.rh_data - avg_stc_right_distractor.rh_data

            # Combine the hemisphere data into a new STC object
            distractor_diff_stc_sub = avg_stc_left_distractor.copy() # Use as a template
            distractor_diff_stc_sub.data = np.vstack([lh_contra_minus_ipsi, rh_contra_minus_ipsi])
            subject_source_estimates['distractor_diff_stc'].append(distractor_diff_stc_sub)

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
if subject_source_estimates['target_diff_stc']:
    ga_target_diff_stc = np.sum(subject_source_estimates['target_diff_stc']) / len(subject_source_estimates['target_diff_stc'])
else:
    ga_target_diff_stc = None
    print("Warning: No target difference STCs to average.")

if subject_source_estimates['distractor_diff_stc']:
    ga_distractor_diff_stc = np.sum(subject_source_estimates['distractor_diff_stc']) / len(subject_source_estimates['distractor_diff_stc'])
else:
    ga_distractor_diff_stc = None
    print("Warning: No distractor difference STCs to average.")

# --- Visualization of Source Estimates ---
print("\\n--- Visualizing Source Estimates ---")

# Plotting grand average STCs
# This will open a 3D brain visualization.

# For N2ac (Target)
if ga_target_diff_stc:
    print("  Plotting Grand Average Target Contra-Ipsi Difference STC (N2ac window)...")
    stc_n2ac_plot = ga_target_diff_stc.copy()

    brain_n2ac = stc_n2ac_plot.plot(
        subject='fsaverage',  # Explicitly tell MNE to use the fsaverage brain
        hemi='both',
        clim="auto",
        subjects_dir=SUBJECTS_DIR,
        smoothing_steps=20,
        colormap="mne"
    )

# For Pd (Distractor)
if ga_distractor_diff_stc:
    print("  Plotting Grand Average Distractor Contra-Ipsi Difference STC (Pd window)...")
    stc_pd_plot = ga_distractor_diff_stc.copy()

    brain_pd = stc_pd_plot.plot(
        subjects_dir=SUBJECTS_DIR,
        subject=FSMRI_SUBJ,
        initial_time=None,
        clim='auto',
        hemi='split',
        views=['lat', 'med'],
        time_unit='s',
        size=(800, 400),
        smoothing_steps=20,
        title='Grand Average Distractor Contra-Ipsi Difference (Pd)',
        time_viewer=True
    )

print("\\nSource localization script finished. Review the generated plots.")

