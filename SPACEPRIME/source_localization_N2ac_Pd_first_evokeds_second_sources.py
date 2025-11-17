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
OUTLIER_RT_THRESHOLD = 2
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

"""# 1. Filter by phase
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    print(f"Filtering out trials from phase {FILTER_PHASE}...")
    df = df[df[PHASE_COL] != FILTER_PHASE]
    print(f"  Trials remaining after phase filter: {len(df)}")

# 2. Remove RT outliers
if REACTION_TIME_COL in df.columns:
    print(f"Removing RT outliers (threshold: {OUTLIER_RT_THRESHOLD} SD)...")
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"  Trials remaining after RT outlier removal: {len(df)}")
"""
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
            """Average epochs at sensor level, then apply inverse solution to the evoked data."""
            if len(epochs_obj) == 0:
                return None
            # First, average the epochs at the sensor level to create an Evoked object
            evoked = epochs_obj.average()

            # Then, apply the inverse solution to the Evoked object
            avg_stc = mne.minimum_norm.apply_inverse(
                evoked, inv_op, lambda2=l2, method=meth, pick_ori=po, verbose=False
            )
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
    peak_val, peak_time = stc.copy().crop(ERP_TMIN, ERP_TMAX).get_peak(mode='abs', time_as_index=False)
    print(f"  Plotting {condition_name} at peak time: {peak_time * 1000:.1f} ms")
    stc.plot(
        subject=FSMRI_SUBJ,
        clim="auto",
        hemi='split',
        size=(800, 400),
        smoothing_steps=10,
        initial_time=peak_time,
        time_label=f'GA {condition_name}\nPeak at {peak_time * 1000:.0f} ms'
    )

plot_ga_condition(ga_left_target, "Left Target")
plot_ga_condition(ga_right_target, "Right Target")
plot_ga_condition(ga_left_distractor, "Left Distractor")
plot_ga_condition(ga_right_distractor, "Right Distractor")

print("\n--- 2. Computing and Plotting Contra-Minus-Ipsi Difference Waves ---")

# --- N2ac (Target) ---
if ga_left_target and ga_right_target:
    ga_target_diff_stc = ga_left_target.copy()  # Use as a template
    # LH: contra (right stim) - ipsi (left stim)
    ga_target_diff_stc.lh_data = ga_right_target.lh_data - ga_left_target.lh_data
    # RH: contra (left stim) - ipsi (right stim)
    ga_target_diff_stc.rh_data = ga_left_target.rh_data - ga_right_target.rh_data

    # Find peak negativity (N2ac) in the difference wave
    _, peak_time_n2ac = ga_target_diff_stc.copy().crop(ERP_TMIN, ERP_TMAX).get_peak(mode='neg', time_as_index=False)
    print(f"  Peak N2ac (contra-ipsi) time found at: {peak_time_n2ac * 1000:.1f} ms")
    ga_target_diff_stc.plot(
        subject=FSMRI_SUBJ, clim="auto", hemi='split', size=(800, 400),
        smoothing_steps=10, initial_time=peak_time_n2ac,
        time_label=f'N2ac (Contra-Ipsi)\nPeak at {peak_time_n2ac * 1000:.0f} ms'
    )
else:
    print("  Skipping N2ac plot: Missing one or both grand average target STCs.")

# --- Pd (Distractor) ---
if ga_left_distractor and ga_right_distractor:
    ga_distractor_diff_stc = ga_left_distractor.copy()  # Use as a template
    # LH: contra (right stim) - ipsi (left stim)
    ga_distractor_diff_stc.lh_data = ga_right_distractor.lh_data - ga_left_distractor.lh_data
    # RH: contra (left stim) - ipsi (right stim)
    ga_distractor_diff_stc.rh_data = ga_left_distractor.rh_data - ga_right_distractor.rh_data

    # Find peak positivity (Pd) in the difference wave
    _, peak_time_pd = ga_distractor_diff_stc.copy().crop(ERP_TMIN, ERP_TMAX).get_peak(mode='pos', time_as_index=False)
    print(f"  Peak Pd (contra-ipsi) time found at: {peak_time_pd * 1000:.1f} ms")
    ga_distractor_diff_stc.plot(
        subject=FSMRI_SUBJ, clim="auto", hemi='split', size=(800, 400),
        smoothing_steps=10, initial_time=peak_time_pd,
        time_label=f'Pd (Contra-Ipsi)\nPeak at {peak_time_pd * 1000:.0f} ms'
    )
else:
    print("  Skipping Pd plot: Missing one or both grand average distractor STCs.")

print("\\nSource localization script finished. Review the generated plots.")