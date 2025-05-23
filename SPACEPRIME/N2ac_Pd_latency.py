import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
import os # For os.path.join
from scipy.signal import savgol_filter
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids

plt.ion()

# --- Parameters ---

# Epoching and Time Windows
EPOCH_TMIN, EPOCH_TMAX = 0.0, 0.7  # Seconds, crop for comparability

# Electrodes for Contra/Ipsi Calculation
# These are used for all three conditions (N2ac-like and Pd-like)
# For a left visual field stimulus, FC6 is contralateral, FC5 is ipsilateral.
# For a right visual field stimulus, FC5 is contralateral, FC6 is ipsilateral.
ELECTRODE_LEFT_HEMISPHERE = "FC5"
ELECTRODE_RIGHT_HEMISPHERE = "FC6"

# Plotting Parameters
SAVGOL_WINDOW_LENGTH = 51
SAVGOL_POLYORDER = 3
AMPLITUDE_SCALE_FACTOR = 1e6  # Volts to Microvolts (10e5 in original was 1e6)
PLOT_COLORS = {
    'distractor_lateral': "darkorange",
    'target_distractor_present': "blue",
    'target_distractor_absent': "green"
}
PLOT_LEGENDS = {
    'distractor_lateral': "Distractor lateral (Pd)",
    'target_distractor_present': "Target lateral (Distractor present, N2ac)",
    'target_distractor_absent': "Target lateral (Distractor absent, N2ac)"
}

# --- End of Parameters ---

# --- Data Storage for Subject-Level Results ---
# Stores the (contra - ipsi) difference wave for each subject and condition
subject_diff_waves = {
    'distractor_lateral': [],
    'target_distractor_present': [],
    'target_distractor_absent': []
}

times_vector = None # To be populated from the first subject
processed_subject_count = 0

# --- Subject Loop ---
print("Starting subject-level processing...")
for subject_id_int in subject_ids:
    subject_str = f"sub-{subject_id_int:02d}" # Assumes subject_ids are integers
    print(f"\n--- Processing Subject: {subject_str} ---")

    try:
        epoch_file_path_pattern = os.path.join(get_data_path(), "derivatives", "epoching",
                                               subject_str, "eeg", f"{subject_str}_task-spaceprime-epo.fif")
        epoch_files = glob.glob(epoch_file_path_pattern)
        if not epoch_files:
            print(f"  Epoch file not found for {subject_str} using pattern: {epoch_file_path_pattern}. Skipping.")
            continue

        epochs_sub = mne.read_epochs(epoch_files[0], preload=True)
        print(f"  Loaded {len(epochs_sub)} epochs.")

        # Apply baseline if needed (currently commented out in original)
        # epochs_sub.apply_baseline((-0.3, 0))

        # Filter for specific trial types if needed (currently commented out)
        # epochs_sub = epochs_sub["select_target==True"]
        # epochs_sub = epochs_sub["Priming==0"]

        epochs_sub.crop(EPOCH_TMIN, EPOCH_TMAX)
        print(f"  Cropped epochs to {EPOCH_TMIN}-{EPOCH_TMAX}s. {len(epochs_sub)} epochs remaining.")

        if not epochs_sub: # If cropping or filtering results in no epochs
            print(f"  No epochs remaining for {subject_str} after preproc. Skipping.")
            continue

        if times_vector is None: # Store times vector from the first successfully processed subject
            times_vector = epochs_sub.times.copy()

        all_conds_sub = list(epochs_sub.event_id.keys())
        subject_contributed_this_loop = False

        # --- 1. Distractor Lateral (Pd-like component) ---
        # Target-2-Singleton-1: Target Central, Distractor Left
        # Target-2-Singleton-3: Target Central, Distractor Right
        left_stim_epochs_dl = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]]
        right_stim_epochs_dl = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]]

        if len(left_stim_epochs_dl) > 0 and len(right_stim_epochs_dl) > 0:
            # Distractor Left: Contra is ELECTRODE_RIGHT_HEMISPHERE, Ipsi is ELECTRODE_LEFT_HEMISPHERE
            contra_dl_left_stim = left_stim_epochs_dl.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_dl_left_stim = left_stim_epochs_dl.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            # Distractor Right: Contra is ELECTRODE_LEFT_HEMISPHERE, Ipsi is ELECTRODE_RIGHT_HEMISPHERE
            contra_dl_right_stim = right_stim_epochs_dl.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_dl_right_stim = right_stim_epochs_dl.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data

            avg_contra_dl = np.mean([contra_dl_left_stim, contra_dl_right_stim], axis=0)
            avg_ipsi_dl = np.mean([ipsi_dl_left_stim, ipsi_dl_right_stim], axis=0)
            diff_wave_dl_sub = avg_contra_dl - avg_ipsi_dl
            subject_diff_waves['distractor_lateral'].append(diff_wave_dl_sub)
            subject_contributed_this_loop = True
            print(f"    Calculated Pd-like diff wave for {subject_str}.")
        else:
            print(f"    Skipping Pd-like for {subject_str} due to insufficient trials.")

        # --- 2. Target Lateral (Distractor Present; N2ac-like component) ---
        # Target-1-Singleton-2: Target Left, Distractor Central
        # Target-3-Singleton-2: Target Right, Distractor Central
        left_stim_epochs_tdp = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]]
        right_stim_epochs_tdp = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]]

        if len(left_stim_epochs_tdp) > 0 and len(right_stim_epochs_tdp) > 0:
            # Target Left: Contra is ELECTRODE_RIGHT_HEMISPHERE, Ipsi is ELECTRODE_LEFT_HEMISPHERE
            contra_tdp_left_stim = left_stim_epochs_tdp.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_tdp_left_stim = left_stim_epochs_tdp.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            # Target Right: Contra is ELECTRODE_LEFT_HEMISPHERE, Ipsi is ELECTRODE_RIGHT_HEMISPHERE
            contra_tdp_right_stim = right_stim_epochs_tdp.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_tdp_right_stim = right_stim_epochs_tdp.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data

            avg_contra_tdp = np.mean([contra_tdp_left_stim, contra_tdp_right_stim], axis=0)
            avg_ipsi_tdp = np.mean([ipsi_tdp_left_stim, ipsi_tdp_right_stim], axis=0)
            diff_wave_tdp_sub = avg_contra_tdp - avg_ipsi_tdp
            subject_diff_waves['target_distractor_present'].append(diff_wave_tdp_sub)
            subject_contributed_this_loop = True
            print(f"    Calculated N2ac-like (distractor present) diff wave for {subject_str}.")
        else:
            print(f"    Skipping N2ac-like (distractor present) for {subject_str} due to insufficient trials.")

        # --- 3. Target Lateral (Distractor Absent; N2ac-like component) ---
        # Target-1-Singleton-0: Target Left, No Distractor
        # Target-3-Singleton-0: Target Right, No Distractor
        left_stim_epochs_tda = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-0" in x]]
        right_stim_epochs_tda = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-0" in x]]

        if len(left_stim_epochs_tda) > 0 and len(right_stim_epochs_tda) > 0:
            # Target Left: Contra is ELECTRODE_RIGHT_HEMISPHERE, Ipsi is ELECTRODE_LEFT_HEMISPHERE
            contra_tda_left_stim = left_stim_epochs_tda.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_tda_left_stim = left_stim_epochs_tda.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            # Target Right: Contra is ELECTRODE_LEFT_HEMISPHERE, Ipsi is ELECTRODE_RIGHT_HEMISPHERE
            contra_tda_right_stim = right_stim_epochs_tda.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_tda_right_stim = right_stim_epochs_tda.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data

            avg_contra_tda = np.mean([contra_tda_left_stim, contra_tda_right_stim], axis=0)
            avg_ipsi_tda = np.mean([ipsi_tda_left_stim, ipsi_tda_right_stim], axis=0)
            diff_wave_tda_sub = avg_contra_tda - avg_ipsi_tda
            subject_diff_waves['target_distractor_absent'].append(diff_wave_tda_sub)
            subject_contributed_this_loop = True
            print(f"    Calculated N2ac-like (distractor absent) diff wave for {subject_str}.")
        else:
            print(f"    Skipping N2ac-like (distractor absent) for {subject_str} due to insufficient trials.")

        if subject_contributed_this_loop:
            processed_subject_count += 1

    except Exception as e:
        print(f"  Error processing subject {subject_str}: {e}. Skipping this subject.")
        # Note: If an error occurs mid-subject, they might have partial contributions.
        # A more robust way would be to collect data in a temp dict for the subject
        # and only append to global lists if the subject processing completes fully.
        continue

print(f"\n--- Finished subject-level processing. Processed data for {processed_subject_count} subjects who contributed to at least one condition. ---")

if processed_subject_count == 0:
    print("No subjects were processed successfully. Exiting.")
    exit()
if times_vector is None:
    print("Times vector could not be determined. Exiting.")
    exit()

# --- Grand Average Calculation ---
print("\nCalculating Grand Averages...")
ga_diff_waves = {}
n_subjects_per_condition = {}

for key, wave_list in subject_diff_waves.items():
    if wave_list: # Check if list is not empty (i.e., at least one subject contributed)
        # Squeeze to remove singleton dimension if present, e.g. (N, 1, times) -> (N, times)
        stacked_waves = np.array(wave_list).squeeze()
        if stacked_waves.ndim == 1: # Handle case where only one subject contributed to this condition
            stacked_waves = stacked_waves[np.newaxis, :]

        if stacked_waves.shape[0] > 0:
            ga_diff_waves[key] = np.mean(stacked_waves, axis=0)
            n_subjects_per_condition[key] = stacked_waves.shape[0]
            print(f"  GA for '{PLOT_LEGENDS.get(key, key)}': {n_subjects_per_condition[key]} subjects.")
        else: # Should not be reached if wave_list was non-empty and stacking worked
            ga_diff_waves[key] = np.full_like(times_vector, np.nan)
            n_subjects_per_condition[key] = 0
            print(f"  No data for GA for '{PLOT_LEGENDS.get(key, key)}' (after stacking).")
    else:
        ga_diff_waves[key] = np.full_like(times_vector, np.nan) # Fill with NaNs if no subjects
        n_subjects_per_condition[key] = 0
        print(f"  No data for GA for '{PLOT_LEGENDS.get(key, key)}'.")


# --- Plotting Grand Averages ---
print("\nPlotting Grand Averages...")
plt.figure(figsize=(10, 6))

for key in subject_diff_waves.keys(): # Iterate in the order they were defined
    if n_subjects_per_condition.get(key, 0) > 0: # Check if there's data to plot
        # The GA is already 1D (times,) due to np.mean(axis=0)
        # and the squeeze/newaxis logic for single subject.
        # So, no [0] indexing needed for ga_diff_waves[key]
        plt.plot(times_vector,
                 savgol_filter(ga_diff_waves[key] * AMPLITUDE_SCALE_FACTOR,
                               window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER),
                 color=PLOT_COLORS.get(key, 'black'), # Default to black if key not in PLOT_COLORS
                 label=f"{PLOT_LEGENDS.get(key, key)} (N={n_subjects_per_condition[key]})")

plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=0, color='k', linestyle=':', linewidth=0.8) # Assuming 0 is stimulus onset
plt.legend(loc='best')
plt.title(f"Grand Average Difference Waves (Contra - Ipsi) at {ELECTRODE_LEFT_HEMISPHERE}/{ELECTRODE_RIGHT_HEMISPHERE}")
plt.ylabel("Amplitude [µV]")
plt.xlabel("Time [s]")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show(block=True)

print("\nDone.")