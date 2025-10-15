import mne
import numpy as np
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns


def _get_subject_lateralized_data(epochs, condition_keyword, ch_left="C3", ch_right="C4"):
    """
    Calculate subject-level lateralized ERP data for a given condition.

    This helper function takes a subject's epochs, isolates trials based on a
    condition keyword (e.g., 'distractor') and stimulus location (left/right),
    and computes the average contralateral and ipsilateral waveforms.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object for a single subject.
    condition_keyword : str
        The keyword to identify the condition in event IDs (e.g., 'distractor', 'target').
    ch_left : str, optional
        The name of the left-hemisphere channel, by default "C3".
    ch_right : str, optional
        The name of the right-hemisphere channel, by default "C4".

    Returns
    -------
    tuple of np.ndarray or (None, None)
        A tuple containing (contra_data, ipsi_data).
        Returns (None, None) if epochs for the condition are not found.
    """
    all_conds = list(epochs.event_id.keys())

    # Separate epochs based on stimulus location for the given condition
    left_epochs = epochs[[c for c in all_conds if f"{condition_keyword}-location-1" in c]]
    right_epochs = epochs[[c for c in all_conds if f"{condition_keyword}-location-3" in c]]

    # Cannot compute lateralized response if one side is missing
    if len(left_epochs) == 0 or len(right_epochs) == 0:
        print(f"Warning: Subject missing '{condition_keyword}' epochs for left or right location. "
              f"Skipping lateralized calculation for this condition.")
        return None, None

    # Calculate contralateral response (response in hemisphere opposite to stimulus)
    # Average of (response to left stim at right channel) and (response to right stim at left channel)
    contra_data = np.mean([
        left_epochs.copy().average(picks=ch_right).get_data(),
        right_epochs.copy().average(picks=ch_left).get_data()
    ], axis=0)

    # Calculate ipsilateral response (response in same hemisphere as stimulus)
    # Average of (response to left stim at left channel) and (response to right stim at right channel)
    ipsi_data = np.mean([
        left_epochs.copy().average(picks=ch_left).get_data(),
        right_epochs.copy().average(picks=ch_right).get_data()
    ], axis=0)

    return contra_data, ipsi_data


def get_passive_listening_ERPs_grand_average():
    """
    Load passive listening data and compute grand-average ERPs.

    This function iterates through each subject, calculates their individual
    contralateral and ipsilateral ERPs for singleton, target, and control
    conditions, and then computes the grand average across all subjects.

    Returns
    -------
    dict
        A dictionary containing the results, with keys:
        'subject_data': Data for each individual subject.
        'grand_average': Grand-averaged data across subjects.
        'difference_waves': Grand-averaged contralateral-minus-ipsilateral difference waves.
        'times': The time points for the ERP data.
    """
    # Dictionary to store subject-level averaged data
    subject_data = {
        'contra_singleton': [], 'ipsi_singleton': [],
        'contra_target': [], 'ipsi_target': [],
        'contra_control': [], 'ipsi_control': []
    }

    times = None  # To store the time array from the first loaded epoch

    for subject in subject_ids:
        # Load one subject's epochs
        try:
            fname = \
            glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-passive-epo.fif")[0]
            epochs = mne.read_epochs(fname, preload=True, verbose=False)
            if times is None:
                times = epochs.times

        except IndexError:
            print(f"Warning: Could not find epochs file for subject {subject}. Skipping.")
            continue

        # --- Process each condition for the current subject ---
        # Singleton
        contra_s, ipsi_s = _get_subject_lateralized_data(epochs, "distractor", ch_left="C3", ch_right="C4")
        if contra_s is not None:
            subject_data['contra_singleton'].append(contra_s)
            subject_data['ipsi_singleton'].append(ipsi_s)

        # Target
        contra_t, ipsi_t = _get_subject_lateralized_data(epochs, "target", ch_left="C3", ch_right="C4")
        if contra_t is not None:
            subject_data['contra_target'].append(contra_t)
            subject_data['ipsi_target'].append(ipsi_t)

        # Control (distractor that is not a singleton)
        contra_c, ipsi_c = _get_subject_lateralized_data(epochs, "control", ch_left="C3", ch_right="C4")
        if contra_c is not None:
            subject_data['contra_control'].append(contra_c)
            subject_data['ipsi_control'].append(ipsi_c)

    # --- Compute Grand Averages ---
    grand_average = {}
    for key, data_list in subject_data.items():
        if data_list:
            # Average across the first dimension (subjects)
            grand_average[key] = np.mean(np.array(data_list), axis=0)
        else:
            grand_average[key] = None  # In case no subjects had data for a condition

    # --- Compute Difference Waves ---
    difference_waves = {
        'singleton': grand_average['contra_singleton'] - grand_average['ipsi_singleton'],
        'target': grand_average['contra_target'] - grand_average['ipsi_target'],
        'control': grand_average['contra_control'] - grand_average['ipsi_control']
    }

    # --- Package and return results ---
    results = {
        'subject_data': subject_data,
        'grand_average': grand_average,
        'difference_waves': difference_waves,
        'times': times
    }

    return results


def calculate_erp_metrics(erp_wave, times, percentage=0.5, plot=False, is_negative_component=False, analysis_window_times=None):
    """
    Calculates key ERP metrics like latency and amplitude from a waveform.

    This function computes:
    1.  **Fractional Area Latency**: The time point where the cumulative area under a
        scaled version of the component's morphology reaches a given percentage. This is
        robust to noise.
    2.  **Amplitude at Latency**: The voltage of the original wave at the calculated
        fractional area latency point.
    3.  **Mean Amplitude**: The average voltage of the wave within the analysis window.

    Args:
        erp_wave (np.ndarray): The ERP waveform.
        times (np.ndarray): The time points for the erp_wave.
        percentage (float, optional): Percentage for fractional area latency. Defaults to 0.5.
        plot (bool, optional): If True, displays a diagnostic plot. Defaults to False.
        is_negative_component (bool): If True, the wave is assumed to be negative-going (e.g., N2ac)
                                    and is inverted for latency calculation. Defaults to False.
        analysis_window_times (tuple, optional): (start, end) in seconds for the analysis.
                                                 If None, the entire waveform is used. Defaults to None.

    Returns:
        dict: A dictionary containing 'latency', 'amplitude_at_latency', and 'mean_amplitude',
              or NaNs if calculation is not possible.
    """
    # --- Initial Checks ---
    default_return = {'latency': np.nan, 'amplitude_at_latency': np.nan, 'mean_amplitude': np.nan}
    if erp_wave is None or times is None or len(erp_wave) == 0:
        return default_return

    original_erp_wave_for_plot = erp_wave.copy()
    original_times_for_plot = times.copy()

    # --- 1. Apply Analysis Time Window ---
    if analysis_window_times is not None:
        start_time, end_time = analysis_window_times
        time_mask = (times >= start_time) & (times <= end_time)
        erp_wave_windowed = erp_wave[time_mask]
        times_windowed = times[time_mask]

        if len(times_windowed) == 0:
            if plot: print("Plotting skipped: Analysis window is empty.")
            return default_return
    else:
        erp_wave_windowed = erp_wave
        times_windowed = times

    # --- 2. Calculate Mean Amplitude (on the windowed, original wave) ---
    mean_amplitude = np.mean(erp_wave_windowed)

    # --- 3. Prepare wave for Latency Calculation ---
    analysis_wave = erp_wave_windowed * -1 if is_negative_component else erp_wave_windowed

    # --- 4. Min-Max Scale for Latency Calculation ---
    min_val, max_val = np.min(analysis_wave), np.max(analysis_wave)
    if max_val <= min_val:
        if plot: print("Plotting skipped: ERP wave is flat.")
        # We can still return mean amplitude even if latency fails
        return {'latency': np.nan, 'amplitude_at_latency': np.nan, 'mean_amplitude': mean_amplitude}

    scaled_wave = (analysis_wave - min_val) / (max_val - min_val)

    # --- 5. Calculate Fractional Area Latency ---
    cum_sum = np.cumsum(scaled_wave)
    total_area = np.max(cum_sum)
    latency = np.nan
    amplitude_at_latency = np.nan

    if total_area > 0:
        target_area = total_area * percentage
        crossings = np.where(cum_sum >= target_area)[0]

        if len(crossings) > 0:
            first_crossing_idx = crossings[0]
            if first_crossing_idx == 0:
                latency = times_windowed[first_crossing_idx]
            else:
                idx_before, idx_after = first_crossing_idx - 1, first_crossing_idx
                t_before, t_after = times_windowed[idx_before], times_windowed[idx_after]
                val_before, val_after = cum_sum[idx_before], cum_sum[idx_after]
                latency = np.interp(target_area, [val_before, val_after], [t_before, t_after])

            # --- 6. Calculate Amplitude at Latency (on original windowed wave) ---
            if not np.isnan(latency):
                amplitude_at_latency = np.interp(latency, times_windowed, erp_wave_windowed)

    # --- 7. Plotting Logic ---
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        erp_type = 'Negative Component (e.g., N2ac)' if is_negative_component else 'Positive Component (e.g., Pd)'
        title = f"ERP Metrics ({percentage*100:.0f}%) on {erp_type}"
        if analysis_window_times:
            title += f"\nAnalysis Window: {analysis_window_times[0]}-{analysis_window_times[1]}s"
        fig.suptitle(title, fontsize=16)

        # Plot 1: Original Waveform
        ax1.set_title("Original Waveform & Analysis Window")
        ax1.plot(original_times_for_plot, original_erp_wave_for_plot, color='navy', label=f'Mean Amp: {mean_amplitude:.2f} µV')
        ax1.set_xlabel("Time (s)"), ax1.set_ylabel("Amplitude (µV)")
        ax1.grid(True, linestyle=':', alpha=0.6), ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        if analysis_window_times:
            ax1.axvspan(analysis_window_times[0], analysis_window_times[1], color='grey', alpha=0.25, label='Analysis Window')
        ax1.legend(loc='best')

        # Plot 2: Latency Calculation
        ax2.set_title("Cumulative Area & Latency (in Window)")
        ax2.plot(times_windowed, cum_sum, color='teal', label='Cumulative Area of Scaled Wave')
        ax2.axhline(target_area, color='red', linestyle='--', label=f'{percentage*100:.0f}% Area Threshold')
        ax2.set_xlabel("Time (s)"), ax2.set_ylabel("Cumulative Area (a.u.)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        if not np.isnan(latency):
            ax2.axvline(latency, color='purple', linestyle='-', lw=2, label=f'Latency: {latency:.3f}s')
            ax2.plot(latency, target_area, 'ro', markersize=8)
            # Also show amplitude at latency on the first plot
            ax1.plot(latency, amplitude_at_latency, 'p', color='purple', markersize=10, label=f'Amp at Latency: {amplitude_at_latency:.2f} µV')
            ax1.legend()
        ax2.legend(loc='best')
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]), plt.show(block=True)

    return {
        'latency': latency,
        'amplitude_at_latency': amplitude_at_latency,
        'mean_amplitude': mean_amplitude
    }


def get_contra_ipsi_diff_wave(trials_df, electrode_pairs, time_window, all_times, lateral_stim_col):
    """
    Calculates the grand-average contralateral-ipsilateral difference wave from a given set of trials.

    This robust version calculates the difference wave for each trial first, then averages
    all trial-level difference waves together. This is the "average of differences" method.

    Args:
        trials_df (pd.DataFrame): DataFrame containing ERP data for a specific condition.
        electrode_pairs (list): List of (left_hemi_el, right_hemi_el) tuples.
        time_window (tuple): The (start, end) time in seconds.
        all_times (np.ndarray): Array of all time points in the epoch.
        lateral_stim_col (str): Column name indicating stimulus location ('TargetLoc' or 'SingletonLoc').

    Returns:
        tuple: (The difference wave, The corresponding time points) or (None, None) if no data.
    """
    if trials_df.empty:
        return None, None

    time_mask = (all_times >= time_window[0]) & (all_times <= time_window[1])
    window_times = all_times[time_mask]

    # If the time window is empty, we can't proceed.
    if not np.any(time_mask):
        return None, None

    all_trial_diff_waves = []

    # Separate trials by stimulus location
    left_stim_trials = trials_df[trials_df[lateral_stim_col] == 'left']
    right_stim_trials = trials_df[trials_df[lateral_stim_col] == 'right']

    # This outer loop pools the results across multiple electrode pairs.
    for left_el, right_el in electrode_pairs:
        # --- Process Left Stimulus Trials ---
        if not left_stim_trials.empty:
            # Contralateral is Right Hemisphere (right_el), Ipsilateral is Left Hemisphere (left_el)
            contra_waves = left_stim_trials[right_el].loc[:, time_mask].values
            ipsi_waves = left_stim_trials[left_el].loc[:, time_mask].values
            # Calculate difference for each trial in this block
            # --- FIX: Explicitly cast to float to avoid dtype issues ---
            diffs = contra_waves.astype(np.float64) - ipsi_waves.astype(np.float64)
            all_trial_diff_waves.append(diffs)

        # --- Process Right Stimulus Trials ---
        if not right_stim_trials.empty:
            # Contralateral is Left Hemisphere (left_el), Ipsilateral is Right Hemisphere (right_el)
            contra_waves = right_stim_trials[left_el].loc[:, time_mask].values
            ipsi_waves = right_stim_trials[right_el].loc[:, time_mask].values
            # Calculate difference for each trial in this block
            # --- FIX: Explicitly cast to float to avoid dtype issues ---
            diffs = contra_waves.astype(np.float64) - ipsi_waves.astype(np.float64)
            all_trial_diff_waves.append(diffs)

    if not all_trial_diff_waves:
        return None, None

    # Concatenate all the difference waves from all conditions and pairs into one big array.
    # The shape will be (total_trials * num_pairs, n_timepoints).
    final_diffs_array = np.concatenate(all_trial_diff_waves, axis=0)

    # Average across all the collected trial-level difference waves.
    mean_diff_wave = np.mean(final_diffs_array, axis=0)

    return mean_diff_wave, window_times


def get_single_trial_contra_ipsi_wave(trial_row, electrode_pairs, time_window, all_times, lateral_stim_loc):
    """
    Calculates the contralateral-ipsilateral difference wave for a single trial.

    Args:
        trial_row (pd.Series): A single row from the merged dataframe, containing ERP data
                               across time for all channels as a MultiIndex.
        electrode_pairs (list): List of tuples, where each tuple is a (left_hemi_el, right_hemi_el) pair.
        time_window (tuple): The (start, end) time in seconds for the analysis.
        all_times (np.ndarray): Array of all time points in the epoch.
        lateral_stim_loc (str): Location of the lateralized stimulus ('left' or 'right').

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final averaged contra-ipsi difference wave for the single trial.
            - np.ndarray: The time points corresponding to the wave.
            Returns (None, None) if data is invalid.
    """
    # Create a boolean mask for the desired time window
    time_mask = (all_times >= time_window[0]) & (all_times <= time_window[1])
    window_times = all_times[time_mask]

    if len(window_times) == 0:
        return None, None

    diff_waves_for_pairs = []
    for left_el, right_el in electrode_pairs:
        try:
            # For a single trial_row (a Series), the index is a MultiIndex of (channel, time).
            # We select the channel, which returns a new Series indexed by time.
            # Then we apply the time mask to get the waveform for the window.
            left_el_wave = trial_row[left_el][time_mask].values
            right_el_wave = trial_row[right_el][time_mask].values

            # Calculate contralateral - ipsilateral difference
            # --- FIX: Explicitly cast to float to avoid dtype issues ---
            if lateral_stim_loc == 'left':
                # Contralateral is Right Hemisphere, Ipsilateral is Left Hemisphere
                diff_wave = right_el_wave.astype(np.float64) - left_el_wave.astype(np.float64)
            else:  # lateral_stim_loc == 'right'
                # Contralateral is Left Hemisphere, Ipsilateral is Right Hemisphere
                diff_wave = left_el_wave.astype(np.float64) - right_el_wave.astype(np.float64)

            diff_waves_for_pairs.append(diff_wave)

        except KeyError as e:
            # This can happen if a channel was marked as bad for a specific subject
            # print(f"Warning: Could not find channel {e} for a trial. Skipping this electrode pair for the trial.")
            continue

    if not diff_waves_for_pairs:
        # This happens if all electrode pairs failed (e.g., all channels were bad)
        return None, None

    # Average the difference waves across all specified electrode pairs for this single trial
    mean_diff_wave = np.mean(diff_waves_for_pairs, axis=0)

    return mean_diff_wave, window_times


# Define some functions
def degrees_va_to_pixels(degrees, screen_pixels, screen_size_cm, viewing_distance_cm):
    """
    Converts degrees visual angle to pixels.

    Args:
        degrees: The visual angle in degrees.
        screen_pixels: The number of pixels on the screen (horizontal or vertical).
        screen_size_cm: The physical size of the screen in centimeters (width or height).
        viewing_distance_cm: The viewing distance in centimeters.

    Returns:
        The number of pixels corresponding to the given visual angle.
    """

    pixels = degrees * (screen_pixels / screen_size_cm) * (viewing_distance_cm * np.tan(np.radians(1)))
    return pixels


def calculate_trial_path_length(trial_group):
    """
    Calculates the total Euclidean path length for a single trial's trajectory.
    Assumes trial_group is a DataFrame with 'x_pixels' and 'y_pixels' columns
    sorted chronologically.
    """
    if len(trial_group) < 2:
        return 0.0  # Path length is 0 if less than 2 points

    # Get x and y coordinates for the current trial
    x_coords = trial_group['x_pixels'].to_numpy()
    y_coords = trial_group['y_pixels'].to_numpy()

    # Calculate the differences between consecutive points (dx, dy)
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    # Calculate the Euclidean distance for each segment: sqrt(dx^2 + dy^2)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    # Total path length is the sum of segment lengths
    total_path_length = np.sum(segment_lengths)
    return total_path_length


def angle_between_vectors(v1, v2):
    """Calculates the absolute angle in degrees between two 2D vectors."""
    # Ensure vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Handle zero vectors to avoid division by zero
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 180.0  # Max angle if one vector is zero

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)

    # Clip the value to handle potential floating point inaccuracies
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def classify_initial_movement(row, locations_map, start_vec):
    """
    Classifies the initial movement direction for a single trial (row).
    Categories are 'target', 'distractor', 'control', and 'other'.
    """
    # Define the initial movement vector
    initial_point = np.array([row['initial_x_dva'], row['initial_y_dva']])
    movement_vector = initial_point - start_vec

    try:
        roi_locations = {
            'target': row['TargetDigit'],
            'distractor': row['SingletonDigit'],
            'control': row['Non-Singleton2Digit']
        }
    except KeyError:
        print("Warning: Columns for ROI locations (e.g., 'TargetLoc') not found. Skipping classification.")
        return None

    # --- NEW: Identify "other" locations ---
    # Get all possible response locations, excluding the center (5)
    all_locations = set(loc for loc in locations_map if loc != 5)
    # Get the locations used by the main ROIs in this trial
    used_locations = {int(loc) for loc in roi_locations.values() if pd.notna(loc) and loc != 5}
    # "other" locations are the ones that are not used
    other_locations = all_locations - used_locations

    angles = {}
    # Calculate angles for the primary ROIs
    for roi_name, roi_digit in roi_locations.items():
        if pd.isna(roi_digit):
            continue  # Skip if the location is not defined

        roi_coord = np.array(locations_map.get(int(roi_digit)))
        roi_vector = roi_coord - start_vec
        angles[roi_name] = angle_between_vectors(movement_vector, roi_vector)

    # --- NEW: Calculate the angle for the "other" category ---
    # This is the *smallest* angle to any of the available "other" locations
    if other_locations:
        other_angles = []
        for other_digit in other_locations:
            other_coord = np.array(locations_map.get(other_digit))
            other_vector = other_coord - start_vec
            other_angles.append(angle_between_vectors(movement_vector, other_vector))

        if other_angles:
            angles['other'] = min(other_angles)

    # Find the ROI with the overall minimum angle
    if not angles:
        return None

    closest_roi = min(angles, key=angles.get)
    return closest_roi


def plot_trial_vectors(trial_row, locations_map):
    """
    Generates a detailed plot visualizing the initial movement vector, ROI vectors,
    and classification for a single trial, including "other" locations.

    Args:
        trial_row (pd.Series): A single row from the analysis_df DataFrame.
        locations_map (dict): The dictionary mapping numpad digits to their
                              (x, y) coordinates in degrees of visual angle.
    """
    # --- 1. Extract Data and Define Vectors ---
    start_point_vec = np.array([0, 0])
    movement_vector = np.array([trial_row['initial_x_dva'], trial_row['initial_y_dva']])

    # Get primary ROI locations for this specific trial
    roi_info = {
        'Target': trial_row.get('TargetDigit'),
        'Distractor': trial_row.get('SingletonDigit'),
        'Control': trial_row.get('Non-Singleton2Digit')
    }

    roi_vectors = {}
    for name, digit in roi_info.items():
        if pd.notna(digit):
            roi_vectors[name] = np.array(locations_map.get(int(digit))) - start_point_vec

    # --- NEW: Identify and process "other" locations ---
    all_locations = set(locations_map.keys()) - {5}  # Exclude center
    used_locations = {int(digit) for digit in roi_info.values() if pd.notna(digit)}
    other_locations = all_locations - used_locations

    other_vectors = {digit: np.array(locations_map.get(digit)) - start_point_vec for digit in other_locations}

    # --- 2. Calculate Angles ---
    angles = {name: angle_between_vectors(movement_vector, vec) for name, vec in roi_vectors.items()}
    other_angles = {digit: angle_between_vectors(movement_vector, vec) for digit, vec in other_vectors.items()}

    # --- 3. Create Plot ---
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot the numpad background
    for digit, (x, y) in locations_map.items():
        ax.plot(x, y, 'o', color='lightgray', markersize=30, zorder=1)
        ax.text(x, y, str(digit), color='black', ha='center', va='center', fontweight='bold', fontsize=12)

    # Plot the actual initial movement vector
    ax.arrow(0, 0, movement_vector[0], movement_vector[1], head_width=0.05, head_length=0.1,
             fc='blue', ec='blue', length_includes_head=True, zorder=10, lw=2)
    ax.text(movement_vector[0] * 1.3, movement_vector[1] * 1.3, 'Initial\nMovement',
             color='blue', ha='center', va='center', fontweight='bold')

    # Plot the ideal primary ROI vectors
    colors = {'Target': 'green', 'Distractor': 'red', 'Control': 'dimgray'}
    for name, vec in roi_vectors.items():
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.1,
                 fc=colors[name], ec=colors[name], linestyle='--', length_includes_head=True, zorder=5, lw=1.5)
        angle_text = f'{name}\nAngle: {angles.get(name, 0):.1f}°'
        ax.text(vec[0] * 1.3, vec[1] * 1.3, angle_text, color=colors[name], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # --- NEW: Plot the "other" ROI vectors ---
    min_other_angle = min(other_angles.values()) if other_angles else float('inf')
    for digit, vec in other_vectors.items():
        angle = other_angles[digit]
        # Highlight the closest "other" vector if the trial was classified as such
        is_closest_other = (trial_row['initial_movement_direction'] == 'other' and np.isclose(angle, min_other_angle))

        color = 'purple' if is_closest_other else '#AAAAAA'
        linestyle = '--'
        zorder = 6 if is_closest_other else 4
        linewidth = 2.0 if is_closest_other else 1.0

        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.1,
                 fc=color, ec=color, linestyle=linestyle, length_includes_head=True, zorder=zorder, lw=linewidth)

        angle_text = f'Angle: {angle:.1f}°'
        # Only show text for the winning "other" to avoid clutter
        if is_closest_other:
            ax.text(vec[0] * 1.3, vec[1] * 1.3, angle_text, color=color, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # --- 4. Final Formatting ---
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Trial Classification Breakdown (sub-{int(trial_row['subject_id'])})", fontsize=16)
    ax.set_xlabel('X coordinate (degrees visual angle)')
    ax.set_ylabel('Y coordinate (degrees visual angle)')

    # Add a conclusion text at the bottom
    conclusion = f"Conclusion: Movement classified as '{trial_row['initial_movement_direction']}'"
    fig.text(0.5, 0.02, conclusion, ha='center', fontsize=12,
             bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.show()


def visualize_full_trajectory(trial_row, raw_df, movement_threshold, target_hz=60):
    """
    Plots a resampled trajectory in absolute coordinates for a single trial,
    highlighting the start of movement and the time of response.

    Args:
        trial_row (pd.Series): A single row from the analysis_df, containing
                               subject_id, block, trial_nr, and rt.
        raw_df (pd.DataFrame): The raw mouse-tracking DataFrame ('df' in your script).
        movement_threshold (float): The movement threshold in dva, measured from the center.
        target_hz (int): The target sampling rate for resampling.
    """
    # --- 1. Get Trial Info and Data ---
    sub = trial_row['subject_id']
    block = trial_row['block']
    trial = trial_row['trial_nr']
    rt = trial_row['rt']  # Reaction time in seconds

    # Isolate the raw trajectory data for this specific trial
    trial_trajectory_df = raw_df[
        (raw_df['subject_id'].astype(int) == int(sub)) &
        (raw_df['block'].astype(int) == int(block)) &
        (raw_df['trial_nr'].astype(int) == int(trial))
    ].copy()

    if trial_trajectory_df.empty:
        print(f"Warning: No raw trajectory data found for sub-{sub}, block-{block}, trial-{trial}.")
        return

    # --- 2. Resample Trajectory ---
    resampled_df = resample_trial_trajectory(trial_trajectory_df, target_hz=target_hz)

    # --- 3. Identify Key Points in Absolute Coordinates ---
    start_point = resampled_df.iloc[0]
    start_x, start_y = start_point['x'], start_point['y']

    # To correctly identify the first movement, calculate displacement from the actual start point.
    # This is robust even if the start point deviates slightly from (0, 0).
    displacement_x = resampled_df['x'] - start_x
    displacement_y = resampled_df['y'] - start_y

    # Find the first point that crossed the movement threshold
    moved_points = resampled_df[
        (displacement_x.abs() > movement_threshold) |
        (displacement_y.abs() > movement_threshold)
    ]
    first_move_point = moved_points.iloc[0] if not moved_points.empty else None

    # Find the point corresponding to the response time (rt)
    response_time_point_idx = (resampled_df['time'] - rt).abs().idxmin()
    response_point = resampled_df.loc[response_time_point_idx]

    # --- 4. Create the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the full absolute trajectory
    ax.plot(resampled_df['x'], resampled_df['y'], '.-', color='lightblue', label='Full Trajectory', zorder=1, markersize=3)

    # Plot the actual starting point
    ax.plot(start_point['x'], start_point['y'], 'o', color='black', markersize=10, label=f"Actual Start ({start_point['x']:.2f}, {start_point['y']:.2f})", zorder=3)

    # Plot the first point detected as movement
    if first_move_point is not None:
        ax.plot(first_move_point['x'], first_move_point['y'], '+', color='red', markersize=15, label='First Movement Detected', zorder=4)

    # Plot the response time point
    ax.plot(response_point['x'], response_point['y'], 'X', color='green', markersize=12, mew=2.5, label=f'Response Time ({rt:.2f}s)', zorder=5)

    # Draw the threshold area around the intended start (0,0)
    threshold_square = patches.Rectangle(
        (-movement_threshold, -movement_threshold),
        2 * movement_threshold, 2 * movement_threshold,
        linewidth=1.5, edgecolor='r', facecolor='red', alpha=0.1,
        linestyle='--', label='Movement Threshold Area'
    )
    ax.add_patch(threshold_square)

    # --- 5. Formatting ---
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(f'Absolute Trajectory for sub-{int(sub)}, block-{int(block)}, trial-{int(trial)}')
    ax.set_xlabel('X coordinate (dva, absolute)')
    ax.set_ylabel('Y coordinate (dva, absolute)')

    lim = 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend()
    plt.show()

def resample_trial_trajectory(trial_df, target_hz=60):
    """
    Resamples a single trial's trajectory to a constant sampling rate using linear interpolation.

    Args:
        trial_df (pd.DataFrame): DataFrame for a single trial, containing 'time', 'x', and 'y'.
        target_hz (int): The target sampling rate in Hz (e.g., 100 Hz).

    Returns:
        pd.DataFrame: A new DataFrame with the resampled trajectory.
    """
    # Ensure data is sorted by time, which is crucial for interpolation
    trial_df = trial_df.sort_values('time').reset_index(drop=True)

    original_time = trial_df['time'].values
    original_x = trial_df['x'].values
    original_y = trial_df['y'].values

    # If there are fewer than two data points, we cannot interpolate
    if len(original_time) < 2:
        return trial_df

    # Create the new, evenly spaced time vector for interpolation
    start_time = original_time[0]
    end_time = original_time[-1]
    step = 1.0 / target_hz
    new_time = np.arange(start_time, end_time, step)

    # Interpolate x and y coordinates onto the new time vector
    new_x = np.interp(new_time, original_time, original_x)
    new_y = np.interp(new_time, original_time, original_y)

    # Create the new resampled DataFrame
    resampled_df = pd.DataFrame({
        'time': new_time,
        'x': new_x,
        'y': new_y
    })

    return resampled_df


def visualize_absolute_trajectory(trial_row, raw_df, locations_map, target_hz=60):
    """
    Plots a resampled trajectory in its original, absolute coordinate space
    to diagnose start and end points.

    Args:
        trial_row (pd.Series): A single row from the analysis_df.
        raw_df (pd.DataFrame): The raw mouse-tracking DataFrame ('df' in your script).
        locations_map (dict): The dictionary mapping numpad digits to coordinates.
        target_hz (int): The target sampling rate for resampling.
    """
    # --- 1. Get Trial Info and Data ---
    sub = trial_row['subject_id']
    block = trial_row['block']
    trial = trial_row['trial_nr']
    rt = trial_row['rt']

    trial_trajectory_df = raw_df[
        (raw_df['subject_id'].astype(int) == int(sub)) &
        (raw_df['block'].astype(int) == int(block)) &
        (raw_df['trial_nr'].astype(int) == int(trial))
    ].copy()

    if trial_trajectory_df.empty:
        print(f"Warning: No raw trajectory data found for sub-{sub}, block-{block}, trial-{trial}.")
        return

    resampled_df = resample_trial_trajectory(trial_trajectory_df, target_hz=target_hz)

    # --- 2. Identify Key Points (No Re-centering) ---
    start_point = resampled_df.iloc[0]
    response_time_point_idx = (resampled_df['time'] - rt).abs().idxmin()
    response_point = resampled_df.loc[response_time_point_idx]

    # --- 3. Create the Plot ---
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot the numpad background for context
    for digit, (x, y) in locations_map.items():
        ax.plot(x, y, 'o', color='lightgray', markersize=40, zorder=1, alpha=0.8)
        ax.text(x, y, str(digit), color='black', ha='center', va='center', fontweight='bold')

    # Plot the full absolute trajectory
    ax.plot(resampled_df['x'], resampled_df['y'], '.-', color='cornflowerblue', label='Absolute Trajectory', zorder=2)

    # Highlight the actual start and end points
    ax.plot(start_point['x'], start_point['y'], 'o', color='red', markersize=12, label=f"Actual Start ({start_point['x']:.2f}, {start_point['y']:.2f})", zorder=4)
    ax.plot(response_point['x'], response_point['y'], 'X', color='green', markersize=12, mew=2.5, label=f"Actual End ({response_point['x']:.2f}, {response_point['y']:.2f})", zorder=5)

    # --- 4. Formatting ---
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(f'Absolute Trajectory for sub-{int(sub)}, block-{int(block)}, trial-{int(trial)}')
    ax.set_xlabel('X coordinate (dva, absolute)')
    ax.set_ylabel('Y coordinate (dva, absolute)')

    lim = 2.0  # Use a larger limit to see the full space
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc='upper right')
    plt.show()


# --- Helper Function for Wave Extraction ---
def get_all_waves(trials_df, electrode_pairs, time_window, all_times, lateral_stim_col):
    """
    Calculates the grand-average contralateral, ipsilateral, and difference waves
    from a given set of trials for a single subject/condition.
    """
    if trials_df.empty:
        return None, None, None, None

    time_mask = (all_times >= time_window[0]) & (all_times <= time_window[1])
    window_times = all_times[time_mask]

    all_contra_activity = []
    all_ipsi_activity = []

    left_stim_trials = trials_df[trials_df[lateral_stim_col] == 'left']
    right_stim_trials = trials_df[trials_df[lateral_stim_col] == 'right']

    for left_el, right_el in electrode_pairs:
        if not left_stim_trials.empty:
            all_contra_activity.append(left_stim_trials[right_el].loc[:, time_mask].values)
            all_ipsi_activity.append(left_stim_trials[left_el].loc[:, time_mask].values)
        if not right_stim_trials.empty:
            all_contra_activity.append(right_stim_trials[left_el].loc[:, time_mask].values)
            all_ipsi_activity.append(right_stim_trials[right_el].loc[:, time_mask].values)

    if not all_contra_activity:
        return None, None, None, None

    mean_contra_wave = np.mean(np.concatenate(all_contra_activity, axis=0), axis=0)
    mean_ipsi_wave = np.mean(np.concatenate(all_ipsi_activity, axis=0), axis=0)
    diff_wave = mean_contra_wave - mean_ipsi_wave

    return diff_wave, mean_contra_wave, mean_ipsi_wave, window_times


def calculate_trajectory_projections(row, locations_map, verbose=False):
    """
    Calculates a comprehensive set of trajectory projection scores for a single trial.
    """
    # --- 1. Get the average movement vector ---
    try:
        avg_vec = np.array([row['avg_x_dva'], row['avg_y_dva']])
    except KeyError:
        return pd.Series(
            {'proj_target': np.nan, 'proj_distractor': np.nan, 'proj_control_max': np.nan, 'proj_control_avg': np.nan})

    if np.linalg.norm(avg_vec) < 1e-6:
        return pd.Series(
            {'proj_target': np.nan, 'proj_distractor': np.nan, 'proj_control_max': np.nan, 'proj_control_avg': np.nan})

    if verbose:
        print("\n--- Calculating Trajectory Projections ---")
        print(
            f"Trial Info: sub-{row.get('subject_id', 'N/A')}, block-{row.get('block', 'N/A')}, trial-{row.get('trial_nr', 'N/A')}")
        print(f"Avg Movement Vector: ({avg_vec[0]:.3f}, {avg_vec[1]:.3f})")

    # --- 2. Helper function ---
    def get_projection(digit, avg_vec, locations_map):
        if pd.isna(digit): return np.nan
        dir_vec = np.array(locations_map.get(int(digit), (0, 0)))
        return np.dot(avg_vec, dir_vec)

    # --- 3. Identify all digits ---
    target_digit = row.get('TargetDigit')
    distractor_digit = row.get('SingletonDigit') if row.get('SingletonPresent') == 1 else np.nan
    all_nontarget_cols = ['SingletonDigit', 'Non-Singleton1Digit', 'Non-Singleton2Digit']
    control_digits = {int(d) for col in all_nontarget_cols if
                      pd.notna(d := row.get(col)) and d != target_digit and d != distractor_digit}

    if verbose:
        print(
            f"Identified Digits -> Target: {target_digit}, Distractor: {distractor_digit}, Controls: {control_digits or 'None'}")

    # --- 4. Calculate projections ---
    results = {}
    results['proj_target'] = get_projection(target_digit, avg_vec, locations_map)
    results['proj_distractor'] = get_projection(distractor_digit, avg_vec, locations_map)

    if verbose:
        print(f"Projection on Target:      {results['proj_target']:.4f}")
        print(f"Projection on Distractor:  {results['proj_distractor']:.4f}")

    if control_digits:
        control_projs = [p for p in [get_projection(d, avg_vec, locations_map) for d in control_digits] if pd.notna(p)]
        if verbose: print(f"Individual Control Projs:  {[round(p, 4) for p in control_projs]}")
        if control_projs:
            results['proj_control_max'] = max(control_projs)
            results['proj_control_avg'] = np.mean(control_projs)
        else:
            results['proj_control_max'], results['proj_control_avg'] = np.nan, np.nan
    else:
        results['proj_control_max'], results['proj_control_avg'] = np.nan, np.nan

    if verbose:
        print(f"Max Control Projection:    {results.get('proj_control_max', np.nan):.4f}")
        print("--------------------------------------")

    return pd.Series(results)


def plot_erp_sanity_check(times, contra_wave, ipsi_wave, diff_wave, meta_df, component_name, time_window,
                          subject_id_col, baseline_window):
    """Helper function to plot grand average ERP waveforms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    times_ms = times * 1000

    subject_diff_waves, subject_contra_waves, subject_ipsi_waves = [], [], []
    for subject in meta_df[subject_id_col].unique():
        mask = (meta_df[subject_id_col] == subject).values
        if np.any(mask):
            subject_contra_waves.append(contra_wave[mask].mean(axis=0))
            subject_ipsi_waves.append(ipsi_wave[mask].mean(axis=0))
            subject_diff_waves.append(diff_wave[mask].mean(axis=0))
            ax.plot(times_ms, diff_wave[mask].mean(axis=0), color='grey', alpha=0.3, lw=1.0)

    if not subject_diff_waves: return

    ax.plot(times_ms, np.mean(subject_contra_waves, axis=0), 'r--', lw=1.5, label='GA Contralateral')
    ax.plot(times_ms, np.mean(subject_ipsi_waves, axis=0), 'b--', lw=1.5, label='GA Ipsilateral')
    ax.plot(times_ms, np.mean(subject_diff_waves, axis=0), 'k-', lw=2.5, label='GA Difference')

    ax.axvspan(baseline_window[0] * 1000, baseline_window[1] * 1000, color='lightblue', alpha=0.5, label='Baseline')
    ax.axvspan(time_window[0] * 1000, time_window[1] * 1000, color='lightcoral', alpha=0.5,
               label=f'{component_name} Window')
    ax.axhline(0, color='black', linestyle='--', lw=0.8)
    ax.axvline(0, color='black', linestyle=':', lw=0.8, label='Stimulus Onset')
    ax.set_title(f'Grand Average {component_name} Waveforms', fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def resample_all_trajectories(raw_df, target_hz=60, trial_cols=None):
    """
    Resamples all trajectories in a raw mouse-tracking dataframe to a constant sampling rate.

    This function works by grouping the data by trial, applying linear interpolation
    to each trial's trajectory individually, and then combining the results into a
    single, clean dataframe. This is the recommended way to prepare trajectory data
    for further analysis, ensuring all subsequent calculations are truly time-weighted.

    Args:
        raw_df (pd.DataFrame): The raw mouse-tracking DataFrame. Must contain time,
                               x, and y coordinates, as well as columns to identify
                               each trial.
        target_hz (int, optional): The target sampling rate in Hz. Defaults to 60.
        trial_cols (list of str, optional): A list of column names that uniquely
                                            identify a trial. If None, defaults to
                                            ['subject_id', 'block', 'trial_nr'].

    Returns:
        pd.DataFrame: A new DataFrame containing all trajectories resampled to the
                      target frequency. The original trial-identifying columns are
                      preserved.
    """
    if trial_cols is None:
        trial_cols = ['subject_id', 'block', 'trial_nr']

    # --- 1. Input Validation ---
    required_cols = trial_cols + ['time', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    # --- 2. Group and Resample ---
    # We group by the trial identifiers. The `groupby` object is an iterator.
    grouped = raw_df.groupby(trial_cols)

    # We will store the resampled dataframes for each trial in a list
    resampled_trials_list = []

    print(f"Resampling {len(grouped)} trials to {target_hz} Hz...")

    for name, group in grouped:
        # 'name' is a tuple of the values from trial_cols (e.g., (1, 1, 1))
        # 'group' is the sub-DataFrame for that specific trial

        # Use the existing single-trial resampling function
        resampled_group = resample_trial_trajectory(group, target_hz=target_hz)

        # If resampling returned an empty or invalid frame, skip it
        if resampled_group.empty:
            continue

        # Add the trial identifier columns back to the resampled data
        for i, col_name in enumerate(trial_cols):
            resampled_group[col_name] = name[i]

        resampled_trials_list.append(resampled_group)

    # --- 3. Combine and Finalize ---
    if not resampled_trials_list:
        print("Warning: No trials could be resampled. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Concatenate all the individual resampled trial DataFrames into one
    final_df = pd.concat(resampled_trials_list, ignore_index=True)

    # Reorder columns for better readability
    final_df = final_df[trial_cols + ['time', 'x', 'y']]

    print(f"Resampling complete. Original samples: {len(raw_df)}, New samples: {len(final_df)}")
    return final_df


def get_vector_length(digit, locations_map):
    """Helper function to get the length of the ideal vector to a digit."""
    if pd.isna(digit) or digit not in locations_map:
        return np.nan
    # Avoid division by zero for the center digit (5), which has length 0
    length = np.linalg.norm(locations_map[digit])
    return length if length > 0 else np.nan


def plot_trajectory_and_vectors(trial_data, full_df, locations_map):
    """A detailed plot to visualize the trajectory, average vector, and scores."""
    sub = trial_data['subject_id']
    blk = trial_data['block']
    trl = trial_data['trial_nr']

    # Get the full trajectory for this specific trial
    trial_trajectory = full_df[
        (full_df['subject_id'] == sub) &
        (full_df['block'] == blk) &
        (full_df['trial_nr'] == trl)
    ]

    # Get key locations
    target_loc = np.array(locations_map[trial_data['TargetDigit']])
    distractor_loc = np.array(locations_map[trial_data['SingletonDigit']])
    avg_vec = np.array([trial_data['avg_x_dva'], trial_data['avg_y_dva']])

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot the full, raw trajectory path
    ax.plot(trial_trajectory['x'], trial_trajectory['y'], color='gray', alpha=0.6, lw=2, label='Full Trajectory Path')

    # Plot ideal locations
    ax.plot(target_loc[0], target_loc[1], 'go', markersize=15, label=f"Target ({trial_data['TargetDigit']})")
    ax.plot(distractor_loc[0], distractor_loc[1], 'ro', markersize=15, label=f"Distractor ({trial_data['SingletonDigit']})")
    ax.plot(0, 0, 'ko', markersize=10)

    # Plot the calculated Average Trajectory Vector
    ax.arrow(0, 0, avg_vec[0], avg_vec[1], color='purple', width=0.02,
             head_width=0.05, length_includes_head=True, label='Avg. Trajectory Vector')

    # Add annotations for the scores
    plt.text(0.95, 0.1,
             f"Proj. on Target: {trial_data['target_capture_score']:.3f}\n"
             f"Proj. on Distractor: {trial_data['distractor_capture_score']:.3f}\n"
             f"Difference: {trial_data['target_distractor_capture_diff']:.3f}",
             transform=ax.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    ax.set_title(f"Trajectory Analysis: sub-{sub}, trial-{trl}\n(Max-Distance Capture Example)")
    ax.set_xlabel("X-position (dva)")
    ax.set_ylabel("Y-position (dva)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--')
    ax.legend()
    sns.despine()
    plt.show()


def get_distance_between_digits(row, locations_map):
    """Calculates the Euclidean distance between the target and distractor."""
    if row['SingletonPresent'] == 0:
        return np.nan # No distractor, no distance

    target_digit = row['TargetDigit']
    distractor_digit = row['SingletonDigit']

    # Ensure both digits are in our location map
    if target_digit not in locations_map or distractor_digit not in locations_map:
        return np.nan

    target_vec = np.array(locations_map[target_digit])
    distractor_vec = np.array(locations_map[distractor_digit])

    # Calculate Euclidean distance
    distance = np.linalg.norm(target_vec - distractor_vec)
    return distance

def plot_jackknife_sanity_check(
    single_trial_wave, single_trial_times,
    jackknife_wave, jackknife_times,
    jackknife_df,
    trial_info,
    component_name,
    electrode_pair,
    time_window,
    all_times,
    lateral_stim_col
):
    """
    Generates a detailed plot to visually inspect and validate the jackknife procedure.

    This plot helps diagnose issues by showing:
    1. All the individual trial waves that constitute the jackknife average.
    2. A direct comparison between the noisy single-trial wave (left out) and the
       (theoretically) smoother jackknife average wave.

    Args:
        single_trial_wave (np.ndarray): The difference wave for the single trial that was left out.
        single_trial_times (np.ndarray): Time points for the single trial wave.
        jackknife_wave (np.ndarray): The averaged difference wave from the jackknife sample.
        jackknife_times (np.ndarray): Time points for the jackknife wave.
        jackknife_df (pd.DataFrame): The DataFrame containing the N-1 trials of the jackknife sample.
        trial_info (dict): Dictionary with info like subject_id, trial_nr for titles.
        component_name (str): Name of the ERP component (e.g., 'N2ac').
        electrode_pair (tuple): The (left, right) electrode pair being analyzed.
        time_window (tuple): The (start, end) time for the analysis.
        all_times (np.ndarray): The full time vector for the epoch.
        lateral_stim_col (str): The column name for stimulus location.
    """
    if jackknife_wave is None or single_trial_wave is None:
        print("Skipping sanity plot due to missing wave data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(
        f"Jackknife Sanity Check: {component_name} on {electrode_pair[0]}-{electrode_pair[1]}\n"
        f"Subject: {trial_info.get('subject_id', 'N/A')}, "
        f"Left-Out Trial Index: {trial_info.get('trial_idx', 'N/A')}",
        fontsize=16, fontweight='bold'
    )

    # --- Plot 1: All waves in the jackknife sample ---
    ax1 = axes[0]
    num_jk_trials = len(jackknife_df)
    ax1.set_title(f"1. Waves in Jackknife Sample (N={num_jk_trials})", fontsize=12)

    # Plot each individual trial wave within the jackknife sample
    for _, jk_trial_row in jackknife_df.iterrows():
        wave, times = get_single_trial_contra_ipsi_wave(
            trial_row=jk_trial_row,
            lateral_stim_loc=jk_trial_row[lateral_stim_col],
            electrode_pairs=[electrode_pair],
            time_window=time_window,
            all_times=all_times
        )
        if wave is not None:
            ax1.plot(times, wave, color='grey', alpha=0.4, lw=1.0)

    # Overlay the jackknife average
    ax1.plot(jackknife_times, jackknife_wave, color='black', lw=2.5, label=f'Jackknife Average (of N={num_jk_trials})')

    ax1.axhline(0, color='black', linestyle='--', lw=0.8)
    ax1.axvline(0, color='black', linestyle=':', lw=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # --- Plot 2: Direct Comparison ---
    ax2 = axes[1]
    ax2.set_title("2. Comparison: Single Trial vs. Jackknife Average", fontsize=12)

    # Plot the single trial that was left out
    ax2.plot(single_trial_times, single_trial_wave, color='cornflowerblue', lw=1.5, label='Single Trial (Left Out)')

    # Plot the jackknife average again for comparison
    ax2.plot(jackknife_times, jackknife_wave, color='red', lw=2.5, label=f'Jackknife Average (of N={num_jk_trials})')

    ax2.axhline(0, color='black', linestyle='--', lw=0.8)
    ax2.axvline(0, color='black', linestyle=':', lw=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # Use block=True to pause the script for inspection. Close the plot window to continue.
    plt.show(block=True)


def calculate_fractional_area_latency(erp_wave, times, percentage=0.5, plot=False, is_target=False, analysis_window_times=None):
    """
    Calculates latency based on a specified percentage of the area under a scaled ERP wave.

    This method first performs a min-max normalization on the ERP waveform to scale it
    between 0 and 1. This emphasizes the shape of the wave relative to its peak. The latency is
    then calculated as the time point where the cumulative area under this *scaled* wave
    reaches the specified percentage of its total area. The calculation can be restricted
    to a specific time window.

    Args:
        erp_wave (np.ndarray): The ERP waveform (can be positive or negative-going).
        times (np.ndarray): The time points corresponding to the erp_wave.
        percentage (float, optional): The percentage of the total area to use as a threshold.
                                      Defaults to 0.5 (for 50%).
        plot (bool, optional): If True, displays a plot illustrating the calculation. Defaults to False.
        is_target (bool): If True, the wave is assumed to be negative-going (e.g., N2ac) and is
                          inverted for the calculation. Defaults to False.
        analysis_window_times (tuple, optional): A (start, end) tuple in seconds to specify the
                                                 time window for the latency calculation.
                                                 If None, the entire waveform is used. Defaults to None.

    Returns:
        float: The calculated latency in seconds, or np.nan if the latency cannot be determined.
    """
    # Keep the full, original wave for plotting context
    original_erp_wave_for_plot = erp_wave.copy()
    original_times_for_plot = times.copy()

    # --- 1. Apply Analysis Time Window if provided ---
    if analysis_window_times is not None:
        start_time, end_time = analysis_window_times
        time_mask = (times >= start_time) & (times <= end_time)
        erp_wave = erp_wave[time_mask]
        times = times[time_mask]

        # If the window is empty or outside the data range, we can't calculate latency.
        if len(times) == 0:
            if plot:
                print("Plotting skipped: The specified analysis window is empty or out of bounds.")
            return np.nan

    # --- 2. Prepare the wave for analysis (handle component polarity) ---
    # To analyze the component's morphology, we make its peak positive.
    # N2ac is negative-going, so we invert it. Pd is positive-going, so we leave it.
    if is_target:
        analysis_wave = erp_wave * -1
    else:
        analysis_wave = erp_wave

    # --- 3. Min-Max Scale the wave within the window to a [0, 1] range ---
    min_val = np.min(analysis_wave)
    max_val = np.max(analysis_wave)

    if max_val <= min_val:
        if plot:
            print("Plotting skipped: ERP wave is flat within the analysis window.")
        return np.nan

    scaled_wave = (analysis_wave - min_val) / (max_val - min_val)

    # --- 4. Calculate Cumulative Area and Latency on the scaled, windowed wave ---
    cum_sum = np.cumsum(scaled_wave)
    total_area = np.max(cum_sum)

    if total_area <= 0:
        if plot:
            print("Plotting skipped: Total area under the scaled curve is zero.")
        return np.nan

    target_area = total_area * percentage
    crossings = np.where(cum_sum >= target_area)[0]

    latency = np.nan
    if len(crossings) > 0:
        first_crossing_idx = crossings[0]
        if first_crossing_idx == 0:
            latency = times[first_crossing_idx]
        else:
            # Linear Interpolation for a more precise latency
            idx_before = first_crossing_idx - 1
            idx_after = first_crossing_idx
            t_before, t_after = times[idx_before], times[idx_after]
            val_before, val_after = cum_sum[idx_before], cum_sum[idx_after]

            if val_after == val_before:
                latency = t_after
            else:
                latency = np.interp(target_area, [val_before, val_after], [t_before, t_after])

    # --- 5. Plotting Logic ---
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # No sharex for different x-ranges
        erp_type = 'N2ac' if is_target else 'Pd'
        title = f"Fractional Area Latency ({percentage*100:.0f}%) on Scaled Waveform ({erp_type})"
        if analysis_window_times:
            title += f"\nAnalysis Window: {analysis_window_times[0]}-{analysis_window_times[1]}s"
        fig.suptitle(title, fontsize=16)

        # Plot 1: Original and Scaled ERP Waveforms
        ax1.set_title("Original Waveform & Analysis Window")
        ax1.plot(original_times_for_plot, original_erp_wave_for_plot, color='navy', label='Original Wave (µV)')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µV)", color='navy')
        ax1.tick_params(axis='y', labelcolor='navy')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        # Highlight the analysis window on the full waveform plot
        if analysis_window_times:
            ax1.axvspan(analysis_window_times[0], analysis_window_times[1], color='grey', alpha=0.25, label='Analysis Window')
        ax1.legend(loc='best')

        # Plot 2: Cumulative Sum and Latency Calculation (on windowed data)
        ax2.set_title("Cumulative Area & Latency (in Window)")
        ax2.plot(times, cum_sum, color='teal', label='Cumulative Area of Scaled Wave')
        ax2.axhline(target_area, color='red', linestyle='--', label=f'{percentage*100:.0f}% Area Threshold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cumulative Area (arbitrary units)")
        ax2.grid(True, linestyle=':', alpha=0.6)

        if not np.isnan(latency):
            ax2.axvline(latency, color='purple', linestyle='-', lw=2, label=f'Latency: {latency:.3f}s')
            ax2.plot(latency, target_area, 'ro', markersize=8)

        ax2.legend(loc='best')
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show(block=True)

    return latency