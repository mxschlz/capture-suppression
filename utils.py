import mne
import numpy as np
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
import matplotlib.pyplot as plt


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

def get_jackknife_contra_ipsi_wave(sample_df, lateral_stim_loc, electrode_pairs, time_window, all_times, plot=False):
    """
    Calculates the average contralateral-ipsilateral difference wave from a jackknife sample.

    Args:
        sample_df (pd.DataFrame): DataFrame containing the ERP data for the jackknife sample (all trials but one).
        lateral_stim_loc (str): Location of the lateralized stimulus ('left' or 'right').
        electrode_pairs (list): List of tuples, where each tuple is a (left_hemi_el, right_hemi_el) pair.
        time_window (tuple): The (start, end) time in seconds for the analysis.
        all_times (np.ndarray): Array of all time points in the epoch.
        plot (bool, optional): If True, displays a plot of the resulting difference wave. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final averaged contra-ipsi difference wave.
            - np.ndarray: The time points corresponding to the wave.
    """
    # Create a boolean mask for the desired time window
    time_mask = (all_times >= time_window[0]) & (all_times <= time_window[1])
    window_times = all_times[time_mask]

    diff_waves_for_pairs = []
    for left_el, right_el in electrode_pairs:
        # Extract ERP data for the electrode pair from the sample trials
        left_el_data = sample_df[left_el].loc[:, time_mask].values
        right_el_data = sample_df[right_el].loc[:, time_mask].values

        # Average across trials to get a single wave for each electrode
        avg_left_wave = np.mean(left_el_data, axis=0)
        avg_right_wave = np.mean(right_el_data, axis=0)

        # Calculate contralateral - ipsilateral difference
        if lateral_stim_loc == 'left':
            # Contralateral is Right Hemisphere, Ipsilateral is Left Hemisphere
            diff_wave = avg_right_wave - avg_left_wave
        else:  # lateral_stim_loc == 'right'
            # Contralateral is Left Hemisphere, Ipsilateral is Right Hemisphere
            diff_wave = avg_left_wave - avg_right_wave
        diff_waves_for_pairs.append(diff_wave)

    # Average the difference waves across all specified electrode pairs
    mean_diff_wave = np.mean(diff_waves_for_pairs, axis=0)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(window_times, mean_diff_wave, label='Contra-Ipsi Difference', color='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        if window_times[0] <= 0 <= window_times[-1]:
            plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"Jackknife Contra-Ipsi Difference Wave (Stimulus: {lateral_stim_loc})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show(block=True)

    return mean_diff_wave, window_times


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


def get_contra_ipsi_diff_wave(trials_df, electrode_pairs, time_window, all_times, lateral_stim_col):
    """
    Calculates the grand-average contralateral-ipsilateral difference wave from a given set of trials.
    This is NOT a jackknife procedure. It averages all trials provided for a condition.

    Args:
        trials_df (pd.DataFrame): DataFrame containing ERP data for a specific condition.
        electrode_pairs (list): List of (left_hemi_el, right_hemi_el) tuples.
        time_window (tuple): The (start, end) time in seconds.
        all_times (np.ndarray): Array of all time points in the epoch.
        lateral_stim_col (str): Column name ('TargetLoc' or 'SingletonLoc') indicating stimulus location.

    Returns:
        tuple: (The difference wave, The corresponding time points) or (None, None) if no data.
    """
    if trials_df.empty:
        return None, None

    time_mask = (all_times >= time_window[0]) & (all_times <= time_window[1])
    window_times = all_times[time_mask]

    all_contra_activity = []
    all_ipsi_activity = []

    # Separate trials by stimulus location
    left_stim_trials = trials_df[trials_df[lateral_stim_col] == 'left']
    right_stim_trials = trials_df[trials_df[lateral_stim_col] == 'right']

    for left_el, right_el in electrode_pairs:
        # For left-stimulus trials, contralateral is RIGHT hemisphere
        if not left_stim_trials.empty:
            all_contra_activity.append(left_stim_trials[right_el].loc[:, time_mask].values)
            all_ipsi_activity.append(left_stim_trials[left_el].loc[:, time_mask].values)

        # For right-stimulus trials, contralateral is LEFT hemisphere
        if not right_stim_trials.empty:
            all_contra_activity.append(right_stim_trials[left_el].loc[:, time_mask].values)
            all_ipsi_activity.append(right_stim_trials[right_el].loc[:, time_mask].values)

    if not all_contra_activity:  # If no trials were found for any condition
        return None, None

    # Concatenate across electrode pairs and average across all trials
    # This pools all contra trials and all ipsi trials together
    mean_contra_wave = np.mean(np.concatenate(all_contra_activity, axis=0), axis=0)
    mean_ipsi_wave = np.mean(np.concatenate(all_ipsi_activity, axis=0), axis=0)

    diff_wave = mean_contra_wave - mean_ipsi_wave
    return diff_wave, window_times
