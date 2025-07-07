import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from stats import remove_outliers, r_squared_mixed_model  # Assuming this is your custom outlier removal function
from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

plt.ion()


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
        # The columns are a MultiIndex: (electrode_name, time_point)
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

    # --- Plotting Logic ---
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(window_times, mean_diff_wave, label='Contra-Ipsi Difference', color='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        # Add a vertical line at time 0 for reference if it's in the plot range
        if window_times[0] <= 0 <= window_times[-1]:
            plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"Jackknife Contra-Ipsi Difference Wave (Stimulus: {lateral_stim_loc})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show(block=True)

    return mean_diff_wave, window_times


def calculate_fractional_area_latency(erp_wave, times, percentage=0.5, plot=False, is_target=False):
    """
    Calculates latency based on a specified percentage of the area under a scaled ERP wave.

    This method first performs a min-max normalization on the absolute ERP waveform to scale it
    between 0 and 1. This emphasizes the shape of the wave relative to its peak. The latency is
    then calculated as the time point where the cumulative area under this *scaled* wave
    reaches the specified percentage of its total area.

    Args:
        erp_wave (np.ndarray): The ERP waveform (can be positive or negative-going).
        times (np.ndarray): The time points corresponding to the erp_wave.
        percentage (float, optional): The percentage of the total area to use as a threshold.
                                      Defaults to 0.5 (for 50%).
        plot (bool, optional): If True, displays a plot illustrating the calculation. Defaults to False.
        is_target (bool): Set plot title according to ERP category. Defaults to False.

    Returns:
        float: The calculated latency in seconds, or np.nan if the latency cannot be determined.
    """
    # Work with the absolute value to make the calculation sign-independent
    # abs_erp_wave = np.abs(erp_wave)
    abs_erp_wave = erp_wave  # do NOT use np.abs() here

    # --- Min-Max Scale the absolute ERP wave to a [0, 1] range ---
    min_val = np.min(abs_erp_wave)
    max_val = np.max(abs_erp_wave)

    # If the wave is flat or has no amplitude, latency cannot be calculated.
    if max_val <= min_val:
        if plot:
            print("Plotting skipped: ERP wave is flat or has no amplitude.")
        return np.nan

    scaled_wave = (abs_erp_wave - min_val) / (max_val - min_val)

    # Calculate the cumulative sum (area under the curve) of the SCALED wave
    cum_sum = np.cumsum(scaled_wave)
    total_area = np.max(cum_sum)

    # If the total area is zero (can happen if scaled_wave is all zeros), we cannot calculate a latency.
    if total_area <= 0:
        if plot:
            print("Plotting skipped: Total area under the scaled curve is zero.")
        return np.nan

    # Determine the area threshold based on the specified percentage
    target_area = total_area * percentage

    # Find the first index where the cumulative sum crosses the threshold
    crossings = np.where(cum_sum >= target_area)[0]

    latency = np.nan
    if len(crossings) > 0:
        first_crossing_idx = crossings[0]

        # If the crossing happens at the very first time point, no interpolation is possible.
        if first_crossing_idx == 0:
            latency = times[first_crossing_idx]
        else:
            # --- Linear Interpolation for a more precise latency ---
            # Get the points just before and at the crossing
            idx_before = first_crossing_idx - 1
            idx_after = first_crossing_idx

            t_before = times[idx_before]
            t_after = times[idx_after]

            # Use the cumulative sum of the scaled wave for interpolation
            val_before = cum_sum[idx_before]
            val_after = cum_sum[idx_after]

            # Avoid division by zero if the cumulative sum is flat in this segment
            if val_after == val_before:
                latency = t_after
            else:
                # Interpolate the time at which the cumulative sum equals the target area
                latency = np.interp(target_area, [val_before, val_after], [t_before, t_after])

    # --- Plotting Logic ---
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        erp_type = 'N2ac' if is_target else 'Pd'
        fig.suptitle(f"Fractional Area Latency ({percentage*100:.0f}%) on Scaled Waveform ({erp_type})", fontsize=16)

        # Plot 1: Original and Scaled ERP Waveforms
        ax1.set_title("Original vs. Scaled Waveform")
        ax1.plot(times, erp_wave, color='navy', label='Original Wave (µV)')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µV)", color='navy')
        ax1.tick_params(axis='y', labelcolor='navy')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Create a second y-axis for the scaled wave
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times, scaled_wave, color='darkorange', linestyle='--', label='Scaled Wave (0-1)')
        ax1_twin.set_ylabel("Scaled Amplitude", color='darkorange')
        ax1_twin.tick_params(axis='y', labelcolor='darkorange')
        ax1_twin.set_ylim(-0.05, 1.05)

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

        # Plot 2: Cumulative Sum and Latency Calculation
        ax2.set_title("Cumulative Area & Latency")
        ax2.plot(times, cum_sum, color='teal', label='Cumulative Area of Scaled Wave')
        ax2.axhline(target_area, color='red', linestyle='--', label=f'{percentage*100:.0f}% Threshold Area')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cumulative Area (arbitrary units)")
        ax2.grid(True, linestyle=':', alpha=0.6)

        if not np.isnan(latency):
            ax2.axvline(latency, color='purple', linestyle='-', lw=2, label=f'Latency: {latency:.3f}s')
            # Mark the exact intersection of the latency and threshold lines
            ax2.plot(latency, target_area, 'ro', markersize=8, label='Crossing Point')

        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        plt.show(block=True)

    return latency


# --- Script Configuration Parameters ---

# --- 1. Data Loading & Preprocessing ---
EPOCH_TMIN = 0.0  # Start time for epoch cropping (seconds)
EPOCH_TMAX = 0.7  # End time for epoch cropping (seconds)
OUTLIER_RT_THRESHOLD = 2.0  # SDs for RT outlier removal (used in remove_outliers)
FILTER_PHASE = 2  # Phase to exclude from analysis (e.g., practice blocks)

# --- 2. Column Names ---
# Existing columns in metadata or to be created
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'  # Target position (e.g., "left", "mid", "right")
DISTRACTOR_COL = 'SingletonLoc'  # Distractor position (e.g., "absent", "left", "mid", "right")
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'  # Original accuracy column (e.g., True/False or 1.0/0.0)
PHASE_COL = 'phase'
# DISTRACTOR_PRESENCE_COL = 'SingletonPresent'  # (e.g., 0 or 1)
PRIMING_COL = 'Priming'  # (e.g., "np", "no-p", "pp")
# TARGET_DIGIT_COL = 'TargetDigit'
# SINGLETON_DIGIT_COL = 'SingletonDigit'
# BLOCK_COL = 'block' # Removed as per request

# ERP component columns (will be created in the script)
ERP_N2AC_LATENCY_COL = 'N2ac_latency' # Changed to latency
ERP_PD_LATENCY_COL = 'Pd_latency'   # Changed to latency

# Derived columns for modeling
TRIAL_NUMBER_COL = 'total_trial_nr'  # Overall trial number per subject (will be created)
ACCURACY_INT_COL = 'select_target_int'  # Integer version of accuracy (0 or 1, will be created)

# --- Mappings and Reference Levels for Categorical Variables ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}

# Define string reference levels based on the original numeric ones
TARGET_REF_VAL_ORIGINAL = 2
TARGET_REF_STR = TARGET_LOC_MAP.get(TARGET_REF_VAL_ORIGINAL) # Should be "mid"

DISTRACTOR_REF_VAL_ORIGINAL = 2 # This reference is typically for when distractor is present
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(DISTRACTOR_REF_VAL_ORIGINAL) # Should be "mid"

PRIMING_REF_VAL_ORIGINAL = 0
PRIMING_REF_STR = PRIMING_MAP.get(PRIMING_REF_VAL_ORIGINAL) # Should be "no-p"

# --- 3. ERP Component Definitions ---
PD_TIME_WINDOW = (0.29, 0.38)  # (start_time, end_time) in seconds
PD_ELECTRODES = [
    ("FC3", "FC4"),
    ("FC5", "FC6"),
    ("C3", "C4"),
    ("C5", "C6"),
    ("CP3", "CP4"),
    ("CP5", "CP6")
]  # Electrodes for Pd. Order: [left_hemisphere_electrode, right_hemisphere_electrode]

# N2ac Component
N2AC_TIME_WINDOW = (0.22, 0.38)  # (start_time, end_time) in seconds
N2AC_ELECTRODES = [
    ("FC3", "FC4"),
    ("FC5", "FC6"),
    ("C3", "C4"),
    ("C5", "C6"),
    ("CP3", "CP4"),
    ("CP5", "CP6")
]  # Electrodes for N2ac. Order: [left_hemisphere_electrode, right_hemisphere_electrode]

# --- New Configuration for Latency Robustness Check ---
PERCENTAGES_TO_TEST = [0.5] # Test thresholds from 20% to 70%

# --- 4. LMM Predictor Variables for N2ac and Pd Models ---
LMM_N2AC_PD_CONTINUOUS_PREDICTORS = [TRIAL_NUMBER_COL, REACTION_TIME_COL]
LMM_N2AC_PD_CATEGORICAL_PREDICTORS = [
    TARGET_COL,
    DISTRACTOR_COL,
    PRIMING_COL,
    ACCURACY_INT_COL
    # BLOCK_COL removed
    # DISTRACTOR_PRESENCE_COL removed
]

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs()
print("Epochs loaded and cropped.")

df = epochs.metadata.copy()
print(f"Original number of trials: {len(df)}")

if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
    print(f"Trials after filtering phase != {FILTER_PHASE}: {len(df)}")
elif FILTER_PHASE is not None:
    print(f"Warning: Column '{PHASE_COL}' not found for filtering.")

if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"Trials after RT outlier removal: {len(df)}")
else:
    print(f"Warning: Column '{REACTION_TIME_COL}' not found for outlier removal.")

if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
else:
    print(f"Warning: '{SUBJECT_ID_COL}' column not found. Cannot create '{TRIAL_NUMBER_COL}'.")

if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
    print(f"Created '{ACCURACY_INT_COL}' from '{ACCURACY_COL}'. Unique values: {df[ACCURACY_INT_COL].unique()}")
else:
    print(f"Warning: Accuracy column '{ACCURACY_COL}' not found. Cannot create '{ACCURACY_INT_COL}'.")

# Process TARGET_COL: Convert to numeric, then map to strings
if TARGET_COL in df.columns:
    numeric_target_col = pd.to_numeric(df[TARGET_COL], errors='coerce')
    if numeric_target_col.isna().sum() > df[TARGET_COL].isna().sum(): # Check if new NaNs were introduced by coercion
        print(f"Warning: NaNs introduced in '{TARGET_COL}' after coercion. Original non-numeric values will become NaN after mapping if not in map.")
    df[TARGET_COL] = numeric_target_col.map(TARGET_LOC_MAP)
    print(f"Unique values in '{TARGET_COL}' after mapping: {df[TARGET_COL].unique()}")
    if TARGET_REF_STR is None:
        print(f"Error: TARGET_REF_STR is None. Original numeric reference {TARGET_REF_VAL_ORIGINAL} might not be in TARGET_LOC_MAP.")
    elif TARGET_REF_STR not in df[TARGET_COL].dropna().unique():
        print(f"Warning: Mapped column '{TARGET_COL}' does not contain the reference value '{TARGET_REF_STR}'. "
              f"Setting '{TARGET_REF_STR}' as reference might cause issues if it's not present in the data for some models.")
else:
    raise ValueError(f"Error: Critical column '{TARGET_COL}' not found in metadata.")

# Process DISTRACTOR_COL: Convert to numeric, then map to strings
if DISTRACTOR_COL in df.columns:
    numeric_distractor_col = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce')
    if numeric_distractor_col.isna().sum() > df[DISTRACTOR_COL].isna().sum():
        print(f"Warning: NaNs introduced in '{DISTRACTOR_COL}' after coercion. Original non-numeric values will become NaN after mapping if not in map.")
    df[DISTRACTOR_COL] = numeric_distractor_col.map(DISTRACTOR_LOC_MAP)
    print(f"Unique values in '{DISTRACTOR_COL}' after mapping: {df[DISTRACTOR_COL].unique()}")
    if DISTRACTOR_REF_STR is None:
         print(f"Error: DISTRACTOR_REF_STR is None. Original numeric reference {DISTRACTOR_REF_VAL_ORIGINAL} might not be in DISTRACTOR_LOC_MAP.")
    # Check for reference 'mid' among present distractors (relevant for Pd model)
    elif DISTRACTOR_LOC_MAP.get(0) is not None and \
         DISTRACTOR_REF_STR not in df[df[DISTRACTOR_COL] != DISTRACTOR_LOC_MAP[0]][DISTRACTOR_COL].dropna().unique():
        print(f"Warning: Mapped column '{DISTRACTOR_COL}' (excluding '{DISTRACTOR_LOC_MAP[0]}') does not contain the reference value '{DISTRACTOR_REF_STR}'. "
              f"This might cause issues for models using '{DISTRACTOR_REF_STR}' as reference for present distractors.")
    elif DISTRACTOR_LOC_MAP.get(0) is None and DISTRACTOR_REF_STR not in df[DISTRACTOR_COL].dropna().unique():
         print(f"Warning: Mapped column '{DISTRACTOR_COL}' does not contain the reference value '{DISTRACTOR_REF_STR}'.")
else:
    raise ValueError(f"Error: Critical column '{DISTRACTOR_COL}' not found in metadata.")

# Process PRIMING_COL: Convert to numeric, then map to strings
if PRIMING_COL in df.columns:
    numeric_priming_col = pd.to_numeric(df[PRIMING_COL], errors='coerce')
    if numeric_priming_col.isna().sum() > df[PRIMING_COL].isna().sum():
        print(f"Warning: NaNs introduced in '{PRIMING_COL}' after coercion. Original non-numeric values will become NaN after mapping if not in map.")
    df[PRIMING_COL] = numeric_priming_col.map(PRIMING_MAP)
    print(f"Unique values in '{PRIMING_COL}' after mapping: {df[PRIMING_COL].unique()}")
    if PRIMING_REF_STR is None:
        print(f"Error: PRIMING_REF_STR is None. Original numeric reference {PRIMING_REF_VAL_ORIGINAL} might not be in PRIMING_MAP.")
    elif PRIMING_REF_STR not in df[PRIMING_COL].dropna().unique():
        print(f"Warning: Mapped column '{PRIMING_COL}' does not contain the reference value '{PRIMING_REF_STR}'. "
              f"Setting '{PRIMING_REF_STR}' as reference might cause issues if it's not present in the data for some models.")
else:
    print(f"Warning: Priming column '{PRIMING_COL}' not found.")

# Get all unique electrodes needed for N2ac and Pd calculations
erp_df_picks_tuples = list(set(N2AC_ELECTRODES + PD_ELECTRODES)) # List of unique (L,R) tuples
erp_df_picks_flat = [item for electrode_pair in erp_df_picks_tuples for item in electrode_pair] # Flatten list of tuples
erp_df_picks_unique_flat = sorted(list(set(erp_df_picks_flat))) # Unique electrode names, sorted

# Extract ERP data for the required electrodes
erp_df = epochs.to_data_frame(picks=erp_df_picks_unique_flat, time_format=None)
# Reshape ERP data from long to wide format
# This makes it easier to access time series for each trial.
# The result has 'epoch' as the index and a multi-level column (channel, time)
print("Reshaping ERP data to wide format...")
erp_wide = erp_df.pivot(index='epoch', columns='time')
# Reorder columns to group by electrode for easier selection
erp_wide = erp_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
# Flatten the multi-index for direct access, e.g., erp_wide['FC3']
erp_wide.columns = erp_wide.columns.droplevel(0)
# Merge metadata with the wide ERP data
# The index of `df` (from `epochs.metadata`) corresponds to the 'epoch' number.
merged_df = df.join(erp_wide)
print(f"Merged metadata with wide ERP data. Shape: {merged_df.shape}")
# Filter for trials relevant to contra-ipsi analysis by separating them into distinct pools
# These are trials where one stimulus is lateral (left/right) and the other is central (mid).
is_target_lateral = merged_df[TARGET_COL].isin(['left', 'right'])
is_distractor_lateral = merged_df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = merged_df[TARGET_COL] == TARGET_REF_STR  # 'mid'
is_distractor_central = merged_df[DISTRACTOR_COL] == 'mid'

# --- Create Separate DataFrames for N2ac and Pd to remove the confound ---
# N2ac trials: Target is lateral, Distractor is central
n2ac_analysis_df = merged_df[is_target_lateral & is_distractor_central].copy()
print(f"Filtered for N2ac analysis (target lateral, distractor central). Number of trials: {len(n2ac_analysis_df)}")

# Pd trials: Distractor is lateral, Target is central
pd_analysis_df = merged_df[is_distractor_lateral & is_target_central].copy()
print(f"Filtered for Pd analysis (distractor lateral, target central). Number of trials: {len(pd_analysis_df)}")

# 4. Initialize new latency columns in the main dataframe `df` for each percentage
print("Initializing latency columns for robustness check...")
for p in PERCENTAGES_TO_TEST:
    p_int = int(p * 100)
    df[f'{ERP_N2AC_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ERP_PD_LATENCY_COL}_{p_int}'] = np.nan

# Get all unique time points from the epoch
all_times = epochs.times

plot_erp_wave = False  # flag to plot results or not

# --- N2ac Latency Calculation Loop (Unconfounded) ---
print("\n--- Calculating N2ac Latencies (Target-Lateral Trials) ---")
n2ac_subjects = n2ac_analysis_df[SUBJECT_ID_COL].unique()
for subject_id in n2ac_subjects:
    print(f"Processing N2ac for subject: {subject_id}...")
    # Get all N2ac-relevant trials for the current subject
    subject_n2ac_df = n2ac_analysis_df[n2ac_analysis_df[SUBJECT_ID_COL] == subject_id]

    # Iterate through each trial to calculate its jackknife latency
    for trial_idx, trial_row in subject_n2ac_df.iterrows():
        # The jackknife sample consists of all *other* N2ac-relevant trials for this subject
        jackknife_sample_df = subject_n2ac_df.drop(index=trial_idx)
        if jackknife_sample_df.empty:
            continue  # Cannot calculate if it's the only trial

        # For N2ac, the lateral stimulus is always the target
        lateral_stim_loc = trial_row[TARGET_COL]

        # Get the jackknife-averaged contra-ipsi difference wave
        n2ac_wave, n2ac_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df,
            lateral_stim_loc=lateral_stim_loc,
            electrode_pairs=N2AC_ELECTRODES,
            time_window=N2AC_TIME_WINDOW,
            all_times=all_times,
            plot=False  # Set to True for debugging a single trial
        )

        # Loop through each percentage threshold and calculate latency
        for p in PERCENTAGES_TO_TEST:
            p_int = int(p * 100)
            col_name = f'{ERP_N2AC_LATENCY_COL}_{p_int}'
            latency = calculate_fractional_area_latency(
                n2ac_wave,
                n2ac_times,
                percentage=p,
                plot=plot_erp_wave,  # Set to True for debugging
                is_target=True
            )
            # Store the calculated latency in the original main dataframe `df`
            df.loc[trial_idx, col_name] = latency

# --- Pd Latency Calculation Loop (Unconfounded) ---
print("\n--- Calculating Pd Latencies (Distractor-Lateral Trials) ---")
pd_subjects = pd_analysis_df[SUBJECT_ID_COL].unique()
for subject_id in pd_subjects:
    print(f"Processing Pd for subject: {subject_id}...")
    # Get all Pd-relevant trials for the current subject
    subject_pd_df = pd_analysis_df[pd_analysis_df[SUBJECT_ID_COL] == subject_id]

    # Iterate through each trial to calculate its jackknife latency
    for trial_idx, trial_row in subject_pd_df.iterrows():
        # The jackknife sample consists of all *other* Pd-relevant trials for this subject
        jackknife_sample_df = subject_pd_df.drop(index=trial_idx)
        if jackknife_sample_df.empty:
            continue  # Cannot calculate if it's the only trial

        # For Pd, the lateral stimulus is always the distractor
        lateral_stim_loc = trial_row[DISTRACTOR_COL]

        # Get the jackknife-averaged contra-ipsi difference wave
        pd_wave, pd_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df,
            lateral_stim_loc=lateral_stim_loc,
            electrode_pairs=PD_ELECTRODES,
            time_window=PD_TIME_WINDOW,
            all_times=all_times,
            plot=False  # Set to True for debugging a single trial
        )

        # Loop through each percentage threshold and calculate latency
        for p in PERCENTAGES_TO_TEST:
            p_int = int(p * 100)
            col_name = f'{ERP_PD_LATENCY_COL}_{p_int}'
            latency = calculate_fractional_area_latency(
                pd_wave,
                pd_times,
                percentage=p,
                plot=plot_erp_wave,  # Set to True for debugging
                is_target=False
            )
            # Store the calculated latency in the original main dataframe `df`
            df.loc[trial_idx, col_name] = latency

# --- Analysis of Latency Robustness ---
print("\n--- Analyzing Robustness of Latency Calculation ---")

# --- N2ac Data Preparation ---
n2ac_latency_cols = [f'{ERP_N2AC_LATENCY_COL}_{int(p*100)}' for p in PERCENTAGES_TO_TEST]
n2ac_df = df[n2ac_latency_cols].copy()
n2ac_long_df = n2ac_df.melt(var_name='Threshold (%)', value_name='Latency (s)')
n2ac_long_df['Threshold (%)'] = n2ac_long_df['Threshold (%)'].str.extract(f'({ERP_N2AC_LATENCY_COL}_(\\d+))')[1].astype(int)
n2ac_long_df.dropna(inplace=True)

# --- Pd Data Preparation ---
pd_latency_cols = [f'{ERP_PD_LATENCY_COL}_{int(p*100)}' for p in PERCENTAGES_TO_TEST]
pd_df = df[pd_latency_cols].copy()
pd_long_df = pd_df.melt(var_name='Threshold (%)', value_name='Latency (s)')
pd_long_df['Threshold (%)'] = pd_long_df['Threshold (%)'].str.extract(f'({ERP_PD_LATENCY_COL}_(\\d+))')[1].astype(int)
pd_long_df.dropna(inplace=True)

# --- Print Statistics and Generate Combined Plot ---
n2ac_has_data = not n2ac_long_df.empty
pd_has_data = not pd_long_df.empty

# Plotting logic: create a combined plot if both have data, otherwise plot individually.
if n2ac_has_data and pd_has_data:
    # Create a figure with two subplots side-by-side, sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle('Distribution of ERP Latencies by Fractional Area Threshold', fontsize=18)

    # N2ac Plot on the first axis (ax1)
    sns.boxplot(data=n2ac_long_df, x='Threshold (%)', y='Latency (s)', ax=ax1)
    ax1.set_title('N2ac Latencies', fontsize=14)
    ax1.set_xlabel('Threshold Percentage (%)')
    ax1.set_ylabel('Calculated Latency (s)')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Pd Plot on the second axis (ax2)
    sns.boxplot(data=pd_long_df, x='Threshold (%)', y='Latency (s)', ax=ax2)
    ax2.set_title('Pd Latencies', fontsize=14)
    ax2.set_xlabel('Threshold Percentage (%)')
    ax2.set_ylabel('')  # Hide y-label as it's shared with the left plot
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show(block=True)

# --- Single-Trial Brain-Behavior Analysis (LMM) ---
print("\n--- Analyzing Single-Trial Brain-Behavior Relationship using LMMs ---")

# Choose a definitive percentage for this analysis. 50% is a standard choice.
definitive_percentage = 50
n2ac_definitive_col = f'{ERP_N2AC_LATENCY_COL}_{definitive_percentage}'
pd_definitive_col = f'{ERP_PD_LATENCY_COL}_{definitive_percentage}'

# --- 1. Prepare single-trial data for N2ac vs. RT ---
n2ac_trials_df = df.dropna(subset=[n2ac_definitive_col, REACTION_TIME_COL, SUBJECT_ID_COL]).copy()
n2ac_trials_df.drop(columns=[pd_definitive_col])
n2ac_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\n2ac_latencies.csv', index=True)

# --- 2. Prepare single-trial data for Pd vs. RT ---
pd_trials_df = df.dropna(subset=[pd_definitive_col, REACTION_TIME_COL, SUBJECT_ID_COL]).copy()
pd_trials_df.drop(columns=[n2ac_definitive_col], inplace=True)
pd_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\pd_latencies.csv', index=True)

# Center variables
#pd_trials_df[pd_definitive_col] = pd_trials_df[pd_definitive_col] - pd_trials_df[pd_definitive_col].mean()
#n2ac_trials_df[n2ac_definitive_col] = n2ac_trials_df[n2ac_definitive_col] - n2ac_trials_df[n2ac_definitive_col].mean()

# Plot the stuff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig.suptitle('Single-Trial Brain-Behavior Relationship (LMM)', fontsize=18)

# --- N2ac LMM and Plot ---
try:
    print("\nFitting LMM for N2ac Latency vs. RT...")
    n2ac_formula = f"{REACTION_TIME_COL} ~ {n2ac_definitive_col}"
    n2ac_model = smf.mixedlm(n2ac_formula, n2ac_trials_df, groups=n2ac_trials_df[SUBJECT_ID_COL]).fit(reml=False)
    print(n2ac_model.summary())

    # Extract results for plotting
    beta = n2ac_model.fe_params[n2ac_definitive_col]
    p_val = n2ac_model.pvalues[n2ac_definitive_col]

    # Create regression plot
    sns.regplot(data=n2ac_trials_df, x=n2ac_definitive_col, y=REACTION_TIME_COL, ax=ax1,
                scatter_kws={'alpha': 0.1}) # Make points transparent to see density
    ax1.set_title('N2ac Latency vs. RT', fontsize=14)
    ax1.set_xlabel(f'N2ac Latency (s) at {definitive_percentage}%')
    ax1.set_ylabel('Reaction Time (s)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.text(0.05, 0.95, f'β = {beta:.3f}\np = {p_val:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
except Exception as e:
    print(f"Could not fit N2ac LMM. Error: {e}")
    ax1.text(0.5, 0.5, "LMM failed to converge", ha='center', va='center', transform=ax1.transAxes)


# --- Pd LMM and Plot ---
try:
    print("\nFitting LMM for Pd Latency vs. RT...")
    pd_formula = f"{REACTION_TIME_COL} ~ {pd_definitive_col}"
    pd_model = smf.mixedlm(pd_formula, pd_trials_df, groups=pd_trials_df[SUBJECT_ID_COL]).fit(reml=False)
    print(pd_model.summary())

    # Extract results for plotting
    beta = pd_model.fe_params[pd_definitive_col]
    p_val = pd_model.pvalues[pd_definitive_col]

    # Create regression plot
    sns.regplot(data=pd_trials_df, x=pd_definitive_col, y=REACTION_TIME_COL, ax=ax2, color='green',
                scatter_kws={'alpha': 0.1})
    ax2.set_title('Pd Latency vs. RT', fontsize=14)
    ax2.set_xlabel(f'Pd Latency (s) at {definitive_percentage}%')
    ax2.set_ylabel('') # Y-axis label is shared
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.text(0.05, 0.95, f'β = {beta:.3f}\np = {p_val:.3f}',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
except Exception as e:
    print(f"Could not fit Pd LMM. Error: {e}")
    ax2.text(0.5, 0.5, "LMM failed to converge", ha='center', va='center', transform=ax2.transAxes)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=True)
