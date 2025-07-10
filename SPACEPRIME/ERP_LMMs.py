import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
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
    #abs_erp_wave = np.abs(erp_wave)
    if is_target:
        abs_erp_wave = erp_wave * -1
    else:
        abs_erp_wave = erp_wave # do NOT use np.abs() here

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
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
PRIMING_COL = 'Priming'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_INT_COL = 'select_target_int'
BLOCK_COL = 'block'


# --- MERGED: ERP component columns for both Latency and Amplitude ---
ERP_N2AC_LATENCY_COL = 'N2ac_latency'
ERP_PD_LATENCY_COL = 'Pd_latency'
ERP_N2AC_AMPLITUDE_COL = 'N2ac_amplitude'
ERP_PD_AMPLITUDE_COL = 'Pd_amplitude'

# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(2)
PRIMING_REF_STR = PRIMING_MAP.get(0)

# --- 3. ERP Component Definitions ---
PD_TIME_WINDOW = (0.20, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_TIME_WINDOW = (0.18, 0.4)
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# --- Latency Robustness Check Configuration ---
PERCENTAGES_TO_TEST = [0.5] # Using 50% as the standard

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs()
df = epochs.metadata.copy()
print(f"Original number of trials: {len(df)}")

# --- Preprocessing Steps (unchanged) ---
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
# Map categorical variables to strings
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map(TARGET_LOC_MAP)
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(DISTRACTOR_LOC_MAP)
df[PRIMING_COL] = pd.to_numeric(df[PRIMING_COL], errors='coerce').map(PRIMING_MAP)
print("Preprocessing and column mapping complete.")

# --- ERP Data Reshaping (unchanged) ---
erp_df_picks_flat = [item for pair in set(N2AC_ELECTRODES + PD_ELECTRODES) for item in pair]
erp_df_picks_unique_flat = sorted(list(set(erp_df_picks_flat)))
erp_df = epochs.to_data_frame(picks=erp_df_picks_unique_flat, time_format=None)
print("Reshaping ERP data to wide format...")
erp_wide = erp_df.pivot(index='epoch', columns='time')
erp_wide = erp_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
erp_wide.columns = erp_wide.columns.droplevel(0)
merged_df = df.join(erp_wide)
print(f"Merged metadata with wide ERP data. Shape: {merged_df.shape}")

# --- Filter for N2ac and Pd trials (unchanged) ---
is_target_lateral = merged_df[TARGET_COL].isin(['left', 'right'])
is_distractor_lateral = merged_df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = merged_df[TARGET_COL] == TARGET_REF_STR
is_distractor_central = merged_df[DISTRACTOR_COL] == 'mid'
n2ac_analysis_df = merged_df[is_target_lateral & is_distractor_central].copy()
pd_analysis_df = merged_df[is_distractor_lateral & is_target_central].copy()
print(f"N2ac trials: {len(n2ac_analysis_df)}, Pd trials: {len(pd_analysis_df)}")

# --- MERGED: Single-Trial ERP Latency & Amplitude Calculation ---
print("\n--- Calculating Single-Trial ERP Latencies and Amplitudes ---")

# Initialize all new columns in the main dataframe `df`
df[ERP_N2AC_AMPLITUDE_COL] = np.nan
df[ERP_PD_AMPLITUDE_COL] = np.nan
for p in PERCENTAGES_TO_TEST:
    p_int = int(p * 100)
    df[f'{ERP_N2AC_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ERP_PD_LATENCY_COL}_{p_int}'] = np.nan

all_times = epochs.times
plot_erp_wave = False  # Set to True to debug/visualize a single trial's calculation

# --- N2ac Calculation Loop ---
print("\n--- Calculating N2ac Latencies & Amplitudes ---")
for subject_id in n2ac_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing N2ac for subject: {subject_id}...")
    subject_n2ac_df = n2ac_analysis_df[n2ac_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_n2ac_df.iterrows():
        jackknife_sample_df = subject_n2ac_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        n2ac_wave, n2ac_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[TARGET_COL],
            electrode_pairs=N2AC_ELECTRODES, time_window=N2AC_TIME_WINDOW, all_times=all_times
        )

        # 1. Calculate and store MEAN AMPLITUDE
        # N2ac is negative, so invert for easier interpretation (larger value = larger N2ac)
        mean_amplitude = np.mean(n2ac_wave) * -1
        df.loc[trial_idx, ERP_N2AC_AMPLITUDE_COL] = mean_amplitude

        # 2. Calculate and store LATENCY for each percentage
        for p in PERCENTAGES_TO_TEST:
            col_name = f'{ERP_N2AC_LATENCY_COL}_{int(p*100)}'
            latency = calculate_fractional_area_latency(
                n2ac_wave, n2ac_times, percentage=p, plot=plot_erp_wave, is_target=True
            )
            df.loc[trial_idx, col_name] = latency

# --- Pd Calculation Loop ---
print("\n--- Calculating Pd Latencies & Amplitudes ---")
for subject_id in pd_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing Pd for subject: {subject_id}...")
    subject_pd_df = pd_analysis_df[pd_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_pd_df.iterrows():
        jackknife_sample_df = subject_pd_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        pd_wave, pd_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[DISTRACTOR_COL],
            electrode_pairs=PD_ELECTRODES, time_window=PD_TIME_WINDOW, all_times=all_times
        )

        # 1. Calculate and store MEAN AMPLITUDE
        # Pd is positive, so no inversion needed.
        mean_amplitude = np.mean(pd_wave)
        df.loc[trial_idx, ERP_PD_AMPLITUDE_COL] = mean_amplitude

        # 2. Calculate and store LATENCY for each percentage
        for p in PERCENTAGES_TO_TEST:
            col_name = f'{ERP_PD_LATENCY_COL}_{int(p*100)}'
            latency = calculate_fractional_area_latency(
                pd_wave, pd_times, percentage=p, plot=plot_erp_wave, is_target=False
            )
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
    plt.show(block=False)

# --- Single-Trial Brain-Behavior Analysis (LMM) ---
print("\n--- Analyzing Single-Trial Brain-Behavior Relationship using LMMs ---")

# Choose a definitive percentage for this analysis. 50% is a standard choice.
definitive_percentage = 50
n2ac_definitive_col = f'{ERP_N2AC_LATENCY_COL}_{definitive_percentage}'
pd_definitive_col = f'{ERP_PD_LATENCY_COL}_{definitive_percentage}'

# --- 1. Prepare single-trial data for N2ac vs. RT ---
n2ac_trials_df = df.dropna(subset=[n2ac_definitive_col, REACTION_TIME_COL, SUBJECT_ID_COL]).copy()
n2ac_trials_df.drop(columns=[pd_definitive_col])
n2ac_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\n2ac_model.csv', index=True)

# --- 2. Prepare single-trial data for Pd vs. RT ---
pd_trials_df = df.dropna(subset=[pd_definitive_col, REACTION_TIME_COL, SUBJECT_ID_COL]).copy()
pd_trials_df.drop(columns=[n2ac_definitive_col], inplace=True)
pd_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\pd_model.csv', index=True)

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
plt.show(block=False)

# --- Single-Trial Brain-Behavior Analysis: 4 Separate Models ---
print("\n--- Analyzing Single-Trial Brain-Behavior Relationship: 4 Separate Models ---")
print("This analysis will build four separate models:")
print("1. N2ac Latency -> Reaction Time (LMM)")
print("2. N2ac Latency -> Accuracy (GLMM)")
print("3. Pd Latency   -> Reaction Time (LMM)")
print("4. Pd Latency   -> Accuracy (GLMM)")

# --- 1. General Setup ---
# Choose a definitive percentage for this analysis. 50% is a standard choice.
definitive_percentage = 50
n2ac_definitive_col = f'{ERP_N2AC_LATENCY_COL}_{definitive_percentage}'
pd_definitive_col = f'{ERP_PD_LATENCY_COL}_{definitive_percentage}'

# Optional: Center continuous predictors for better model convergence and interpretation.
# If you uncomment this, use 'df' in the models below, as the changes are in-place.
# df[n2ac_definitive_col] = df[n2ac_definitive_col].fillna(df[n2ac_definitive_col].mean()) - df[n2ac_definitive_col].mean()
# df[pd_definitive_col] = df[pd_definitive_col].fillna(df[pd_definitive_col].mean()) - df[pd_definitive_col].mean()
# df[BLOCK_COL] = df[BLOCK_COL] - df[BLOCK_COL].mean()


# --- 2. N2ac Latency Models ---
print("\n" + "="*25 + " N2ac Latency Models " + "="*25)

# On these trials, SingletonLoc is always 'mid', so it cannot be a predictor.
# We control for TargetLoc ('left' vs 'right') instead.
n2ac_formula_predictors = (f"{n2ac_definitive_col} + {BLOCK_COL} + "
                           f"C({PRIMING_COL}, Treatment('{PRIMING_REF_STR}')) + {ERP_N2AC_AMPLITUDE_COL} + "
                           f"C({TARGET_COL})")

# --- Model 1: N2ac -> Reaction Time (LMM) ---
print("\n--- Fitting Model 1: N2ac Latency -> Reaction Time (LMM) ---")
if not n2ac_trials_df.empty:
    try:
        # We add ACCURACY_INT_COL as a predictor for RT models
        n2ac_rt_formula = f"{REACTION_TIME_COL} ~ {n2ac_formula_predictors} + C({ACCURACY_INT_COL})"
        print(f"Formula: {n2ac_rt_formula}")

        n2ac_rt_model = smf.mixedlm(n2ac_rt_formula, n2ac_trials_df,
                                    groups=n2ac_trials_df[SUBJECT_ID_COL])
        n2ac_rt_fit = n2ac_rt_model.fit(reml=False)
        print(n2ac_rt_fit.summary())
    except Exception as e:
        print(f"Could not fit N2ac RT LMM. Error: {e}")
else:
    print("Skipping N2ac RT model: No data available.")


# --- 3. Pd Latency Models ---
print("\n" + "="*25 + " Pd Latency Models " + "="*25)

# On these trials, TargetLoc is always 'mid', so it cannot be a predictor.
# We control for SingletonLoc ('left' vs 'right') and its interaction with block.
pd_formula_predictors = (f"{pd_definitive_col} + {BLOCK_COL} + "
                         f"C({PRIMING_COL}, Treatment('{PRIMING_REF_STR}')) + "
                         f"C({DISTRACTOR_COL}) + {ERP_PD_AMPLITUDE_COL}")

# --- Model 3: Pd -> Reaction Time (LMM) ---
print("\n--- Fitting Model 3: Pd Latency -> Reaction Time (LMM) ---")
if not pd_trials_df.empty:
    try:
        # We add ACCURACY_INT_COL as a predictor for RT models
        pd_rt_formula = f"{REACTION_TIME_COL} ~ {pd_formula_predictors} + C({ACCURACY_INT_COL})"
        print(f"Formula: {pd_rt_formula}")

        pd_rt_model = smf.mixedlm(pd_rt_formula, pd_trials_df,
                                  groups=pd_trials_df[SUBJECT_ID_COL])
        # For LMMs, it's conventional to use REML (Restricted Maximum Likelihood)
        pd_rt_fit = pd_rt_model.fit(reml=True)
        print(pd_rt_fit.summary())
    except Exception as e:
        print(f"Could not fit Pd RT LMM. Error: {e}")
else:
    print("Skipping Pd RT model: No data available.")
