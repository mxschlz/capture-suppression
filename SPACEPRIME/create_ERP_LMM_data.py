import matplotlib.pyplot as plt
import SPACEPRIME
from utils import get_jackknife_contra_ipsi_wave, calculate_fractional_area_latency, get_single_trial_contra_ipsi_wave
import pandas as pd
import seaborn as sns
import numpy as np
from stats import remove_outliers, add_within_between_predictors  # Assuming this is your custom outlier removal function

plt.ion()

# --- Script Configuration Parameters ---

# --- 1. Data Loading & Preprocessing ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = None

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
# NEW: Electrode column name
ELECTRODE_COL = 'electrode'


# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(2)
PRIMING_REF_STR = PRIMING_MAP.get(0)

# --- 3. ERP Component Definitions ---
PD_TIME_WINDOW = (0.2, 0.4)
N2AC_TIME_WINDOW = (0.2, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4")]
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4")]
# NEW: Define reference electrode pair for LMM
N2AC_ELECTRODE_REF = f"{N2AC_ELECTRODES[0][0]}-{N2AC_ELECTRODES[0][1]}"
PD_ELECTRODE_REF = f"{PD_ELECTRODES[0][0]}-{PD_ELECTRODES[0][1]}"


# --- Latency Robustness Check Configuration ---
PERCENTAGES_TO_TEST = [0.3, 0.5, 0.7] # Using 50% as the standard

# --- Main Script ---
print("Loading and concatenating epochs...")
paradigm = "spaceprime"
epochs = SPACEPRIME.load_concatenated_epochs(paradigm)
df = epochs.metadata.copy()
sfreq = epochs.info["sfreq"]
print(f"Original number of trials: {len(df)}")

# --- Preprocessing Steps (largely unchanged) ---
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
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

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


# --- MODIFIED: ERP Calculation Section ---
# We will now build lists of dictionaries, which will be converted to long-format DataFrames.
# This is more efficient and cleaner than trying to populate a pre-allocated wide dataframe.
n2ac_results_list = []
pd_results_list = []

all_times = epochs.times
plot_erp_wave = False  # Set to True to debug/visualize a single trial's calculation

# --- N2ac Calculation Loop ---
print("\n--- Calculating N2ac Latencies & Amplitudes (per electrode) ---")
grand_mean_rt_n2ac = n2ac_analysis_df[REACTION_TIME_COL].mean()
grand_std_rt_n2ac = n2ac_analysis_df[REACTION_TIME_COL].std(ddof=1)
grand_mean_acc_n2ac = n2ac_analysis_df[ACCURACY_INT_COL].mean()
grand_std_acc_n2ac = n2ac_analysis_df[ACCURACY_INT_COL].std(ddof=1)

for subject_id in n2ac_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing N2ac for subject: {subject_id}...")
    subject_n2ac_df = n2ac_analysis_df[n2ac_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_n2ac_df.iterrows():
        jackknife_sample_df = subject_n2ac_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        # Calculate trial-level jackknifed behavioral metrics once
        jackknifed_rt = jackknife_sample_df[REACTION_TIME_COL].mean()
        jackknifed_acc = jackknife_sample_df[ACCURACY_INT_COL].mean()
        z_rt = (jackknifed_rt - grand_mean_rt_n2ac) / grand_std_rt_n2ac
        z_acc = (jackknifed_acc - grand_mean_acc_n2ac) / grand_std_acc_n2ac
        jackknifed_bis = (z_acc - z_rt)

        # NEW: Loop over each electrode pair to calculate metrics individually
        for electrode_pair in N2AC_ELECTRODES:
            result_row = {
                SUBJECT_ID_COL: subject_id,
                REACTION_TIME_COL: trial_row[REACTION_TIME_COL],
                ACCURACY_INT_COL: trial_row[ACCURACY_INT_COL],
                PRIMING_COL: trial_row[PRIMING_COL],
                TRIAL_NUMBER_COL: trial_row[TRIAL_NUMBER_COL],
                ELECTRODE_COL: f"{electrode_pair[0]}-{electrode_pair[1]}",
                'jackknifed_rt': jackknifed_rt * -1,
                'jackknifed_acc': jackknifed_acc * -1,
                'jackknifed_bis': jackknifed_bis * -1,
            }

            # Jackknife ERP calculation for the specific pair
            n2ac_wave, n2ac_times = get_jackknife_contra_ipsi_wave(
                sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[TARGET_COL],
                electrode_pairs=[electrode_pair],  # MODIFIED: Pass only the current pair
                time_window=N2AC_TIME_WINDOW, all_times=all_times)

            # Single-Trial ERP calculation for the specific pair
            st_n2ac_wave, st_n2ac_times = get_single_trial_contra_ipsi_wave(
                trial_row=trial_row, lateral_stim_loc=trial_row[TARGET_COL],
                electrode_pairs=[electrode_pair], # MODIFIED: Pass only the current pair
                time_window=N2AC_TIME_WINDOW, all_times=all_times
            )

            for p in PERCENTAGES_TO_TEST:
                p_int = int(p * 100)
                # Jackknife metrics
                jk_lat = calculate_fractional_area_latency(n2ac_wave, n2ac_times, percentage=p, plot=plot_erp_wave, is_target=True) if n2ac_wave is not None else np.nan
                jk_amp = np.interp(jk_lat, n2ac_times, n2ac_wave) if not np.isnan(jk_lat) else np.nan
                result_row[f'jk_latency_{p_int}'] = jk_lat * -1
                result_row[f'jk_amplitude_{p_int}'] = jk_amp * -1

                # Single-trial metrics
                st_lat = calculate_fractional_area_latency(st_n2ac_wave, st_n2ac_times, percentage=p, plot=False, is_target=True) if st_n2ac_wave is not None else np.nan
                st_amp = np.interp(st_lat, st_n2ac_times, st_n2ac_wave.astype(float)) if not np.isnan(st_lat) else np.nan
                result_row[f'st_latency_{p_int}'] = st_lat
                result_row[f'st_amplitude_{p_int}'] = st_amp

            n2ac_results_list.append(result_row)


# --- Pd Calculation Loop (Apply the same logic) ---
print("\n--- Calculating Pd Latencies & Amplitudes (per electrode) ---")
grand_mean_rt_pd = pd_analysis_df[REACTION_TIME_COL].mean()
grand_std_rt_pd = pd_analysis_df[REACTION_TIME_COL].std(ddof=1)
grand_mean_acc_pd = pd_analysis_df[ACCURACY_INT_COL].mean()
grand_std_acc_pd = pd_analysis_df[ACCURACY_INT_COL].std(ddof=1)

for subject_id in pd_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing Pd for subject: {subject_id}...")
    subject_pd_df = pd_analysis_df[pd_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_pd_df.iterrows():
        jackknife_sample_df = subject_pd_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        jackknifed_rt = jackknife_sample_df[REACTION_TIME_COL].mean()
        jackknifed_acc = jackknife_sample_df[ACCURACY_INT_COL].mean()
        z_rt = (jackknifed_rt - grand_mean_rt_pd) / grand_std_rt_pd
        z_acc = (jackknifed_acc - grand_mean_acc_pd) / grand_std_acc_pd
        jackknifed_bis = (z_acc - z_rt)

        # NEW: Loop over each electrode pair
        for electrode_pair in PD_ELECTRODES:
            result_row = {
                SUBJECT_ID_COL: subject_id,
                REACTION_TIME_COL: trial_row[REACTION_TIME_COL],
                ACCURACY_INT_COL: trial_row[ACCURACY_INT_COL],
                PRIMING_COL: trial_row[PRIMING_COL],
                TRIAL_NUMBER_COL: trial_row[TRIAL_NUMBER_COL],
                ELECTRODE_COL: f"{electrode_pair[0]}-{electrode_pair[1]}",
                'jackknifed_rt': jackknifed_rt * -1,
                'jackknifed_acc': jackknifed_acc * -1,
                'jackknifed_bis': jackknifed_bis * -1,
            }

            # Jackknife ERP calculation for the specific pair
            pd_wave, pd_times = get_jackknife_contra_ipsi_wave(
                sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[DISTRACTOR_COL],
                electrode_pairs=[electrode_pair], # MODIFIED
                time_window=PD_TIME_WINDOW, all_times=all_times)

            # Single-Trial ERP calculation for the specific pair
            st_pd_wave, st_pd_times = get_single_trial_contra_ipsi_wave(
                trial_row=trial_row, lateral_stim_loc=trial_row[DISTRACTOR_COL],
                electrode_pairs=[electrode_pair], # MODIFIED
                time_window=PD_TIME_WINDOW, all_times=all_times
            )

            for p in PERCENTAGES_TO_TEST:
                p_int = int(p * 100)
                # Jackknife metrics
                jk_lat = calculate_fractional_area_latency(pd_wave, pd_times, percentage=p, plot=plot_erp_wave, is_target=False) if pd_wave is not None else np.nan
                jk_amp = np.interp(jk_lat, pd_times, pd_wave) if not np.isnan(jk_lat) else np.nan
                result_row[f'jk_latency_{p_int}'] = jk_lat * -1
                result_row[f'jk_amplitude_{p_int}'] = jk_amp * -1

                # Single-trial metrics
                st_lat = calculate_fractional_area_latency(st_pd_wave, st_pd_times, percentage=p, plot=False, is_target=False) if st_pd_wave is not None else np.nan
                st_amp = np.interp(st_lat, st_pd_times, st_pd_wave.astype(float)) if not np.isnan(st_lat) else np.nan
                result_row[f'st_latency_{p_int}'] = st_lat
                result_row[f'st_amplitude_{p_int}'] = st_amp

            pd_results_list.append(result_row)

# --- MODIFIED: Final Data Export ---
print("\n--- Saving Long-Format Data to CSV ---")

# Convert the lists of dictionaries into DataFrames
n2ac_final_df = pd.DataFrame(n2ac_results_list)
pd_final_df = pd.DataFrame(pd_results_list)

# --- NEW: Split predictors into within- and between-subject components ---
n2ac_final_df = add_within_between_predictors(n2ac_final_df, SUBJECT_ID_COL, PERCENTAGES_TO_TEST)
pd_final_df = add_within_between_predictors(pd_final_df, SUBJECT_ID_COL, PERCENTAGES_TO_TEST)

# Save the new long-format dataframes
n2ac_output_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_n2ac_erp_behavioral_lmm_long_data_between-within.csv'
n2ac_final_df.to_csv(n2ac_output_path, index=False)
print(f"N2ac-specific long-format data saved to:\n{n2ac_output_path}")

# Note: Corrected the filename for Pd to be distinct from N2ac
pd_output_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_pd_erp_behavioral_lmm_long_data_between-within.csv'
pd_final_df.to_csv(pd_output_path, index=False)
print(f"Pd-specific long-format data saved to:\n{pd_output_path}")


# --- NEW: Calculate and Save Subject Averages for Correlation Analysis ---
print("\n--- Calculating and Saving Subject-Averaged Data ---")

def calculate_and_save_subject_averages(df, component_name, paradigm_name):
    """
    Calculates subject-level averages grouped by condition and saves to a CSV file.

    This function groups the data by subject, priming condition, and electrode,
    then calculates the mean for all numeric metrics. The resulting dataframe is
    saved to a new CSV file, suitable for import into statistical software like
    JASP or jamovi.
    """
    if df.empty:
        print(f"Skipping subject-average calculation for {component_name}: DataFrame is empty.")
        return

    # Define grouping variables. We want averages per subject, per condition, per electrode.
    grouping_vars = [SUBJECT_ID_COL, PRIMING_COL, ELECTRODE_COL]

    # Ensure all grouping columns exist in the dataframe
    if not all(col in df.columns for col in grouping_vars):
        print(f"Skipping subject-average calculation for {component_name}: Missing one or more grouping columns.")
        return

    # Group by subject, priming, and electrode, then calculate the mean of all other numeric columns.
    # as_index=False keeps the grouping variables as columns.
    subject_avg_df = df.groupby(grouping_vars, as_index=False).mean()

    # Construct a descriptive output filename
    output_path = (f'{SPACEPRIME.get_data_path()}concatenated\\'
                   f'{paradigm_name}_{component_name}_subject_averages.csv')

    # Save to CSV
    subject_avg_df.to_csv(output_path, index=False)
    print(f"Subject-averaged {component_name} data saved to:\n{output_path}")
    print(f"Data shape: {subject_avg_df.shape}")


# Calculate and save for N2ac
calculate_and_save_subject_averages(
    df=n2ac_final_df,
    component_name='n2ac',
    paradigm_name=paradigm
)

# Calculate and save for Pd
calculate_and_save_subject_averages(
    df=pd_final_df,
    component_name='pd',
    paradigm_name=paradigm
)

# --- MODIFIED: Analysis of Latency Robustness ---
print("\n--- Analyzing Robustness of Latency Calculation (by Electrode) ---")

def prepare_robustness_df(df, component_name):
    """Melts the dataframe for robustness plotting."""
    latency_cols = [f'jk_latency_{int(p*100)}' for p in PERCENTAGES_TO_TEST]
    id_vars = [ELECTRODE_COL]
    value_vars = [col for col in latency_cols if col in df.columns]
    if not value_vars:
        return pd.DataFrame() # Return empty if no latency columns found

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Threshold', value_name='Latency (s)')
    long_df['Threshold'] = long_df['Threshold'].str.extract(r'_(\d+)').astype(int)
    long_df.dropna(inplace=True)
    return long_df

n2ac_robust_df = prepare_robustness_df(n2ac_final_df, 'N2ac')
pd_robust_df = prepare_robustness_df(pd_final_df, 'Pd')

n2ac_has_data = not n2ac_robust_df.empty
pd_has_data = not pd_robust_df.empty

if n2ac_has_data or pd_has_data:
    num_plots = sum([n2ac_has_data, pd_has_data])
    fig, axes = plt.subplots(1, num_plots, figsize=(9 * num_plots, 8), sharey=True, squeeze=False)
    ax_idx = 0
    fig.suptitle('Distribution of ERP Latencies by Fractional Area Threshold and Electrode', fontsize=18)

    if n2ac_has_data:
        ax = axes[0, ax_idx]
        sns.boxplot(data=n2ac_robust_df, x='Threshold', y='Latency (s)', hue=ELECTRODE_COL, ax=ax)
        ax.set_title('N2ac Latencies', fontsize=14)
        ax.set_xlabel('Threshold Percentage (%)')
        ax.set_ylabel('Calculated Latency (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(title='Electrode')
        ax_idx += 1

    if pd_has_data:
        ax = axes[0, ax_idx]
        sns.boxplot(data=pd_robust_df, x='Threshold', y='Latency (s)', hue=ELECTRODE_COL, ax=ax)
        ax.set_title('Pd Latencies', fontsize=14)
        ax.set_xlabel('Threshold Percentage (%)')
        if ax_idx == 0: # Only set Y label if it's the first plot
            ax.set_ylabel('Calculated Latency (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(title='Electrode')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("No data available for latency robustness plots.")
