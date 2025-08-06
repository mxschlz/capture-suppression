import matplotlib.pyplot as plt
import SPACEPRIME
from utils import get_jackknife_contra_ipsi_wave, calculate_fractional_area_latency, get_single_trial_contra_ipsi_wave
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from stats import remove_outliers  # Assuming this is your custom outlier removal function
from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

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


# --- MERGED: ERP component columns for both Latency and Amplitude ---
# Jackknifed metrics
ERP_N2AC_LATENCY_COL = 'jk_N2ac_latency'
ERP_PD_LATENCY_COL = 'jk_Pd_latency'
ERP_N2AC_AMPLITUDE_COL = 'jk_N2ac_amplitude'
ERP_PD_AMPLITUDE_COL = 'jk_Pd_amplitude'
# NEW: Single-Trial metrics
ST_ERP_N2AC_LATENCY_COL = 'st_N2ac_latency'
ST_ERP_PD_LATENCY_COL = 'st_Pd_latency'
ST_ERP_N2AC_AMPLITUDE_COL = 'st_N2ac_amplitude'
ST_ERP_PD_AMPLITUDE_COL = 'st_Pd_amplitude'


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

# --- Latency Robustness Check Configuration ---
PERCENTAGES_TO_TEST = [0.3, 0.5, 0.7] # Using 50% as the standard

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime_desc-csd")
df = epochs.metadata.copy()
sfreq = epochs.info["sfreq"]
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

# --- MERGED: Single-Trial ERP Latency & Amplitude Calculation ---
print("\n--- Calculating Single-Trial ERP Latencies and Amplitudes ---")

# Initialize all new columns in the main dataframe `df`
for p in PERCENTAGES_TO_TEST:
    p_int = int(p * 100)
    # Jackknifed columns
    df[f'{ERP_N2AC_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ERP_PD_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ERP_N2AC_AMPLITUDE_COL}_{p_int}'] = np.nan
    df[f'{ERP_PD_AMPLITUDE_COL}_{p_int}'] = np.nan
    # NEW: Single-trial columns
    df[f'{ST_ERP_N2AC_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ST_ERP_PD_LATENCY_COL}_{p_int}'] = np.nan
    df[f'{ST_ERP_N2AC_AMPLITUDE_COL}_{p_int}'] = np.nan
    df[f'{ST_ERP_PD_AMPLITUDE_COL}_{p_int}'] = np.nan


# NEW: Initialize columns for jackknifed RT
df['jackknifed_rt_n2ac'] = np.nan
df['jackknifed_rt_pd'] = np.nan
df['jackknifed_acc_n2ac'] = np.nan
df['jackknifed_acc_pd'] = np.nan
df['jackknifed_bis_n2ac'] = np.nan
df['jackknifed_bis_pd'] = np.nan

all_times = epochs.times
plot_erp_wave = False  # Set to True to debug/visualize a single trial's calculation

# --- N2ac Calculation Loop ---
print("\n--- Calculating N2ac Latencies & Amplitudes ---")

# These are the reference means and standard deviations for the entire N2ac dataset.
grand_mean_rt_n2ac = n2ac_analysis_df[REACTION_TIME_COL].mean()
grand_std_rt_n2ac = n2ac_analysis_df[REACTION_TIME_COL].std(ddof=1)
grand_mean_acc_n2ac = n2ac_analysis_df[ACCURACY_INT_COL].mean()
grand_std_acc_n2ac = n2ac_analysis_df[ACCURACY_INT_COL].std(ddof=1)

for subject_id in n2ac_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing N2ac for subject: {subject_id}...")
    subject_n2ac_df = n2ac_analysis_df[n2ac_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_n2ac_df.iterrows():
        # --- Jackknife Calculation ---
        jackknife_sample_df = subject_n2ac_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        jackknifed_rt = jackknife_sample_df[REACTION_TIME_COL].mean()
        df.loc[trial_idx, 'jackknifed_rt_n2ac'] = jackknifed_rt * -1
        jackknifed_acc = jackknife_sample_df[ACCURACY_INT_COL].mean()
        df.loc[trial_idx, 'jackknifed_acc_n2ac'] = jackknifed_acc * -1
        z_rt = (jackknifed_rt - grand_mean_rt_n2ac) / grand_std_rt_n2ac
        z_acc = (jackknifed_acc - grand_mean_acc_n2ac) / grand_std_acc_n2ac
        df.loc[trial_idx, 'jackknifed_bis_n2ac'] = (z_acc - z_rt) * -1

        n2ac_wave, n2ac_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[TARGET_COL],
            electrode_pairs=N2AC_ELECTRODES, time_window=N2AC_TIME_WINDOW, all_times=all_times)

        if n2ac_wave is not None:
            for p in PERCENTAGES_TO_TEST:
                col_name_latency = f'{ERP_N2AC_LATENCY_COL}_{int(p * 100)}'
                latency = calculate_fractional_area_latency(
                    n2ac_wave, n2ac_times, percentage=p, plot=plot_erp_wave, is_target=True)
                df.loc[trial_idx, col_name_latency] = latency

                col_name_amplitude = f'{ERP_N2AC_AMPLITUDE_COL}_{int(p * 100)}'
                if not np.isnan(latency):
                    amplitude_at_latency = np.interp(latency, n2ac_times, n2ac_wave)
                    df.loc[trial_idx, col_name_amplitude] = amplitude_at_latency

        # --- NEW: Single-Trial Calculation ---
        st_n2ac_wave, st_n2ac_times = get_single_trial_contra_ipsi_wave(
            trial_row=trial_row, electrode_pairs=N2AC_ELECTRODES, time_window=N2AC_TIME_WINDOW,
            all_times=all_times, lateral_stim_loc=trial_row[TARGET_COL]
        )

        if st_n2ac_wave is not None:
            for p in PERCENTAGES_TO_TEST:
                st_col_name_latency = f'{ST_ERP_N2AC_LATENCY_COL}_{int(p * 100)}'
                st_latency = calculate_fractional_area_latency(
                    st_n2ac_wave, st_n2ac_times, percentage=p, plot=False, is_target=True)
                df.loc[trial_idx, st_col_name_latency] = st_latency

                st_col_name_amplitude = f'{ST_ERP_N2AC_AMPLITUDE_COL}_{int(p * 100)}'
                if not np.isnan(st_latency):
                    st_amplitude = np.interp(st_latency, st_n2ac_times, st_n2ac_wave.astype(float))
                    df.loc[trial_idx, st_col_name_amplitude] = st_amplitude


# --- Pd Calculation Loop (Apply the same logic) ---
print("\n--- Calculating Pd Latencies & Amplitudes ---")

grand_mean_rt_pd = pd_analysis_df[REACTION_TIME_COL].mean()
grand_std_rt_pd = pd_analysis_df[REACTION_TIME_COL].std(ddof=1)
grand_mean_acc_pd = pd_analysis_df[ACCURACY_INT_COL].mean()
grand_std_acc_pd = pd_analysis_df[ACCURACY_INT_COL].std(ddof=1)

for subject_id in pd_analysis_df[SUBJECT_ID_COL].unique():
    print(f"Processing Pd for subject: {subject_id}...")
    subject_pd_df = pd_analysis_df[pd_analysis_df[SUBJECT_ID_COL] == subject_id]
    for trial_idx, trial_row in subject_pd_df.iterrows():
        # --- Jackknife Calculation ---
        jackknife_sample_df = subject_pd_df.drop(index=trial_idx)
        if jackknife_sample_df.empty: continue

        jackknifed_rt = jackknife_sample_df[REACTION_TIME_COL].mean()
        df.loc[trial_idx, 'jackknifed_rt_pd'] = jackknifed_rt * -1
        jackknifed_acc = jackknife_sample_df[ACCURACY_INT_COL].mean()
        df.loc[trial_idx, 'jackknifed_acc_pd'] = jackknifed_acc * -1
        z_rt = (jackknifed_rt - grand_mean_rt_pd) / grand_std_rt_pd
        z_acc = (jackknifed_acc - grand_mean_acc_pd) / grand_std_acc_pd
        df.loc[trial_idx, 'jackknifed_bis_pd'] = (z_acc - z_rt) * -1

        pd_wave, pd_times = get_jackknife_contra_ipsi_wave(
            sample_df=jackknife_sample_df, lateral_stim_loc=trial_row[DISTRACTOR_COL],
            electrode_pairs=PD_ELECTRODES, time_window=PD_TIME_WINDOW, all_times=all_times)

        if pd_wave is not None:
            for p in PERCENTAGES_TO_TEST:
                col_name_latency = f'{ERP_PD_LATENCY_COL}_{int(p * 100)}'
                latency = calculate_fractional_area_latency(
                    pd_wave, pd_times, percentage=p, plot=plot_erp_wave, is_target=False)
                df.loc[trial_idx, col_name_latency] = latency

                col_name_amplitude = f'{ERP_PD_AMPLITUDE_COL}_{int(p * 100)}'
                if not np.isnan(latency):
                    amplitude_at_latency = np.interp(latency, pd_times, pd_wave)
                    df.loc[trial_idx, col_name_amplitude] = amplitude_at_latency

        # --- NEW: Single-Trial Calculation ---
        st_pd_wave, st_pd_times = get_single_trial_contra_ipsi_wave(
            trial_row=trial_row, electrode_pairs=PD_ELECTRODES, time_window=PD_TIME_WINDOW,
            all_times=all_times, lateral_stim_loc=trial_row[DISTRACTOR_COL]
        )

        if st_pd_wave is not None:
            for p in PERCENTAGES_TO_TEST:
                st_col_name_latency = f'{ST_ERP_PD_LATENCY_COL}_{int(p * 100)}'
                st_latency = calculate_fractional_area_latency(
                    st_pd_wave, st_pd_times, percentage=p, plot=False, is_target=False)
                df.loc[trial_idx, st_col_name_latency] = st_latency

                st_col_name_amplitude = f'{ST_ERP_PD_AMPLITUDE_COL}_{int(p * 100)}'
                if not np.isnan(st_latency):
                    st_amplitude = np.interp(st_latency, st_pd_times, st_pd_wave.astype(float))
                    df.loc[trial_idx, st_col_name_amplitude] = st_amplitude

# --- Final Data Export ---
print("\n--- Saving Data to CSV ---")

# Split the data and save separately
print("\n--- Splitting and saving N2ac and Pd data separately ---")

# Use the indices from the analysis dataframes created earlier to select the correct rows
# from the final, fully-populated dataframe `df`.

# Create and save the N2ac-only dataframe
n2ac_final_df = df.loc[n2ac_analysis_df.index].copy()
n2ac_output_path = f'{SPACEPRIME.get_data_path()}concatenated\\n2ac_erp_behavioral_mixed_model_data.csv'
n2ac_final_df.to_csv(n2ac_output_path, index=True)
print(f"N2ac-specific data saved to:\n{n2ac_output_path}")

# Create and save the Pd-only dataframe
pd_final_df = df.loc[pd_analysis_df.index].copy()
pd_output_path = f'{SPACEPRIME.get_data_path()}concatenated\\pd_erp_behavioral_mixed_model_data.csv'
pd_final_df.to_csv(pd_output_path, index=True)
print(f"Pd-specific data saved to:\n{pd_output_path}")

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

# --- Statistical Analysis (LMM) ---
print("\n--- Running Linear Mixed Models ---")


def prepare_lmm_data(df, latency_col, amplitude_col, trial_num_col, rt_col, priming_col, subject_col):
    """
    Prepares the dataframe for LMM by selecting columns, dropping NaNs, and z-scoring.
    NOW INCLUDES AMPLITUDE AND TRIAL NUMBER.
    """
    # Select necessary columns for the model
    cols_to_select = [subject_col, rt_col, latency_col, amplitude_col, trial_num_col, priming_col]
    lmm_df = df[cols_to_select].copy()

    # Drop rows with missing values in any of the key columns
    cols_to_check_na = [rt_col, latency_col, amplitude_col, trial_num_col, priming_col]
    lmm_df.dropna(subset=cols_to_check_na, inplace=True)

    if lmm_df.empty:
        return None

    # Z-score (standardize) the continuous variables
    # This makes the model coefficients (betas) directly comparable
    lmm_df['z_rt'] = (lmm_df[rt_col] - lmm_df[rt_col].mean()) / lmm_df[rt_col].std()
    lmm_df['z_latency'] = (lmm_df[latency_col] - lmm_df[latency_col].mean()) / lmm_df[latency_col].std()
    lmm_df['z_amplitude'] = (lmm_df[amplitude_col] - lmm_df[amplitude_col].mean()) / lmm_df[amplitude_col].std()
    lmm_df['z_trial_num'] = (lmm_df[trial_num_col] - lmm_df[trial_num_col].mean()) / lmm_df[trial_num_col].std()

    return lmm_df


# --- 1. N2ac Model: Predicting Reaction Time ---
print("\n--- Model 1: N2ac Latency, Amplitude, & Trial Number -> Reaction Time ---")

# Define the predictor columns (using 50% fractional area metrics)
n2ac_latency_predictor_col = f'{ERP_N2AC_LATENCY_COL}_50'
n2ac_amplitude_predictor_col = f'{ERP_N2AC_AMPLITUDE_COL}_50'

# Prepare the data using our updated helper function
n2ac_lmm_df = prepare_lmm_data(
    df=n2ac_final_df,
    latency_col=n2ac_latency_predictor_col,
    amplitude_col=n2ac_amplitude_predictor_col,
    trial_num_col=TRIAL_NUMBER_COL,
    rt_col=REACTION_TIME_COL,
    priming_col=PRIMING_COL,
    subject_col=SUBJECT_ID_COL
)

if n2ac_lmm_df is not None and not n2ac_lmm_df.empty:
    try:
        # UPDATED FORMULA: Includes z_amplitude and z_trial_num as predictors.
        # We predict z-scored RT from z-scored latency, amplitude, trial number, and priming.
        formula = f"z_rt ~ z_latency + z_amplitude + z_trial_num + C({PRIMING_COL}, Treatment(reference='{PRIMING_REF_STR}'))"

        # Create and fit the model
        n2ac_model = smf.mixedlm(
            formula,
            n2ac_lmm_df,
            groups=n2ac_lmm_df[SUBJECT_ID_COL]
        )
        n2ac_results = n2ac_model.fit(reml=False)

        # Print the results summary
        print(n2ac_results.summary())

    except Exception as e:
        print(f"Could not fit N2ac LMM. Error: {e}")
else:
    print("Skipping N2ac LMM: Not enough data after cleaning.")


# --- 2. Pd Model: Predicting Reaction Time ---
print("\n--- Model 2: Pd Latency, Amplitude, & Trial Number -> Reaction Time ---")

# Define the predictor columns (using 50% fractional area metrics)
pd_latency_predictor_col = f'{ERP_PD_LATENCY_COL}_50'
pd_amplitude_predictor_col = f'{ERP_PD_AMPLITUDE_COL}_50'

# Prepare the data
pd_lmm_df = prepare_lmm_data(
    df=pd_final_df,
    latency_col=pd_latency_predictor_col,
    amplitude_col=pd_amplitude_predictor_col,
    trial_num_col=TRIAL_NUMBER_COL,
    rt_col=REACTION_TIME_COL,
    priming_col=PRIMING_COL,
    subject_col=SUBJECT_ID_COL
)

if pd_lmm_df is not None and not pd_lmm_df.empty:
    try:
        # UPDATED FORMULA: Includes z_amplitude and z_trial_num as predictors.
        formula = f"z_rt ~ z_latency + z_amplitude + z_trial_num + C({PRIMING_COL}, Treatment(reference='{PRIMING_REF_STR}'))"

        # Create and fit the model
        pd_model = smf.mixedlm(
            formula,
            pd_lmm_df,
            groups=pd_lmm_df[SUBJECT_ID_COL]
        )
        pd_results = pd_model.fit(reml=False)

        # Print the results summary
        print(pd_results.summary())

    except Exception as e:
        print(f"Could not fit Pd LMM. Error: {e}")
else:
    print("Skipping Pd LMM: Not enough data after cleaning.")
