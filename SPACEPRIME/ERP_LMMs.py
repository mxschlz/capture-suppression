import matplotlib.pyplot as plt
import SPACEPRIME
from utils import get_jackknife_contra_ipsi_wave, calculate_fractional_area_latency
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from stats import remove_outliers, r_squared_mixed_model  # Assuming this is your custom outlier removal function
from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

plt.ion()

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
PD_TIME_WINDOW = (0.2, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_TIME_WINDOW = (0.2, 0.4)
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# --- Latency Robustness Check Configuration ---
PERCENTAGES_TO_TEST = [0.3, 0.5, 0.7] # Using 50% as the standard

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs()
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
            electrode_pairs=N2AC_ELECTRODES, time_window=N2AC_TIME_WINDOW, all_times=all_times)
        # Calculate and store LATENCY for each percentage
        for p in PERCENTAGES_TO_TEST:
            col_name_latency = f'{ERP_N2AC_LATENCY_COL}_{int(p*100)}'
            latency = calculate_fractional_area_latency(
                n2ac_wave, n2ac_times, percentage=p, plot=plot_erp_wave, is_target=True)
            df.loc[trial_idx, col_name_latency] = latency
            # Use np.interp for a more precise amplitude estimate between sample points.
            col_name_amplitude = f'{ERP_N2AC_AMPLITUDE_COL}_{int(p*100)}'
            # calculate amplitude at exactly that latency
            amplitude_at_latency = np.interp(latency, n2ac_times, n2ac_wave)
            # insert into dataframe location
            df.loc[trial_idx, col_name_amplitude] = amplitude_at_latency


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
            electrode_pairs=PD_ELECTRODES, time_window=PD_TIME_WINDOW, all_times=all_times)

        # Calculate and store LATENCY for each percentage
        for p in PERCENTAGES_TO_TEST:
            col_name = f'{ERP_PD_LATENCY_COL}_{int(p*100)}'
            latency = calculate_fractional_area_latency(
                pd_wave, pd_times, percentage=p, plot=plot_erp_wave, is_target=False)
            df.loc[trial_idx, col_name] = latency
            # Use np.interp for a more precise amplitude estimate between sample points.
            col_name_amplitude = f'{ERP_PD_AMPLITUDE_COL}_{int(p*100)}'
            # calculate amplitude at exactly that latency
            amplitude_at_latency = np.interp(latency, pd_times, pd_wave)
            # insert into dataframe location
            df.loc[trial_idx, col_name_amplitude] = amplitude_at_latency

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
# n2ac_trials_df.drop(columns=[pd_definitive_col, ERP_PD_AMPLITUDE_COL], inplace=True)
n2ac_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\n2ac_model.csv', index=True)

# --- 2. Prepare single-trial data for Pd vs. RT ---
pd_trials_df = df.dropna(subset=[pd_definitive_col, REACTION_TIME_COL, SUBJECT_ID_COL]).copy()
# pd_trials_df.drop(columns=[n2ac_definitive_col, ERP_N2AC_AMPLITUDE_COL], inplace=True)
pd_trials_df.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\pd_model.csv', index=True)

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

# Define the corresponding definitive amplitude columns
n2ac_definitive_amplitude_col = f'{ERP_N2AC_AMPLITUDE_COL}_{definitive_percentage}'
pd_definitive_amplitude_col = f'{ERP_PD_AMPLITUDE_COL}_{definitive_percentage}'
# --- 2. N2ac Latency Models ---
print("\n" + "="*25 + " N2ac Latency Models " + "="*25)

# On these trials, SingletonLoc is always 'mid', so it cannot be a predictor.
# We control for TargetLoc ('left' vs 'right') instead.
n2ac_formula_predictors = (f"{n2ac_definitive_col} + {BLOCK_COL} + "
                           f"C({PRIMING_COL}, Treatment('{PRIMING_REF_STR}')) + {n2ac_definitive_amplitude_col} + "
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
        n2ac_rt_fit = n2ac_rt_model.fit(reml=True)
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
                         f"C({DISTRACTOR_COL}) + {pd_definitive_amplitude_col}")

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
