import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from stats import remove_outliers  # Assuming this is your custom outlier removal function
from patsy.contrasts import Treatment  # Import Treatment for specifying reference levels

plt.ion()

# --- Script Configuration Parameters ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = None

WINDOW_DURATION_S = 0.100  # Duration of the running average window in seconds (e.g., 100ms)

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
PRIMING_COL = 'Priming'
ERP_N2AC_COL = 'N2ac'
ERP_PD_COL = 'Pd'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_INT_COL = 'select_target_int'

# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}

TARGET_REF_VAL_ORIGINAL = 2
TARGET_REF_STR = TARGET_LOC_MAP.get(TARGET_REF_VAL_ORIGINAL)

DISTRACTOR_REF_VAL_ORIGINAL = 2
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(DISTRACTOR_REF_VAL_ORIGINAL)

PRIMING_REF_VAL_ORIGINAL = 0
PRIMING_REF_STR = PRIMING_MAP.get(PRIMING_REF_VAL_ORIGINAL)

# --- 3. ERP Component Definitions ---
# Define the electrode pairs. Both N2ac and Pd will be calculated as the
# mean of the differences (Left - Right) from these pairs.
ERP_ELECTRODE_PAIRS = [
    ("FC3", "FC4"),
    ("FC5", "FC6"),
    ("C3", "C4"),
    ("C5", "C6"),
    ("CP3", "CP4"),
    ("CP5", "CP6")
]

# --- 4. LMM Predictor Variables ---
LMM_N2AC_PD_CONTINUOUS_PREDICTORS = [TRIAL_NUMBER_COL, REACTION_TIME_COL]
LMM_N2AC_PD_CATEGORICAL_PREDICTORS = [
    TARGET_COL,
    DISTRACTOR_COL,
    PRIMING_COL,
    ACCURACY_INT_COL
]

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs()
print("Epochs loaded and cropped.")

df = epochs.metadata.copy()

# --- Metadata Preprocessing (largely unchanged) ---
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
elif FILTER_PHASE is not None:
    print(f"Warning: Column '{PHASE_COL}' not found for filtering.")

if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
else:
    print(f"Warning: Column '{REACTION_TIME_COL}' not found for outlier removal.")

if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
else:
    print(f"Warning: '{SUBJECT_ID_COL}' column not found. Cannot create '{TRIAL_NUMBER_COL}'.")

if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
else:
    print(f"Warning: Accuracy column '{ACCURACY_COL}' not found. Cannot create '{ACCURACY_INT_COL}'.")

if TARGET_COL in df.columns:
    numeric_target_col = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df[TARGET_COL] = numeric_target_col.map(TARGET_LOC_MAP)
    if TARGET_REF_STR is None or TARGET_REF_STR not in df[TARGET_COL].dropna().unique():
        print(f"Warning: Target reference '{TARGET_REF_STR}' issue. Available: {df[TARGET_COL].dropna().unique()}")
else:
    raise ValueError(f"Error: Critical column '{TARGET_COL}' not found in metadata.")

if DISTRACTOR_COL in df.columns:
    numeric_distractor_col = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce')
    df[DISTRACTOR_COL] = numeric_distractor_col.map(DISTRACTOR_LOC_MAP)
    if DISTRACTOR_REF_STR is None or DISTRACTOR_REF_STR not in df[DISTRACTOR_COL].dropna().unique():
        print(f"Warning: Distractor reference '{DISTRACTOR_REF_STR}' issue. Available: {df[DISTRACTOR_COL].dropna().unique()}")
else:
    raise ValueError(f"Error: Critical column '{DISTRACTOR_COL}' not found in metadata.")

if PRIMING_COL in df.columns:
    numeric_priming_col = pd.to_numeric(df[PRIMING_COL], errors='coerce')
    df[PRIMING_COL] = numeric_priming_col.map(PRIMING_MAP)
    if PRIMING_REF_STR is None or PRIMING_REF_STR not in df[PRIMING_COL].dropna().unique():
        print(f"Warning: Priming reference '{PRIMING_REF_STR}' issue. Available: {df[PRIMING_COL].dropna().unique()}")
else:
    print(f"Warning: Priming column '{PRIMING_COL}' not found.")

df_base = df.copy()

print("-" * 30)
print(f"Starting time-course LMM analysis with {WINDOW_DURATION_S}ms running average window...")

# --- ERP Data Extraction ---
# Collect all unique electrodes from the defined pairs
all_electrodes_from_pairs = []
for left, right in ERP_ELECTRODE_PAIRS:
    all_electrodes_from_pairs.append(left)
    all_electrodes_from_pairs.append(right)
erp_df_picks = list(set(all_electrodes_from_pairs))

all_times_erp_df = epochs.to_data_frame(picks=erp_df_picks, time_format=None) # time is in seconds
trial_interval = epochs.times

# --- Storage for Z-values and Time Points ---
time_points_for_plot = []
# Initialize dictionaries for z-values (structure remains the same)
n2ac_target_z_values = {key: [] for _, key in TARGET_LOC_MAP.items() if key != TARGET_REF_STR}
n2ac_distractor_z_values = {key: [] for _, key in DISTRACTOR_LOC_MAP.items() if key != DISTRACTOR_REF_STR}
pd_distractor_z_values = {key: [] for _, key in DISTRACTOR_LOC_MAP.items() if key != DISTRACTOR_REF_STR}
pd_target_z_values = {key: [] for _, key in TARGET_LOC_MAP.items() if key != TARGET_REF_STR}


# --- Main Loop Over Time Points (Window Centers) ---
for center_time_point in trial_interval:
    print(f"\nProcessing window centered at: {center_time_point:.4f} s / {trial_interval[-1]:.4f} s")
    time_points_for_plot.append(center_time_point)

    window_start_time = center_time_point - (WINDOW_DURATION_S / 2)
    window_end_time = center_time_point + (WINDOW_DURATION_S / 2)
    window_start_time = max(window_start_time, trial_interval[0])
    window_end_time = min(window_end_time, trial_interval[-1])

    erp_data_in_window = all_times_erp_df[
        (all_times_erp_df['time'] >= window_start_time) &
        (all_times_erp_df['time'] <= window_end_time)
    ]

    if erp_data_in_window.empty:
        print(f"Warning: No ERP data in window [{window_start_time:.4f}s - {window_end_time:.4f}s]. Appending NaNs.")
        for key_list in [n2ac_target_z_values, n2ac_distractor_z_values, pd_distractor_z_values, pd_target_z_values]:
            for key in key_list:
                key_list[key].append(np.nan)
        continue

    # Average ERP data for all picked electrodes across time points within the window for each epoch
    electrode_means_in_window = erp_data_in_window.groupby('epoch')[erp_df_picks].mean()

    df_current_iter = df_base.copy()
    df_current_iter = df_current_iter.merge(electrode_means_in_window, left_index=True, right_index=True, how='left')

    # Calculate ERP component values (N2ac, Pd) from the window-averaged electrode data
    df_current_iter[ERP_N2AC_COL] = np.nan
    df_current_iter[ERP_PD_COL] = np.nan

    # Calculate N2ac and Pd using the mean of differences from ERP_ELECTRODE_PAIRS
    # Assuming both components are calculated with the same logic from these pairs
    for component_col_name in [ERP_N2AC_COL, ERP_PD_COL]:
        pair_differences_series_list = []
        for left_elec, right_elec in ERP_ELECTRODE_PAIRS:
            if left_elec in df_current_iter.columns and right_elec in df_current_iter.columns:
                # Ensure data is numeric before subtraction, handle potential NaNs from merge
                left_data = pd.to_numeric(df_current_iter[left_elec], errors='coerce')
                right_data = pd.to_numeric(df_current_iter[right_elec], errors='coerce')
                difference = left_data - right_data
                pair_differences_series_list.append(difference.rename(f"{left_elec}-{right_elec}")) # Name series for clarity if debugging
            else:
                print(f"Warning: Electrodes {left_elec} or {right_elec} not found in df_current_iter for {component_col_name} calculation at t={center_time_point:.3f}s.")

        if pair_differences_series_list:
            all_pair_differences_df = pd.concat(pair_differences_series_list, axis=1)
            # Calculate the mean of the differences (row-wise mean across the pair-difference columns)
            df_current_iter[component_col_name] = all_pair_differences_df.mean(axis=1, skipna=True)
        else:
            print(f"Warning: No valid electrode pairs processed for {component_col_name} at t={center_time_point:.3f}s. Column will be NaNs.")

    # --- LMM Fitting (N2ac and Pd) ---
    # The LMM fitting logic for N2ac and Pd remains largely the same,
    # using df_current_iter which now has ERP_N2AC_COL and ERP_PD_COL
    # calculated based on the mean of multiple electrode pair differences.

    # --- 2. N2ac LMM ---
    n2ac_model_cols_iter = [ERP_N2AC_COL, SUBJECT_ID_COL] + \
                           LMM_N2AC_PD_CONTINUOUS_PREDICTORS + \
                           LMM_N2AC_PD_CATEGORICAL_PREDICTORS
    df_n2ac_model_iter = df_current_iter[list(set(n2ac_model_cols_iter))].dropna(
        subset=[ERP_N2AC_COL] + LMM_N2AC_PD_CONTINUOUS_PREDICTORS + LMM_N2AC_PD_CATEGORICAL_PREDICTORS)

    if SUBJECT_ID_COL in df_n2ac_model_iter:
        df_n2ac_model_iter[SUBJECT_ID_COL] = df_n2ac_model_iter[SUBJECT_ID_COL].astype(str)

    temp_n2ac_target_z = {key: np.nan for key in n2ac_target_z_values}
    temp_n2ac_dist_z = {key: np.nan for key in n2ac_distractor_z_values}

    if not df_n2ac_model_iter.empty and df_n2ac_model_iter[ERP_N2AC_COL].notna().any() and len(
            df_n2ac_model_iter[SUBJECT_ID_COL].unique()) > 1:
        formula_parts_n2ac_iter = [f"{ERP_N2AC_COL} ~"]
        formula_parts_n2ac_iter.extend(LMM_N2AC_PD_CONTINUOUS_PREDICTORS)
        for cat_pred in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
            ref_str_n2ac = None
            if cat_pred == PRIMING_COL: ref_str_n2ac = PRIMING_REF_STR
            elif cat_pred == TARGET_COL: ref_str_n2ac = TARGET_REF_STR
            elif cat_pred == DISTRACTOR_COL: ref_str_n2ac = DISTRACTOR_REF_STR
            treatment_str_n2ac = ""
            if cat_pred != ACCURACY_INT_COL:
                if ref_str_n2ac and ref_str_n2ac in df_n2ac_model_iter[cat_pred].dropna().unique():
                    treatment_str_n2ac = f", Treatment(reference='{ref_str_n2ac}')"
            formula_parts_n2ac_iter.append(f"C({cat_pred}{treatment_str_n2ac})")
        n2ac_full_formula_iter = " + ".join(formula_parts_n2ac_iter)

        try:
            n2ac_lmm_iter = smf.mixedlm(formula=n2ac_full_formula_iter, data=df_n2ac_model_iter, groups=SUBJECT_ID_COL)
            n2ac_lmm_results_iter = n2ac_lmm_iter.fit(reml=True, method=['lbfgs'])
            coeffs_n2ac = pd.read_html(n2ac_lmm_results_iter.summary().tables[1].to_html(), header=0, index_col=0)[0]

            for target_level_str in n2ac_target_z_values.keys():
                coeff_name_treat = f"C({TARGET_COL}, Treatment(reference='{TARGET_REF_STR}'))[T.{target_level_str}]"
                coeff_name_default_ref = f"C({TARGET_COL})[T.{target_level_str}]"
                if coeff_name_treat in coeffs_n2ac.index:
                    temp_n2ac_target_z[target_level_str] = coeffs_n2ac.loc[coeff_name_treat, 'z']
                elif coeff_name_default_ref in coeffs_n2ac.index:
                    temp_n2ac_target_z[target_level_str] = coeffs_n2ac.loc[coeff_name_default_ref, 'z']

            for dist_level_str in n2ac_distractor_z_values.keys():
                coeff_name_treat = f"C({DISTRACTOR_COL}, Treatment(reference='{DISTRACTOR_REF_STR}'))[T.{dist_level_str}]"
                coeff_name_default_ref = f"C({DISTRACTOR_COL})[T.{dist_level_str}]"
                if coeff_name_treat in coeffs_n2ac.index:
                    temp_n2ac_dist_z[dist_level_str] = coeffs_n2ac.loc[coeff_name_treat, 'z']
                elif coeff_name_default_ref in coeffs_n2ac.index:
                    temp_n2ac_dist_z[dist_level_str] = coeffs_n2ac.loc[coeff_name_default_ref, 'z']
        except Exception as e:
            print(f"Error fitting N2ac LMM for window t={center_time_point:.3f}s: {e}")

    for key, val in temp_n2ac_target_z.items(): n2ac_target_z_values[key].append(val)
    for key, val in temp_n2ac_dist_z.items(): n2ac_distractor_z_values[key].append(val)

    # --- 3. Pd LMM ---
    pd_model_cols_iter = [ERP_PD_COL, SUBJECT_ID_COL] + \
                         LMM_N2AC_PD_CONTINUOUS_PREDICTORS + \
                         LMM_N2AC_PD_CATEGORICAL_PREDICTORS
    df_pd_model_iter = df_current_iter[list(set(pd_model_cols_iter))].dropna(
        subset=[ERP_PD_COL] + LMM_N2AC_PD_CONTINUOUS_PREDICTORS + LMM_N2AC_PD_CATEGORICAL_PREDICTORS)

    if SUBJECT_ID_COL in df_pd_model_iter:
        df_pd_model_iter[SUBJECT_ID_COL] = df_pd_model_iter[SUBJECT_ID_COL].astype(str)

    temp_pd_dist_z = {key: np.nan for key in pd_distractor_z_values}
    temp_pd_target_z = {key: np.nan for key in pd_target_z_values}

    if not df_pd_model_iter.empty and df_pd_model_iter[ERP_PD_COL].notna().any() and len(
            df_pd_model_iter[SUBJECT_ID_COL].unique()) > 1:
        formula_parts_pd_iter = [f"{ERP_PD_COL} ~"]
        formula_parts_pd_iter.extend(LMM_N2AC_PD_CONTINUOUS_PREDICTORS)
        for cat_pred in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
            ref_str_pd = None
            if cat_pred == PRIMING_COL: ref_str_pd = PRIMING_REF_STR
            elif cat_pred == TARGET_COL: ref_str_pd = TARGET_REF_STR
            elif cat_pred == DISTRACTOR_COL: ref_str_pd = DISTRACTOR_REF_STR
            treatment_str_pd = ""
            if cat_pred != ACCURACY_INT_COL:
                if ref_str_pd and ref_str_pd in df_pd_model_iter[cat_pred].dropna().unique():
                    treatment_str_pd = f", Treatment(reference='{ref_str_pd}')"
            formula_parts_pd_iter.append(f"C({cat_pred}{treatment_str_pd})")
        pd_full_formula_iter = " + ".join(formula_parts_pd_iter)

        try:
            pd_lmm_iter = smf.mixedlm(formula=pd_full_formula_iter, data=df_pd_model_iter, groups=SUBJECT_ID_COL)
            pd_lmm_results_iter = pd_lmm_iter.fit(reml=True, method=['lbfgs'])
            coeffs_pd = pd.read_html(pd_lmm_results_iter.summary().tables[1].to_html(), header=0, index_col=0)[0]

            for dist_level_str in pd_distractor_z_values.keys():
                coeff_name_treat = f"C({DISTRACTOR_COL}, Treatment(reference='{DISTRACTOR_REF_STR}'))[T.{dist_level_str}]"
                coeff_name_default_ref = f"C({DISTRACTOR_COL})[T.{dist_level_str}]"
                if coeff_name_treat in coeffs_pd.index:
                    temp_pd_dist_z[dist_level_str] = coeffs_pd.loc[coeff_name_treat, 'z']
                elif coeff_name_default_ref in coeffs_pd.index:
                    temp_pd_dist_z[dist_level_str] = coeffs_pd.loc[coeff_name_default_ref, 'z']

            for target_level_str in pd_target_z_values.keys():
                coeff_name_treat = f"C({TARGET_COL}, Treatment(reference='{TARGET_REF_STR}'))[T.{target_level_str}]"
                coeff_name_default_ref = f"C({TARGET_COL})[T.{target_level_str}]"
                if coeff_name_treat in coeffs_pd.index:
                    temp_pd_target_z[target_level_str] = coeffs_pd.loc[coeff_name_treat, 'z']
                elif coeff_name_default_ref in coeffs_pd.index:
                    temp_pd_target_z[target_level_str] = coeffs_pd.loc[coeff_name_default_ref, 'z']
        except Exception as e:
            print(f"Error fitting Pd LMM for window t={center_time_point:.3f}s: {e}")

    for key, val in temp_pd_dist_z.items(): pd_distractor_z_values[key].append(val)
    for key, val in temp_pd_target_z.items(): pd_target_z_values[key].append(val)

# --- End of Main Loop ---
print("-" * 30)
print("Time-course LMM analysis complete. Plotting results...")

# --- Plotting Z-values over Time (Plotting logic remains the same) ---
# N2ac Model Z-values
plt.figure(figsize=(14, 8))
for target_level, z_vals in n2ac_target_z_values.items():
    plt.plot(time_points_for_plot, z_vals, label=f"Target: {target_level} (vs {TARGET_REF_STR})")
for dist_level, z_vals in n2ac_distractor_z_values.items():
    plt.plot(time_points_for_plot, z_vals, label=f"Distractor: {dist_level} (vs {DISTRACTOR_REF_STR})", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Z-value (from N2ac LMM)")
plt.title(
    f"Z-values for Target & Distractor Location (refs: T='{TARGET_REF_STR}', D='{DISTRACTOR_REF_STR}')\n"
    f"Time Course ({WINDOW_DURATION_S}s running average window, mean of {len(ERP_ELECTRODE_PAIRS)} electrode pairs)"
)
plt.axhline(0, color='grey', linestyle='-', linewidth=0.8)
plt.legend(loc='best', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Pd Model Z-values
plt.figure(figsize=(14, 8))
for dist_level, z_vals in pd_distractor_z_values.items():
    if dist_level != "absent":
        plt.plot(time_points_for_plot, z_vals, label=f"Distractor: {dist_level} (vs {DISTRACTOR_REF_STR})")
for target_level, z_vals in pd_target_z_values.items():
    plt.plot(time_points_for_plot, z_vals, label=f"Target: {target_level} (vs {TARGET_REF_STR})", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Z-value (from Pd LMM)")
plt.title(
    f"Pd Model: Z-values for Distractor & Target Location (refs: D='{DISTRACTOR_REF_STR}', T='{TARGET_REF_STR}')\n"
    f"Time Course ({WINDOW_DURATION_S}ms running average window, mean of {len(ERP_ELECTRODE_PAIRS)} pairs)"
)
plt.axhline(0, color='grey', linestyle='-', linewidth=0.8)
plt.legend(loc='best', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
