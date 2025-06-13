import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import statsmodels.genmod.bayes_mixed_glm as sm_bayes
from stats import remove_outliers, r_squared_mixed_model  # Assuming this is your custom outlier removal function
from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

plt.ion()

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
ERP_N2AC_COL = 'N2ac'
ERP_PD_COL = 'Pd'

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

# Calculate N2ac means
n2ac_time_mask = (erp_df['time'] >= N2AC_TIME_WINDOW[0]) & (erp_df['time'] <= N2AC_TIME_WINDOW[1])
n2ac_filtered_df = erp_df[n2ac_time_mask]
print(f"Calculating N2ac means using electrodes: {N2AC_ELECTRODES} in window: {N2AC_TIME_WINDOW}")
# Calculate means for all relevant unique electrodes within the N2ac time window
n2ac_means = n2ac_filtered_df.groupby('epoch')[erp_df_picks_unique_flat].mean()
# Rename columns to specify they are from the N2ac time window
n2ac_means = n2ac_means.rename(columns={elec: f"{elec}_N2acWindow" for elec in n2ac_means.columns})
print("N2ac means calculated and columns renamed.")

# Calculate Pd means
pd_time_mask = (erp_df['time'] >= PD_TIME_WINDOW[0]) & (erp_df['time'] <= PD_TIME_WINDOW[1])
pd_filtered_df = erp_df[pd_time_mask]
print(f"Calculating Pd means using electrodes: {PD_ELECTRODES} in window: {PD_TIME_WINDOW}")
# Calculate means for all relevant unique electrodes within the Pd time window
pd_means = pd_filtered_df.groupby('epoch')[erp_df_picks_unique_flat].mean()
# Rename columns to specify they are from the Pd time window
pd_means = pd_means.rename(columns={elec: f"{elec}_PdWindow" for elec in pd_means.columns})
print("Pd means calculated and columns renamed.")

# Merge the window-specific means into the main dataframe
df = df.merge(n2ac_means, left_index=True, right_index=True, how='left')
df = df.merge(pd_means, left_index=True, right_index=True, how='left')
print("ERP means merged into metadata with window-specific column names.")

# Initialize ERP component columns
df[ERP_N2AC_COL] = np.nan
df[ERP_PD_COL] = np.nan

# Calculate N2ac based on the mean of differences from specified electrode pairs
n2ac_pair_difference_columns = []
for left_electrode, right_electrode in N2AC_ELECTRODES:
    left_col_name = f"{left_electrode}_N2acWindow"
    right_col_name = f"{right_electrode}_N2acWindow"
    if left_col_name in df.columns and right_col_name in df.columns:
        temp_diff_col = f"temp_n2ac_diff_{left_electrode}_{right_electrode}"
        df[temp_diff_col] = df[left_col_name] - df[right_col_name]
        n2ac_pair_difference_columns.append(temp_diff_col)
    else:
        print(f"Warning: Columns for N2ac pair ({left_col_name}, {right_col_name}) not found. This pair will be skipped for N2ac calculation.")

if n2ac_pair_difference_columns:
    df[ERP_N2AC_COL] = df[n2ac_pair_difference_columns].mean(axis=1)
    print(f"{ERP_N2AC_COL} calculated as the mean of {len(n2ac_pair_difference_columns)} electrode pair differences.")
    # Optionally, drop the temporary difference columns
    # df = df.drop(columns=n2ac_pair_difference_columns)
else:
    print(f"Warning: No valid electrode pairs found for N2ac calculation. {ERP_N2AC_COL} will remain NaN.")

# Calculate Pd based on the mean of differences from specified electrode pairs
pd_pair_difference_columns = []
for left_electrode, right_electrode in PD_ELECTRODES:
    left_col_name = f"{left_electrode}_PdWindow"
    right_col_name = f"{right_electrode}_PdWindow"
    if left_col_name in df.columns and right_col_name in df.columns:
        temp_diff_col = f"temp_pd_diff_{left_electrode}_{right_electrode}"
        df[temp_diff_col] = df[left_col_name] - df[right_col_name]
        pd_pair_difference_columns.append(temp_diff_col)
    else:
        print(f"Warning: Columns for Pd pair ({left_col_name}, {right_col_name}) not found. This pair will be skipped for Pd calculation.")

if pd_pair_difference_columns:
    df[ERP_PD_COL] = df[pd_pair_difference_columns].mean(axis=1)
    print(f"{ERP_PD_COL} calculated as the mean of {len(pd_pair_difference_columns)} electrode pair differences.")
    # Optionally, drop the temporary difference columns
    # df = df.drop(columns=pd_pair_difference_columns)
else:
    print(f"Warning: No valid electrode pairs found for Pd calculation. {ERP_PD_COL} will remain NaN.")


print(f"\n{ERP_N2AC_COL} Calculation Summary (Trials with calculated {ERP_N2AC_COL} per {TARGET_COL}):")
if ERP_N2AC_COL in df and TARGET_COL in df:
    print(df.groupby(TARGET_COL)[ERP_N2AC_COL].apply(lambda x: x.notna().sum()))
else:
    print(f"Cannot display N2ac summary; '{ERP_N2AC_COL}' or '{TARGET_COL}' missing.")

print(f"\n{ERP_PD_COL} Calculation Summary (Trials with calculated {ERP_PD_COL} per {DISTRACTOR_COL}):")
if ERP_PD_COL in df and DISTRACTOR_COL in df:
    print(df.groupby(DISTRACTOR_COL)[ERP_PD_COL].apply(lambda x: x.notna().sum()))
else:
    print(f"Cannot display Pd summary; '{ERP_PD_COL}' or '{DISTRACTOR_COL}' missing.")
print("-" * 30)

# --- 6. Statistical Modeling: N2ac and Pd LMMs ---
n2ac_model_cols = [ERP_N2AC_COL, SUBJECT_ID_COL] + \
                  LMM_N2AC_PD_CONTINUOUS_PREDICTORS + \
                  LMM_N2AC_PD_CATEGORICAL_PREDICTORS
df_n2ac_model = df[list(set(n2ac_model_cols))].dropna()
df_n2ac_model[SUBJECT_ID_COL] = df_n2ac_model[SUBJECT_ID_COL].astype(int).astype(str)
# save to csv
df_n2ac_model.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\df_n2ac_model.csv', index=True)


if df_n2ac_model.empty or df_n2ac_model[ERP_N2AC_COL].isna().all():
    print(f"N2ac model dataframe is empty or all '{ERP_N2AC_COL}' are NaN. Check N2ac calculation.")
else:
    print(f"Data for N2ac model: {len(df_n2ac_model)} trials")
    formula_parts_n2ac = [f"{ERP_N2AC_COL} ~"]
    formula_parts_n2ac.extend(LMM_N2AC_PD_CONTINUOUS_PREDICTORS)
    for cat_pred in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
        if cat_pred == PRIMING_COL:
            if PRIMING_REF_STR and PRIMING_REF_STR in df_n2ac_model[PRIMING_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference='{PRIMING_REF_STR}'))")
            else:
                print(f"Warning: Reference '{PRIMING_REF_STR}' not found for {PRIMING_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        elif cat_pred == TARGET_COL:
            if TARGET_REF_STR and TARGET_REF_STR in df_n2ac_model[TARGET_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference='{TARGET_REF_STR}'))")
            else:
                print(f"Warning: Reference '{TARGET_REF_STR}' not found for {TARGET_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        elif cat_pred == DISTRACTOR_COL:
            if DISTRACTOR_REF_STR and DISTRACTOR_REF_STR in df_n2ac_model[DISTRACTOR_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference='{DISTRACTOR_REF_STR}'))")
            else:
                print(f"Warning: Reference '{DISTRACTOR_REF_STR}' not found for {DISTRACTOR_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        else: # For ACCURACY_INT_COL, which is already 0/1
            formula_parts_n2ac.append(f"C({cat_pred})")
    # Add interaction term for ACCURACY_INT_COL and TARGET_COL
    if TARGET_COL in df_n2ac_model.columns and ACCURACY_INT_COL in df_n2ac_model.columns:
        target_treatment_for_interaction = ""
        # Check if the reference level for TARGET_COL is present in the current model's data
        if TARGET_REF_STR and TARGET_REF_STR in df_n2ac_model[TARGET_COL].unique():
            target_treatment_for_interaction = f", Treatment(reference='{TARGET_REF_STR}')"

        interaction_term_n2ac = f"C({ACCURACY_INT_COL}):C({TARGET_COL}{target_treatment_for_interaction})"
        formula_parts_n2ac.append(interaction_term_n2ac)
        # print(f"Added interaction to N2ac: {interaction_term_n2ac}") # Optional: for confirmation
    else:
        print(f"Warning: Could not add {ACCURACY_INT_COL}:{TARGET_COL} interaction to N2ac model, "
              f"as one or both columns are missing from df_n2ac_model after filtering.")

    n2ac_full_formula = " + ".join(formula_parts_n2ac)
    print(f"\nN2ac LMM Formula: {n2ac_full_formula}")
    print("Fitting N2ac LMM...")
    try:
        n2ac_lmm = smf.mixedlm(formula=n2ac_full_formula, data=df_n2ac_model, groups=SUBJECT_ID_COL)
        n2ac_lmm_results = n2ac_lmm.fit(reml=True)
        print("\n--- N2ac LMM Summary (Full Model) ---")
        print(n2ac_lmm_results.summary())
        # Calculate and print R-squared
        r2_n2ac = r_squared_mixed_model(n2ac_lmm_results)
        if r2_n2ac:
            print(f"N2ac LMM Marginal R-squared (fixed effects): {r2_n2ac['marginal_r2']:.3f}")
            print(f"N2ac LMM Conditional R-squared (fixed + random effects): {r2_n2ac['conditional_r2']:.3f}")

    except Exception as e:
        print(f"Error fitting N2ac LMM: {e}")
        print("Review data for N2ac model (first 5 rows):")
        print(df_n2ac_model.head())
        df_n2ac_model.info()
        for col in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
            if col in df_n2ac_model:
                print(f"Value counts for {col} in N2ac model data:\n{df_n2ac_model[col].value_counts(dropna=False)}")

# Filter for Pd modeling: exclude trials where distractor was "absent"
df_for_pd_modeling = df.copy()  # Use all trials for Pd modeling

pd_model_cols = [ERP_PD_COL, SUBJECT_ID_COL] + \
                LMM_N2AC_PD_CONTINUOUS_PREDICTORS + \
                LMM_N2AC_PD_CATEGORICAL_PREDICTORS
df_pd_model = df_for_pd_modeling[list(set(pd_model_cols))].dropna()
df_pd_model[SUBJECT_ID_COL] = df_pd_model[SUBJECT_ID_COL].astype(int).astype(str)
# save to csv
df_pd_model.to_csv('G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\concatenated\\df_pd_model.csv', index=True)

if df_pd_model.empty or df_pd_model[ERP_PD_COL].isna().all():
    print(f"Pd model dataframe is empty or all '{ERP_PD_COL}' are NaN. Check Pd calculation and filtering.")
else:
    print(f"Data for Pd model: {len(df_pd_model)} trials")
    if DISTRACTOR_COL in df_pd_model:
        print(f"Info: Unique '{DISTRACTOR_COL}' values in data for Pd model: {df_pd_model[DISTRACTOR_COL].unique()}")
    else:
        print(f"Warning: '{DISTRACTOR_COL}' not found in df_pd_model after processing.")

    formula_parts_pd = [f"{ERP_PD_COL} ~"]
    formula_parts_pd.extend(LMM_N2AC_PD_CONTINUOUS_PREDICTORS) # TRIAL_NUMBER_COL is in here
    pd_model_categorical_predictors = LMM_N2AC_PD_CATEGORICAL_PREDICTORS # Use the defined list

    for cat_pred in pd_model_categorical_predictors:
        if cat_pred == PRIMING_COL:
            if PRIMING_REF_STR and PRIMING_REF_STR in df_pd_model[PRIMING_COL].unique():
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference='{PRIMING_REF_STR}'))")
            else:
                print(f"Warning: Reference '{PRIMING_REF_STR}' not found for {PRIMING_COL} in Pd model data. Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        elif cat_pred == TARGET_COL:
            if TARGET_REF_STR and TARGET_REF_STR in df_pd_model[TARGET_COL].unique():
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference='{TARGET_REF_STR}'))")
            else:
                print(f"Warning: Reference '{TARGET_REF_STR}' not found for {TARGET_COL} in Pd model data. Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        elif cat_pred == DISTRACTOR_COL:
            if DISTRACTOR_REF_STR and DISTRACTOR_REF_STR in df_pd_model[DISTRACTOR_COL].unique():
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference='{DISTRACTOR_REF_STR}'))")
            else:
                print(f"Warning: Reference '{DISTRACTOR_REF_STR}' not found for {DISTRACTOR_COL} in Pd model data (after filtering). Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        else: # For ACCURACY_INT_COL
            formula_parts_pd.append(f"C({cat_pred})")
    # Add interaction term for ACCURACY_INT_COL and TARGET_COL
    if TARGET_COL in df_pd_model.columns and ACCURACY_INT_COL in df_pd_model.columns:
        target_treatment_for_interaction_pd = ""
        # Check if the reference level for TARGET_COL is present in the current model's data
        if TARGET_REF_STR and TARGET_REF_STR in df_pd_model[TARGET_COL].unique():
            target_treatment_for_interaction_pd = f", Treatment(reference='{TARGET_REF_STR}')"

        interaction_term_pd = f"C({ACCURACY_INT_COL}):C({TARGET_COL}{target_treatment_for_interaction_pd})"
        formula_parts_pd.append(interaction_term_pd)
        # print(f"Added interaction to Pd: {interaction_term_pd}") # Optional: for confirmation
    else:
        print(f"Warning: Could not add {ACCURACY_INT_COL}:{TARGET_COL} interaction to Pd model, "
              f"as one or both columns are missing from df_pd_model after filtering.")

    pd_full_formula = " + ".join(formula_parts_pd)
    print(f"\nPd LMM Formula: {pd_full_formula}")
    print("Fitting Pd LMM...")
    try:
        pd_lmm = smf.mixedlm(formula=pd_full_formula, data=df_pd_model, groups=SUBJECT_ID_COL)
        pd_lmm_results = pd_lmm.fit(reml=True)
        print("\n--- Pd LMM Summary (Full Model) ---")
        print(pd_lmm_results.summary())
        # Calculate and print R-squared
        r2_pd = r_squared_mixed_model(pd_lmm_results)
        if r2_pd:
            print(f"Pd LMM Marginal R-squared (fixed effects): {r2_pd['marginal_r2']:.3f}")
            print(f"Pd LMM Conditional R-squared (fixed + random effects): {r2_pd['conditional_r2']:.3f}")
    except Exception as e:
        print(f"Error fitting Pd LMM: {e}")
        print("Review data for Pd model (first 5 rows):")
        print(df_pd_model.head())
        df_pd_model.info()
        debug_cols_pd = [col for col in pd_model_categorical_predictors + LMM_N2AC_PD_CONTINUOUS_PREDICTORS if col in df_pd_model]
        for col in set(debug_cols_pd):
            if df_pd_model[col].dtype == 'object' or df_pd_model[col].nunique() < 10:
                print(f"Value counts for {col} in Pd model data:\n{df_pd_model[col].value_counts(dropna=False)}")
            else:
                print(f"Summary for {col} in Pd model data:\n{df_pd_model[col].describe()}")
