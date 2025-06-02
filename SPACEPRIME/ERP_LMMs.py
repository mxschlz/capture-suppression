import matplotlib.pyplot as plt
from SPACEPRIME import get_data_path, concatenated_epochs_data  # Assuming this function returns your data path
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import statsmodels.genmod.bayes_mixed_glm as sm_bayes
from stats import remove_outliers  # Assuming this is your custom outlier removal function
# from patsy.contrasts import Treatment # Import Treatment for specifying reference levels

plt.ion()

# --- Script Configuration Parameters ---

# --- 1. Data Loading & Preprocessing ---
EPOCH_TMIN = 0.0  # Start time for epoch cropping (seconds)
EPOCH_TMAX = 0.7  # End time for epoch cropping (seconds)
OUTLIER_RT_THRESHOLD = 2.0  # SDs for RT outlier removal (used in remove_outliers)
FILTER_PHASE = None  # Phase to exclude from analysis (e.g., practice blocks)

# --- 2. Column Names ---
# Existing columns in metadata or to be created
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'  # Target position (e.g., 1, 2, 3)
DISTRACTOR_COL = 'SingletonLoc'  # Distractor position (e.g., 0, 1, 2, 3)
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'  # Original accuracy column (e.g., True/False or 1.0/0.0)
PHASE_COL = 'phase'
DISTRACTOR_PRESENCE_COL = 'SingletonPresent'  # (e.g., 0 or 1)
PRIMING_COL = 'Priming'  # (e.g., -1, 0, 1)
#TARGET_DIGIT_COL = 'TargetDigit'
#SINGLETON_DIGIT_COL = 'SingletonDigit'
# BLOCK_COL = 'block' # Removed as per request

# ERP component columns (will be created in the script)
ERP_N2AC_COL = 'N2ac'
ERP_PD_COL = 'Pd'

# Derived columns for modeling
TRIAL_NUMBER_COL = 'total_trial_nr'  # Overall trial number per subject (will be created)
ACCURACY_INT_COL = 'select_target_int'  # Integer version of accuracy (0 or 1, will be created)

# --- 3. ERP Component Definitions ---
# Pd Component
PD_TIME_WINDOW = (0.29, 0.38)  # (start_time, end_time) in seconds
PD_ELECTRODES = ["C3",
                 "C4"]  # Electrodes for Pd. Order: [left_hemisphere_electrode, right_hemisphere_electrode]

# N2ac Component
N2AC_TIME_WINDOW = (0.22, 0.29)  # (start_time, end_time) in seconds
N2AC_ELECTRODES = ["FC5",
                   "FC6"]  # Electrodes for N2ac. Order: [left_hemisphere_electrode, right_hemisphere_electrode]


# --- 4. LMM Predictor Variables for N2ac and Pd Models ---
LMM_N2AC_PD_CONTINUOUS_PREDICTORS = [TRIAL_NUMBER_COL, REACTION_TIME_COL]
LMM_N2AC_PD_CATEGORICAL_PREDICTORS = [
    DISTRACTOR_PRESENCE_COL,
    TARGET_COL,
    DISTRACTOR_COL,
    PRIMING_COL,
    ACCURACY_INT_COL
    # BLOCK_COL removed
]

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = concatenated_epochs_data.load_data().crop(tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
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

for col_name in [TARGET_COL, DISTRACTOR_COL]:
    if col_name not in df.columns:
        raise ValueError(f"Error: Critical column '{col_name}' not found in metadata.")
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    if df[col_name].isna().any():
        print(f"Warning: NaNs introduced in '{col_name}' after coercion. Check data.")
    print(f"Unique values in '{col_name}': {df[col_name].unique()}")
    if 2 not in df[col_name].unique():
         print(f"Warning: Column '{col_name}' does not contain the value 2. "
               f"Setting 2 as reference might cause issues if it's not present in the data for some models.")


if PRIMING_COL in df.columns:
     if 0 not in df[PRIMING_COL].unique():
         print(f"Warning: Priming column '{PRIMING_COL}' does not contain the value 0. "
               f"Setting 0 as reference might cause issues if it's not present in the data.")
     print(f"Unique values in '{PRIMING_COL}': {df[PRIMING_COL].unique()}")
else:
     print(f"Warning: Priming column '{PRIMING_COL}' not found.")

# Removed BLOCK_COL processing logic

print("-" * 30)

erp_df_picks = list(set(N2AC_ELECTRODES + PD_ELECTRODES))
erp_df = epochs.to_data_frame(picks=erp_df_picks, time_format=None)

n2ac_time_mask = (erp_df['time'] >= N2AC_TIME_WINDOW[0]) & (erp_df['time'] <= N2AC_TIME_WINDOW[1])
n2ac_filtered_df = erp_df[n2ac_time_mask]
print(f"Calculating N2ac means using electrodes: {N2AC_ELECTRODES} in window: {N2AC_TIME_WINDOW}")
n2ac_means = n2ac_filtered_df.groupby('epoch')[N2AC_ELECTRODES].mean()
print("N2ac means calculated.")

pd_time_mask = (erp_df['time'] >= PD_TIME_WINDOW[0]) & (erp_df['time'] <= PD_TIME_WINDOW[1])
pd_filtered_df = erp_df[pd_time_mask]
print(f"Calculating Pd means using electrodes: {PD_ELECTRODES} in window: {PD_TIME_WINDOW}")
pd_means = pd_filtered_df.groupby('epoch')[PD_ELECTRODES].mean()
print("Pd means calculated.")

df = df.merge(n2ac_means, left_index=True, right_index=True, how='left')
df = df.merge(pd_means, left_index=True, right_index=True, how='left')
print("ERP means merged into metadata.")

df[ERP_N2AC_COL] = np.nan
df[ERP_PD_COL] = np.nan

n2ac_elec_L = N2AC_ELECTRODES[0]
n2ac_elec_R = N2AC_ELECTRODES[1]
pd_elec_L = PD_ELECTRODES[0]
pd_elec_R = PD_ELECTRODES[1]

if n2ac_elec_L in df.columns and n2ac_elec_R in df.columns:
    df[ERP_N2AC_COL] = df[n2ac_elec_L] - df[n2ac_elec_R]
    print(f"{ERP_N2AC_COL} calculated as {n2ac_elec_L} - {n2ac_elec_R}.")
else:
    print(f"Warning: Columns for N2ac electrodes ({n2ac_elec_L}, {n2ac_elec_R}) not found. {ERP_N2AC_COL} will remain NaN.")

if pd_elec_L in df.columns and pd_elec_R in df.columns:
    df[ERP_PD_COL] = df[pd_elec_L] - df[pd_elec_R]
    print(f"{ERP_PD_COL} calculated as {pd_elec_L} - {pd_elec_R}.")
else:
    print(f"Warning: Columns for Pd electrodes ({pd_elec_L}, {pd_elec_R}) not found. {ERP_PD_COL} will remain NaN.")

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

if df_n2ac_model.empty or df_n2ac_model[ERP_N2AC_COL].isna().all():
    print(f"N2ac model dataframe is empty or all '{ERP_N2AC_COL}' are NaN. Check N2ac calculation.")
else:
    print(f"Data for N2ac model: {len(df_n2ac_model)} trials")
    formula_parts_n2ac = [f"{ERP_N2AC_COL} ~"]
    formula_parts_n2ac.extend(LMM_N2AC_PD_CONTINUOUS_PREDICTORS)
    for cat_pred in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
        if cat_pred == PRIMING_COL:
            if 0 in df_n2ac_model[PRIMING_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference=0))")
            else:
                print(f"Warning: Reference 0 not found for {PRIMING_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        elif cat_pred == TARGET_COL:
            if 2 in df_n2ac_model[TARGET_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference=2))")
            else:
                print(f"Warning: Reference 2 not found for {TARGET_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        elif cat_pred == DISTRACTOR_COL:
            if 2 in df_n2ac_model[DISTRACTOR_COL].unique():
                formula_parts_n2ac.append(f"C({cat_pred}, Treatment(reference=2))")
            else:
                print(f"Warning: Reference 2 not found for {DISTRACTOR_COL} in N2ac model data. Using default.")
                formula_parts_n2ac.append(f"C({cat_pred})")
        # Removed BLOCK_COL handling
        else:
            formula_parts_n2ac.append(f"C({cat_pred})")
    n2ac_full_formula = " + ".join(formula_parts_n2ac)
    print(f"\nN2ac LMM Formula: {n2ac_full_formula}")
    print("Fitting N2ac LMM...")
    try:
        n2ac_lmm = smf.mixedlm(formula=n2ac_full_formula, data=df_n2ac_model, groups=SUBJECT_ID_COL)
        n2ac_lmm_results = n2ac_lmm.fit(reml=True)
        print("\n--- N2ac LMM Summary (Full Model) ---")
        print(n2ac_lmm_results.summary())
    except Exception as e:
        print(f"Error fitting N2ac LMM: {e}")
        print("Review data for N2ac model (first 5 rows):")
        print(df_n2ac_model.head())
        df_n2ac_model.info()
        for col in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
            if col in df_n2ac_model:
                print(f"Value counts for {col} in N2ac model data:\n{df_n2ac_model[col].value_counts(dropna=False)}")

df_for_pd_modeling = df[df[DISTRACTOR_COL] != 0].copy() # Original line for Pd specific filtering
# print(f"Trials after excluding '{DISTRACTOR_COL} == 0' for Pd modeling: {len(df_for_pd_modeling)}")
# df_for_pd_modeling = df # Using all data for Pd model as per previous change, or revert if needed
pd_model_cols = [ERP_PD_COL, SUBJECT_ID_COL] + \
                LMM_N2AC_PD_CONTINUOUS_PREDICTORS + \
                LMM_N2AC_PD_CATEGORICAL_PREDICTORS
df_pd_model = df_for_pd_modeling[list(set(pd_model_cols))].dropna()

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
    pd_model_categorical_predictors = []
    for cat_pred in LMM_N2AC_PD_CATEGORICAL_PREDICTORS:
        if cat_pred == DISTRACTOR_PRESENCE_COL:
            if DISTRACTOR_PRESENCE_COL in df_pd_model and df_pd_model[DISTRACTOR_PRESENCE_COL].nunique() < 2:
                print(f"Warning: '{DISTRACTOR_PRESENCE_COL}' has only one unique value in Pd model data. Excluding from Pd formula.")
                continue
        pd_model_categorical_predictors.append(cat_pred)

    for cat_pred in pd_model_categorical_predictors:
        if cat_pred == PRIMING_COL:
            if 0 in df_pd_model[PRIMING_COL].unique():
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference=0))")
            else:
                print(f"Warning: Reference 0 not found for {PRIMING_COL} in Pd model data. Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        elif cat_pred == TARGET_COL:
            if 2 in df_pd_model[TARGET_COL].unique():
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference=2))")
            else:
                print(f"Warning: Reference 2 not found for {TARGET_COL} in Pd model data. Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        elif cat_pred == DISTRACTOR_COL:
            if 2 in df_pd_model[DISTRACTOR_COL].unique(): # Check if this is still relevant if not filtering distractor==0
                formula_parts_pd.append(f"C({cat_pred}, Treatment(reference=2))")
            else:
                print(f"Warning: Reference 2 not found for {DISTRACTOR_COL} in Pd model data. Using default.")
                formula_parts_pd.append(f"C({cat_pred})")
        # Removed BLOCK_COL handling
        else:
            formula_parts_pd.append(f"C({cat_pred})")

    pd_full_formula = " + ".join(formula_parts_pd)
    """
    # Add the interaction C(SingletonPresent):total_trial_nr
    if DISTRACTOR_PRESENCE_COL in df_pd_model.columns and TRIAL_NUMBER_COL in df_pd_model.columns:
        interaction_term_pd = f"C({DISTRACTOR_PRESENCE_COL}):{TRIAL_NUMBER_COL}"
        pd_full_formula += f" + {interaction_term_pd}"
        print(f"Info: Added interaction '{interaction_term_pd}' to Pd formula.")
    else:
        print(f"Warning: Could not add C({DISTRACTOR_PRESENCE_COL}):{TRIAL_NUMBER_COL} interaction to Pd model, "
              f"one or both columns missing from df_pd_model.")
    """
    print(f"\nPd LMM Formula: {pd_full_formula}")
    print("Fitting Pd LMM...")
    try:
        pd_lmm = smf.mixedlm(formula=pd_full_formula, data=df_pd_model, groups=SUBJECT_ID_COL)
        pd_lmm_results = pd_lmm.fit(reml=True)
        print("\n--- Pd LMM Summary (Full Model) ---")
        print(pd_lmm_results.summary())
    except Exception as e:
        print(f"Error fitting Pd LMM: {e}")
        print("Review data for Pd model (first 5 rows):")
        print(df_pd_model.head())
        df_pd_model.info()
        # Adjusted debug print loop
        debug_cols_pd = [col for col in pd_model_categorical_predictors + [TRIAL_NUMBER_COL, DISTRACTOR_PRESENCE_COL] if col in df_pd_model]
        for col in set(debug_cols_pd): # Use set to avoid duplicates
            if df_pd_model[col].dtype == 'object' or df_pd_model[col].nunique() < 10:
                print(f"Value counts for {col} in Pd model data:\n{df_pd_model[col].value_counts(dropna=False)}")
            else:
                print(f"Summary for {col} in Pd model data:\n{df_pd_model[col].describe()}")


print("-" * 30)
# --- 7. Additional Analyses ---

if all(col in df.columns for col in [SUBJECT_ID_COL, REACTION_TIME_COL, ACCURACY_COL, ERP_PD_COL, ERP_N2AC_COL]):
    df_corr = df.groupby(SUBJECT_ID_COL)[
        [REACTION_TIME_COL, ACCURACY_COL, ERP_PD_COL, ERP_N2AC_COL]].mean().reset_index().astype(float)
    if REACTION_TIME_COL in df_corr.columns and ERP_N2AC_COL in df_corr.columns and not df_corr[[REACTION_TIME_COL, ERP_N2AC_COL]].isna().any().any():
        corr_rt_n2ac, p_val_rt_n2ac = pearsonr(df_corr[REACTION_TIME_COL], df_corr[ERP_N2AC_COL])
        print(f"\nPearson correlation between mean {REACTION_TIME_COL} and mean {ERP_N2AC_COL} per subject: r={corr_rt_n2ac:.3f}, p={p_val_rt_n2ac:.3f}")
        sns.lmplot(data=df_corr, x=REACTION_TIME_COL, y=ERP_N2AC_COL)
        plt.title(f"Correlation: Mean {REACTION_TIME_COL} vs. Mean {ERP_N2AC_COL}")
        plt.show(block=False) # Use block=False if running interactively
    else:
        print(f"Skipping RT vs N2ac correlation due to missing data or columns in aggregated df_corr.")
else:
    print("Skipping correlation analysis due to missing columns for aggregation.")

if all(col in df.columns for col in [ERP_PD_COL, REACTION_TIME_COL, SUBJECT_ID_COL]):
    pd_beh_df = df[[ERP_PD_COL, REACTION_TIME_COL, SUBJECT_ID_COL]].dropna()
    if not pd_beh_df.empty:
        pd_beh_formula = f"{ERP_PD_COL} ~ {REACTION_TIME_COL}"
        print(f"\nFitting LMM: {pd_beh_formula}")
        pd_beh_model = smf.mixedlm(formula=pd_beh_formula, data=pd_beh_df, groups=SUBJECT_ID_COL)
        try:
            pd_beh_result = pd_beh_model.fit(reml=True)
            print("\n--- Pd ~ RT Model Summary ---")
            print(pd_beh_result.summary())
            sns.lmplot(data=pd_beh_df, x=REACTION_TIME_COL, y=ERP_PD_COL, hue=SUBJECT_ID_COL, legend=False)
            plt.title(f"LMM: {ERP_PD_COL} vs. {REACTION_TIME_COL}")
            plt.show(block=False)
        except Exception as e:
            print(f"Error fitting {ERP_PD_COL} ~ {REACTION_TIME_COL} model: {e}")
    else:
        print(f"Skipping {ERP_PD_COL} ~ {REACTION_TIME_COL} LMM: Dataframe empty after dropna.")
else:
    print(f"Skipping {ERP_PD_COL} ~ {REACTION_TIME_COL} LMM: Missing required columns.")

if all(col in df.columns for col in [ERP_PD_COL, TRIAL_NUMBER_COL, SUBJECT_ID_COL]):
    pd_trial_df = df[[ERP_PD_COL, TRIAL_NUMBER_COL, SUBJECT_ID_COL]].dropna()
    if not pd_trial_df.empty:
        pd_trial_formula = f"{ERP_PD_COL} ~ {TRIAL_NUMBER_COL}" # Main effect of trial number
        print(f"\nFitting LMM: {pd_trial_formula}")
        pd_trial_model = smf.mixedlm(formula=pd_trial_formula, data=pd_trial_df, groups=SUBJECT_ID_COL)
        try:
            pd_trial_result = pd_trial_model.fit(reml=True)
            print("\n--- Pd ~ Trial Number Model Summary ---")
            print(pd_trial_result.summary())
        except Exception as e:
            print(f"Error fitting {ERP_PD_COL} ~ {TRIAL_NUMBER_COL} model: {e}")
    else:
        print(f"Skipping {ERP_PD_COL} ~ {TRIAL_NUMBER_COL} LMM: Dataframe empty after dropna.")
else:
    print(f"Skipping {ERP_PD_COL} ~ {TRIAL_NUMBER_COL} LMM: Missing required columns.")

# Add TRIAL_NUMBER_COL to rt_lmm_cols for the RT model
rt_lmm_cols = [DISTRACTOR_PRESENCE_COL, TARGET_COL, DISTRACTOR_COL, PRIMING_COL,
               REACTION_TIME_COL, SUBJECT_ID_COL, TRIAL_NUMBER_COL] # BLOCK_COL removed
# Check if all essential columns for RT LMM are present
essential_rt_cols = [REACTION_TIME_COL, SUBJECT_ID_COL] # Minimal set for model to run
if all(col in df.columns for col in essential_rt_cols):
    rt_lmm_cols_existing = [col for col in rt_lmm_cols if col in df.columns]
    rt_stat_df = df[rt_lmm_cols_existing].dropna()

    if not rt_stat_df.empty:
        target_ref_rt_str = ""
        if TARGET_COL in rt_stat_df and 2 in rt_stat_df[TARGET_COL].unique():
            target_ref_rt_str = ", Treatment(reference=2)"
        elif TARGET_COL in rt_stat_df:
            print(f"Warning: Reference 2 not found for {TARGET_COL} in RT model data. Using default.")

        distractor_ref_rt_str = ""
        if DISTRACTOR_COL in rt_stat_df and 2 in rt_stat_df[DISTRACTOR_COL].unique():
            distractor_ref_rt_str = ", Treatment(reference=2)"
        elif DISTRACTOR_COL in rt_stat_df:
            print(f"Warning: Reference 2 not found for {DISTRACTOR_COL} in RT model data. Using default.")

        priming_ref_rt_str = ""
        if PRIMING_COL in rt_stat_df and 0 in rt_stat_df[PRIMING_COL].unique():
            priming_ref_rt_str = ", Treatment(reference=0)"
        elif PRIMING_COL in rt_stat_df:
            print(f"Warning: Reference 0 not found for {PRIMING_COL} in RT model data. Using default.")

        # Base formula parts
        rt_formula_parts = [f"{REACTION_TIME_COL} ~"]
        if DISTRACTOR_PRESENCE_COL in rt_stat_df:
            rt_formula_parts.append(f"C({DISTRACTOR_PRESENCE_COL})")
        if TARGET_COL in rt_stat_df:
            rt_formula_parts.append(f"C({TARGET_COL}{target_ref_rt_str})")
        if DISTRACTOR_COL in rt_stat_df:
            rt_formula_parts.append(f"C({DISTRACTOR_COL}{distractor_ref_rt_str})")
        if PRIMING_COL in rt_stat_df:
            rt_formula_parts.append(f"C({PRIMING_COL}{priming_ref_rt_str})")
        # BLOCK_COL main effect removed

        # Add main effect of TRIAL_NUMBER_COL and its interaction with DISTRACTOR_PRESENCE_COL
        interaction_term_rt = ""
        if TRIAL_NUMBER_COL in rt_stat_df.columns:
            rt_formula_parts.append(TRIAL_NUMBER_COL) # Main effect of trial number
            if DISTRACTOR_PRESENCE_COL in rt_stat_df.columns:
                interaction_term_rt = f"C({DISTRACTOR_PRESENCE_COL}):{TRIAL_NUMBER_COL}"
                rt_formula_parts.append(interaction_term_rt)
                print(f"Info: Added main effect '{TRIAL_NUMBER_COL}' and interaction '{interaction_term_rt}' to RT formula.")
            else:
                print(f"Warning: {DISTRACTOR_PRESENCE_COL} not in RT model data, cannot add interaction with {TRIAL_NUMBER_COL}.")
        else:
            print(f"Warning: {TRIAL_NUMBER_COL} not in RT model data, cannot add its main effect or interaction.")

        rt_formula = " + ".join(rt_formula_parts)

        print(f"\nFitting LMM for RT: {rt_formula}")
        rt_model = smf.mixedlm(formula=rt_formula, data=rt_stat_df, groups=SUBJECT_ID_COL)
        try:
            rt_result = rt_model.fit(reml=True)
            print("\n--- RT LMM Summary ---")
            print(rt_result.summary())
        except Exception as e:
            print(f"Error fitting RT LMM: {e}")
            print("Review data for RT model (first 5 rows):")
            print(rt_stat_df.head())
            rt_stat_df.info()
            # Adjusted debug print loop
            debug_cols_rt = [col for col in rt_lmm_cols_existing if col in rt_stat_df]
            for col in set(debug_cols_rt): # Use set to avoid duplicates
                 if rt_stat_df[col].dtype == 'object' or rt_stat_df[col].nunique() < 10:
                    print(f"Value counts for {col} in RT model data:\n{rt_stat_df[col].value_counts(dropna=False)}")
                 else:
                    print(f"Summary for {col} in RT model data:\n{rt_stat_df[col].describe()}")
    else:
        print("Skipping RT LMM: Dataframe empty after dropna.")
else:
    print(f"Skipping RT LMM: Missing one or more essential columns for initial selection (e.g., {', '.join(essential_rt_cols)}).")


acc_glmm_cols = [DISTRACTOR_PRESENCE_COL, TARGET_COL, DISTRACTOR_COL,
                 PRIMING_COL, ACCURACY_INT_COL, SUBJECT_ID_COL, TRIAL_NUMBER_COL] # BLOCK_COL removed
essential_acc_cols = [ACCURACY_INT_COL, SUBJECT_ID_COL] # Minimal set for model to run
if all(col in df.columns for col in essential_acc_cols) and \
   all(col in df.columns for col in acc_glmm_cols if col not in essential_acc_cols): # Check other desired cols exist
    acc_glmm_cols_existing = [col for col in acc_glmm_cols if col in df.columns]
    acc_stat_df = df[acc_glmm_cols_existing].dropna()

    if not acc_stat_df.empty:
        target_ref_acc_str = ""
        if TARGET_COL in acc_stat_df and 2 in acc_stat_df[TARGET_COL].unique():
            target_ref_acc_str = ", Treatment(reference=2)"
        elif TARGET_COL in acc_stat_df:
            print(f"Warning: Reference 2 not found for {TARGET_COL} in Accuracy model data. Using default.")

        distractor_ref_acc_str = ""
        if DISTRACTOR_COL in acc_stat_df and 2 in acc_stat_df[DISTRACTOR_COL].unique():
            distractor_ref_acc_str = ", Treatment(reference=2)"
        elif DISTRACTOR_COL in acc_stat_df:
            print(f"Warning: Reference 2 not found for {DISTRACTOR_COL} in Accuracy model data. Using default.")

        priming_ref_acc_str = ""
        if PRIMING_COL in acc_stat_df and 0 in acc_stat_df[PRIMING_COL].unique():
            priming_ref_acc_str = ", Treatment(reference=0)"
        elif PRIMING_COL in acc_stat_df:
            print(f"Warning: Reference 0 not found for {PRIMING_COL} in Accuracy model data. Using default.")

        # Base formula parts
        acc_formula_parts = [f"{ACCURACY_INT_COL} ~"]
        if DISTRACTOR_PRESENCE_COL in acc_stat_df:
            acc_formula_parts.append(f"C({DISTRACTOR_PRESENCE_COL})")
        if TARGET_COL in acc_stat_df:
            acc_formula_parts.append(f"C({TARGET_COL}{target_ref_acc_str})")
        if DISTRACTOR_COL in acc_stat_df:
            acc_formula_parts.append(f"C({DISTRACTOR_COL}{distractor_ref_acc_str})")
        if PRIMING_COL in acc_stat_df:
            acc_formula_parts.append(f"C({PRIMING_COL}{priming_ref_acc_str})")
        # BLOCK_COL main effect removed

        # Add main effect of TRIAL_NUMBER_COL and its interaction with DISTRACTOR_PRESENCE_COL
        interaction_term_acc = ""
        if TRIAL_NUMBER_COL in acc_stat_df.columns:
            acc_formula_parts.append(TRIAL_NUMBER_COL) # Main effect of trial number
            if DISTRACTOR_PRESENCE_COL in acc_stat_df.columns:
                interaction_term_acc = f"C({DISTRACTOR_PRESENCE_COL}):{TRIAL_NUMBER_COL}"
                acc_formula_parts.append(interaction_term_acc)
                print(f"Info: Added main effect '{TRIAL_NUMBER_COL}' and interaction '{interaction_term_acc}' to ACC formula.")
            else:
                print(f"Warning: {DISTRACTOR_PRESENCE_COL} not in ACC model data, cannot add interaction with {TRIAL_NUMBER_COL}.")
        else:
            print(f"Warning: {TRIAL_NUMBER_COL} not in ACC model data, cannot add its main effect or interaction.")

        acc_formula_glmm = " + ".join(acc_formula_parts)
        vc_f = {SUBJECT_ID_COL: f"0 + C({SUBJECT_ID_COL})"} # Random intercept for subject_id

        print(f"\nFitting Bayesian GLMM for Accuracy: {acc_formula_glmm}")
        try:
            model_bayes_glmm = sm_bayes.BinomialBayesMixedGLM.from_formula(
                formula=acc_formula_glmm,
                vc_formulas=vc_f,
                data=acc_stat_df)
            result_bayes_glmm = model_bayes_glmm.fit_vb()
            print("\n--- Accuracy Bayesian GLMM Summary ---")
            print(result_bayes_glmm.summary())
        except Exception as e:
            print(f"Error fitting Accuracy Bayesian GLMM: {e}")
            print("Check data for accuracy GLMM (first 5 rows):")
            print(acc_stat_df.head())
            acc_stat_df.info()
            # Adjusted debug print loop
            debug_cols_acc = [col for col in acc_glmm_cols_existing if col in acc_stat_df]
            for col in set(debug_cols_acc): # Use set to avoid duplicates
                 if acc_stat_df[col].dtype == 'object' or acc_stat_df[col].nunique() < 10:
                    print(f"Value counts for {col} in ACC model data:\n{acc_stat_df[col].value_counts(dropna=False)}")
                 else:
                    print(f"Summary for {col} in ACC model data:\n{acc_stat_df[col].describe()}")
    else:
        print("Skipping Accuracy Bayesian GLMM: Dataframe empty after dropna.")
else:
    print(f"Skipping Accuracy Bayesian GLMM: Missing one or more essential columns for initial selection (e.g., {', '.join(essential_acc_cols)}), or ACCURACY_INT_COL not created.")

print("\nScript finished.")
if plt.get_fignums(): # Check if any figures are open
    plt.show(block=True) # Keep plots open until manually closed