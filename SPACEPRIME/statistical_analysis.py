import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, zscore
from stats import remove_outliers
plt.ion()


# In this script, we are going to load up some eeg together with behavioral data. Combined, these data will be the basis
# for a couple of linear mixed models. Overall, we want to aim for a mixed model which contains all possible relevant
# predictors and dependent variables. Let's go.
# First, we load up the EEG data, which already includes the behavior as metadata attribute.
# crop to save RAM
tmin = 0  # tmin
tmax = 0.5  # tmax
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=True).crop(tmin, tmax) for subject in subject_ids])
# We want to calculate some average values for N2ac and Pd, therefore, we have to do some data transformation magic.
# First, we save the dataframe in a separate variable to facilitate modification.
# --- 1. Prepare Metadata Dataframe ---
df = epochs.metadata.copy() # Work on a copy
df = df[df["phase"]!=2]
df = remove_outliers(df, column_name="rt", threshold=2)

# Add total_trial_nr per subject (if needed)
# Check if 'subject_id' exists first
if 'subject_id' in df.columns:
    df['total_trial_nr'] = df.groupby('subject_id').cumcount() # cumcount starts at 0
else:
    print("Warning: 'subject_id' column not found. Cannot create 'total_trial_nr'.")
    # Optional: Create a dummy subject ID if analyzing only one subject's data
    # df['subject_id'] = 0
    # df['total_trial_nr'] = df.index

# --- 2. Verify Target and Singleton Location Columns ---
# We assume 'TargetLoc' and 'SingletonLoc' columns already exist based on your input.

target_col_name = 'TargetLoc'
distractor_col_name = 'SingletonLoc'

# Check if columns actually exist (important safety check)
if target_col_name not in df.columns or distractor_col_name not in df.columns:
    raise ValueError(f"Error: Columns '{target_col_name}' or '{distractor_col_name}' not found in metadata, although expected.")

# Optional but Recommended: Ensure columns are numeric
# If they might be strings or objects, uncomment and adapt the following lines:
df[target_col_name] = pd.to_numeric(df[target_col_name], errors='coerce')
df[distractor_col_name] = pd.to_numeric(df[distractor_col_name], errors='coerce')
# Check for any NaNs introduced by coercion if needed:
print(f"NaNs in {target_col_name} after coercion: {df[target_col_name].isna().sum()}")
print(f"NaNs in {distractor_col_name} after coercion: {df[distractor_col_name].isna().sum()}")

# --- Sanity Check ---
print("Verifying existing columns:")
print(f"Unique values found in '{target_col_name}': ", df[target_col_name].unique())
print(f"Unique values found in '{distractor_col_name}': ", df[distractor_col_name].unique())
# Ensure these unique values match your expectations (e.g., [1, 2, 3] for Target, [0, 1, 2, 3] for Singleton)
print("-" * 30) # Separator

# --- 3. Calculate Mean Amplitudes in Time Windows ---
Pd_window = (0.25, 0.35)
Pd_elecs = ["C3", "C4"]
N2ac_window = (0.2, 0.3)
N2ac_elecs = ["FC5", "FC6"]

# Get data for all relevant electrodes in wide format, indexed by epoch
erp_df = epochs.to_data_frame(picks=N2ac_elecs + Pd_elecs, time_format=None)

# --- Calculate N2ac means ---
# Filter for the N2ac time window (convert window from seconds to milliseconds)
n2ac_time_mask = (erp_df['time'] >= N2ac_window[0]) & (erp_df['time'] <= N2ac_window[1])
n2ac_filtered_df = erp_df[n2ac_time_mask]

# Group by epoch and calculate the mean for N2ac electrodes
print("Calculating N2ac means...")
n2ac_means = n2ac_filtered_df.groupby('epoch')[N2ac_elecs].mean()
print("N2ac means calculated.")

# --- Calculate Pd means ---
# Filter for the Pd time window (convert window from seconds to milliseconds)
pd_time_mask = (erp_df['time'] >= Pd_window[0]) & (erp_df['time'] <= Pd_window[1])
pd_filtered_df = erp_df[pd_time_mask]

# Group by epoch and calculate the mean for Pd electrodes
print("Calculating Pd means...")
pd_means = pd_filtered_df.groupby('epoch')[Pd_elecs].mean()
print("Pd means calculated.")

# --- 4. Merge ERP Means into Metadata Dataframe ---
# Assuming df's index corresponds to epoch number (0 to N-1)
df = df.merge(n2ac_means[[f"{elec}" for elec in N2ac_elecs]], left_index=True, right_index=True, how='left')
df = df.merge(pd_means[[f"{elec}" for elec in Pd_elecs]], left_index=True, right_index=True, how='left')

# --- 5. Calculate Difference Waves (Vectorized) ---
# Initialize difference wave columns with NaN
df['N2ac'] = np.nan
df['Pd'] = np.nan

# Define electrode names based on merged dataframe
fc5_col_n2ac = f"{N2ac_elecs[0]}" # e.g., "FC5_N2ac"
fc6_col_n2ac = f"{N2ac_elecs[1]}" # e.g., "FC6_N2ac"
c3_col_pd = f"{Pd_elecs[0]}"     # e.g., "C3_Pd"
c4_col_pd = f"{Pd_elecs[1]}"     # e.g., "C4_Pd"

# --- N2ac Calculation (Contra - Ipsi relative to TARGET) ---
# Target Left (Loc 1): Contra = FC6 (Right Hemi), Ipsi = FC5 (Left Hemi)
target_left_mask = (df[target_col_name] == 1)
df.loc[target_left_mask, 'N2ac'] = df.loc[target_left_mask, fc6_col_n2ac] - df.loc[target_left_mask, fc5_col_n2ac]

# Target Right (Loc 3): Contra = FC5 (Left Hemi), Ipsi = FC6 (Right Hemi)
target_right_mask = (df[target_col_name] == 3)
df.loc[target_right_mask, 'N2ac'] = df.loc[target_right_mask, fc5_col_n2ac] - df.loc[target_right_mask, fc6_col_n2ac]

# --- Pd Calculation (Contra - Ipsi relative to DISTRACTOR) ---
# Distractor Left (Loc 1): Contra = C4 (Right Hemi), Ipsi = C3 (Left Hemi)
distractor_left_mask = (df[distractor_col_name] == 1)
df.loc[distractor_left_mask, 'Pd'] = df.loc[distractor_left_mask, c4_col_pd] - df.loc[distractor_left_mask, c3_col_pd]

# Distractor Right (Loc 3): Contra = C3 (Left Hemi), Ipsi = C4 (Right Hemi)
distractor_right_mask = (df[distractor_col_name] == 3)
df.loc[distractor_right_mask, 'Pd'] = df.loc[distractor_right_mask, c3_col_pd] - df.loc[distractor_right_mask, c4_col_pd]

# Distractor Center (Loc 2) or Absent (Loc 0): Pd difference wave is not applicable -> remains NaN
# Display counts to verify calculations
print("\nN2ac Calculation Summary (Trials with calculated N2ac per TargetLoc):")
print(df['N2ac'].notna().groupby(df[target_col_name]).sum())
print("\nPd Calculation Summary (Trials with calculated Pd per SingletonLoc):")
print(df['Pd'].notna().groupby(df[distractor_col_name]).sum())
print("-" * 30) # Separator

# --- 6. Statistical Modeling ---
# Now use the 'N2ac' and 'Pd' columns
# Example Model for Pd (predicting Pd difference wave by distractor location)
# Filter data for modeling - automatically drops trials with NaN in Pd_diff or SingletonLateral
pd_stat_df = df[['Pd', 'SingletonLoc', 'subject_id']].dropna()

# Check if data remains after filtering
if pd_stat_df.empty:
    print("Warning: No data available for Pd model after filtering NaNs. Check calculations and data.")
else:
    pd_formula = "Pd ~ C(SingletonLoc)"
    pd_model = smf.mixedlm(formula=pd_formula, data=pd_stat_df, groups="subject_id")
    try:
        pd_result = pd_model.fit() # Use ML for model comparison if needed
        print("\n--- Pd Model Summary ---")
        print(pd_result.summary())
    except Exception as e:
        print(f"\nError fitting Pd model: {e}")
        print("Check the data in pd_stat_df:")
        print(pd_stat_df.head())
        print(pd_stat_df.info())

# Plotting
sns.lmplot(data=pd_stat_df, x="SingletonLoc", y="Pd", hue="subject_id")

# Example Model for N2ac (predicting N2ac difference wave by target laterality)
# Filter data for modeling
n2ac_stat_df = df[['N2ac', 'TargetLoc', 'subject_id']].dropna()

# Check if data remains after filtering
if n2ac_stat_df.empty:
    print("\nWarning: No data available for N2ac model after filtering NaNs. Check calculations and data.")
else:
    n2ac_formula = "N2ac ~ C(TargetLoc)"
    n2ac_model = smf.mixedlm(formula=n2ac_formula, data=n2ac_stat_df, groups="subject_id")
    try:
        n2ac_result = n2ac_model.fit(reml=False)
        print("\n--- N2ac Model Summary ---")
        print(n2ac_result.summary())
    except Exception as e:
        print(f"\nError fitting N2ac model: {e}")
        print("Check the data in n2ac_stat_df:")
        print(n2ac_stat_df.head())
        print(n2ac_stat_df.info())

# Plotting
sns.lmplot(data=n2ac_stat_df, x="TargetLoc", y="N2ac", hue="subject_id")

# Do some nice correlation (behavior against neural data)
df_corr = df.groupby(["subject_id"])[["rt", "select_target", "Pd", "N2ac"]].mean().reset_index().astype(float)
# do z-score transformation
cols_to_zscore = ['rt', 'select_target', 'Pd', 'N2ac']
# create new columns for storing the z-scores
z_cols = [col + '_z' for col in cols_to_zscore]
# Apply the zscore function to the selected columns
# .apply() runs the zscore function on each column individually
df_corr[z_cols] = df_corr[cols_to_zscore].apply(zscore)
# do stats
pearsonr(df_corr.rt_z, df_corr.N2ac_z)
# Plotting
sns.lmplot(data=df_corr, x="select_target_z", y="N2ac_z")

# Now, do one giant mixed model where we predict reaction time/accuracy with the following predictors:
# distractor presence, priming type, target identity/location, distractor identity/location
rt_stat_df = df[["SingletonPresent", "TargetDigit", "TargetLoc", "SingletonDigit", "SingletonLoc", "Priming", "rt", "subject_id"]].dropna()
rt_formula = "rt ~ C(SingletonPresent) + C(TargetDigit) + C(TargetLoc) + C(SingletonDigit) + C(SingletonLoc) + C(Priming)"
rt_model = smf.mixedlm(formula=rt_formula, data=rt_stat_df, groups="subject_id")
rt_result = rt_model.fit()
rt_result.summary()
