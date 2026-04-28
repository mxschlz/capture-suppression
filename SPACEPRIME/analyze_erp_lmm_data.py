import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import SPACEPRIME
import statsmodels.formula.api as smf

# Define the paradigm
paradigm = "spaceprime"

# Function to compute effect size r from t-value and degrees of freedom
def compute_effect_size_r(t, df):
    """
    Computes the standardized partial effect size r for mixed-effects models.
    
    r = √(t² / (t² + df))
    """
    return np.sqrt(t**2 / (t**2 + df))

def run_lmm_analysis(n2ac_path, pd_path, dataset_name="Main Dataset"):
    print(f"\n\n{'='*80}")
    print(f"--- RUNNING ANALYSIS FOR: {dataset_name.upper()} ---")
    print(f"{'='*80}\n")

    # Load the dataframes
    print("Loading N2ac and Pd dataframes...")
    n2ac_df = pd.read_csv(n2ac_path)
    pd_df = pd.read_csv(pd_path)

    print(f"N2ac DataFrame shape: {n2ac_df.shape}")
    print(f"Pd DataFrame shape: {pd_df.shape}")

    # Clean the data: drop rows with NaN in key columns
    key_cols_n2ac = ['target_towardness', 'Priming', 'st_latency_50', 'st_mean_amp_50', 'total_trial_nr', 'subject_id']
    n2ac_df = n2ac_df.dropna(subset=key_cols_n2ac).reset_index(drop=True)

    key_cols_pd = ['target_towardness', 'Priming', 'st_latency_50', 'st_mean_amp_50', 'total_trial_nr', 'subject_id']
    pd_df = pd_df.dropna(subset=key_cols_pd).reset_index(drop=True)

    print(f"After cleaning, N2ac DataFrame shape: {n2ac_df.shape}")
    print(f"After cleaning, Pd DataFrame shape: {pd_df.shape}")

    # Display descriptive statistics
    print("\n--- Descriptive Statistics for N2ac DataFrame ---")
    print(n2ac_df.describe())

    print("\n--- Descriptive Statistics for Pd DataFrame ---")
    print(pd_df.describe())

    # Print specific ranges and means for the text description (Single-trial level)
    print("\n--- Specific Descriptive Stats for Manuscript Text (Single-Trial Level) ---")
    print(f"N2ac Latency (st_latency_50): Mean = {n2ac_df['st_latency_50'].mean():.3f}s, "
          f"Median = {n2ac_df['st_latency_50'].median():.3f}s, "
          f"Range = [{n2ac_df['st_latency_50'].min():.3f}s, {n2ac_df['st_latency_50'].max():.3f}s]")
    print(f"N2ac Amplitude (st_mean_amp_50): Mean = {n2ac_df['st_mean_amp_50'].mean():.3f}µV, "
          f"Median = {n2ac_df['st_mean_amp_50'].median():.3f}µV, "
          f"Range = [{n2ac_df['st_mean_amp_50'].min():.3f}µV, {n2ac_df['st_mean_amp_50'].max():.3f}µV]")

    print("-" * 40)

    print(f"Pd Latency (st_latency_50): Mean = {pd_df['st_latency_50'].mean():.3f}s, "
          f"Median = {pd_df['st_latency_50'].median():.3f}s, "
          f"Range = [{pd_df['st_latency_50'].min():.3f}s, {pd_df['st_latency_50'].max():.3f}s]")
    print(f"Pd Amplitude (st_mean_amp_50): Mean = {pd_df['st_mean_amp_50'].mean():.3f}µV, "
          f"Median = {pd_df['st_mean_amp_50'].median():.3f}µV, "
          f"Range = [{pd_df['st_mean_amp_50'].min():.3f}µV, {pd_df['st_mean_amp_50'].max():.3f}µV]\n")

    # Example: Run a simple mixed-effects model for N2ac latency
    if 'st_latency_50' in n2ac_df.columns:
        print("\n--- Mixed-Effects Model for N2ac ---")
        print(f"Data types:\n{n2ac_df[['target_towardness', 'Priming', 'st_latency_50', 'st_mean_amp_50', 'total_trial_nr', 'subject_id']].dtypes}")
        print(f"Unique subjects: {n2ac_df['subject_id'].nunique()}")
        try:
            model = smf.mixedlm("target_towardness ~ Priming + st_latency_50 + st_mean_amp_50 + total_trial_nr", data=n2ac_df, groups=n2ac_df["subject_id"])
            result = model.fit()
            print(result.summary())
            
            # Extract and compute r for coefficients
            for param in result.tvalues.index:
                if 'Priming' in param or 'st_latency_50' in param or 'st_mean_amp_50' in param or 'total_trial_nr' in param or 'st_mean_amp_50:st_latency_50' in param:
                    t_val = result.tvalues[param]
                    df_val = result.df_resid
                    r_effect = compute_effect_size_r(t_val, df_val)
                    print(f"{param}: t={t_val:.3f}, df={df_val:.1f}, r={r_effect:.3f}")
        except Exception as e:
            print(f"Error fitting model for N2ac: {e}")

    # Example: Run a simple mixed-effects model for Pd latency
    if 'st_latency_50' in pd_df.columns:
        print("\n--- Mixed-Effects Model for Pd ---")
        print(f"Data types:\n{pd_df[['target_towardness', 'Priming', 'st_latency_50', 'st_mean_amp_50', 'total_trial_nr', 'subject_id']].dtypes}")
        print(f"Unique subjects: {pd_df['subject_id'].nunique()}")
        try:
            model = smf.mixedlm("target_towardness ~ Priming + st_latency_50 + st_mean_amp_50 + total_trial_nr", data=pd_df, groups=pd_df["subject_id"])
            result = model.fit()
            print(result.summary())
            
            for param in result.tvalues.index:
                if 'Priming' in param or 'st_latency_50' in param or 'st_mean_amp_50' in param or 'total_trial_nr' in param or 'st_mean_amp_50:st_latency_50' in param:
                    t_val = result.tvalues[param]
                    df_val = result.df_resid
                    r_effect = compute_effect_size_r(t_val, df_val)
                    print(f"{param}: t={t_val:.3f}, df={df_val:.1f}, r={r_effect:.3f}")
        except Exception as e:
            print(f"Error fitting model for Pd: {e}")


# --- Define File Paths ---

# 1. Main Data Paths
n2ac_main_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_n2ac_erp_behavioral_lmm_long_data_between-within.csv'
pd_main_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_pd_erp_behavioral_lmm_long_data_between-within.csv'

# 2. Control Data Paths
n2ac_control_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_n2ac_control_erp_behavioral_data.csv'
pd_control_path = f'{SPACEPRIME.get_data_path()}concatenated\\{paradigm}_pd_control_erp_behavioral_data.csv'

# --- Execute Analysis ---

# Uncomment this line if you ever want to run the original data:
run_lmm_analysis(n2ac_main_path, pd_main_path, dataset_name="Main Dataset")

# Run the Control Datasets
run_lmm_analysis(n2ac_control_path, pd_control_path, dataset_name="Control Dataset")
