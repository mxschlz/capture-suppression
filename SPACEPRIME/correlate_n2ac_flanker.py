import matplotlib.pyplot as plt
import numpy as np
import SPACEPRIME
import pandas as pd
import seaborn as sns
from stats import remove_outliers  # Assuming this is your custom outlier removal function
from scipy import stats

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
PHASE_COL = 'phase'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_INT_COL = 'select_target_int'

# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)

# --- 3. ERP Component Definitions ---
# Baseline window for correction (e.g., -150ms to 0ms)
BASELINE_WINDOW = (-0.15, 0.0)

# Flanker task configuration
FLANKER_EXPERIMENT_NAME = 'flanker_data.csv'  # The key for load_concatenated_csv
FLANKER_CONGRUENCY_COL = 'congruency'  # e.g., 'congruent', 'incongruent'
FLANKER_ACC_COL = 'correct'           # e.g., 1 for correct, 0 for incorrect

# N2ac component definition
N2AC_TIME_WINDOW = (0.2, 0.4)
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")
df = epochs.metadata.copy().reset_index(drop=True)
print(f"Original number of trials: {len(df)}")

# --- Preprocessing Steps ---
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
# Map categorical variables to strings
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map(TARGET_LOC_MAP)
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(DISTRACTOR_LOC_MAP)
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

print("Preprocessing and column mapping complete.")

# --- Filter for N2ac-eliciting trials (metadata only) ---
print("\nFiltering for N2ac trials (lateral target, central distractor)...")
is_target_lateral = df[TARGET_COL].isin(['left', 'right'])
is_distractor_central = df[DISTRACTOR_COL] == 'mid'
n2ac_trials_meta = df[is_target_lateral & is_distractor_central].copy()
print(f"Found {len(n2ac_trials_meta)} trials for N2ac analysis.")

# --- Calculate N2ac "Pure Area" per trial ---
print("Calculating N2ac component (Contra-minus-Ipsi)...")

# 1. Select the actual Epochs objects corresponding to our metadata
n2ac_epochs = epochs[n2ac_trials_meta.index]

# 2. Define contralateral and ipsilateral channels based on TARGET side
contra_chans_left_target = [p[1] for p in N2AC_ELECTRODES]  # Right hemisphere channels
ipsi_chans_left_target = [p[0] for p in N2AC_ELECTRODES]   # Left hemisphere channels
contra_chans_right_target = [p[0] for p in N2AC_ELECTRODES] # Left hemisphere channels
ipsi_chans_right_target = [p[1] for p in N2AC_ELECTRODES]  # Right hemisphere channels

# 3. Get EEG data as a NumPy array for efficient computation
all_n2ac_chans = sorted(list(set(sum(N2AC_ELECTRODES, ()))))
n2ac_data = n2ac_epochs.get_data(picks=all_n2ac_chans)  # (trials, channels, times)
n2ac_data = n2ac_data * 1e6  # Convert from V to µV

# Create a map from channel name to its index in the data array
ch_map = {ch_name: i for i, ch_name in enumerate(n2ac_epochs.copy().pick(all_n2ac_chans).ch_names)}

# 4. Get channel indices for contra/ipsi groups
contra_idx_left = [ch_map[ch] for ch in contra_chans_left_target]
ipsi_idx_left = [ch_map[ch] for ch in ipsi_chans_left_target]
contra_idx_right = [ch_map[ch] for ch in contra_chans_right_target]
ipsi_idx_right = [ch_map[ch] for ch in ipsi_chans_right_target]

# 5. Calculate contralateral, ipsilateral, and difference waves for each trial
is_left_target = (n2ac_trials_meta[TARGET_COL] == 'left').values
is_right_target = (n2ac_trials_meta[TARGET_COL] == 'right').values

# Initialize empty arrays to store the waves for each trial
diff_wave = np.zeros_like(n2ac_data[:, 0, :])  # Shape: (n_trials, n_times)
contra_wave = np.zeros_like(n2ac_data[:, 0, :])
ipsi_wave = np.zeros_like(n2ac_data[:, 0, :])

# Calculate and fill for left target trials
contra_mean_left = n2ac_data[is_left_target][:, contra_idx_left, :].mean(axis=1)
ipsi_mean_left = n2ac_data[is_left_target][:, ipsi_idx_left, :].mean(axis=1)
contra_wave[is_left_target] = contra_mean_left
ipsi_wave[is_left_target] = ipsi_mean_left
diff_wave[is_left_target] = contra_mean_left - ipsi_mean_left

# Calculate and fill for right target trials
contra_mean_right = n2ac_data[is_right_target][:, contra_idx_right, :].mean(axis=1)
ipsi_mean_right = n2ac_data[is_right_target][:, ipsi_idx_right, :].mean(axis=1)
contra_wave[is_right_target] = contra_mean_right
ipsi_wave[is_right_target] = ipsi_mean_right
diff_wave[is_right_target] = contra_mean_right - ipsi_mean_right

# 6. Calculate "Pure Signed Area" based on the provided paragraph's methodology
n2ac_times_idx = n2ac_epochs.time_as_index(N2AC_TIME_WINDOW)
baseline_times_idx = n2ac_epochs.time_as_index(BASELINE_WINDOW)


def _calculate_signed_area(data_window, polarity='positive'):
    """
    Calculates the signed area by summing only positive or negative values.

    Args:
        data_window (np.ndarray): The data slice for the time window (trials, times).
        polarity (str): 'positive' to sum positive values, 'negative' to sum negative.

    Returns:
        np.ndarray: A 1D array of the signed area for each trial.
    """
    area_data = data_window.copy()
    if polarity == 'positive':
        area_data[area_data < 0] = 0
    elif polarity == 'negative':
        area_data[area_data > 0] = 0
    return area_data.sum(axis=1)


# Calculate the "raw" signed area for the N2ac component (sum of NEGATIVE values)
raw_n2ac_area = _calculate_signed_area(diff_wave[:, n2ac_times_idx[0]:n2ac_times_idx[1]], polarity='negative')

# Calculate the signed area of the baseline to subtract as noise (sum of NEGATIVE values)
baseline_noise_area = _calculate_signed_area(diff_wave[:, baseline_times_idx[0]:baseline_times_idx[1]], polarity='negative')

# The "N2ac pure area" is the baseline-subtracted signed area
n2ac_pure_area = raw_n2ac_area - baseline_noise_area

# 7. Add the calculated N2ac area to our metadata and aggregate by subject
print("\n--- Sanity Check for Signed Area Calculation (First 5 Trials) ---")
print(f"{'Trial #':<8} | {'Raw N2ac Area':<15} | {'Baseline Noise Area':<20} | {'Pure N2ac Area'}")
print("-" * 70)
for i in range(min(5, len(n2ac_trials_meta))):
    print(f"{i:<8} | {raw_n2ac_area[i]:<15.4e} | {baseline_noise_area[i]:<20.4e} | {n2ac_pure_area[i]:.4e}")

n2ac_trials_meta['n2ac_pure_area'] = n2ac_pure_area
n2ac_subject_df = n2ac_trials_meta.groupby(SUBJECT_ID_COL)['n2ac_pure_area'].mean().reset_index()
print("N2ac pure area calculated and averaged per subject.")

# --- Sanity-Check Visualization: Difference Wave ---
print("\nGenerating sanity-check plot for N2ac difference wave...")

fig, ax = plt.subplots(figsize=(10, 6))
times_ms = n2ac_epochs.times * 1000  # Convert times to milliseconds for plotting

subject_diff_waves = []
subject_contra_waves = []
subject_ipsi_waves = []
# Get unique subjects to iterate over
unique_subjects = n2ac_subject_df[SUBJECT_ID_COL].unique()

# Plot individual subject difference waves in the background
for subject in unique_subjects:
    subject_trial_mask = (n2ac_trials_meta[SUBJECT_ID_COL] == subject).values
    if np.any(subject_trial_mask):
        subject_mean_diff = diff_wave[subject_trial_mask].mean(axis=0)
        subject_mean_contra = contra_wave[subject_trial_mask].mean(axis=0)
        subject_mean_ipsi = ipsi_wave[subject_trial_mask].mean(axis=0)

        subject_diff_waves.append(subject_mean_diff)
        subject_contra_waves.append(subject_mean_contra)
        subject_ipsi_waves.append(subject_mean_ipsi)

        ax.plot(times_ms, subject_mean_diff, color='grey', alpha=0.3, linewidth=1.0, label='_nolegend_')

# Calculate and plot the grand average waves
if subject_diff_waves:
    grand_average_contra = np.mean(subject_contra_waves, axis=0)
    grand_average_ipsi = np.mean(subject_ipsi_waves, axis=0)
    grand_average_diff = np.mean(subject_diff_waves, axis=0)

    ax.plot(times_ms, grand_average_contra, color='red', linestyle='--', linewidth=1.5, label='Grand Average Contralateral')
    ax.plot(times_ms, grand_average_ipsi, color='blue', linestyle='--', linewidth=1.5, label='Grand Average Ipsilateral')
    ax.plot(times_ms, grand_average_diff, color='black', linewidth=2.5, label='Grand Average Difference (Contra-Ipsi)')

# --- Highlight the analysis windows ---
ax.axvspan(BASELINE_WINDOW[0] * 1000, BASELINE_WINDOW[1] * 1000,
           color='lightblue', alpha=0.5, label=f'Baseline ({int(BASELINE_WINDOW[0]*1000)} to {int(BASELINE_WINDOW[1]*1000)} ms)')
ax.axvspan(N2AC_TIME_WINDOW[0] * 1000, N2AC_TIME_WINDOW[1] * 1000,
           color='lightcoral', alpha=0.5, label=f'N2ac Analysis ({int(N2AC_TIME_WINDOW[0]*1000)} to {int(N2AC_TIME_WINDOW[1]*1000)} ms)')

# --- Plot aesthetics ---
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(0, color='black', linestyle=':', linewidth=0.8, label='Stimulus Onset')
ax.set_title('Grand Average N2ac Waveforms', fontweight='bold', fontsize=14)
ax.set_xlabel('Time from Stimulus Onset (ms)', fontsize=12)
ax.set_ylabel('Amplitude (µV)', fontsize=12)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Load and Process Flanker Task Data ---
print("\nLoading and processing flanker task data...")
flanker_df = SPACEPRIME.load_concatenated_csv(FLANKER_EXPERIMENT_NAME)

# Ensure subject ID is a string for consistent merging
flanker_df[SUBJECT_ID_COL] = flanker_df[SUBJECT_ID_COL].astype(float).astype(str)

flanker_rt_by_subject = flanker_df.groupby(
    [SUBJECT_ID_COL, FLANKER_CONGRUENCY_COL]
)[REACTION_TIME_COL].mean().unstack()

# Calculate the flanker effect
flanker_rt_by_subject['flanker_effect'] = (
    flanker_rt_by_subject['incongruent'] - flanker_rt_by_subject['congruent']) * 1000

flanker_subject_df = flanker_rt_by_subject.reset_index()
flanker_subject_df.columns.name = None
print("Flanker effect calculated per subject.")

# --- DIAGNOSTIC STEP: Check for matching subjects before merging ---
n2ac_subjects = set(n2ac_subject_df[SUBJECT_ID_COL])
flanker_subjects = set(flanker_subject_df[SUBJECT_ID_COL])
common_subjects = n2ac_subjects.intersection(flanker_subjects)

print(f"\n--- Subject ID Sanity Check ---")
print(f"Found {len(n2ac_subjects)} subjects in EEG data: {sorted(list(n2ac_subjects))}")
print(f"Found {len(flanker_subjects)} subjects in Flanker data: {sorted(list(flanker_subjects))}")
print(f"Found {len(common_subjects)} common subjects for correlation: {sorted(list(common_subjects))}")

# --- Correlate N2ac with Flanker Effect ---
print("\nCorrelating N2ac pure area with Flanker effect...")

correlation_df = pd.merge(
    n2ac_subject_df,
    flanker_subject_df[[SUBJECT_ID_COL, 'flanker_effect']],
    on=SUBJECT_ID_COL
)

correlation_df.dropna(inplace=True)

# --- Final Analysis: Run only if there are enough subjects for a correlation ---
if len(correlation_df) < 2:
    print("\n--- Correlation Analysis Skipped ---")
    print(f"Found only {len(correlation_df)} subject with complete data. A correlation requires at least 2 subjects.")
    if not correlation_df.empty:
        print("\nData for the single subject:")
        print(correlation_df.to_string())
else:
    print("\n--- Correlation Results ---")
    print(f"Number of subjects in analysis: {len(correlation_df)}")

    # Calculate Pearson correlation
    corr_coef, p_value = stats.pearsonr(
        correlation_df['n2ac_pure_area'],
        correlation_df['flanker_effect']
    )
    print(f"Pearson Correlation Coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")

    # --- Visualization ---
    plt.figure(figsize=(9, 7))
    sns.regplot(data=correlation_df, x='n2ac_pure_area', y='flanker_effect', color='darkgreen')

    # Add subject ID labels to each point
    for idx, row in correlation_df.iterrows():
        plt.text(
            x=row['n2ac_pure_area'],
            y=row['flanker_effect'],
            s=row[SUBJECT_ID_COL],
            ha='left',
            va='bottom',
            fontsize=8,
            color='dimgray'
        )

    plt.title(f"Correlation between N2ac Area and Flanker Effect\n"
              f"r = {corr_coef:.3f}, p = {p_value:.3f}",
              fontweight='bold')
    plt.xlabel("N2ac Pure Area (Arbitrary Units)", fontsize=12)
    plt.ylabel("Flanker Effect (RT difference ms)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()