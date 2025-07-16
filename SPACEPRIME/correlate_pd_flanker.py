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
OUTLIER_RT_THRESHOLD = 2
FILTER_PHASE = 2

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_COL = 'correct'


# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(2)

# --- 3. ERP Component Definitions ---
# Baseline window for correction (e.g., -150ms to 0ms)
BASELINE_WINDOW = (-0.15, 0.0)

# Flanker task configuration (ASSUMPTIONS - PLEASE VERIFY)
FLANKER_EXPERIMENT_NAME = 'flanker_data.csv'  # The key for load_concatenated_csv
FLANKER_CONGRUENCY_COL = 'congruency' # e.g., 'congruent', 'incongruent'
FLANKER_ACC_COL = 'correct'          # e.g., 1 for correct, 0 for incorrect
# Aligned with the paragraph's methodology (200-350 ms)
PD_TIME_WINDOW = (0.2, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")
df = epochs.metadata.copy().reset_index(drop=True)
print(f"Original number of trials: {len(df)}")

# --- Preprocessing Steps (unchanged) ---
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

# --- Filter for Pd-eliciting trials (metadata only) ---
is_distractor_lateral = df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = df[TARGET_COL] == TARGET_REF_STR
pd_trials_meta = df[is_distractor_lateral & is_target_central].copy()
print(f"Found {len(pd_trials_meta)} trials for Pd analysis.")

# --- Calculate Pd "Pure Area" per trial ---
print("Calculating Pd component (Contra-minus-Ipsi)...")

# 1. Select the actual Epochs objects corresponding to our metadata
pd_epochs = epochs[pd_trials_meta.index]

# 2. Define contralateral and ipsilateral channels based on distractor side
contra_chans_left_dist = [p[1] for p in PD_ELECTRODES]
ipsi_chans_left_dist = [p[0] for p in PD_ELECTRODES]
contra_chans_right_dist = [p[0] for p in PD_ELECTRODES]
ipsi_chans_right_dist = [p[1] for p in PD_ELECTRODES]

# 3. Get EEG data as a NumPy array for efficient computation
all_pd_chans = sorted(list(set(sum(PD_ELECTRODES, ()))))
pd_data = pd_epochs.get_data(picks=all_pd_chans)  # (trials, channels, times)
# multiply the Pd pure area by 10e6 to have µV
pd_data = pd_data * 1e6

# Create a map from channel name to its index in the data array
ch_map = {ch_name: i for i, ch_name in enumerate(pd_epochs.copy().pick(all_pd_chans).ch_names)}

# 4. Get channel indices for contra/ipsi groups
contra_idx_left = [ch_map[ch] for ch in contra_chans_left_dist]
ipsi_idx_left = [ch_map[ch] for ch in ipsi_chans_left_dist]
contra_idx_right = [ch_map[ch] for ch in contra_chans_right_dist]
ipsi_idx_right = [ch_map[ch] for ch in ipsi_chans_right_dist]

# 5. Calculate contralateral, ipsilateral, and difference waves for each trial
is_left_dist = (pd_trials_meta[DISTRACTOR_COL] == 'left').values
is_right_dist = (pd_trials_meta[DISTRACTOR_COL] == 'right').values

# Initialize empty arrays to store the waves for each trial
diff_wave = np.zeros_like(pd_data[:, 0, :])  # Shape: (n_trials, n_times)
contra_wave = np.zeros_like(pd_data[:, 0, :])
ipsi_wave = np.zeros_like(pd_data[:, 0, :])

# Calculate and fill for left distractor trials
contra_mean_left = pd_data[is_left_dist][:, contra_idx_left, :].mean(axis=1)
ipsi_mean_left = pd_data[is_left_dist][:, ipsi_idx_left, :].mean(axis=1)
contra_wave[is_left_dist] = contra_mean_left
ipsi_wave[is_left_dist] = ipsi_mean_left
diff_wave[is_left_dist] = contra_mean_left - ipsi_mean_left

# Calculate and fill for right distractor trials
contra_mean_right = pd_data[is_right_dist][:, contra_idx_right, :].mean(axis=1)
ipsi_mean_right = pd_data[is_right_dist][:, ipsi_idx_right, :].mean(axis=1)
contra_wave[is_right_dist] = contra_mean_right
ipsi_wave[is_right_dist] = ipsi_mean_right
diff_wave[is_right_dist] = contra_mean_right - ipsi_mean_right

# 6. Calculate "Pure Signed Area" based on the provided paragraph's methodology
pd_times_idx = pd_epochs.time_as_index(PD_TIME_WINDOW)
baseline_times_idx = pd_epochs.time_as_index(BASELINE_WINDOW)


def _calculate_signed_area(data_window, polarity='positive'):
    """
    Calculates the signed area by summing only positive or negative values.

    Args:
        data_window (np.ndarray): The data slice for the time window (trials, times).
        polarity (str): 'positive' to sum positive values, 'negative' to sum negative.

    Returns:
        np.ndarray: A 1D array of the signed area for each trial.
    """
    # Make a copy to avoid modifying the original data
    area_data = data_window.copy()
    if polarity == 'positive':
        # Set all negative values to zero
        area_data[area_data < 0] = 0
    elif polarity == 'negative':
        # Set all positive values to zero
        area_data[area_data > 0] = 0
    # Sum across the time dimension to get the area
    return area_data.sum(axis=1)


# Calculate the "raw" signed area for the Pd component (sum of positive values)
raw_pd_area = _calculate_signed_area(diff_wave[:, pd_times_idx[0]:pd_times_idx[1]], polarity='positive')

# Calculate the signed area of the baseline to subtract as noise
baseline_noise_area = _calculate_signed_area(diff_wave[:, baseline_times_idx[0]:baseline_times_idx[1]], polarity='positive')

# The "Pd pure area" is the baseline-subtracted signed area.
# This gives a result in (µV * samples).
pd_pure_area_samples = raw_pd_area - baseline_noise_area

# Convert from (µV * samples) to (µV * ms) for physically meaningful units.
# This is a linear scaling and does not affect the correlation result, but makes the values reportable.
sampling_interval_ms = (1 / epochs.info['sfreq']) * 1000
pd_pure_area = pd_pure_area_samples * sampling_interval_ms

# 7. Add the calculated Pd area to our metadata and aggregate by subject
# Note on units: The 'area' is the sum of voltage points over a time window.
# The physical unit is technically (Volts * samples). To get a true area in (Volts * seconds),
# one would need to multiply by the sampling interval (1/sfreq). However, since this
# scaling is linear, it does not affect the correlation result. We treat it as arbitrary units (a.u.).
print("\n--- Sanity Check for Signed Area Calculation (First 5 Trials) ---")
print(f"{'Trial #':<8} | {'Raw PD Area':<15} | {'Baseline Noise Area':<20} | {'Pure PD Area'}")
print("-" * 70)
for i in range(min(5, len(pd_trials_meta))):
    print(f"{i:<8} | {raw_pd_area[i]:<15.4e} | {baseline_noise_area[i]:<20.4e} | {pd_pure_area_samples[i]:.4e}")


pd_trials_meta['pd_pure_area'] = pd_pure_area
pd_subject_df = pd_trials_meta.groupby(SUBJECT_ID_COL)['pd_pure_area'].mean().reset_index()
print("Pd pure area calculated and averaged per subject.")

# --- Sanity-Check Visualization: Difference Wave ---
print("\nGenerating sanity-check plot for Pd difference wave...")

fig, ax = plt.subplots(figsize=(10, 6))
times_ms = pd_epochs.times * 1000  # Convert times to milliseconds for plotting

subject_diff_waves = []
subject_contra_waves = []
subject_ipsi_waves = []
# Get unique subjects to iterate over
unique_subjects = pd_subject_df[SUBJECT_ID_COL].unique()

# Plot individual subject difference waves in the background
for subject in unique_subjects:
    # Find the indices for trials belonging to the current subject
    subject_trial_mask = (pd_trials_meta[SUBJECT_ID_COL] == subject).values

    if np.any(subject_trial_mask):
        # Calculate the average waves for this subject
        subject_mean_diff = diff_wave[subject_trial_mask].mean(axis=0)
        subject_mean_contra = contra_wave[subject_trial_mask].mean(axis=0)
        subject_mean_ipsi = ipsi_wave[subject_trial_mask].mean(axis=0)

        subject_diff_waves.append(subject_mean_diff)
        subject_contra_waves.append(subject_mean_contra)
        subject_ipsi_waves.append(subject_mean_ipsi)

        # Plot subject's average difference wave (use label='_nolegend_' to hide from legend)
        ax.plot(times_ms, subject_mean_diff, color='grey', alpha=0.3, linewidth=1.0, label='_nolegend_')

# Calculate and plot the grand average waves
if subject_diff_waves:
    grand_average_contra = np.mean(subject_contra_waves, axis=0)
    grand_average_ipsi = np.mean(subject_ipsi_waves, axis=0)
    grand_average_diff = np.mean(subject_diff_waves, axis=0)

    # Plot the contra and ipsi waves first, with dashed lines
    ax.plot(times_ms, grand_average_contra, color='red', linestyle='--', linewidth=1.5, label='Grand Average Contralateral')
    ax.plot(times_ms, grand_average_ipsi, color='blue', linestyle='--', linewidth=1.5, label='Grand Average Ipsilateral')

    # Plot the main difference wave on top
    ax.plot(times_ms, grand_average_diff, color='black', linewidth=2.5, label='Grand Average Difference (Contra-Ipsi)')

# --- Highlight the analysis windows to show where the data comes from ---
# Shaded region for baseline
ax.axvspan(BASELINE_WINDOW[0] * 1000, BASELINE_WINDOW[1] * 1000,
           color='lightblue', alpha=0.5, label=f'Baseline ({int(BASELINE_WINDOW[0]*1000)} to {int(BASELINE_WINDOW[1]*1000)} ms)')

# Shaded region for Pd component
ax.axvspan(PD_TIME_WINDOW[0] * 1000, PD_TIME_WINDOW[1] * 1000,
           color='lightcoral', alpha=0.5, label=f'Pd Analysis ({int(PD_TIME_WINDOW[0]*1000)} to {int(PD_TIME_WINDOW[1]*1000)} ms)')

# --- Plot aesthetics ---
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(0, color='black', linestyle=':', linewidth=0.8, label='Stimulus Onset')
ax.set_title('Grand Average Pd Waveforms', fontweight='bold', fontsize=14)
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

# As requested, we will calculate the mean RT on all trials, not just correct ones.
# You are right that groupby().unstack() can create a DataFrame with a "weird shape".
# This is because it creates a named index for the columns. The code below handles this robustly.
flanker_rt_by_subject = flanker_df.groupby([SUBJECT_ID_COL, FLANKER_CONGRUENCY_COL])[REACTION_TIME_COL].mean().unstack()

# Calculate the flanker effect
flanker_rt_by_subject['flanker_effect'] = (
    flanker_rt_by_subject['incongruent'] - flanker_rt_by_subject['congruent']) * 1000

# Convert the index ('subject_id') to a regular column for merging
flanker_subject_df = flanker_rt_by_subject.reset_index()
# This optional step cleans up the column index name left over from the unstack operation
flanker_subject_df.columns.name = None

print("Flanker effect calculated per subject.")

# --- DIAGNOSTIC STEP: Check for matching subjects before merging ---
# An empty correlation_df usually means the subject IDs do not match between the two data files.
# The following lines will print the subjects found in each dataset to help you debug.
pd_subjects = set(pd_subject_df[SUBJECT_ID_COL])
flanker_subjects = set(flanker_subject_df[SUBJECT_ID_COL])
common_subjects = pd_subjects.intersection(flanker_subjects)

print(f"\n--- Subject ID Sanity Check ---")
print(f"Found {len(pd_subjects)} subjects in EEG data: {sorted(list(pd_subjects))}")
print(f"Found {len(flanker_subjects)} subjects in Flanker data: {sorted(list(flanker_subjects))}")
print(f"Found {len(common_subjects)} common subjects for correlation: {sorted(list(common_subjects))}")

# --- Correlate Pd with Flanker Effect ---
print("\nCorrelating Pd pure area with Flanker effect...")

# Merge the two dataframes on subject_id
correlation_df = pd.merge(
    pd_subject_df,
    flanker_subject_df[[SUBJECT_ID_COL, 'flanker_effect']],
    on=SUBJECT_ID_COL)

# Drop any subjects who might be missing data in one of the tasks
correlation_df.dropna(inplace=True)

# correlation_df = correlation_df.drop(index=10)

# Calculate Pearson correlation
corr_coef, p_value = stats.pearsonr(
    correlation_df['pd_pure_area'],
    correlation_df['flanker_effect'])

# --- Visualization ---
plt.figure(figsize=(9, 7))
sns.regplot(data=correlation_df, x='pd_pure_area', y='flanker_effect', color='royalblue')

# Add subject ID labels to each point
for idx, row in correlation_df.iterrows():
    plt.text(
        x=row['pd_pure_area'],
        y=row['flanker_effect'],
        s=row[SUBJECT_ID_COL],
        ha='left',          # Horizontal alignment
        va='bottom',        # Vertical alignment
        fontsize=8,
        color='dimgray')

plt.title(f"Correlation between Pd Area and Flanker Effect\n"
          f"r = {corr_coef:.3f}, p = {p_value:.3f}",
          fontweight='bold')
plt.xlabel("Pd Pure Area (µV*ms)", fontsize=12)
plt.ylabel("Flanker Effect (RT difference ms)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n--- Correlation Results ---")
print(f"Number of subjects in analysis: {len(correlation_df)}")
print(f"Pearson Correlation Coefficient: {corr_coef:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")


# --- Final Analysis: Plot Waveforms by Flanker Effect Group ---
print("\n--- Final Analysis: Splitting by Flanker Effect ---")

# 1. Split subjects into tertiles based on their flanker effect
if len(correlation_df) >= 3:
    correlation_df['flanker_group'] = pd.qcut(
        correlation_df['flanker_effect'],
        q=3,
        labels=['Low Flanker Effect', 'Middle Flanker Effect', 'High Flanker Effect'],
        duplicates='drop'  # Handle cases with identical flanker scores
    )
    print("Subjects split into groups based on flanker effect:")
    print(correlation_df.groupby('flanker_group')[SUBJECT_ID_COL].apply(list).to_string())

    # 2. Create the 2x2 visualization grid
    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharey=True)
    colors = {'Low Flanker Effect': 'green', 'Middle Flanker Effect': 'orange', 'High Flanker Effect': 'firebrick'}
    group_names = ['Low Flanker Effect', 'Middle Flanker Effect', 'High Flanker Effect']
    
    # Map groups to specific subplots
    ax_map = {
        'Low Flanker Effect': axes[0],
        'Middle Flanker Effect': axes[1],
        'High Flanker Effect': axes[2]
    }
    diff_ax = axes[3]

    for group_name in group_names:
        if group_name in correlation_df['flanker_group'].values:
            # Get the list of subjects in the current group
            subjects_in_group = correlation_df[correlation_df['flanker_group'] == group_name][SUBJECT_ID_COL].tolist()
            # Create a boolean mask to select trials from these subjects
            trial_mask = pd_trials_meta[SUBJECT_ID_COL].isin(subjects_in_group)

            # Calculate the grand average contra, ipsi, and difference waves for the group
            ga_contra = contra_wave[trial_mask].mean(axis=0)
            ga_ipsi = ipsi_wave[trial_mask].mean(axis=0)
            ga_diff = ga_contra - ga_ipsi

            # Plot contra and ipsi on the group's dedicated subplot
            ax = ax_map[group_name]
            ax.set_title(f'{group_name} (n={len(subjects_in_group)})', fontweight='bold')
            ax.plot(times_ms, ga_contra, color='red', linestyle='--', linewidth=2, label='Contralateral')
            ax.plot(times_ms, ga_ipsi, color='blue', linestyle='--', linewidth=2, label='Ipsilateral')

            # Plot the group's difference wave on the summary subplot
            diff_ax.plot(times_ms, ga_diff, color=colors[group_name], linewidth=2.5, label=f'{group_name}')

    # 3. Finalize plot aesthetics
    diff_ax.set_title('Difference Waves by Group', fontweight='bold')

    # Highlight the Pd analysis window on the difference wave plot
    diff_ax.axvspan(PD_TIME_WINDOW[0] * 1000, PD_TIME_WINDOW[1] * 1000,
                    color='lightcoral', alpha=0.3, label=f'Pd Analysis Window')

    for ax in axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='black', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Time from Stimulus Onset (ms)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('Amplitude (µV)', fontsize=12)
    fig.suptitle('Grand Average Pd Waveforms by Flanker Effect Group', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.show()

else:
    print("Skipping flanker group analysis: not enough subjects (need at least 3).")
