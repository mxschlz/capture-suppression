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
FILTER_PHASE = None

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
SINGLETON_PRESENT_COL = 'SingletonPresent'
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'
ACCURACY_COL = 'select_target'
ACCURACY_INT_COL = 'select_target_int'

# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(2)

# --- 3. ERP Component Definitions ---
BASELINE_WINDOW = (-0.2, 0.0)

# Flanker task configuration
FLANKER_EXPERIMENT_NAME = 'flanker_data.csv'
FLANKER_CONGRUENCY_COL = 'congruency'
FLANKER_ACC_COL = 'correct'

# Pd component definition (Distractor-locked)
PD_TIME_WINDOW = (0.2, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

### MERGED ### - N2ac component definition (Target-locked)
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
# Map categorical variables to strings
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map(TARGET_LOC_MAP)
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(DISTRACTOR_LOC_MAP)
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

print("Preprocessing and column mapping complete.")

# --- Filter for Pd-eliciting trials (lateral distractor, central target) ---
print("\nFiltering for Pd trials...")
is_distractor_lateral = df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = df[TARGET_COL] == TARGET_REF_STR
pd_trials_meta = df[is_distractor_lateral & is_target_central].copy()
print(f"Found {len(pd_trials_meta)} trials for Pd analysis.")

# --- Calculate Pd "Pure Area" ---
print("Calculating Pd component (Contra-minus-Ipsi)...")
pd_epochs = epochs[pd_trials_meta.index]
contra_chans_left_dist = [p[1] for p in PD_ELECTRODES]
ipsi_chans_left_dist = [p[0] for p in PD_ELECTRODES]
contra_chans_right_dist = [p[0] for p in PD_ELECTRODES]
ipsi_chans_right_dist = [p[1] for p in PD_ELECTRODES]
all_pd_chans = sorted(list(set(sum(PD_ELECTRODES, ()))))
pd_data = pd_epochs.get_data(picks=all_pd_chans) * 1e6  # in µV
ch_map_pd = {ch_name: i for i, ch_name in enumerate(pd_epochs.copy().pick(all_pd_chans).ch_names)}
contra_idx_left_pd = [ch_map_pd[ch] for ch in contra_chans_left_dist]
ipsi_idx_left_pd = [ch_map_pd[ch] for ch in ipsi_chans_left_dist]
contra_idx_right_pd = [ch_map_pd[ch] for ch in contra_chans_right_dist]
ipsi_idx_right_pd = [ch_map_pd[ch] for ch in ipsi_chans_right_dist]
is_left_dist = (pd_trials_meta[DISTRACTOR_COL] == 'left').values
is_right_dist = (pd_trials_meta[DISTRACTOR_COL] == 'right').values
pd_diff_wave = np.zeros_like(pd_data[:, 0, :])
pd_contra_wave = np.zeros_like(pd_data[:, 0, :])
pd_ipsi_wave = np.zeros_like(pd_data[:, 0, :])
pd_contra_wave[is_left_dist] = pd_data[is_left_dist][:, contra_idx_left_pd, :].mean(axis=1)
pd_ipsi_wave[is_left_dist] = pd_data[is_left_dist][:, ipsi_idx_left_pd, :].mean(axis=1)
pd_contra_wave[is_right_dist] = pd_data[is_right_dist][:, contra_idx_right_pd, :].mean(axis=1)
pd_ipsi_wave[is_right_dist] = pd_data[is_right_dist][:, ipsi_idx_right_pd, :].mean(axis=1)
pd_diff_wave[is_left_dist] = pd_contra_wave[is_left_dist] - pd_ipsi_wave[is_left_dist]
pd_diff_wave[is_right_dist] = pd_contra_wave[is_right_dist] - pd_ipsi_wave[is_right_dist]


# --- Area Calculation Function ---
def _calculate_signed_area(data_window, polarity='positive'):
    area_data = data_window.copy()
    if polarity == 'positive':
        area_data[area_data < 0] = 0
    elif polarity == 'negative':
        area_data[area_data > 0] = 0
    return area_data.sum(axis=-1)


# --- Calculate Pd Area per Subject ---
pd_times_idx = pd_epochs.time_as_index(PD_TIME_WINDOW)
baseline_times_idx_pd = pd_epochs.time_as_index(BASELINE_WINDOW)
sampling_interval_ms = (1 / epochs.info['sfreq']) * 1000
unique_subjects_pd = pd_trials_meta[SUBJECT_ID_COL].unique()
subject_results_pd = []
print("Averaging ERPs and calculating Pd area for each subject...")
for subject in unique_subjects_pd:
    subject_trial_mask = (pd_trials_meta[SUBJECT_ID_COL] == subject).values
    subject_diff_waves_for_avg = pd_diff_wave[subject_trial_mask]
    if subject_diff_waves_for_avg.shape[0] == 0: continue
    subject_avg_diff_wave = subject_diff_waves_for_avg.mean(axis=0)
    avg_wave_pd_window = subject_avg_diff_wave[pd_times_idx[0]:pd_times_idx[1]]
    avg_wave_baseline_window = subject_avg_diff_wave[baseline_times_idx_pd[0]:baseline_times_idx_pd[1]]
    raw_pd_area = _calculate_signed_area(avg_wave_pd_window, polarity='positive')
    baseline_noise_area = _calculate_signed_area(avg_wave_baseline_window, polarity='positive')
    pd_pure_area_samples = raw_pd_area - baseline_noise_area
    subject_results_pd.append({
        SUBJECT_ID_COL: subject,
        'pd_pure_area': pd_pure_area_samples * sampling_interval_ms
    })
pd_subject_df = pd.DataFrame(subject_results_pd)
print("Pd pure area calculated for each subject.")

### MERGED ### --- Start of N2ac Calculation Section ---

# --- Filter for N2ac-eliciting trials (lateral target, central distractor) ---
print("\nFiltering for N2ac trials...")
is_target_lateral = df[TARGET_COL].isin(['left', 'right'])
is_distractor_central = df[DISTRACTOR_COL] == 'mid'
n2ac_trials_meta = df[is_target_lateral & is_distractor_central].copy()
print(f"Found {len(n2ac_trials_meta)} trials for N2ac analysis.")

# --- Calculate N2ac "Pure Area" ---
print("Calculating N2ac component (Contra-minus-Ipsi)...")
n2ac_epochs = epochs[n2ac_trials_meta.index]
# Define contra/ipsi channels based on TARGET side
contra_chans_left_target = [p[1] for p in N2AC_ELECTRODES]
ipsi_chans_left_target = [p[0] for p in N2AC_ELECTRODES]
contra_chans_right_target = [p[0] for p in N2AC_ELECTRODES]
ipsi_chans_right_target = [p[1] for p in N2AC_ELECTRODES]
all_n2ac_chans = sorted(list(set(sum(N2AC_ELECTRODES, ()))))
n2ac_data = n2ac_epochs.get_data(picks=all_n2ac_chans) * 1e6  # in µV
ch_map_n2ac = {ch_name: i for i, ch_name in enumerate(n2ac_epochs.copy().pick(all_n2ac_chans).ch_names)}
contra_idx_left_n2ac = [ch_map_n2ac[ch] for ch in contra_chans_left_target]
ipsi_idx_left_n2ac = [ch_map_n2ac[ch] for ch in ipsi_chans_left_target]
contra_idx_right_n2ac = [ch_map_n2ac[ch] for ch in contra_chans_right_target]
ipsi_idx_right_n2ac = [ch_map_n2ac[ch] for ch in ipsi_chans_right_target]
is_left_target = (n2ac_trials_meta[TARGET_COL] == 'left').values
is_right_target = (n2ac_trials_meta[TARGET_COL] == 'right').values
n2ac_diff_wave = np.zeros_like(n2ac_data[:, 0, :])
n2ac_contra_wave = np.zeros_like(n2ac_data[:, 0, :])
n2ac_ipsi_wave = np.zeros_like(n2ac_data[:, 0, :])
n2ac_contra_wave[is_left_target] = n2ac_data[is_left_target][:, contra_idx_left_n2ac, :].mean(axis=1)
n2ac_ipsi_wave[is_left_target] = n2ac_data[is_left_target][:, ipsi_idx_left_n2ac, :].mean(axis=1)
n2ac_contra_wave[is_right_target] = n2ac_data[is_right_target][:, contra_idx_right_n2ac, :].mean(axis=1)
n2ac_ipsi_wave[is_right_target] = n2ac_data[is_right_target][:, ipsi_idx_right_n2ac, :].mean(axis=1)
n2ac_diff_wave[is_left_target] = n2ac_contra_wave[is_left_target] - n2ac_ipsi_wave[is_left_target]
n2ac_diff_wave[is_right_target] = n2ac_contra_wave[is_right_target] - n2ac_ipsi_wave[is_right_target]

# --- Calculate N2ac Area per Subject ---
n2ac_times_idx = n2ac_epochs.time_as_index(N2AC_TIME_WINDOW)
baseline_times_idx_n2ac = n2ac_epochs.time_as_index(BASELINE_WINDOW)
unique_subjects_n2ac = n2ac_trials_meta[SUBJECT_ID_COL].unique()
subject_results_n2ac = []
print("Averaging ERPs and calculating N2ac area for each subject...")
for subject in unique_subjects_n2ac:
    subject_trial_mask = (n2ac_trials_meta[SUBJECT_ID_COL] == subject).values
    subject_diff_waves_for_avg = n2ac_diff_wave[subject_trial_mask]
    if subject_diff_waves_for_avg.shape[0] == 0: continue
    subject_avg_diff_wave = subject_diff_waves_for_avg.mean(axis=0)
    avg_wave_n2ac_window = subject_avg_diff_wave[n2ac_times_idx[0]:n2ac_times_idx[1]]
    avg_wave_baseline_window = subject_avg_diff_wave[baseline_times_idx_n2ac[0]:baseline_times_idx_n2ac[1]]
    # N2ac is a NEGATIVE component
    raw_n2ac_area = _calculate_signed_area(avg_wave_n2ac_window, polarity='negative')
    baseline_noise_area = _calculate_signed_area(avg_wave_baseline_window, polarity='negative')
    n2ac_pure_area_samples = raw_n2ac_area - baseline_noise_area
    subject_results_n2ac.append({
        SUBJECT_ID_COL: subject,
        'n2ac_pure_area': n2ac_pure_area_samples * sampling_interval_ms
    })
n2ac_subject_df = pd.DataFrame(subject_results_n2ac)
print("N2ac pure area calculated for each subject.")

### MERGED ### --- Behavioral Data Processing ---

# --- 1. Calculate Flanker Effect ---
print("\nLoading and processing flanker task data...")
flanker_df = SPACEPRIME.load_concatenated_csv(FLANKER_EXPERIMENT_NAME)
flanker_df[SUBJECT_ID_COL] = flanker_df[SUBJECT_ID_COL].astype(str)
flanker_rt_by_subject = flanker_df.groupby(
    [SUBJECT_ID_COL, FLANKER_CONGRUENCY_COL]
)[REACTION_TIME_COL].mean().unstack()
flanker_rt_by_subject['flanker_effect'] = (flanker_rt_by_subject['incongruent'] - flanker_rt_by_subject['congruent']) * 1000
flanker_subject_df = flanker_rt_by_subject.reset_index()[[SUBJECT_ID_COL, 'flanker_effect']]
print("Flanker effect calculated per subject.")

# --- Calculate Singleton Presence Effect (RT and Accuracy) ---
print("\nCalculating singleton presence effect (RT & Accuracy) per subject...")

# The main 'df' contains the trial-by-trial data from the spaceprime task
# Ensure SingletonPresent is numeric for calculations
df[SINGLETON_PRESENT_COL] = pd.to_numeric(df[SINGLETON_PRESENT_COL], errors='coerce')

# Group by subject and singleton presence, then calculate mean RT and Accuracy
singleton_perf = df.groupby([SUBJECT_ID_COL, SINGLETON_PRESENT_COL])[[REACTION_TIME_COL, ACCURACY_COL]].mean()

# Unstack to get singleton present (1) / absent (0) as columns
singleton_perf_unstacked = singleton_perf.unstack(level=SINGLETON_PRESENT_COL)

# Flatten the multi-level column index (e.g., from ('rt', 0) to 'rt_0')
singleton_perf_unstacked.columns = [f'{col}_{int(level)}' for col, level in singleton_perf_unstacked.columns]

# Calculate the difference (Present - Absent).
# We multiply RT by 1000 to get ms, and accuracy by 100 to get percentage points.
singleton_perf_unstacked['singleton_rt_effect'] = (singleton_perf_unstacked[f'{REACTION_TIME_COL}_1'] - singleton_perf_unstacked[f'{REACTION_TIME_COL}_0']) * 1000
singleton_perf_unstacked['singleton_acc_effect'] = (singleton_perf_unstacked[f'{ACCURACY_COL}_1'] - singleton_perf_unstacked[f'{ACCURACY_COL}_0']).astype(float) * 100

# Reset index to make 'subject_id' a column for merging
singleton_effect_df = singleton_perf_unstacked.reset_index()

print("Singleton presence effect calculated.")

# --- 3. Load Questionnaire Data ---
print("Loading questionnaire data...")
questionnaire_data = SPACEPRIME.load_concatenated_csv("combined_questionnaire_results.csv")
questionnaire_data[SUBJECT_ID_COL] = questionnaire_data[SUBJECT_ID_COL].astype(str)

### MERGED ### --- Merge All Data for Correlation ---
print("\nMerging all EEG and behavioral data sources...")
# Start with a list of all dataframes to merge
data_frames = [pd_subject_df, n2ac_subject_df, flanker_subject_df, singleton_effect_df, questionnaire_data]

correlation_df = pd.concat(data_frames, axis=1).drop(columns=[SUBJECT_ID_COL])
correlation_df.dropna(inplace=True)
print(f"Found {len(correlation_df)} subjects with complete data across all measures.")

# --- Load, Reshape, and Merge ERP Latency/Amplitude Data ---
print("\nLoading and reshaping ERP latency/amplitude data...")
erp_long_df = SPACEPRIME.load_concatenated_csv("erp_latency_amplitude_subject_mean.csv")

# The goal is to transform the data from a "long" format (2 rows per subject)
# to a "wide" format (1 row per subject with dedicated columns for each metric).

# Based on the old code, the subject identifier might be 'subject'. Let's standardize it.
if 'subject' in erp_long_df.columns and SUBJECT_ID_COL not in erp_long_df.columns:
    erp_long_df.rename(columns={'subject': SUBJECT_ID_COL}, inplace=True)

# Ensure the subject ID is a string for consistent merging
erp_long_df[SUBJECT_ID_COL] = erp_long_df[SUBJECT_ID_COL].astype(str)

# Use pivot_table to reshape the data.
# This will make 'component' values into new columns.
# I'm assuming your columns are 'subject_id', 'component', 'latency', and 'amplitude'.
erp_wide_df = erp_long_df.pivot_table(
    index=SUBJECT_ID_COL,
    columns='component',
    values=['latency', 'amplitude']
)

# The pivot creates a multi-level column index, e.g., ('latency', 'N2pc').
# We'll flatten this into a single-level index, e.g., 'latency_N2pc' for clarity.
erp_wide_df.columns = [f'{value}_{component}' for value, component in erp_wide_df.columns]

# Reset the index to make 'subject_id' a column again, ready for merging.
erp_wide_df.reset_index(inplace=True)

# Merge the new wide-format ERP data with the main subject dataframe.
# An 'inner' merge ensures we only analyze subjects present in both datasets.
final_df = pd.concat([correlation_df,erp_wide_df], axis=1)

# This hardcoded outlier removal was in the original script.
# A more robust approach would be to identify outliers and remove by subject_id.
# TODO: to remove or not to remove ...
if 10 in final_df.index:
    print("Warning: Removing subject at hardcoded index 10 from correlation analysis.")
    #final_df = final_df.drop(index=10).reset_index(drop=True)

### MERGED ### --- Sanity-Check ERP Visualizations ---
def plot_erp_sanity_check(times, contra_wave, ipsi_wave, diff_wave, meta_df, component_name, time_window):
    """Helper function to plot grand average ERP waveforms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    times_ms = times * 1000

    subject_diff_waves, subject_contra_waves, subject_ipsi_waves = [], [], []
    for subject in meta_df[SUBJECT_ID_COL].unique():
        mask = (meta_df[SUBJECT_ID_COL] == subject).values
        if np.any(mask):
            subject_contra_waves.append(contra_wave[mask].mean(axis=0))
            subject_ipsi_waves.append(ipsi_wave[mask].mean(axis=0))
            subject_diff_waves.append(diff_wave[mask].mean(axis=0))
            ax.plot(times_ms, diff_wave[mask].mean(axis=0), color='grey', alpha=0.3, lw=1.0)

    if not subject_diff_waves: return

    ax.plot(times_ms, np.mean(subject_contra_waves, axis=0), 'r--', lw=1.5, label='GA Contralateral')
    ax.plot(times_ms, np.mean(subject_ipsi_waves, axis=0), 'b--', lw=1.5, label='GA Ipsilateral')
    ax.plot(times_ms, np.mean(subject_diff_waves, axis=0), 'k-', lw=2.5, label='GA Difference')

    ax.axvspan(BASELINE_WINDOW[0] * 1000, BASELINE_WINDOW[1] * 1000, color='lightblue', alpha=0.5, label='Baseline')
    ax.axvspan(time_window[0] * 1000, time_window[1] * 1000, color='lightcoral', alpha=0.5,
               label=f'{component_name} Window')
    ax.axhline(0, color='black', linestyle='--', lw=0.8)
    ax.axvline(0, color='black', linestyle=':', lw=0.8, label='Stimulus Onset')
    ax.set_title(f'Grand Average {component_name} Waveforms', fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


print("\nGenerating sanity-check plots for ERP components...")
plot_erp_sanity_check(pd_epochs.times, pd_contra_wave, pd_ipsi_wave, pd_diff_wave, pd_trials_meta, 'Pd', PD_TIME_WINDOW)
plot_erp_sanity_check(n2ac_epochs.times, n2ac_contra_wave, n2ac_ipsi_wave, n2ac_diff_wave, n2ac_trials_meta, 'N2ac',
                      N2AC_TIME_WINDOW)

### MERGED ### --- Visualize ERPs by Behavioral Split ---
print("\nVisualizing Pd ERPs split by singleton accuracy effect...")

# --- 1. Define the behavioral variable and create groups ---
behavioral_metric = 'singleton_acc_effect'

# Use a quartile split to create four groups for the behavioral metric
# Ensure the metric exists and has enough unique values for a quartile split
if behavioral_metric in final_df.columns and final_df[behavioral_metric].nunique() > 3:
    metric_title = behavioral_metric.replace("_", " ").title()
    # Corrected to a true quartile (4 groups) split for better analysis
    group_labels = [f'Q1', f'Q2', f'Q3']
    final_df['behavioral_group'] = pd.qcut(final_df[behavioral_metric],
                                           q=3,
                                           labels=group_labels,
                                           duplicates='drop')
    print(f"Successfully created {final_df['behavioral_group'].nunique()} groups based on quartiles of '{behavioral_metric}'.")
    print("Group sizes:\n", final_df['behavioral_group'].value_counts().sort_index())


    # --- 2. Aggregate trial-level ERPs to subject-level averages ---
    subject_avg_contra = {}
    subject_avg_ipsi = {}
    subject_avg_diff = {}

    # Use the original pd_trials_meta which is aligned with the wave arrays
    for subject in pd_trials_meta[SUBJECT_ID_COL].unique():
        mask = (pd_trials_meta[SUBJECT_ID_COL] == subject).values
        if np.any(mask):
            subject_avg_contra[subject] = pd_contra_wave[mask].mean(axis=0)
            subject_avg_ipsi[subject] = pd_ipsi_wave[mask].mean(axis=0)
            subject_avg_diff[subject] = pd_diff_wave[mask].mean(axis=0)

    # --- 3. Plot Grand Averages for each behavioral group ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(f'Pd Component by Quartiles of {metric_title}', fontsize=16, fontweight='bold')
    times_ms = pd_epochs.times * 1000

    # Create a color palette for the quartile groups
    n_groups = final_df['behavioral_group'].nunique()
    colors = plt.cm.Accent(np.linspace(0.1, 0.9, n_groups))

    # Sort group labels to ensure Q1 is plotted first with the first color, etc.
    sorted_groups = sorted(final_df['behavioral_group'].dropna().unique())

    for i, group_label in enumerate(sorted_groups):
        # Get subjects belonging to the current group
        # We need to look up subject IDs from final_df, which is at the subject level
        subjects_in_group = final_df[final_df['behavioral_group'] == group_label][SUBJECT_ID_COL].tolist()

        # Collect the average waves for these subjects
        group_contra_waves = [subject_avg_contra[s] for s in subjects_in_group if s in subject_avg_contra]
        group_ipsi_waves = [subject_avg_ipsi[s] for s in subjects_in_group if s in subject_avg_ipsi]
        group_diff_waves = [subject_avg_diff[s] for s in subjects_in_group if s in subject_avg_diff]

        if not group_diff_waves:
            print(f"Warning: No ERP data found for group '{group_label}'. Skipping.")
            continue

        # Calculate grand average for the group
        ga_contra = np.mean(group_contra_waves, axis=0)
        ga_ipsi = np.mean(group_ipsi_waves, axis=0)
        ga_diff = np.mean(group_diff_waves, axis=0)

        color = colors[i]

        # Plot on the 'Contra vs Ipsi' subplot
        axes[0].plot(times_ms, ga_contra, color=color, linestyle='--', label=f'Contra ({group_label})', lw=2)
        axes[0].plot(times_ms, ga_ipsi, color=color, linestyle=':', label=f'Ipsi ({group_label})', lw=2)

        # Plot on the 'Difference Wave' subplot
        axes[1].plot(times_ms, ga_diff, color=color, linestyle='-', lw=2.5, label=f'Difference ({group_label})')

    # --- 4. Formatting and Final Touches ---
    for ax in axes:
        ax.axvspan(PD_TIME_WINDOW[0] * 1000, PD_TIME_WINDOW[1] * 1000, color='lightcoral', alpha=0.3, label='Pd Window')
        ax.axhline(0, color='black', linestyle='--', lw=0.8)
        ax.axvline(0, color='black', linestyle=':', lw=0.8)
        ax.set_xlabel('Time (ms)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(title="Groups")
        # Set the x-axis to the requested time interval (0 to 0.7 seconds)
        ax.set_xlim(0, 700)

    axes[0].set_title('Contralateral vs. Ipsilateral Waveforms')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[1].set_title('Difference Wave (Contra - Ipsi)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

### MERGED ### --- Multivariate Correlation Analysis ---
print("\n--- Multivariate Correlation Analysis ---")
# --- 2. Perform Correlation Analysis ---
cols_to_correlate = [
    'pd_pure_area', 'n2ac_pure_area', 'latency_N2ac', 'latency_Pd', 'amplitude_N2ac', 'amplitude_Pd', 'flanker_effect',
    'singleton_rt_effect', 'singleton_acc_effect',
    'wnss_noise_resistance', 'ssq_speech_mean', "ssq_spatial_mean",
    "ssq_quality_mean"
]
existing_cols = [col for col in cols_to_correlate if col in final_df.columns]
print(f"Analyzing correlations between: {existing_cols}")

# --- 1. Visualization: Focused Subplots for Significant Correlations ---
print("Generating focused plots for significant correlations (p < 0.05)...")

# --- Step 1: Identify all significant pairs ---
# Use the pre-calculated correlation and p-value matrices
corr_matrix = final_df[existing_cols].corr(method='spearman')
p_values = final_df[existing_cols].corr(method=lambda x, y: stats.spearmanr(x, y)[1])
SIGNIFICANCE_THRESHOLD = 0.05

# Get the lower triangle of the p-value matrix to avoid duplicate pairs
p_lower = p_values.where(np.tril(np.ones(p_values.shape), k=-1).astype(bool))

# Find row, col indices where the p-value is below the threshold
sig_rows, sig_cols = np.where(p_lower < SIGNIFICANCE_THRESHOLD)

# Create a list of (var1, var2) tuples for the significant pairs
significant_pairs = [
    (p_lower.index[r], p_lower.columns[c]) for r, c in zip(sig_rows, sig_cols)
]

n_significant = len(significant_pairs)

# --- Step 2: Create a dynamic subplot grid for the results ---
if n_significant == 0:
    print("No significant correlations found to plot.")
    # Create a plot that explicitly states this finding
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.text(0.5, 0.5, "No significant correlations found\n(Spearman, p < 0.05)",
            ha='center', va='center', fontsize=14, style='italic', color='grey')
    ax.set_axis_off()
    plt.show()
else:
    print(f"Found {n_significant} significant correlations. Generating plots...")
    # Determine grid size to be as square as possible
    cols = int(np.ceil(np.sqrt(n_significant)))
    rows = int(np.ceil(n_significant / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5), squeeze=False)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # --- Step 3: Plot each significant pair in its own subplot ---
    for i, (var1, var2) in enumerate(significant_pairs):
        ax = axes[i]

        # Draw the regression plot showing the distribution of data points
        sns.regplot(
            data=final_df, x=var1, y=var2, ax=ax,
            scatter_kws={'alpha': 0.7, 'edgecolor': 'w', 's': 50, 'color': 'steelblue'},
            line_kws={'color': 'crimson', 'lw': 2.5}
        )

        # Add clear annotations for r and p values
        r = corr_matrix.loc[var1, var2]
        p = p_values.loc[var1, var2]
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        annotation = f"Spearman's r = {r:.2f}\n{p_text}"
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.6))

        # Clean up titles and labels for readability
        ax.set_title(f"{var1.replace('_', ' ').title()}\nvs\n{var2.replace('_', ' ').title()}",
                     fontweight='bold', fontsize=12)
        ax.set_xlabel(var1.replace('_', ' '), fontsize=10)
        ax.set_ylabel(var2.replace('_', ' '), fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- Step 4: Clean up the overall figure ---
    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_significant, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Significant Brain-Behavior Correlations', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()

# --- 2. Visualization: Correlation Heatmap ---
print("Generating correlation matrix heatmap...")
corr_matrix = final_df[existing_cols].corr()
p_values = final_df[existing_cols].corr(method=lambda x, y: stats.spearmanr(x, y)[1])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, cmap='vlag', center=0, annot=False, linewidths=.5, vmin=-1, vmax=1, ax=ax)

# Add custom annotations with significance
for i in range(len(corr_matrix)):
    for j in range(i):  # Only iterate through lower triangle
        r, p = corr_matrix.iloc[i, j], p_values.iloc[i, j]
        text = f"{r:.2f}"
        if p < 0.001:
            text += "***"
        elif p < 0.01:
            text += "**"
        elif p < 0.05:
            text += "*"
        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="white" if abs(r) > 0.6 else "black")

ax.set_title('Spearman correlation matrix', fontsize=15, fontweight='bold')
ax.text(0.98, 0.98, "* p<0.05\n** p<0.01\n*** p<0.001", transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()

print("\n--- Correlation Matrix (r-values) ---")
print(corr_matrix.round(3))
print("\n--- P-Values ---")
print(p_values.round(3))
