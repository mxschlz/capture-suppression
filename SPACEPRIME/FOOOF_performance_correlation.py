from SPACEPRIME import load_concatenated_epochs
import numpy as np
from scipy.stats import ttest_rel
from scipy.signal import welch
import fooof
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import warnings
import SPACEPRIME

plt.ion()

# --- Configuration ---
# 1. Define parameters for PSD computation and FOOOF fitting
FMIN_PSD, FMAX_PSD = 1, 40  # Frequency range for PSD computation
FMIN_FOOOF, FMAX_FOOOF = 3, 40  # Frequency range for FOOOF fitting

# IMPORTANT: Define the column names in your epochs.metadata
ACCURACY_COLUMN = 'select_target'
SUBJECT_ID_COLUMN = 'subject_id'
### MODIFICATION: Add the column name for reaction times.
REACTION_TIME_COLUMN = 'rt'

# Set to True to save a plot for each subject's FOOOF fit.
PLOT_SUBJECT_FITS = True  # Keep False to avoid clutter

# Define an output directory and create it if it doesn't exist.
PLOTS_OUTPUT_DIR = "fooof_within_subject_analysis"
if not os.path.exists(PLOTS_OUTPUT_DIR):
    os.makedirs(PLOTS_OUTPUT_DIR)
print(f"\nAll plots will be saved to the '{PLOTS_OUTPUT_DIR}/' directory.")


def process_subject_psd_and_fooof(epochs_subject, analysis_name, subject_id):
    """
    Computes PSD and fits FOOOF for a single subject's epochs.
    This is the memory-intensive part of the analysis.

    Args:
        epochs_subject (mne.Epochs): The epochs object for one subject.
        analysis_name (str): A descriptive name for the analysis (e.g., "pre-stimulus_correct").
        subject_id (str): The ID of the subject being processed.

    Returns:
        float: The calculated aperiodic exponent. Returns None if fitting fails or no epochs.
    """
    if len(epochs_subject) < 2:  # Need at least 2 epochs for a reliable PSD
        if len(epochs_subject) > 0:
            print(
                f"      - Skipping {analysis_name} for subject {subject_id}: Not enough epochs ({len(epochs_subject)} found).")
        return None

    # --- PSD Computation with Padding for 0.1 Hz Resolution ---
    sfreq = epochs_subject.info['sfreq']
    eeg_data = epochs_subject.get_data()
    n_epochs, n_channels, n_samples_original = eeg_data.shape

    target_duration_s = 10.0
    target_n_samples = int(target_duration_s * sfreq)

    if n_samples_original < target_n_samples:
        pad_length = target_n_samples - n_samples_original
        pad_width = ((0, 0), (0, 0), (0, pad_length))
        eeg_data_padded = np.pad(eeg_data, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        eeg_data_padded = eeg_data
        target_n_samples = n_samples_original

    # Define parameters for Welch's method
    n_per_seg = target_n_samples
    n_fft = int(2 ** np.ceil(np.log2(n_per_seg)))
    eeg_data_reshaped = eeg_data_padded.reshape(n_epochs * n_channels, target_n_samples)
    full_freqs, psds_reshaped = welch(
        eeg_data_reshaped, fs=sfreq, nperseg=n_per_seg, nfft=n_fft, axis=-1
    )
    psds_full_spectrum = psds_reshaped.reshape(n_epochs, n_channels, -1)

    # Filter frequencies and PSD data
    freq_mask = (full_freqs >= FMIN_PSD) & (full_freqs <= FMAX_PSD)
    all_freqs = full_freqs[freq_mask]
    all_psds_data = psds_full_spectrum[:, :, freq_mask]

    # --- FOOOF Fitting ---
    avg_psd_subject = np.mean(all_psds_data, axis=(0, 1))
    fm = fooof.FOOOF(aperiodic_mode='fixed', peak_threshold=2.0, verbose=False, max_n_peaks=4,
                     peak_width_limits=[2, 15])
    fm.fit(all_freqs, avg_psd_subject, freq_range=[FMIN_FOOOF, FMAX_FOOOF])

    if PLOT_SUBJECT_FITS:
        fig, ax = plt.subplots(figsize=(10, 8))
        fm.plot(ax=ax, plt_log=False)
        exponent_val = fm.get_params('aperiodic', 'exponent')
        r_squared_val = fm.get_params('r_squared')
        ax.set_title(f"FOOOF Fit for Subject: {subject_id} ({analysis_name})\n"
                     f"Exponent: {exponent_val:.3f}, R-squared: {r_squared_val:.3f}",
                     fontsize=14)
        plot_filename = os.path.join(PLOTS_OUTPUT_DIR, f"fooof_fit_subject_{subject_id}_{analysis_name}.png")
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fm.get_params('aperiodic', 'exponent')


### MODIFICATION: This function is now more generic to handle different comparisons.
def analyze_and_plot_within_subject(results_df, period, analysis_type, groups):
    """
    Performs paired t-test and plots the within-subject differences.

    Args:
        results_df (pd.DataFrame): DataFrame with all results.
        period (str): The period to analyze, "pre-stimulus" or "post-stimulus".
        analysis_type (str): The column defining the condition (e.g., 'outcome' or 'rt_split').
        groups (list): A list of two strings for the group names (e.g., ['correct', 'incorrect']).
    """
    title_map = {
        'outcome': 'by Trial Outcome',
        'rt_split': 'by Reaction Time (Correct Trials)'
    }
    print(f"\n--- Analyzing {title_map[analysis_type]} for: {period.upper()} ---")

    # Filter for the period and pivot the table to get groups side-by-side
    df_period = results_df[(results_df['period'] == period) & (results_df[analysis_type].notna())]
    df_pivot = df_period.pivot_table(index='subject_id', columns=analysis_type, values='exponent').dropna()

    if len(df_pivot) < 3:  # Need at least 3 pairs for a meaningful test
        print(f"Not enough paired data for this analysis (found {len(df_pivot)} subjects). Skipping.")
        return

    group1_data = df_pivot[groups[0]]
    group2_data = df_pivot[groups[1]]

    # Perform paired t-test
    t_stat, p_val = ttest_rel(group1_data, group2_data)
    print(f"Paired t-test results ({len(df_pivot)} subjects):")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.stripplot(data=df_pivot[groups], palette=['skyblue', 'salmon'], size=7, jitter=0.15, ax=ax)

    # Draw lines connecting paired points for each subject
    for _, row in df_pivot.iterrows():
        ax.plot([0, 1], [row[groups[0]], row[groups[1]]], color='gray', linestyle='-', linewidth=1, alpha=0.6)

    ax.set_title(f"Within-Subject Aperiodic Exponent ({period.replace('-', ' ').title()})\n"
                 f"{title_map[analysis_type]}\n"
                 f"Paired t-test: t={t_stat:.2f}, p={p_val:.3f} (n={len(df_pivot)})",
                 fontsize=14)
    ax.set_ylabel("Aperiodic Exponent")
    ax.set_xlabel("Condition")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g.title() for g in groups])
    plt.tight_layout()

    plot_filename = os.path.join(PLOTS_OUTPUT_DIR, f"comparison_{period}_{analysis_type}.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to: {plot_filename}")


# --- Main Execution ---
print("Loading concatenated epochs...")
all_epochs = load_concatenated_epochs("spaceprime_desc-csd")
# Make sure metadata is a DataFrame
if not isinstance(all_epochs.metadata, pd.DataFrame):
    raise TypeError("Epochs metadata must be a pandas DataFrame for this analysis.")
print("Epochs loaded.")

if REACTION_TIME_COLUMN not in all_epochs.metadata.columns:
    raise ValueError(f"Reaction time column '{REACTION_TIME_COLUMN}' not found in epochs metadata.")

unique_subject_ids = all_epochs.metadata[SUBJECT_ID_COLUMN].unique()
print(f"\nFound {len(unique_subject_ids)} unique subjects. Starting subject-by-subject processing...")

all_results = []

# ### MODIFICATION: Main loop now handles median split and runs all conditions.
for subject_id in unique_subject_ids:
    print(f"\n--- Processing Subject: {subject_id} ---")
    subject_epochs_all = all_epochs[all_epochs.metadata[SUBJECT_ID_COLUMN] == subject_id].copy()

    # --- Perform Subject-Specific Median Split for Reaction Time ---
    # Use only correct trials to calculate the median RT
    correct_trials_meta = subject_epochs_all.metadata[subject_epochs_all.metadata[ACCURACY_COLUMN] == True]
    if not correct_trials_meta.empty and correct_trials_meta[REACTION_TIME_COLUMN].notna().any():
        with warnings.catch_warnings():  # Suppress warning if all RTs are NaN for a subject
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rt_median = np.nanmedian(correct_trials_meta[REACTION_TIME_COLUMN])

        # Add 'rt_split' column to this subject's metadata
        is_fast = (subject_epochs_all.metadata[ACCURACY_COLUMN] == True) & (
                    subject_epochs_all.metadata[REACTION_TIME_COLUMN] <= rt_median)
        is_slow = (subject_epochs_all.metadata[ACCURACY_COLUMN] == True) & (
                    subject_epochs_all.metadata[REACTION_TIME_COLUMN] > rt_median)
        subject_epochs_all.metadata['rt_split'] = np.nan
        subject_epochs_all.metadata.loc[is_fast, 'rt_split'] = 'fast'
        subject_epochs_all.metadata.loc[is_slow, 'rt_split'] = 'slow'
    else:
        subject_epochs_all.metadata['rt_split'] = np.nan  # Ensure column exists even if no RT data

    # --- Define all conditions to be processed for this subject ---
    conditions = [
        # Analysis 1: Correct vs Incorrect
        {'name': 'pre-stimulus_correct', 'query': f"{ACCURACY_COLUMN} == True", 'tmin': None, 'tmax': 0,
         'period': 'pre-stimulus', 'outcome': 'correct', 'rt_split': None},
        {'name': 'pre-stimulus_incorrect', 'query': f"{ACCURACY_COLUMN} == False", 'tmin': None, 'tmax': 0,
         'period': 'pre-stimulus', 'outcome': 'incorrect', 'rt_split': None},
        {'name': 'post-stimulus_correct', 'query': f"{ACCURACY_COLUMN} == True", 'tmin': 0, 'tmax': None,
         'period': 'post-stimulus', 'outcome': 'correct', 'rt_split': None},
        {'name': 'post-stimulus_incorrect', 'query': f"{ACCURACY_COLUMN} == False", 'tmin': 0, 'tmax': None,
         'period': 'post-stimulus', 'outcome': 'incorrect', 'rt_split': None},
        # Analysis 2: Fast vs Slow on Correct Trials
        {'name': 'pre-stimulus_fast', 'query': "rt_split == 'fast'", 'tmin': None, 'tmax': 0, 'period': 'pre-stimulus',
         'outcome': None, 'rt_split': 'fast'},
        {'name': 'pre-stimulus_slow', 'query': "rt_split == 'slow'", 'tmin': None, 'tmax': 0, 'period': 'pre-stimulus',
         'outcome': None, 'rt_split': 'slow'},
        {'name': 'post-stimulus_fast', 'query': "rt_split == 'fast'", 'tmin': 0, 'tmax': None,
         'period': 'post-stimulus', 'outcome': None, 'rt_split': 'fast'},
        {'name': 'post-stimulus_slow', 'query': "rt_split == 'slow'", 'tmin': 0, 'tmax': None,
         'period': 'post-stimulus', 'outcome': None, 'rt_split': 'slow'},
    ]

    for cond in conditions:
        print(f"  - Analyzing: {cond['name']}")
        # Use MNE's query system to select epochs, then crop
        epochs_for_analysis = subject_epochs_all[cond['query']].copy().crop(tmin=cond['tmin'], tmax=cond['tmax'])

        exponent = process_subject_psd_and_fooof(epochs_for_analysis, cond['name'], subject_id)

        if exponent is not None:
            all_results.append({
                "subject_id": subject_id,
                "period": cond['period'],
                "outcome": cond['outcome'],
                "rt_split": cond['rt_split'],
                "exponent": exponent
            })

print("\n\nAll subject processing complete. Analyzing and plotting results...")

# --- Final Analysis and Visualization ---
if all_results:
    results_df = pd.DataFrame(all_results)

    # Run and plot the two different analyses
    analyze_and_plot_within_subject(results_df, "pre-stimulus", 'outcome', ['correct', 'incorrect'])
    analyze_and_plot_within_subject(results_df, "post-stimulus", 'outcome', ['correct', 'incorrect'])
    analyze_and_plot_within_subject(results_df, "pre-stimulus", 'rt_split', ['fast', 'slow'])
    analyze_and_plot_within_subject(results_df, "post-stimulus", 'rt_split', ['fast', 'slow'])
    ### MODIFICATION: Create and display the wide-format DataFrame as requested ###
    print("\n\n--- Generating Wide-Format DataFrame ---")


    # Define a helper function to create a single, descriptive condition label
    def create_condition_label(row):
        if pd.notna(row['outcome']):
            return f"{row['period']}_{row['outcome']}"
        elif pd.notna(row['rt_split']):
            return f"{row['period']}_{row['rt_split']}"
        return None


    # Apply the function to create the new 'condition' column
    results_df['condition'] = results_df.apply(create_condition_label, axis=1)

    # Pivot the table to create the wide format
    wide_results_df = results_df.pivot_table(
        index='subject_id',
        columns='condition',
        values='exponent'
    )

    # Add a suffix to all columns for better clarity
    wide_results_df = wide_results_df.add_suffix('_exponent')

    # Reset the index to make 'subject_id' a regular column
    wide_results_df = wide_results_df.reset_index()
    wide_results_df.reset_index(names="subject_id", inplace=True)
    print("Wide-format DataFrame created successfully. Displaying head:")
    print(wide_results_df.head())
    output_path = f'{SPACEPRIME.get_data_path()}concatenated\\fooof_exponents.csv'
    # You can now save this wide DataFrame to a CSV file if you wish:
    wide_results_df.to_csv(output_path, index=False)
    # print("\nSaved wide-format DataFrame to CSV.")
else:
    print("No results were generated. Cannot perform final analysis.")

print("\nAll analyses complete.")
