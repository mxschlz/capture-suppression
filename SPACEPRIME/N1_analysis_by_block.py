import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from SPACEPRIME import load_concatenated_epochs
from SPACEPRIME.subjects import subject_ids
from stats import remove_outliers

plt.ion()

# --- Script Parameters ---

# --- 1. Preprocessing Parameters (adopted from ERPs_lateral_stimuli.py) ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2
REACTION_TIME_COL = 'rt'
PHASE_COL = 'phase'

# --- 2. N1 Analysis Parameters ---
# Time window for N1 component analysis (in seconds)
N1_WINDOW = (0.05, 0.15)

# ROIs for Contra/Ipsi Calculation
LEFT_ROI = ["C3"]
RIGHT_ROI = ["C4"]

# Epoching time window (should encompass the N1 window)
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.7

# --- 3. Plotting Parameters ---
AMPLITUDE_SCALE_FACTOR = 1e6  # To convert V to µV
PALETTE = "viridis"

# --- End of Parameters ---


def calculate_n1_amplitude(epochs, tmin, tmax):
    """
    Calculates the mean amplitude within a specified time window.

    Args:
        epochs (mne.Epochs): The epochs object to analyze.
        tmin (float): The start of the time window in seconds.
        tmax (float): The end of the time window in seconds.

    Returns:
        float: The mean amplitude over the specified channels and time window.
               Returns np.nan if no trials are present.
    """
    if len(epochs) == 0:
        return np.nan
    # Crop to the N1 window, get the data, and average over time points and trials
    return epochs.copy().crop(tmin, tmax).get_data().mean()


# --- Data Storage ---
# We will build a list of dictionaries and convert to a DataFrame at the end
results_list = []

# --- Load and Preprocess Data ---
print("--- Loading and Preprocessing Data ---")
epochs = load_concatenated_epochs("spaceprime").crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Original number of trials: {len(epochs)}")

# Get metadata for preprocessing
df_meta = epochs.metadata.copy().reset_index(drop=True)

# 1. Filter by phase
if PHASE_COL in df_meta.columns and FILTER_PHASE is not None:
    print(f"Filtering out trials from phase {FILTER_PHASE}...")
    df_meta = df_meta[df_meta[PHASE_COL] != FILTER_PHASE]
    print(f"  Trials remaining after phase filter: {len(df_meta)}")

# 2. Remove RT outliers
if REACTION_TIME_COL in df_meta.columns:
    print(f"Removing RT outliers (threshold: {OUTLIER_RT_THRESHOLD} SD)...")
    df_meta = remove_outliers(df_meta, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"  Trials remaining after RT outlier removal: {len(df_meta)}")

# 3. Apply the filter back to the epochs object
epochs = epochs[df_meta.index]
print(f"Final number of trials after preprocessing: {len(epochs)}")
# --- End of Preprocessing ---

# --- Subject and Block Loop ---
print("\n--- Analyzing N1 Amplitude per Subject per Block ---")
for subject_id_num in subject_ids:
    subject_str = f"sub-{subject_id_num}"
    print(f"Processing Subject: {subject_str}")

    try:
        epochs_sub = epochs[f"subject_id=={subject_id_num}"]
        if len(epochs_sub) == 0:
            print(f"  No epochs found for {subject_str} after preprocessing. Skipping.")
            continue

        all_conds_sub = list(epochs_sub.event_id.keys())

        # Get unique blocks for this subject from their metadata
        subject_blocks = sorted(epochs_sub.metadata['block'].unique())
        print(f"  Found blocks: {subject_blocks}")

        for block_num in subject_blocks:
            epochs_block = epochs_sub[f"block == {block_num}"]

            for loc_type in ['Singleton', 'Target']:
                # We define "left" and "right" trials based on the location type
                # 'loc' == 1 -> Left; 'loc' == 3 -> Right
                # Define conditions for source estimation
                if loc_type == "Singleton":
                    left_stim_epochs = epochs_block[
                        [x for x in all_conds_sub if "Target-2-Singleton-1" in x]].copy()
                    right_stim_epochs = epochs_block[
                        [x for x in all_conds_sub if f"Target-2-Singleton-3" in x]].copy()
                elif loc_type == "Target":
                    left_stim_epochs = epochs_block[
                        [x for x in all_conds_sub if "Target-1-Singleton-2" in x]].copy()
                    right_stim_epochs = epochs_block[
                        [x for x in all_conds_sub if f"Target-3-Singleton-2" in x]].copy()
                else: raise ValueError

                # --- Calculate Contra and Ipsi N1 Amplitudes ---

                # For LEFT stimuli: Contra is RIGHT_ROI, Ipsi is LEFT_ROI
                contra_n1_left_stim = calculate_n1_amplitude(
                    left_stim_epochs.copy().pick(RIGHT_ROI), N1_WINDOW[0], N1_WINDOW[1]
                )
                ipsi_n1_left_stim = calculate_n1_amplitude(
                    left_stim_epochs.copy().pick(LEFT_ROI), N1_WINDOW[0], N1_WINDOW[1]
                )

                # For RIGHT stimuli: Contra is LEFT_ROI, Ipsi is RIGHT_ROI
                contra_n1_right_stim = calculate_n1_amplitude(
                    right_stim_epochs.copy().pick(LEFT_ROI), N1_WINDOW[0], N1_WINDOW[1]
                )
                ipsi_n1_right_stim = calculate_n1_amplitude(
                    right_stim_epochs.copy().pick(RIGHT_ROI), N1_WINDOW[0], N1_WINDOW[1]
                )

                # Average across stimulus sides to get a single contra and ipsi value per block
                # We use np.nanmean to handle cases where one side might have no trials
                avg_contra_n1 = np.nanmean([contra_n1_left_stim, contra_n1_right_stim])
                avg_ipsi_n1 = np.nanmean([ipsi_n1_left_stim, ipsi_n1_right_stim])

                # Calculate the difference
                diff_n1 = avg_contra_n1 - avg_ipsi_n1

                # Append results for this subject and block
                results_list.append({
                    'subject': subject_id_num,
                    'block': block_num,
                    'loc_type': loc_type,
                    'n1_contra': avg_contra_n1,
                    'n1_ipsi': avg_ipsi_n1,
                    'n1_diff': diff_n1
                })

    except Exception as e:
        print(f"  An error occurred while processing {subject_str}: {e}")
        continue

# Convert the list of results into a pandas DataFrame
results_df = pd.DataFrame(results_list)

# Scale amplitudes to microvolts for easier interpretation
for col in ['n1_contra', 'n1_ipsi', 'n1_diff']:
    if col in results_df.columns:
        results_df[col] *= AMPLITUDE_SCALE_FACTOR

print("\n--- Analysis Complete ---")
print("First 5 rows of the results DataFrame:")
print(results_df.head())

# --- Visualization ---
if not results_df.empty:
    print("\n--- Generating Plots ---")
    n_subjects = len(results_df['subject'].unique())
    n_blocks = len(results_df['block'].unique())

    # --- Plot 1: Bar plot of Contra-Ipsi N1 difference across blocks ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=results_df, x='block', y='n1_diff', hue='loc_type', palette=PALETTE, ax=ax1, errorbar='se')
    ax1.set_title(f'Mean Contra-Ipsi N1 Amplitude Across Blocks (N={n_subjects})', fontsize=16)
    ax1.set_xlabel('Experimental Block', fontsize=12)
    ax1.set_ylabel('N1 Amplitude Difference (Contra - Ipsi) [µV]', fontsize=12)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.legend(title='Lateralized Stimulus')
    sns.despine(ax=ax1)
    fig1.tight_layout()
    fig1.canvas.draw_idle()

    # --- Plot 2: Point plot showing trend and individual subject data ---
    # Using catplot to create facets for each location type.
    g = sns.catplot(
        data=results_df,
        x='block',
        y='n1_diff',
        col='loc_type',
        kind='point',
        errorbar='se',
        capsize=0.1,
        join=True,
        height=6,
        aspect=1,
        color='black'
    )

    # Overlay stripplot on each facet
    for ax, loc_type in zip(g.axes.flat, g.col_names):
        sns.stripplot(
            data=results_df[results_df['loc_type'] == loc_type],
            x='block',
            y='n1_diff',
            hue='subject',
            palette='tab20',
            jitter=0.1,
            alpha=0.6,
            ax=ax,
            legend=False
        )
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

    g.fig.suptitle(f'Contra-Ipsi N1 Amplitude Trend Across Blocks (N={n_subjects})', fontsize=18, y=1.03)
    g.set_axis_labels('Experimental Block', 'N1 Amplitude Difference (Contra - Ipsi) [µV]')
    g.set_titles("Lateralized Stimulus: {col_name}")
    g.fig.tight_layout(rect=[0, 0, 1, 0.97])
    g.fig.canvas.draw_idle()

else:
    print("\nNo data was processed, skipping plot generation.")
