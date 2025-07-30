import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import sem

import SPACEPRIME
from utils import calculate_fractional_area_latency, get_all_waves
import pingouin as pg

plt.ion()

# ===================================================================
# --- SCRIPT CONFIGURATION ---
# ===================================================================

# 4. ERP Component Definitions
COMPONENT_TIME_WINDOW = (0.0, 0.7)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# 5. ERP Metric & Statistical Parameters
LATENCY_ANALYSIS_WINDOW = (0.2, 0.4)
LATENCY_PERCENTAGE = 0.5

# ===================================================================
# --- STEP 1: LOAD PRE-COMPUTED MOUSE TRAJECTORY CLASSIFICATIONS ---
# ===================================================================
print("\n--- Step 1: Loading Pre-computed Initial Movement Classifications ---")

trajectory_classifications_df = SPACEPRIME.load_concatenated_csv("initial_movement_classifications.csv")
capture_scores = SPACEPRIME.load_concatenated_csv("capture_scores.csv")

# ===================================================================
# --- STEP 2: LOAD ERP DATA AND MERGE WITH TRAJECTORY INFO ---
# ===================================================================
print("\n--- Step 2: Loading ERP Data and Merging ---")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime").crop(COMPONENT_TIME_WINDOW[0], COMPONENT_TIME_WINDOW[1])
all_times = epochs.times
metadata_df = epochs.metadata.copy()

# Reshape ERP data to wide format
print("Reshaping ERP data to wide format...")
erp_df_picks_flat = [item for pair in set(N2AC_ELECTRODES + PD_ELECTRODES) for item in pair]
erp_df_picks_unique_flat = sorted(list(set(erp_df_picks_flat)))
erp_df = epochs.to_data_frame(picks=erp_df_picks_unique_flat, time_format=None)
erp_wide = erp_df.pivot(index='epoch', columns='time')
erp_wide = erp_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
erp_wide.columns = erp_wide.columns.droplevel(0)

# Merge metadata, ERP data, and trajectory classifications
print("Merging all data sources...")
merged_df = metadata_df.join(erp_wide)
final_df = pd.merge(
    merged_df,
    trajectory_classifications_df,
    on=['subject_id', 'block', 'trial_nr'],
    how='inner' # Keep only trials that have both ERP and valid trajectory data
)
print(f"Final merged data shape: {final_df.shape}")

# ===================================================================
# --- STEP 3: CALCULATE ERP METRICS BY MOVEMENT DIRECTION ---
# ===================================================================
print("\n--- Step 3: Calculating ERPs by Movement Direction ---")

# Filter for trials relevant to each component
final_df['TargetLoc'] = pd.to_numeric(final_df['TargetLoc'], errors='coerce').map({1: "left", 2: "mid", 3: "right"})
final_df['SingletonLoc'] = pd.to_numeric(final_df['SingletonLoc'], errors='coerce').map({0: "absent", 1: "left", 2: "mid", 3: "right"})
n2ac_base_df = final_df[(final_df['TargetLoc'].isin(['left', 'right'])) & (final_df['SingletonLoc'] == 'mid')].copy()
pd_base_df = final_df[(final_df['SingletonLoc'].isin(['left', 'right'])) & (final_df['TargetLoc'] == 'mid')].copy()

subject_agg_data = []
for subject_id in final_df['subject_id'].unique():
    print(f"Processing subject: {subject_id}")
    components_to_process = {
        'N2ac': {'df': n2ac_base_df[n2ac_base_df['subject_id'] == subject_id], 'stim_col': 'TargetLoc', 'electrodes': N2AC_ELECTRODES, 'is_target': True},
        'Pd':   {'df': pd_base_df[pd_base_df['subject_id'] == subject_id], 'stim_col': 'SingletonLoc', 'electrodes': PD_ELECTRODES, 'is_target': False}
    }

    for comp_name, params in components_to_process.items():
        # Split by initial movement direction
        for move_dir in ['target', 'distractor', 'control']:
            cond_df = params['df'][params['df']['initial_movement_direction'] == move_dir]
            if cond_df.empty: continue

            # Capture all returned waves: diff, contra, and ipsi
            diff_wave, contra_wave, ipsi_wave, times = get_all_waves(
                cond_df, params['electrodes'], COMPONENT_TIME_WINDOW, all_times, params['stim_col']
            )

            latency, amplitude = np.nan, np.nan
            if diff_wave is not None:
                latency = calculate_fractional_area_latency(diff_wave, times, percentage=LATENCY_PERCENTAGE,
                                                            is_target=params['is_target'],
                                                            analysis_window_times=LATENCY_ANALYSIS_WINDOW)
                if not np.isnan(latency) and (times[0] <= latency <= times[-1]):
                    amplitude = np.interp(latency, times, diff_wave)

            subject_agg_data.append({
                'subject': subject_id, 'component': comp_name, 'condition': move_dir,
                'latency': latency, 'amplitude': amplitude, 'total_trials': len(cond_df),
                'wave': diff_wave,
                'contra_wave': contra_wave,  # Store contra wave
                'ipsi_wave': ipsi_wave,  # Store ipsi wave
                'times': times
            })

agg_df = pd.DataFrame(subject_agg_data)

# ===================================================================
# --- STEP 4: PLOT GRAND-AVERAGE DIFFERENCE WAVES ---
# ===================================================================
print("\n--- Step 4: Plotting Grand-Average Difference Waves ---")
fig_diff, axes_diff = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
fig_diff.suptitle('Grand-Average Difference Waves by Initial Movement Direction', fontsize=20, y=1.06)
axes_diff = axes_diff.flatten()
components_to_plot = ['N2ac', 'Pd']
colors = {'target': '#2ca02c', 'distractor': '#d62728', 'control': '#1f77b4'} # Green, Red, Blue

for i, comp_name in enumerate(components_to_plot):
    ax = axes_diff[i]
    ax.set_title(comp_name, fontsize=16)
    plot_df = agg_df[agg_df['component'] == comp_name]

    for move_dir, color in colors.items():
        cond_df = plot_df[plot_df['condition'] == move_dir].dropna(subset=['wave'])
        if cond_df.empty: continue

        ga_wave = np.mean(np.stack(cond_df['wave'].values), axis=0)
        sem_wave = sem(np.stack(cond_df['wave'].values), axis=0)
        times_for_plot = cond_df['times'].iloc[0]

        ax.plot(times_for_plot, ga_wave, color=color, lw=2.5, label=f'{move_dir.capitalize()} (N={len(cond_df)})')
        ax.fill_between(times_for_plot, ga_wave - sem_wave, ga_wave + sem_wave, color=color, alpha=0.2)

    ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.2, zorder=0)
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axvline(0, color='k', linestyle=':', lw=1)
    ax.legend(loc='best')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

# ===================================================================
# --- STEP 5: PLOT & ANALYZE ERP METRICS ---
# ===================================================================
print("\n--- Step 5: Plotting and Analyzing ERP Metrics ---")
fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
# Update the title to reflect the new analysis
fig_metrics.suptitle('ERP Metrics by Initial Movement Direction (Pairwise Comparisons)', fontsize=20, y=1.06)
axes_metrics = axes_metrics.flatten()
components_to_plot = ['N2ac', 'Pd']
colors = {'target': '#2ca02c', 'distractor': '#d62728', 'control': '#1f77b4'}

for i, comp_name in enumerate(components_to_plot):
    ax = axes_metrics[i]
    ax.set_title(comp_name, fontsize=16)
    plot_df = agg_df[agg_df['component'] == comp_name].copy()

    # --- Perform Pairwise Comparisons ---
    if pg:
        print(f"\n{'=' * 20} STATISTICAL ANALYSIS FOR: {comp_name.upper()} {'=' * 20}")

        # --- Latency Analysis ---
        print(f"\n--- Pairwise Comparisons for Latency ({comp_name}) ---")
        # Prepare data in long format, dropping subjects who don't have data for all conditions
        # This ensures the pairwise tests are properly paired.
        pairwise_df_lat = plot_df.pivot(index='subject', columns='condition', values='latency').dropna()

        if len(pairwise_df_lat) > 1:
            # Convert back to long format for pingouin's pairwise_tests function
            long_df_lat = pairwise_df_lat.reset_index().melt(
                id_vars='subject',
                value_vars=['target', 'distractor', 'control'],
                var_name='condition',
                value_name='latency'
            )

            # Directly run pairwise t-tests between all conditions with Holm correction
            pairwise_lat = pg.pairwise_tests(
                data=long_df_lat,
                dv='latency',
                within='condition',
                subject='subject',
                padjust='holm'
            )
            print("Pairwise T-Tests for Latency (Holm-corrected p-values):")
            print(pairwise_lat.round(4))
        else:
            print("Not enough subjects with complete latency data to run pairwise tests.")

        # --- Amplitude Analysis ---
        print(f"\n--- Pairwise Comparisons for Amplitude ({comp_name}) ---")
        pairwise_df_amp = plot_df.pivot(index='subject', columns='condition', values='amplitude').dropna()

        if len(pairwise_df_amp) > 1:
            long_df_amp = pairwise_df_amp.reset_index().melt(
                id_vars='subject',
                value_vars=['target', 'distractor', 'control'],
                var_name='condition',
                value_name='amplitude'
            )

            pairwise_amp = pg.pairwise_tests(
                data=long_df_amp,
                dv='amplitude',
                within='condition',
                subject='subject',
                padjust='holm'
            )
            print("\nPairwise T-Tests for Amplitude (Holm-corrected p-values):")
            print(pairwise_amp.round(4))
        else:
            print("Not enough subjects with complete amplitude data to run pairwise tests.")
        print(f"{'=' * 60}")

    # --- Plotting (Visual representation remains the same) ---
    # Latency Plot
    sns.pointplot(data=plot_df, x='condition', y='latency', order=colors.keys(), ax=ax, join=True, errorbar=('se', 1),
                  scale=0.8, errwidth=1.5, capsize=0.1, color='C0')
    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel(f'{int(LATENCY_PERCENTAGE * 100)}% Latency (s)', fontsize=14, color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.grid(axis='x', linestyle=':', alpha=0.7)

    # Amplitude Plot on a second y-axis
    ax_amp = ax.twinx()
    sns.pointplot(data=plot_df, x='condition', y='amplitude', order=colors.keys(), ax=ax_amp,
                  join=True, errorbar=('se', 1), scale=0.8, errwidth=1.5, capsize=0.1,
                  color='C1', linestyles='--')
    ax_amp.set_ylabel('Amplitude (µV)', fontsize=14, color='C1')
    ax_amp.tick_params(axis='y', labelcolor='C1')
    ax_amp.grid(False)

    # Create a clean, informative legend
    legend_handles = [
        Line2D([0], [0], color='C0', lw=2, label='Latency'),
        Line2D([0], [0], color='C1', lw=2, linestyle='--', label='Amplitude')
    ]
    ax.legend(handles=legend_handles, loc='best')

plt.show()

# ===================================================================
# --- STEP 6: PLOT GRAND-AVERAGE CONTRA/IPSI WAVES (SEPARATE SUBPLOTS) ---
# ===================================================================
print("\n--- Step 6: Visualizing Grand-Average Contralateral and Ipsilateral ERPs in detail ---")

# Create a 2x3 grid of subplots for a clearer view
# Rows: N2ac, Pd
# Columns: Target, Distractor, Control
fig_contra_ipsi, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True, constrained_layout=True)
fig_contra_ipsi.suptitle('Grand-Average Contralateral & Ipsilateral ERPs by Component and Movement',
                         fontsize=20, y=1.04)

conditions_to_plot = ['target', 'distractor', 'control']

# Loop through rows (components: N2ac, Pd)
for row_idx, comp_name in enumerate(components_to_plot):
    # Loop through columns (movement directions: target, distractor, control)
    for col_idx, move_dir in enumerate(conditions_to_plot):
        ax = axes[row_idx, col_idx]
        color = colors[move_dir]

        # Set title for the subplot
        ax.set_title(f"{comp_name} - {move_dir.capitalize()} Movement", fontsize=14)

        # Filter data for this specific component and condition
        cond_df = agg_df[
            (agg_df['component'] == comp_name) &
            (agg_df['condition'] == move_dir)
        ].dropna(subset=['contra_wave', 'ipsi_wave'])

        if cond_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        n_subjects = len(cond_df)
        times_for_plot = cond_df['times'].iloc[0]

        # Grand average waves
        ga_contra_wave = np.mean(np.stack(cond_df['contra_wave'].values), axis=0)
        ga_ipsi_wave = np.mean(np.stack(cond_df['ipsi_wave'].values), axis=0)
        sem_contra = sem(np.stack(cond_df['contra_wave'].values), axis=0)
        sem_ipsi = sem(np.stack(cond_df['ipsi_wave'].values), axis=0)

        # Plot Contra and Ipsi waves
        ax.plot(times_for_plot, ga_contra_wave, color=color, lw=2.5, linestyle='-', label=f'Contralateral (N={n_subjects})')
        ax.fill_between(times_for_plot, ga_contra_wave - sem_contra, ga_contra_wave + sem_contra, color=color, alpha=0.2)

        ax.plot(times_for_plot, ga_ipsi_wave, color=color, lw=2.5, linestyle='--', label=f'Ipsilateral (N={n_subjects})')
        ax.fill_between(times_for_plot, ga_ipsi_wave - sem_ipsi, ga_ipsi_wave + sem_ipsi, color=color, alpha=0.2)

        # Aesthetics
        ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.2, zorder=0)
        ax.axhline(0, color='k', linestyle='--', lw=1)
        ax.axvline(0, color='k', linestyle=':', lw=1)
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Add labels only to the outer plots to avoid clutter
        if col_idx == 0:
            ax.set_ylabel("Amplitude (µV)", fontsize=12)
        if row_idx == 1:
            ax.set_xlabel("Time (s)", fontsize=12)

plt.show()

# ===================================================================
# --- STEP 7: BRAIN-BEHAVIOR ANALYSIS (BINNING BY CAPTURE SCORE) ---
# ===================================================================
print("\n--- Step 7: Brain-Behavior Analysis ---")
print("Analyzing relationship between ERP components and continuous capture score.")

# --- 7.1: Merge Capture Scores into the main DataFrame ---
# The capture_scores df should have subject_id, block, trial_nr, and the score itself.
# We'll merge it into the base dataframes for N2ac and Pd.

# First, ensure the score column has a consistent name (e.g., 'capture_score').
if 'attraction_score' in capture_scores.columns:
    capture_scores.rename(columns={'attraction_score': 'capture_score'}, inplace=True)

# To prevent creating duplicate columns with _x/_y suffixes, we select only the
# essential columns from the capture_scores dataframe before merging.
merge_keys = ['subject_id', 'block', 'trial_nr']
score_col = 'capture_score'

# Ensure the score column actually exists before trying to select it
if score_col not in capture_scores.columns:
    raise ValueError(f"Column '{score_col}' not found in capture_scores.csv. Please check the file.")

# Create a lean DataFrame with only the keys and the new score column
capture_scores_to_merge = capture_scores[merge_keys + [score_col]].copy()

# Ensure key columns are of the same type for merging across all dataframes
for df_to_fix in [n2ac_base_df, pd_base_df, capture_scores_to_merge]:
    for col in merge_keys:
        # Use pd.to_numeric to handle potential type mismatches (e.g., float vs int)
        df_to_fix[col] = pd.to_numeric(df_to_fix[col], errors='coerce')

print("Merging capture scores into ERP data...")
n2ac_binned_df = pd.merge(n2ac_base_df, capture_scores_to_merge, on=merge_keys, how='inner')
pd_binned_df = pd.merge(pd_base_df, capture_scores_to_merge, on=merge_keys, how='inner')

print(f"Successfully merged capture scores. New N2ac shape: {n2ac_binned_df.shape}, New Pd shape: {pd_binned_df.shape}")
# --- 7.2: Bin Trials by Capture Score for each Subject ---
print("Binning trials by capture score into quartiles for each subject...")
N_BINS = 2
bin_labels = [f'Q{i + 1}' for i in range(N_BINS)]


# Define a function to apply qcut safely within each group
def bin_subject_data(df, col_to_bin='capture_score', n_bins=N_BINS, labels=bin_labels):
    # Need at least as many data points as bins to create meaningful bins
    if len(df) < n_bins:
        return None
    try:
        # qcut creates bins with an equal number of data points
        df['capture_bin'] = pd.qcut(df[col_to_bin], q=n_bins, labels=labels, duplicates='drop')
        return df
    except ValueError:  # This can happen if there are not enough unique values to create n_bins
        return None


# Apply the binning function to each subject's data
n2ac_binned_df = n2ac_binned_df.groupby('subject_id', group_keys=False).apply(bin_subject_data).dropna(
    subset=['capture_bin'])
pd_binned_df = pd_binned_df.groupby('subject_id', group_keys=False).apply(bin_subject_data).dropna(
    subset=['capture_bin'])

print(f"Successfully binned N2ac trials: {len(n2ac_binned_df)}, Pd trials: {len(pd_binned_df)}")

# --- 7.3: Calculate ERPs for each Capture Score Bin ---
print("Calculating ERPs for each capture score bin...")
binned_agg_data = []
components_to_process_binned = {
    'N2ac': {'df': n2ac_binned_df, 'stim_col': 'TargetLoc', 'electrodes': N2AC_ELECTRODES, 'is_target': True},
    'Pd': {'df': pd_binned_df, 'stim_col': 'SingletonLoc', 'electrodes': PD_ELECTRODES, 'is_target': False}
}

for comp_name, params in components_to_process_binned.items():
    for subject_id in params['df']['subject_id'].unique():
        subject_df = params['df'][params['df']['subject_id'] == subject_id]
        for bin_label in bin_labels:
            cond_df = subject_df[subject_df['capture_bin'] == bin_label]
            if cond_df.empty: continue

            diff_wave, _, _, times = get_all_waves(
                cond_df, params['electrodes'], COMPONENT_TIME_WINDOW, all_times, params['stim_col']
            )

            latency, amplitude = np.nan, np.nan
            if diff_wave is not None:
                latency = calculate_fractional_area_latency(diff_wave, times, percentage=LATENCY_PERCENTAGE,
                                                            is_target=params['is_target'],
                                                            analysis_window_times=LATENCY_ANALYSIS_WINDOW)
                if not np.isnan(latency) and (times[0] <= latency <= times[-1]):
                    amplitude = np.interp(latency, times, diff_wave)

            binned_agg_data.append({
                'subject': subject_id, 'component': comp_name, 'capture_bin': bin_label,
                'latency': latency, 'amplitude': amplitude, 'total_trials': len(cond_df),
                'wave': diff_wave, 'times': times
            })

binned_agg_df = pd.DataFrame(binned_agg_data)

# --- 7.4: Plot Grand-Average Waves by Capture Score Bin ---
print("Plotting grand-average waves by capture score bin...")
fig_binned_waves, axes_binned_waves = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
fig_binned_waves.suptitle('Brain-Behavior: ERPs by Capture Score Quartile', fontsize=20, y=1.06)
axes_binned_waves = axes_binned_waves.flatten()
bin_colors = sns.color_palette("viridis_r", n_colors=N_BINS)

for i, comp_name in enumerate(components_to_plot):
    ax = axes_binned_waves[i]
    ax.set_title(comp_name, fontsize=16)
    plot_df = binned_agg_df[binned_agg_df['component'] == comp_name]

    for j, bin_label in enumerate(bin_labels):
        cond_df = plot_df[plot_df['capture_bin'] == bin_label].dropna(subset=['wave'])
        if cond_df.empty: continue

        ga_wave = np.mean(np.stack(cond_df['wave'].values), axis=0)
        sem_wave = sem(np.stack(cond_df['wave'].values), axis=0)
        times_for_plot = cond_df['times'].iloc[0]

        ax.plot(times_for_plot, ga_wave, color=bin_colors[j], lw=2.5, label=f'{bin_label}')
        ax.fill_between(times_for_plot, ga_wave - sem_wave, ga_wave + sem_wave, color=bin_colors[j], alpha=0.2)

    ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.2, zorder=0)
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axvline(0, color='k', linestyle=':', lw=1)
    ax.legend(title="Capture Score Quartile\n(Q1=Capture -> Q4=Suppress)", loc='best')
    ax.set_xlabel("Time (s)", fontsize=12)
    if i == 0: ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.show()

# --- 7.5: Plot & Analyze ERP Metrics by Capture Score Bin ---
print("Plotting and analyzing ERP metrics by capture score bin...")
fig_binned_metrics, axes_binned_metrics = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
fig_binned_metrics.suptitle('Brain-Behavior: ERP Metrics by Capture Score Quartile', fontsize=20, y=1.06)
axes_binned_metrics = axes_binned_metrics.flatten()

for i, comp_name in enumerate(components_to_plot):
    ax = axes_binned_metrics[i]
    ax.set_title(comp_name, fontsize=16)
    plot_df = binned_agg_df[binned_agg_df['component'] == comp_name].copy()

    # --- Statistical Analysis: RM-ANOVA on Bins ---
    print(f"\n{'=' * 20} STATISTICAL ANALYSIS FOR: {comp_name.upper()} BY CAPTURE BIN {'=' * 20}")
    for metric in ['latency', 'amplitude']:
        # Ensure there's enough data to run the ANOVA
        if plot_df[metric].dropna().empty:
            print(f"Skipping {metric} analysis for {comp_name} due to missing data.")
            continue
        print(f"\n--- RM-ANOVA for {metric.capitalize()} ({comp_name}) ---")
        # Drop subjects who don't have data for all bins to ensure a balanced design
        metric_df = plot_df.pivot(index='subject', columns='capture_bin', values=metric).dropna()
        if len(metric_df) < 2:
            print(f"Not enough subjects with complete data for {metric} ANOVA.")
            continue

        long_metric_df = metric_df.reset_index().melt(id_vars='subject', value_name=metric, var_name='capture_bin')
        rm_anova_results = pg.rm_anova(data=long_metric_df, dv=metric, within='capture_bin', subject='subject',
                                       detailed=True)
        print(rm_anova_results.round(4))

    # --- Plotting Metrics ---
    sns.pointplot(data=plot_df, x='capture_bin', y='latency', order=bin_labels, ax=ax, join=True, errorbar=('se', 1),
                  scale=0.8, errwidth=1.5, capsize=0.1, color='C0')
    ax.set_xlabel('Capture Score Quartile (Q1=Capture -> Q4=Suppress)', fontsize=14)
    ax.set_ylabel(f'{int(LATENCY_PERCENTAGE * 100)}% Latency (s)', fontsize=14, color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.grid(axis='x', linestyle=':', alpha=0.7)

    ax_amp = ax.twinx()
    sns.pointplot(data=plot_df, x='capture_bin', y='amplitude', order=bin_labels, ax=ax_amp,
                  join=True, errorbar=('se', 1), scale=0.8, errwidth=1.5, capsize=0.1,
                  color='C1', linestyles='--')
    ax_amp.set_ylabel('Amplitude (µV)', fontsize=14, color='C1')
    ax_amp.tick_params(axis='y', labelcolor='C1')
    ax_amp.grid(False)

    legend_handles = [
        Line2D([0], [0], color='C0', lw=2, label='Latency'),
        Line2D([0], [0], color='C1', lw=2, linestyle='--', label='Amplitude')
    ]
    ax.legend(handles=legend_handles, loc='best')

plt.show()
