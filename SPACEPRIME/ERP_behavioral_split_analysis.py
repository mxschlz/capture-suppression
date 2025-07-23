import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Import for creating custom legend handles
import SPACEPRIME
import pandas as pd
import seaborn as sns
import numpy as np
from stats import remove_outliers # Assuming this is your custom outlier removal function
from utils import get_contra_ipsi_diff_wave, calculate_fractional_area_latency
from mne.stats import permutation_t_test

plt.ion()

# --- Script Configuration ---

# 1. Data Loading & Preprocessing
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2

# 2. Column Names
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
ACCURACY_INT_COL = 'select_target_int'
RT_SPLIT_COL = 'rt_split'

# 3. ERP Component Definitions
# This window is for extracting the wave from the epoch data
COMPONENT_TIME_WINDOW = (0.0, 0.7)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# 4. Statistical & Analysis Parameters
# This window is for the fractional area latency calculation itself
LATENCY_ANALYSIS_WINDOW = (0.2, 0.4)
LATENCY_PERCENTAGE = 0.5
N_PERMUTATIONS = 10000
P_VAL_ALPHA = 0.05
SEED = 42

# --- Main Script ---

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime").crop(COMPONENT_TIME_WINDOW[0], COMPONENT_TIME_WINDOW[1])
df = epochs.metadata.copy()

# Preprocessing
df = df[df[PHASE_COL] != FILTER_PHASE]
df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map({1: "left", 2: "mid", 3: "right"})
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(
    {0: "absent", 1: "left", 2: "mid", 3: "right"})
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

# --- ERP Data Reshaping ---
erp_df_picks_flat = [item for pair in set(N2AC_ELECTRODES + PD_ELECTRODES) for item in pair]
erp_df_picks_unique_flat = sorted(list(set(erp_df_picks_flat)))
erp_df = epochs.to_data_frame(picks=erp_df_picks_unique_flat, time_format=None)
print("Reshaping ERP data to wide format...")
erp_wide = erp_df.pivot(index='epoch', columns='time')
erp_wide = erp_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
erp_wide.columns = erp_wide.columns.droplevel(0)
merged_df = df.join(erp_wide)
print(f"Merged metadata with wide ERP data. Shape: {merged_df.shape}")
all_times = epochs.times

# --- 2. Perform Behavioral Splits and Aggregate ERP Metrics by Subject ---
print("\n--- Step 2: Performing Splits and Aggregating ERP Metrics ---")

# Filter for N2ac and Pd-relevant trials
is_target_lateral = merged_df[TARGET_COL].isin(['left', 'right'])
is_distractor_lateral = merged_df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = merged_df[TARGET_COL] == 'mid'
is_distractor_central = merged_df[DISTRACTOR_COL] == 'mid'
n2ac_base_df = merged_df[is_target_lateral & is_distractor_central].copy()
pd_base_df = merged_df[is_distractor_lateral & is_target_central].copy()

print("Performing component-specific median splits...")
n2ac_base_df[RT_SPLIT_COL] = n2ac_base_df.groupby(SUBJECT_ID_COL)[REACTION_TIME_COL].transform(
    lambda x: pd.qcut(x, 2, labels=['fast', 'slow'], duplicates='drop'))
pd_base_df[RT_SPLIT_COL] = pd_base_df.groupby(SUBJECT_ID_COL)[REACTION_TIME_COL].transform(
    lambda x: pd.qcut(x, 2, labels=['fast', 'slow'], duplicates='drop'))

# --- Loop through subjects and conditions to calculate ERP metrics ---
subject_agg_data = []
for subject_id in merged_df[SUBJECT_ID_COL].unique():
    print(f"Processing subject: {subject_id}")

    components_to_process = {
        'N2ac': {'df': n2ac_base_df[n2ac_base_df[SUBJECT_ID_COL] == subject_id],
                 'stim_col': TARGET_COL, 'electrodes': N2AC_ELECTRODES, 'is_target': True},
        'Pd':   {'df': pd_base_df[pd_base_df[SUBJECT_ID_COL] == subject_id],
                 'stim_col': DISTRACTOR_COL, 'electrodes': PD_ELECTRODES, 'is_target': False}
    }

    for comp_name, params in components_to_process.items():
        comp_df = params['df']
        if comp_df.empty: continue

        splits = {'RT': ['fast', 'slow'], 'Accuracy': ['correct', 'incorrect']}
        for split_by, conditions in splits.items():
            for cond_name in conditions:
                if split_by == 'RT':
                    mask = (comp_df[RT_SPLIT_COL] == cond_name)
                else: # Accuracy
                    acc_val = 1 if cond_name == 'correct' else 0
                    mask = (comp_df[ACCURACY_INT_COL] == acc_val)
                cond_df = comp_df[mask]

                if cond_df.empty: continue

                wave, times = get_contra_ipsi_diff_wave(
                    cond_df, params['electrodes'], COMPONENT_TIME_WINDOW, all_times, params['stim_col']
                )

                latency, amplitude = np.nan, np.nan
                if wave is not None:
                    # Calculate 50% fractional area latency within the specified analysis window
                    latency = calculate_fractional_area_latency(
                        wave, times,
                        percentage=LATENCY_PERCENTAGE,
                        is_target=params['is_target'],
                        analysis_window_times=LATENCY_ANALYSIS_WINDOW
                    )
                    # Get amplitude at that specific latency
                    if not np.isnan(latency) and (times[0] <= latency <= times[-1]):
                        amplitude = np.interp(latency, times, wave)

                subject_agg_data.append({
                    'subject': subject_id,
                    'component': comp_name,
                    'split_by': split_by,
                    'condition': cond_name,
                    'latency': latency,
                    'amplitude': amplitude,
                    'total_trials': len(cond_df)
                })

agg_df = pd.DataFrame(subject_agg_data)
print("Aggregation complete. Resulting data shape:", agg_df.shape)


# --- 3. Analyze and Plot ERP Metrics with Permutation Tests ---
print("\n--- Step 3: Analyzing and Plotting ERP Metrics ---")

comparisons = [
    {'title': 'N2ac by RT', 'component': 'N2ac', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'title': 'N2ac by Accuracy', 'component': 'N2ac', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
    {'title': 'Pd by RT', 'component': 'Pd', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'title': 'Pd by Accuracy', 'component': 'Pd', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
]

# REFACTORED: Create a 2x2 grid of plots with a more balanced size
fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
fig.suptitle('ERP Metric Analysis by Behavioral Split (Paired Permutation Test)', fontsize=20, y=1.06)
axes = axes.flatten()

# Define colors for clarity
LATENCY_COLOR = 'C0'  # Blue
AMPLITUDE_COLOR = 'C1' # Orange

for i, comp_info in enumerate(comparisons):
    ax = axes[i]
    ax.set_title(comp_info['title'], fontsize=16)

    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])].copy()
    cond1_name, cond2_name = comp_info['conds']

    legend_handles = []

    # --- Latency Analysis & Plot (Left Y-axis) ---
    pivot_lat = plot_df.pivot(index='subject', columns='condition', values='latency').dropna()
    if len(pivot_lat) > 2:
        sns.pointplot(data=plot_df, x='condition', y='latency', order=comp_info['conds'], ax=ax,
                      join=True, errorbar=('ci', 95), scale=0.8, errwidth=1.5, capsize=0.1, color=LATENCY_COLOR)

        diffs_lat = pivot_lat[cond1_name].values - pivot_lat[cond2_name].values
        X_lat = diffs_lat[:, np.newaxis]
        _, p_val_lat, _ = permutation_t_test(X_lat, n_permutations=N_PERMUTATIONS, seed=SEED)

        # Create the legend label for latency with p-value
        p_text = f"p={p_val_lat[0]:.3f}"
        if p_val_lat[0] < P_VAL_ALPHA:
            p_text += "*" # Add significance marker
        lat_label = f'Latency ({p_text})'
        legend_handles.append(Line2D([0], [0], color=LATENCY_COLOR, lw=2, label=lat_label))
    else:
        ax.text(0.5, 0.75, "No Latency Data", ha='center', va='center', color=LATENCY_COLOR)

    ax.set_xlabel('Condition', fontsize=14)
    ax.set_ylabel(f'{int(LATENCY_PERCENTAGE*100)}% Latency (s)', fontsize=14, color=LATENCY_COLOR)
    ax.tick_params(axis='y', labelcolor=LATENCY_COLOR)
    ax.grid(axis='x', linestyle=':', alpha=0.7)

    # --- Amplitude Analysis & Plot (Right Y-axis) ---
    ax_amp = ax.twinx()  # Create the second y-axis

    # Invert N2ac amplitude for plotting and set the correct label
    if comp_info['component'] == 'N2ac':
        plot_df['amplitude'] *= -1
        amp_label = 'Inverted Amplitude (µV)'
    else:
        amp_label = 'Amplitude (µV)'

    pivot_amp = plot_df.pivot(index='subject', columns='condition', values='amplitude').dropna()
    if len(pivot_amp) > 2:
        sns.pointplot(data=plot_df, x='condition', y='amplitude', order=comp_info['conds'], ax=ax_amp,
                      join=True, errorbar=('ci', 95), scale=0.8, errwidth=1.5, capsize=0.1,
                      color=AMPLITUDE_COLOR, linestyles='--') # Use a different linestyle

        diffs_amp = pivot_amp[cond1_name].values - pivot_amp[cond2_name].values
        X_amp = diffs_amp[:, np.newaxis]
        _, p_val_amp, _ = permutation_t_test(X_amp, n_permutations=N_PERMUTATIONS, seed=SEED)

        # Create the legend label for amplitude with p-value
        p_text = f"p={p_val_amp[0]:.3f}"
        if p_val_amp[0] < P_VAL_ALPHA:
            p_text += "*" # Add significance marker
        amp_label_legend = f'Amplitude ({p_text})'
        legend_handles.append(Line2D([0], [0], color=AMPLITUDE_COLOR, lw=2, linestyle='--', label=amp_label_legend))
    else:
        ax.text(0.5, 0.25, "No Amplitude Data", ha='center', va='center', color=AMPLITUDE_COLOR)

    ax_amp.set_ylabel(amp_label, fontsize=14, color=AMPLITUDE_COLOR)
    ax_amp.tick_params(axis='y', labelcolor=AMPLITUDE_COLOR)
    ax_amp.grid(False) # Turn off grid for the secondary axis to avoid clutter

    # Add the combined legend to the plot
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best')


# --- 4. Plot Trial Count Balance ---
print("\n--- Step 4: Visualizing Trial Count Balance ---")

fig_counts, axes_counts = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
fig_counts.suptitle('Trial Counts per Condition, Showing Within-Subject Change', fontsize=18, y=1.02)
axes_counts = axes_counts.flatten()

count_comparisons = [
    {'ax_idx': 0, 'component': 'N2ac', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'ax_idx': 1, 'component': 'N2ac', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
    {'ax_idx': 2, 'component': 'Pd', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'ax_idx': 3, 'component': 'Pd', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
]

for comp_info in count_comparisons:
    ax = axes_counts[comp_info['ax_idx']]
    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])]

    # Use stripplot to show individual subject lines, which is better for within-subject designs
    # This requires pivoting the data to get one row per subject
    pivot_counts = plot_df.pivot(index='subject', columns='condition', values='total_trials')
    if not pivot_counts.empty:
        # Plot lines connecting paired points for each subject
        for subject_id, row in pivot_counts.iterrows():
            ax.plot(comp_info['conds'], row[comp_info['conds']], marker='o', color='gray', alpha=0.3, linestyle='-')

    # Overlay the pointplot to show the mean and CI
    sns.pointplot(data=plot_df, x='condition', y='total_trials', order=comp_info['conds'],
                  join=False, errorbar=('ci', 95), scale=1.2, errwidth=2,
                  capsize=0.1, ax=ax, color='black')

    ax.set_title(f"{comp_info['component']} by {comp_info['split_by']} Split", fontsize=14)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Total Trial Count' if comp_info['ax_idx'] in [0, 2] else '')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show(block=True)
