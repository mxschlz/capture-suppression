import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Import for creating custom legend handles
import SPACEPRIME
import pandas as pd
import seaborn as sns
import numpy as np
from stats import remove_outliers # Assuming this is your custom outlier removal function
from utils import calculate_fractional_area_latency, get_all_waves
from mne.stats import permutation_t_test
from scipy.stats import sem

plt.ion()

# --- Script Configuration ---

# 1. Data Loading & Preprocessing
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = None

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
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4")]
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4")]

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
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime_desc-csd").crop(COMPONENT_TIME_WINDOW[0], COMPONENT_TIME_WINDOW[1])
df = epochs.metadata.copy()

# Preprocessing
if FILTER_PHASE:
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

                # Use the helper function to get all three waves
                diff_wave, contra_wave, ipsi_wave, times = get_all_waves(
                    cond_df, params['electrodes'], COMPONENT_TIME_WINDOW, all_times, params['stim_col']
                )

                latency, amplitude = np.nan, np.nan
                if diff_wave is not None:
                    # Calculate metrics from the difference wave
                    latency = calculate_fractional_area_latency(
                        diff_wave, times,
                        percentage=LATENCY_PERCENTAGE,
                        is_target=params['is_target'],
                        analysis_window_times=LATENCY_ANALYSIS_WINDOW
                    )
                    if not np.isnan(latency) and (times[0] <= latency <= times[-1]):
                        amplitude = np.interp(latency, times, diff_wave)

                subject_agg_data.append({
                    'subject': subject_id,
                    'component': comp_name,
                    'split_by': split_by,
                    'condition': cond_name,
                    'latency': latency,
                    'amplitude': amplitude,
                    'total_trials': len(cond_df),
                    'wave': diff_wave,
                    'contra_wave': contra_wave, # Store contra wave
                    'ipsi_wave': ipsi_wave,     # Store ipsi wave
                    'times': times
                })


agg_df = pd.DataFrame(subject_agg_data)
print("Aggregation complete. Resulting data shape:", agg_df.shape)
# save the stuff
df_save = agg_df.groupby(["subject", "component"])[["latency", "amplitude"]].mean()
output_path = f'{SPACEPRIME.get_data_path()}concatenated\\erp_latency_amplitude_subject_mean.csv'
df_save.to_csv(output_path, index=True)


# --- 3. Analyze and Plot ERP Metrics with Permutation Tests ---
# This step has been removed. The statistical analysis is now integrated
# into the grand-average difference wave plots in Step 5 for a more
# consolidated and informative visualization.
print("\n--- Step 3: Merged into Step 5 ---")

comparisons = [
    {'title': 'N2ac by RT', 'component': 'N2ac', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'title': 'N2ac by Accuracy', 'component': 'N2ac', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
    {'title': 'Pd by RT', 'component': 'Pd', 'split_by': 'RT', 'conds': ['fast', 'slow']},
    {'title': 'Pd by Accuracy', 'component': 'Pd', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
]


# --- 4. Plot Grand-Average ERP Waves with Contra/Ipsi Detail ---
print("\n--- Step 4: Visualizing Grand-Average ERP Waveforms ---")
fig_waves, axes_waves = plt.subplots(2, 2, figsize=(20, 12), sharey=True, constrained_layout=True)
fig_waves.suptitle('Grand-Average Contralateral and Ipsilateral ERPs by Behavioral Split', fontsize=20, y=1.06)
axes_waves = axes_waves.flatten()

for i, comp_info in enumerate(comparisons):
    ax = axes_waves[i]
    ax.set_title(comp_info['title'], fontsize=16)

    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])]
    colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange

    # --- Plotting ---
    legend_handles = [
        Line2D([0], [0], color='k', lw=2, label='Contralateral'),
        Line2D([0], [0], color='k', lw=2, linestyle='--', label='Ipsilateral'),
    ]

    for j, cond_name in enumerate(comp_info['conds']):
        color = colors[j]
        cond_df = plot_df[plot_df['condition'] == cond_name].dropna(
            subset=['contra_wave', 'ipsi_wave', 'latency', 'amplitude'])

        if cond_df.empty:
            continue

        n_subjects = len(cond_df)
        times_for_plot = cond_df['times'].iloc[0]

        # Grand average waves
        ga_contra_wave = np.mean(np.stack(cond_df['contra_wave'].values), axis=0)
        ga_ipsi_wave = np.mean(np.stack(cond_df['ipsi_wave'].values), axis=0)
        sem_contra = sem(np.stack(cond_df['contra_wave'].values), axis=0)
        sem_ipsi = sem(np.stack(cond_df['ipsi_wave'].values), axis=0)

        # Plot Contra and Ipsi waves
        ax.plot(times_for_plot, ga_contra_wave, color=color, lw=2.5, linestyle='-')
        ax.fill_between(times_for_plot, ga_contra_wave - sem_contra, ga_contra_wave + sem_contra, color=color,
                        alpha=0.2)

        ax.plot(times_for_plot, ga_ipsi_wave, color=color, lw=2.5, linestyle='--')
        ax.fill_between(times_for_plot, ga_ipsi_wave - sem_ipsi, ga_ipsi_wave + sem_ipsi, color=color, alpha=0.2)

        # Add a colored legend entry for the condition
        legend_handles.append(Line2D([0], [0], color=color, lw=4, label=f'{cond_name} (N={n_subjects})'))

    # Aesthetics
    ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.25, zorder=0)
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axvline(0, color='k', linestyle=':', lw=1)

    # Reorder legend handles to be more logical
    final_legend_handles = [h for h in legend_handles if ' (N=' in h.get_label()]
    final_legend_handles.extend([h for h in legend_handles if ' (N=' not in h.get_label()])
    ax.legend(handles=final_legend_handles, loc='best')

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)


# --- 5. Plot Grand-Average Difference Waves with Metric Crosshairs & Stats ---
print("\n--- Step 5: Visualizing Grand-Average Difference Waves with Stats ---")
fig_diff, axes_diff = plt.subplots(2, 2, figsize=(20, 12), sharey=True, constrained_layout=True)
fig_diff.suptitle('Grand-Average Difference Waves with Latency/Amplitude Markers and Paired Stats', fontsize=20, y=1.06)
axes_diff = axes_diff.flatten()

for i, comp_info in enumerate(comparisons):
    ax = axes_diff[i]
    ax.set_title(comp_info['title'], fontsize=16)
    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])]
    colors = ['#1f77b4', '#ff7f0e']
    cond1_name, cond2_name = comp_info['conds']

    # --- Perform Paired Permutation Tests ---
    # Latency Analysis
    pivot_lat = plot_df.pivot(index='subject', columns='condition', values='latency')
    p_text_lat = "p=n.s."
    # Ensure both conditions are present for paired comparison
    if cond1_name in pivot_lat.columns and cond2_name in pivot_lat.columns:
        valid_pairs = pivot_lat[[cond1_name, cond2_name]].dropna()
        if len(valid_pairs) > 2:
            diffs_lat = valid_pairs[cond1_name].values - valid_pairs[cond2_name].values
            X_lat = diffs_lat[:, np.newaxis]
            _, p_val_lat, _ = permutation_t_test(X_lat, n_permutations=N_PERMUTATIONS, seed=SEED)
            p_text_lat = f"p={p_val_lat[0]:.3f}{'*' if p_val_lat[0] < P_VAL_ALPHA else ''}"

    # Amplitude Analysis
    pivot_amp = plot_df.pivot(index='subject', columns='condition', values='amplitude')
    p_text_amp = "p=n.s."
    if cond1_name in pivot_amp.columns and cond2_name in pivot_amp.columns:
        valid_pairs = pivot_amp[[cond1_name, cond2_name]].dropna()
        if len(valid_pairs) > 2:
            diffs_amp = valid_pairs[cond1_name].values - valid_pairs[cond2_name].values
            X_amp = diffs_amp[:, np.newaxis]
            _, p_val_amp, _ = permutation_t_test(X_amp, n_permutations=N_PERMUTATIONS, seed=SEED)
            p_text_amp = f"p={p_val_amp[0]:.3f}{'*' if p_val_amp[0] < P_VAL_ALPHA else ''}"

    # --- Plotting Loop ---
    for j, cond_name in enumerate(comp_info['conds']):
        color = colors[j]
        cond_df = plot_df[plot_df['condition'] == cond_name].dropna(subset=['wave'])

        if cond_df.empty:
            continue

        n_subjects = len(cond_df)
        times_for_plot = cond_df['times'].iloc[0]

        # Grand average of the difference wave
        ga_wave = np.mean(np.stack(cond_df['wave'].values), axis=0)
        sem_wave = sem(np.stack(cond_df['wave'].values), axis=0)

        # Calculate latency and amplitude ON THE GRAND-AVERAGE WAVE for visualization
        is_target_comp = comp_info['component'] == 'N2ac'
        latency_on_ga = calculate_fractional_area_latency(
            ga_wave, times_for_plot,
            percentage=LATENCY_PERCENTAGE,
            is_target=is_target_comp,
            analysis_window_times=LATENCY_ANALYSIS_WINDOW
        )
        amplitude_on_ga = np.interp(latency_on_ga, times_for_plot, ga_wave) if not np.isnan(latency_on_ga) else np.nan

        # Plot the difference wave
        ax.plot(times_for_plot, ga_wave, color=color, lw=2.5, label=f'{cond_name} (N={n_subjects})')
        ax.fill_between(times_for_plot, ga_wave - sem_wave, ga_wave + sem_wave, color=color, alpha=0.2)

        # Plot crosshair lines
        if not np.isnan(latency_on_ga) and not np.isnan(amplitude_on_ga):
            # Vertical line from x-axis to the point
            ax.plot([latency_on_ga, latency_on_ga], [0, amplitude_on_ga],
                    color=color, linestyle=':', linewidth=1.5, zorder=10)
            # Horizontal line from y-axis to the point
            ax.plot([ax.get_xlim()[0], latency_on_ga], [amplitude_on_ga, amplitude_on_ga],
                    color=color, linestyle=':', linewidth=1.5, zorder=10)
            # Plot the central point itself
            ax.plot(latency_on_ga, amplitude_on_ga, 'o',
                    markerfacecolor=color, markeredgecolor='k', markersize=8, zorder=11)

    # --- Aesthetics and Legend with Stats ---
    ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.25, zorder=0)
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axvline(0, color='k', linestyle=':', lw=1)
    if 'times_for_plot' in locals():
        ax.set_xlim(times_for_plot[0], times_for_plot[-1])

    # Create legend handles
    line_handles, _ = ax.get_legend_handles_labels()
    # Add a handle for the crosshair marker
    #line_handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
                               #label='Metric on GA Wave', markersize=8, linestyle='None'))
    # Add text-only handles for the stats by creating invisible lines
    line_handles.append(Line2D([0], [0], color='w', label=f'Latency Comp: {p_text_lat}'))
    line_handles.append(Line2D([0], [0], color='w', label=f'Amplitude Comp: {p_text_amp}'))

    ax.legend(handles=line_handles, loc='best', title="Paired Comparisons")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)


# --- 6. Plot Trial Count Balance ---
print("\n--- Step 6: Visualizing Trial Count Balance ---")
fig_counts, axes_counts = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
fig_counts.suptitle('Trial Counts per Condition, Showing Within-Subject Change', fontsize=18, y=1.02)
axes_counts = axes_counts.flatten()

for i, comp_info in enumerate(comparisons):
    ax = axes_counts[i]
    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])]
    pivot_counts = plot_df.pivot(index='subject', columns='condition', values='total_trials')
    if not pivot_counts.empty:
        for subject_id, row in pivot_counts.iterrows():
            ax.plot(comp_info['conds'], row[comp_info['conds']], marker='o', color='gray', alpha=0.3, linestyle='-')
    sns.pointplot(data=plot_df, x='condition', y='total_trials', order=comp_info['conds'],
                  join=False, errorbar=('ci', 95), scale=1.2, errwidth=2,
                  capsize=0.1, ax=ax, color='black')
    ax.set_title(comp_info['title'], fontsize=14)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Total Trial Count' if i in [0, 2] else '')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show(block=True)
