import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import seaborn as sns
import numpy as np
from stats import remove_outliers # Assuming this is your custom outlier removal function
from utils import get_contra_ipsi_diff_wave, calculate_fractional_area_latency
from scipy.stats import sem

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
# Using the full epoch for analysis now, as we plot the whole wave.
PD_TIME_WINDOW = (0.0, 0.7)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_TIME_WINDOW = (0.0, 0.7)
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# 4. Statistical & Analysis Parameters
N_PERMUTATIONS = 10000
CLUSTER_STAT_ALPHA = 0.05 # Alpha for forming clusters and for final cluster p-values
CONFIDENCE_LEVEL = 0.95 # For confidence interval plots
TAIL = 0 # Two-tailed test
N_JOBS = 5
SEED = 42

# --- Main Script ---

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
epochs = SPACEPRIME.load_concatenated_epochs().crop(0.0, 0.7)
df = epochs.metadata.copy()

# Preprocessing
df = df[df[PHASE_COL] != FILTER_PHASE]
df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map({1: "left", 2: "mid", 3: "right"})
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(
    {0: "absent", 1: "left", 2: "mid", 3: "right"})

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

# --- 2. Perform Behavioral Splits and Aggregate Data by Subject ---
print("\n--- Step 2: Performing Behavioral Splits and Aggregating Data ---")

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

# --- Loop through subjects and conditions to calculate ERPs ---
subject_agg_data = []
for subject_id in merged_df[SUBJECT_ID_COL].unique():
    print(f"Processing subject: {subject_id}")

    components_to_process = {
        'N2ac': {'df': n2ac_base_df[n2ac_base_df[SUBJECT_ID_COL] == subject_id],
                 'stim_col': TARGET_COL, 'electrodes': N2AC_ELECTRODES,
                 'time_window': N2AC_TIME_WINDOW, 'is_target': True},
        'Pd':   {'df': pd_base_df[pd_base_df[SUBJECT_ID_COL] == subject_id],
                 'stim_col': DISTRACTOR_COL, 'electrodes': PD_ELECTRODES,
                 'time_window': PD_TIME_WINDOW, 'is_target': False}
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

                stim_col = params['stim_col']
                balance_counts = cond_df[stim_col].value_counts()
                left_count = balance_counts.get('left', 0)
                right_count = balance_counts.get('right', 0)
                print(f"    - {comp_name} | {split_by} Split: {cond_name:<9} | "
                      f"Total: {len(cond_df):<3} (L: {left_count}, R: {right_count})")

                wave, times = get_contra_ipsi_diff_wave(
                    cond_df, params['electrodes'], params['time_window'], all_times, params['stim_col']
                )
                wave = wave * 10
                if wave is not None:
                    # Store the entire wave and trial counts
                    subject_agg_data.append({
                        'subject': subject_id,
                        'component': comp_name,
                        'split_by': split_by,
                        'condition': cond_name,
                        'wave': wave,
                        'total_trials': len(cond_df), # <-- This is the required addition
                        'left_trials': left_count,
                        'right_trials': right_count
                    })

agg_df = pd.DataFrame(subject_agg_data)
print("Aggregation complete. Resulting data shape:", agg_df.shape)

# --- 3. Plot Grand-Average ERP Waves and Run Cluster Statistics ---
print("\n--- Step 3: Plotting Grand-Average ERP Waves with Cluster Statistics ---")

comparisons = [
    {'component': 'N2ac', 'split_by': 'RT', 'conds': ['fast', 'slow'], 'colors': ['#1f77b4', '#ff7f0e']},
    {'component': 'N2ac', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect'], 'colors': ['#2ca02c', '#d62728']},
    {'component': 'Pd', 'split_by': 'RT', 'conds': ['fast', 'slow'], 'colors': ['#1f77b4', '#ff7f0e']},
    {'component': 'Pd', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect'], 'colors': ['#2ca02c', '#d62728']},
]

PERCENTAGE_TO_PLOT = 0.5

fig, axes = plt.subplots(2, 2, figsize=(6, 10), sharey=True, constrained_layout=True)
axes = axes.flatten()
fig.suptitle('Grand-Average ERP Difference Waves by Behavioral Split', fontsize=20, y=1.03)

for i, comp_info in enumerate(comparisons):
    ax = axes[i]
    component, split_by = comp_info['component'], comp_info['split_by']
    cond1_name, cond2_name = comp_info['conds']
    colors = comp_info['colors']

    ax.set_title(f"{component} by {split_by} Split", fontsize=14)

    plot_df = agg_df[(agg_df['component'] == component) & (agg_df['split_by'] == split_by)]
    pivot_df = plot_df.pivot(index='subject', columns='condition', values='wave').dropna()

    if len(pivot_df) < 5:
        ax.text(0.5, 0.5, "Not enough subjects for comparison", ha='center', va='center')
        continue

    cond1_waves = np.stack(pivot_df[cond1_name].values)
    cond2_waves = np.stack(pivot_df[cond2_name].values)

    # Invert N2ac for plotting so positive is up
    if component == 'N2ac':
        cond1_waves *= 1
        cond2_waves *= 1

    n_subjects_in_comp = len(pivot_df)

    for j, (cond_name, waves, color) in enumerate(zip(comp_info['conds'], [cond1_waves, cond2_waves], colors)):
        # --- 1. Calculate Wave Properties ---
        ga_wave = np.mean(waves, axis=0)
        latency = calculate_fractional_area_latency(ga_wave, all_times, percentage=PERCENTAGE_TO_PLOT, plot=False, is_target=True if component == "N2ac" else False)
        if n_subjects_in_comp > 1:
            sem_wave = sem(waves, axis=0)
        else:
            sem_wave = np.zeros_like(ga_wave)

        # --- 2. Plot the Main Wave and SEM ---
        ax.plot(all_times, ga_wave, label=f"{cond_name} (N={n_subjects_in_comp})", color=color, lw=2, alpha=0.5)
        ax.fill_between(all_times, ga_wave - sem_wave, ga_wave + sem_wave, color=color, alpha=0.1)

        # --- 3. Plot Latency/Amplitude Markers (Your Requested Change) ---
        if latency is not None:
            amplitude = np.interp(latency, all_times, ga_wave)

            # Draw partial lines from axes to the wave
            ax.plot([latency, latency], [0, amplitude], color=color, linestyle='--', lw=1.2)
            ax.plot([0, latency], [amplitude, amplitude], color=color, linestyle='--', lw=1.2)

            # Add a curved arrow pointing to the intersection
            # Use different offsets for each wave to avoid overlap
            offset = (30, 30) if j == 0 else (30, -30)
            ax.annotate(
                '',  # No text needed, just the arrow
                xy=(latency, amplitude),
                xycoords='data',
                xytext=offset,
                textcoords='offset points',
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=.2",
                    color=color,
                    lw=1.5
                )
            )

    # Aesthetics
    ax.axhline(0, color='k', linestyle='--', lw=0.8)
    ax.axvline(0, color='k', linestyle=':', lw=0.8)
    ax.legend(loc='upper left')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({'Inverted ' if component == 'N2ac' else ''}µV)")
    ax.grid(True, linestyle=':', alpha=0.6)


# --- 4. Plot Trial Count Balance (Improved with individual subject lines) ---
print("\n--- Step 4: Visualizing Trial Count Balance with Individual Subject Data ---")

# We will use the aggregated dataframe 'agg_df' which now contains 'total_trials'.
# A 2x2 figure is a clear way to show the four main comparisons.
fig_counts, axes_counts = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
fig_counts.suptitle('Trial Counts per Condition, Showing Within-Subject Change', fontsize=18, y=1.02)
axes_counts = axes_counts.flatten()

# Define the comparisons to plot
count_comparisons = [
    {'ax_idx': 0, 'component': 'N2ac', 'split_by': 'RT',       'conds': ['fast', 'slow']},
    {'ax_idx': 1, 'component': 'N2ac', 'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
    {'ax_idx': 2, 'component': 'Pd',   'split_by': 'RT',       'conds': ['fast', 'slow']},
    {'ax_idx': 3, 'component': 'Pd',   'split_by': 'Accuracy', 'conds': ['correct', 'incorrect']},
]

for comp_info in count_comparisons:
    ax = axes_counts[comp_info['ax_idx']]
    component = comp_info['component']
    split_by = comp_info['split_by']
    conditions = comp_info['conds']

    # Filter the aggregated data for the current comparison
    # This plot_df will now correctly contain the 'total_trials' column
    plot_df = agg_df[(agg_df['component'] == component) & (agg_df['split_by'] == split_by)]

    # 1. Plot individual subject data points with some jitter
    sns.stripplot(
        data=plot_df,
        x='condition',
        y='total_trials',
        order=conditions,
        jitter=0.1,
        alpha=0.4,
        color='gray',
        ax=ax
    )

    # 2. Overlay the mean, CI, and connecting lines for the within-subject trend
    sns.pointplot(
        data=plot_df,
        x='condition',
        y='total_trials',
        order=conditions,
        join=True,         # This draws the connecting lines
        errorbar=('ci', 95), # Show 95% confidence interval
        scale=0.7,         # Make points and lines a bit smaller
        errwidth=1.5,      # Width of the CI bars
        capsize=0.1,       # Caps on the CI bars
        ax=ax,
        color='black'
    )

    # Aesthetics
    ax.set_title(f'{component} by {split_by} Split', fontsize=14)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Total Trial Count' if comp_info['ax_idx'] in [0, 2] else '')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show(block=True)


# --- 4. Plot Trial Count Balance ---
print("\n--- Step 4: Visualizing Trial Count Balance ---")

trial_counts_df = agg_df.melt(
    id_vars=['subject', 'component', 'split_by', 'condition'],
    value_vars=['left_trials', 'right_trials'],
    var_name='stimulus_side', value_name='count'
)
trial_counts_df['stimulus_side'] = trial_counts_df['stimulus_side'].str.replace('_trials', '')
trial_counts_df['x_label'] = trial_counts_df['split_by'] + '\n' + trial_counts_df['condition']

fig_counts, axes_counts = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig_counts.suptitle('Average Trial Counts per Condition Across Subjects', fontsize=18, y=0.98)

components = ['N2ac', 'Pd']
x_order = ['RT\nfast', 'RT\nslow', 'Accuracy\ncorrect', 'Accuracy\nincorrect']

for i, component in enumerate(components):
    ax = axes_counts[i]
    component_df = trial_counts_df[trial_counts_df['component'] == component]
    sns.barplot(
        data=component_df, x='x_label', y='count', hue='stimulus_side',
        ax=ax, palette='muted', order=x_order, errorbar='sd'
    )
    ax.set_title(f'{component} Trial Counts', fontsize=14)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Average Trial Count' if i == 0 else '')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Stimulus Side')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show(block=True)

