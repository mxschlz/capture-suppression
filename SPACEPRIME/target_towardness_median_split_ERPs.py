import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import SPACEPRIME
import pandas as pd
import seaborn as sns
import numpy as np
from stats import remove_outliers  # Assuming this is your custom outlier removal function
from utils import calculate_fractional_area_latency, get_all_waves
from scipy.stats import sem
from scipy.signal import savgol_filter  # Import for smoothing

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
COMPONENT_TIME_WINDOW = (0.1, 0.5)
PD_ELECTRODES = [("C3", "C4")]
N2AC_ELECTRODES = [("C3", "C4")]

# 4. Analysis Parameters
LATENCY_ANALYSIS_WINDOW = (0.2, 0.4)
LATENCY_PERCENTAGE = 0.5

# --- NEW: Configuration for Poster Plots ---

# Place your p-values from your external analysis (e.g., jamovi) here.
MANUAL_P_VALUES = {
    'N2ac': {
        'Target towardness': {
            'latency': 0.001,  # Replace with your p-value for N2ac latency
            'amplitude': 0.667  # Replace with your p-value for N2ac amplitude
        }
    },
    'Pd': {
        'Target towardness': {
            'latency': 0.092,  # Replace with your p-value for Pd latency
            'amplitude': 0.49  # Replace with your p-value for Pd amplitude
        }
    }
}

# --- NEW: Configuration for Plotting Aesthetics ---
# Smoothing parameters for ERP waves (use window_length=1 for no smoothing)
SMOOTHING_WINDOW_LENGTH = 21  # Must be an odd integer (e.g., 11, 15, 21)
SMOOTHING_POLYORDER = 3  # Polynomial order for Savitzky-Golay filter


# --- UPDATED: Helper function for p-value formatting ---
def format_p_value_with_asterisks(p_val):
    """Formats a p-value with appropriate significance asterisks."""
    if p_val is None or not isinstance(p_val, (int, float)) or p_val > 1 or p_val < 0:
        return "p = n.s."

    # Determine the asterisks first
    if p_val < 0.001:
        asterisks = '***'
    elif p_val < 0.01:
        asterisks = '**'
    elif p_val < 0.05:
        asterisks = '*'
    else:
        asterisks = ''

    # Format the p-value string
    if p_val < 0.001:
        p_string = "p < .001"
    else:
        p_string = f"p = {p_val:.3f}"

    return f"{p_string}{asterisks}"


# --- Main Script ---

# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime").crop(COMPONENT_TIME_WINDOW[0], COMPONENT_TIME_WINDOW[1])
df = epochs.metadata.copy()

# Preprocessing
if FILTER_PHASE:
    df = df[df[PHASE_COL] != FILTER_PHASE]
df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map({1: "left", 2: "mid", 3: "right"})
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(
    {0: "absent", 1: "left", 2: "mid", 3: "right"})
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(int).astype(str)

# ERP Data Reshaping
erp_df_picks_flat = [item for pair in set(N2AC_ELECTRODES + PD_ELECTRODES) for item in pair]
erp_df_picks_unique_flat = sorted(list(set(erp_df_picks_flat)))
erp_df = epochs.to_data_frame(picks=erp_df_picks_unique_flat, time_format=None)
erp_wide = erp_df.pivot(index='epoch', columns='time')
erp_wide = erp_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
erp_wide.columns = erp_wide.columns.droplevel(0)
merged_df = df.join(erp_wide)
all_times = epochs.times
print(f"Merged metadata with wide ERP data. Shape: {merged_df.shape}")

# --- 2. Perform Behavioral Splits and Aggregate ERP Metrics by Subject ---
print("\n--- Step 2: Performing Splits and Aggregating ERP Metrics ---")

# Filter for N2ac and Pd-relevant trials
is_target_lateral = merged_df[TARGET_COL].isin(['left', 'right'])
is_distractor_lateral = merged_df[DISTRACTOR_COL].isin(['left', 'right'])
is_target_central = merged_df[TARGET_COL] == 'mid'
is_distractor_central = merged_df[DISTRACTOR_COL] == 'mid'
n2ac_base_df = merged_df[is_target_lateral & is_distractor_central].copy()
pd_base_df = merged_df[is_distractor_lateral & is_target_central].copy()

# Load and merge target towardness values
target_towardness = SPACEPRIME.load_concatenated_csv("target_towardness.csv", index_col=0)
merge_cols = [SUBJECT_ID_COL, "block", 'trial_nr']
target_towardness[SUBJECT_ID_COL] = target_towardness[SUBJECT_ID_COL].astype(int).astype(str)
n2ac_base_df[SUBJECT_ID_COL] = n2ac_base_df[SUBJECT_ID_COL].astype(int).astype(str)
pd_base_df[SUBJECT_ID_COL] = pd_base_df[SUBJECT_ID_COL].astype(int).astype(str)
n2ac_final_df = pd.merge(n2ac_base_df, target_towardness, on=merge_cols, how='left')
pd_final_df = pd.merge(pd_base_df, target_towardness, on=merge_cols, how='left')

# Perform median splits
n2ac_final_df["target_towardness_split"] = n2ac_final_df.groupby(SUBJECT_ID_COL)["target_towardness"].transform(
    lambda x: pd.qcut(x, 2, labels=['low', 'high'], duplicates='drop'))
pd_final_df["target_towardness_split"] = pd_final_df.groupby(SUBJECT_ID_COL)["target_towardness"].transform(
    lambda x: pd.qcut(x, 2, labels=['low', 'high'], duplicates='drop'))

# Aggregate ERP metrics
subject_agg_data = []
for subject_id in merged_df[SUBJECT_ID_COL].unique():
    components_to_process = {
        'N2ac': {'df': n2ac_final_df[n2ac_final_df[SUBJECT_ID_COL] == subject_id],
                 'stim_col': TARGET_COL, 'electrodes': N2AC_ELECTRODES, 'is_target': True},
        'Pd': {'df': pd_final_df[pd_final_df[SUBJECT_ID_COL] == subject_id],
               'stim_col': DISTRACTOR_COL, 'electrodes': PD_ELECTRODES, 'is_target': False}
    }
    for comp_name, params in components_to_process.items():
        comp_df = params['df']
        if comp_df.empty: continue
        for cond_name in ['low', 'high']:
            mask = (comp_df['target_towardness_split'] == cond_name)
            cond_df = comp_df[mask]
            if cond_df.empty: continue
            diff_wave, _, _, times = get_all_waves(
                cond_df, params['electrodes'], COMPONENT_TIME_WINDOW, all_times, params['stim_col']
            )
            latency, amplitude = np.nan, np.nan
            if diff_wave is not None:
                latency = calculate_fractional_area_latency(
                    diff_wave, times, percentage=LATENCY_PERCENTAGE,
                    is_target=params['is_target'], analysis_window_times=LATENCY_ANALYSIS_WINDOW
                )
                if not np.isnan(latency) and (times[0] <= latency <= times[-1]):
                    amplitude = np.interp(latency, times, diff_wave)
            subject_agg_data.append({
                'subject': subject_id, 'component': comp_name, 'split_by': 'Target towardness',
                'condition': cond_name, 'latency': latency, 'amplitude': amplitude, 'wave': diff_wave, 'times': times
            })
agg_df = pd.DataFrame(subject_agg_data)
print("Aggregation complete. Resulting data shape:", agg_df.shape)

# --- 3. Generate Minimalist Plots with Analysis Details and Axes ---
print("\n--- Step 3: Generating Minimalist Plots with Analysis Details and Axes ---")

# Define the specific comparisons
poster_comparisons = [
    {'component': 'N2ac', 'split_by': 'Target towardness', 'conds': ['low', 'high']},
    {'component': 'Pd', 'split_by': 'Target towardness', 'conds': ['low', 'high']},
]

# Create a 1x2 figure for the two plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True, sharey=True)
axes = axes.flatten()

# Use a colorblind-friendly palette
colors = sns.color_palette("magma", 2)

for i, comp_info in enumerate(poster_comparisons):
    ax = axes[i]
    plot_df = agg_df[(agg_df['component'] == comp_info['component']) & (agg_df['split_by'] == comp_info['split_by'])]

    # --- Plotting Loop ---
    for j, cond_name in enumerate(comp_info['conds']):
        color = colors[j]
        cond_df = plot_df[plot_df['condition'] == cond_name].dropna(subset=['wave'])
        if cond_df.empty: continue

        times_for_plot = cond_df['times'].iloc[0]
        ga_wave = np.mean(np.stack(cond_df['wave'].values), axis=0)
        sem_wave = sem(np.stack(cond_df['wave'].values), axis=0)

        # --- Smooth the grand average wave and SEM for cleaner plotting ---
        if SMOOTHING_WINDOW_LENGTH > 1 and len(ga_wave) > SMOOTHING_WINDOW_LENGTH:
            ga_wave_smooth = savgol_filter(ga_wave, SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER)
            sem_upper_smooth = savgol_filter(ga_wave + sem_wave, SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER)
            sem_lower_smooth = savgol_filter(ga_wave - sem_wave, SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER)
        else:
            ga_wave_smooth = ga_wave
            sem_upper_smooth = ga_wave + sem_wave
            sem_lower_smooth = ga_wave - sem_wave

        # --- Calculate latency and amplitude for crosshairs ---
        is_target_comp = comp_info['component'] == 'N2ac'
        latency_on_ga = calculate_fractional_area_latency(
            ga_wave, times_for_plot, percentage=LATENCY_PERCENTAGE,
            is_target=is_target_comp, analysis_window_times=LATENCY_ANALYSIS_WINDOW
        )
        amplitude_for_plot = np.interp(latency_on_ga, times_for_plot, ga_wave_smooth) if not np.isnan(
            latency_on_ga) else np.nan

        # Plot the smoothed difference wave and SEM band
        ax.plot(times_for_plot, ga_wave_smooth, color=color, lw=3)
        ax.fill_between(times_for_plot, sem_lower_smooth, sem_upper_smooth, color=color, alpha=0.2)

        # --- Plot crosshair lines and marker ---
        if not np.isnan(latency_on_ga) and not np.isnan(amplitude_for_plot):
            ax.plot([latency_on_ga, latency_on_ga], [0, amplitude_for_plot],
                    color=color, linestyle=':', linewidth=4, zorder=10, alpha=1.0)
            ax.plot([times_for_plot[0], latency_on_ga], [amplitude_for_plot, amplitude_for_plot],
                    color=color, linestyle=':', linewidth=4, zorder=10, alpha=1.0)
            ax.plot(latency_on_ga, amplitude_for_plot, 'o',
                    markerfacecolor='black', markeredgecolor=color, markeredgewidth=2, markersize=10, zorder=11)

    # --- Minimalist Aesthetics ---
    # Draw the analysis window
    ax.axvspan(LATENCY_ANALYSIS_WINDOW[0], LATENCY_ANALYSIS_WINDOW[1], color='grey', alpha=0.2, zorder=0)

    ax.axhline(0, color='k', linestyle='--', lw=1.5)
    ax.axvline(0, color='k', linestyle='-', lw=1)
    if 'times_for_plot' in locals():
        ax.set_xlim(times_for_plot[0], times_for_plot[-1])

    # --- MODIFIED: Remove ticks and labels, but keep the main axis lines ---
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    sns.despine(ax=ax)  # Removes top and right spines, keeping the bottom and left ones.

# --- 4. Save and Show Plot ---
# Save the figure (updated filename for this specific version)
output_filename = "G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\ERP_Towardness_Split_Minimalist_With_Axes.svg"
plt.savefig(output_filename, bbox_inches='tight', transparent=True)
print(f"\n--- Minimalist plot with axes saved as {output_filename} ---")

