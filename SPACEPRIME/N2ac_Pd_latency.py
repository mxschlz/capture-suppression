import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import t, sem
from matplotlib.lines import Line2D
import itertools

from SPACEPRIME import load_concatenated_epochs
from stats import remove_outliers
from utils import calculate_fractional_area_latency
from mne.stats import permutation_t_test

plt.ion()

# --- Parameters ---

# Epoching and Time Windows
EPOCH_TMIN, EPOCH_TMAX = 0.0, 0.7  # Seconds, crop for comparability

# Electrodes for Contra/Ipsi Calculation
ELECTRODE_LEFT_HEMISPHERE = "C3"
ELECTRODE_RIGHT_HEMISPHERE = "C4"
OUTLIER_RT_THRESHOLD = 2.0
REACTION_TIME_COL = 'rt'
SUBJECT_ID_COL = 'subject_id'

# Plotting Parameters
SAVGOL_WINDOW_LENGTH = 51
SAVGOL_POLYORDER = 3
AMPLITUDE_SCALE_FACTOR = 1e6  # Volts to Microvolts
PLOT_COLORS = {
    'distractor_lateral': "darkorange",
    'target_distractor_present': "blue",
    'target_distractor_absent': "green"
}
PLOT_LEGENDS = {
    'distractor_lateral': "Distractor lateral (Pd)",
    'target_distractor_present': "Target lateral (Distractor present, N2ac)",
    'target_distractor_absent': "Target lateral (Distractor absent, N2ac)"
}

# --- Latency & Permutation Test Parameters (from ERP_behavioral_split_analysis.py) ---
# Analysis windows for latency calculation
N2AC_ANALYSIS_WINDOW = (0.2, 0.4)
PD_ANALYSIS_WINDOW = (0.2, 0.4)
LATENCY_PERCENTAGE = 0.5  # 50% fractional area latency

# MNE Permutation t-test parameters
N_PERMUTATIONS_TTEST = 10000
P_VAL_ALPHA = 0.05
SEED = 42

# --- End of Parameters ---


# --- Data Storage for Subject-Level Results ---
# Format: subject_diff_waves['sub-01']['distractor_lateral'] = wave_array
subject_diff_waves = {}

times_vector = None  # To be populated from the first subject

# load epochs and preprocess
epochs = load_concatenated_epochs("spaceprime").crop(EPOCH_TMIN, EPOCH_TMAX)
print(f"Loaded {len(epochs)} epochs.")

df = epochs.metadata.copy().reset_index(drop=True)
if REACTION_TIME_COL in df.columns:
    print(f"Removing RT outliers (threshold: {OUTLIER_RT_THRESHOLD} SD)...")
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
    print(f"  Trials remaining after RT outlier removal: {len(df)}")

epochs = epochs[df.index]
print(f"Final number of trials after preprocessing: {len(epochs)}")

# --- Subject Loop ---
print("\n--- Starting subject-level processing ---")
all_subject_ids = epochs.metadata[SUBJECT_ID_COL].unique()

for subject_id_int in all_subject_ids:
    subject_id_str = f"sub-{int(subject_id_int):02d}"
    subject_diff_waves[subject_id_str] = {}

    try:
        epochs_sub = epochs[f"subject_id=={subject_id_int}"]
        if not epochs_sub:
            print(f"  No epochs for sub-{subject_id_int} after preproc. Skipping.")
            continue

        if times_vector is None:
            times_vector = epochs_sub.times.copy()

        all_conds_sub = list(epochs_sub.event_id.keys())

        # --- 1. Distractor Lateral (Pd-like component) ---
        left_stim_epochs_dl = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-1" in x]]
        right_stim_epochs_dl = epochs_sub[[x for x in all_conds_sub if "Target-2-Singleton-3" in x]]

        if len(left_stim_epochs_dl) > 0 and len(right_stim_epochs_dl) > 0:
            contra_dl_left = left_stim_epochs_dl.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_dl_left = left_stim_epochs_dl.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            contra_dl_right = right_stim_epochs_dl.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_dl_right = right_stim_epochs_dl.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            diff_wave_dl_sub = np.mean([contra_dl_left, contra_dl_right], axis=0) - np.mean(
                [ipsi_dl_left, ipsi_dl_right], axis=0)
            subject_diff_waves[subject_id_str]['distractor_lateral'] = diff_wave_dl_sub.squeeze()

        # --- 2. Target Lateral (Distractor Present; N2ac-like component) ---
        left_stim_epochs_tdp = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-2" in x]]
        right_stim_epochs_tdp = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-2" in x]]

        if len(left_stim_epochs_tdp) > 0 and len(right_stim_epochs_tdp) > 0:
            contra_tdp_left = left_stim_epochs_tdp.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_tdp_left = left_stim_epochs_tdp.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            contra_tdp_right = right_stim_epochs_tdp.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_tdp_right = right_stim_epochs_tdp.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            diff_wave_tdp_sub = np.mean([contra_tdp_left, contra_tdp_right], axis=0) - np.mean(
                [ipsi_tdp_left, ipsi_tdp_right], axis=0)
            subject_diff_waves[subject_id_str]['target_distractor_present'] = diff_wave_tdp_sub.squeeze()

        # --- 3. Target Lateral (Distractor Absent; N2ac-like component) ---
        left_stim_epochs_tda = epochs_sub[[x for x in all_conds_sub if "Target-1-Singleton-0" in x]]
        right_stim_epochs_tda = epochs_sub[[x for x in all_conds_sub if "Target-3-Singleton-0" in x]]

        if len(left_stim_epochs_tda) > 0 and len(right_stim_epochs_tda) > 0:
            contra_tda_left = left_stim_epochs_tda.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            ipsi_tda_left = left_stim_epochs_tda.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            contra_tda_right = right_stim_epochs_tda.copy().average(picks=ELECTRODE_LEFT_HEMISPHERE).data
            ipsi_tda_right = right_stim_epochs_tda.copy().average(picks=ELECTRODE_RIGHT_HEMISPHERE).data
            diff_wave_tda_sub = np.mean([contra_tda_left, contra_tda_right], axis=0) - np.mean(
                [ipsi_tda_left, ipsi_tda_right], axis=0)
            subject_diff_waves[subject_id_str]['target_distractor_absent'] = diff_wave_tda_sub.squeeze()

        if not subject_diff_waves[subject_id_str]:
            del subject_diff_waves[subject_id_str]
        else:
            print(
                f"  Processed subject {subject_id_str}. Found data for: {list(subject_diff_waves[subject_id_str].keys())}")

    except Exception as e:
        print(f"  Error processing subject {subject_id_str}: {e}. Skipping.")
        if subject_id_str in subject_diff_waves:
            del subject_diff_waves[subject_id_str]
        continue

print(f"\n--- Finished. Processed data for {len(subject_diff_waves)} subjects. ---")

# --- NEW: Calculate Subject-Level Latency and Amplitude Metrics ---
print("\n--- Calculating Subject-Level Fractional Area Latencies & Amplitudes ---")
subject_metrics = {}
for sub_id, cond_data in subject_diff_waves.items():
    subject_metrics[sub_id] = {}
    for cond_key, wave in cond_data.items():
        is_target_component = 'target' in cond_key
        analysis_window = N2AC_ANALYSIS_WINDOW if is_target_component else PD_ANALYSIS_WINDOW

        latency = calculate_fractional_area_latency(
            wave, times_vector,
            percentage=LATENCY_PERCENTAGE,
            is_target=is_target_component,
            analysis_window_times=analysis_window
        )
        amplitude = np.nan
        if not np.isnan(latency) and (times_vector[0] <= latency <= times_vector[-1]):
            amplitude = np.interp(latency, times_vector, wave) * AMPLITUDE_SCALE_FACTOR

        subject_metrics[sub_id][cond_key] = {'latency': latency, 'amplitude': amplitude}

# --- Grand Average Calculation ---
print("\n--- Calculating Grand Averages ---")
ga_diff_waves = {}
stacked_diff_waves = {}
condition_keys = list(PLOT_LEGENDS.keys())[1:]

for key in condition_keys:
    waves_list = [data[key] for sub, data in subject_diff_waves.items() if key in data]
    if waves_list:
        stacked_data = np.array(waves_list)
        ga_diff_waves[key] = np.mean(stacked_data, axis=0)
        stacked_diff_waves[key] = stacked_data
        print(f"  GA for '{PLOT_LEGENDS.get(key, key)}': {len(waves_list)} subjects.")

# --- Plotting and Statistical Analysis ---
print("\n--- Plotting Grand Averages and Running Paired Permutation t-tests ---")
fig, ax = plt.subplots(figsize=(14, 9))

# --- Plotting Loop ---
for key in condition_keys:
    if key in ga_diff_waves:
        ga_wave = ga_diff_waves[key]
        stacked_data = stacked_diff_waves[key]
        n_subs = stacked_data.shape[0]

        ga_wave_plot = savgol_filter(ga_wave, SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER) * AMPLITUDE_SCALE_FACTOR
        sem_wave = sem(stacked_data, axis=0)
        t_crit = t.ppf(1 - 0.05 / 2, n_subs - 1)
        ci_range = sem_wave * t_crit * AMPLITUDE_SCALE_FACTOR

        ax.plot(times_vector, ga_wave_plot, color=PLOT_COLORS[key], lw=2.5, label=f"{PLOT_LEGENDS[key]} (N={n_subs})")
        ax.fill_between(times_vector, ga_wave_plot - ci_range, ga_wave_plot + ci_range, color=PLOT_COLORS[key],
                        alpha=0.1)

        # Add crosshairs based on GA wave metrics
        is_target_comp = 'target' in key
        analysis_window = N2AC_ANALYSIS_WINDOW if is_target_comp else PD_ANALYSIS_WINDOW
        latency_on_ga = calculate_fractional_area_latency(
            ga_wave, times_vector, percentage=LATENCY_PERCENTAGE, is_target=is_target_comp,
            analysis_window_times=analysis_window)

        if not np.isnan(latency_on_ga) and (times_vector[0] <= latency_on_ga <= times_vector[-1]):
            amplitude_on_ga = np.interp(latency_on_ga, times_vector, ga_wave_plot)
            ax.plot([latency_on_ga, latency_on_ga], [0, amplitude_on_ga], color=PLOT_COLORS[key], linestyle=':', lw=1.5)
            ax.plot([ax.get_xlim()[0], latency_on_ga], [amplitude_on_ga, amplitude_on_ga], color=PLOT_COLORS[key],
                    linestyle=':', lw=1.5)
            ax.plot(latency_on_ga, amplitude_on_ga, 'o', markerfacecolor=PLOT_COLORS[key], markeredgecolor='k',
                    markersize=8, zorder=11)

# --- Paired Permutation t-tests on Metrics ---
print("\n--- Running Paired Permutation t-tests on Metrics ---")
comparisons = list(itertools.combinations(condition_keys, 2))
stats_text_handles = []

for cond1, cond2 in comparisons:
    print(f"\n--- Comparing: {PLOT_LEGENDS[cond1]} vs {PLOT_LEGENDS[cond2]} ---")


    def run_metric_test(metric_name):
        common_subjects = [s for s, m in subject_metrics.items() if cond1 in m and cond2 in m and
                           not np.isnan(m[cond1][metric_name]) and not np.isnan(m[cond2][metric_name])]

        if len(common_subjects) < 3:
            print(f"  Not enough common subjects ({len(common_subjects)}) for {metric_name} test. Skipping.")
            return "p=n.s."

        metric1 = [subject_metrics[s][cond1][metric_name] for s in common_subjects]
        metric2 = [subject_metrics[s][cond2][metric_name] for s in common_subjects]
        diffs = np.array(metric1) - np.array(metric2)

        _, p_val, _ = permutation_t_test(diffs[:, np.newaxis], n_permutations=N_PERMUTATIONS_TTEST, seed=SEED)
        p_val_float = p_val[0]

        p_str = f"p={p_val_float:.3f}" if p_val_float >= 0.001 else "p<0.001"
        if p_val_float < P_VAL_ALPHA:
            p_str += '*'
        return p_str


    p_text_lat = run_metric_test('latency')
    p_text_amp = run_metric_test('amplitude')

    name1_short = PLOT_LEGENDS[cond1].split('(')[0].strip()
    name2_short = PLOT_LEGENDS[cond2].split('(')[0].strip()

    stats_text_handles.append(Line2D([0], [0], color='w', label=f"'{name1_short}' vs '{name2_short}':"))
    stats_text_handles.append(Line2D([0], [0], color='w', label=f"  Latency: {p_text_lat}"))
    stats_text_handles.append(Line2D([0], [0], color='w', label=f"  Amplitude: {p_text_amp}"))
    stats_text_handles.append(Line2D([0], [0], color='w', label=" "))  # Spacer

# --- Finalize Plot ---
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(0, color="black", linestyle=":", linewidth=1)

# Create legend
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Metric on GA Wave', markersize=8,
                      linestyle='None'))
handles.extend(stats_text_handles)
ax.legend(handles=handles, loc='best', title="Paired Comparisons", fontsize=10)

ax.set_title(
    f"Grand Average Difference Waves (Contra - Ipsi) at {ELECTRODE_LEFT_HEMISPHERE}/{ELECTRODE_RIGHT_HEMISPHERE}",
    fontsize=16, weight='bold')
ax.set_ylabel("Amplitude (µV)", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show(block=True)
