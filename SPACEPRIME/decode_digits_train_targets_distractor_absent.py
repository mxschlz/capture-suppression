import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore, LinearModel, get_coef
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats
from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from SPACEPRIME.subjects import subject_ids as subjects
from SPACEPRIME import get_data_path

import seaborn as sns
sns.set_theme(context="talk", style="ticks")

# ==========================================
# 1. SETUP AND INITIALIZATION
# ==========================================
n_subjects = len(subjects)
tmin = -0.2
tmax = 1.0
sf = 250
samples = int((tmin.__abs__() + tmax) * 250 + 1)
times = np.linspace(-0.2, 1.0, samples)

run_temporal_generalization = True

# Get the base data directory and convert it to a Path object
base_data_path = Path(get_data_path())

# Initialize arrays to store the decoding accuracy for each subject
all_subject_scores_test_target = np.zeros((n_subjects, len(times)))
all_subject_scores_test_control = np.zeros((n_subjects, len(times)))
all_subject_scores_test_distractor = np.zeros((n_subjects, len(times)))
all_subject_scores_train_cv = np.zeros((n_subjects, len(times)))  # Sanity check

if run_temporal_generalization:
    all_subject_tg_scores_test_distractor = np.zeros((n_subjects, len(times), len(times)))

# Initialize for spatial patterns over time
time_bins = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
all_subject_patterns_target_train = {i: [] for i in range(len(time_bins))}

# Set up the decoders (using 3-fold cross-validation)
clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
time_decoder = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

if run_temporal_generalization:
    tg_decoder = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')
else:
    tg_decoder = None

n_classes = 9

print(f"Starting cross-condition decoding (Train: Target (distractor-absent) -> Test: Targets/Controls/Distractors) for {n_subjects} subjects...")


def get_cross_condition_scores(X_train, y_train, X_test, y_test_target, y_test_control, y_test_distractor, cv, estimator, tg_estimator=None):
    # Cross-validation for training data (sanity check)
    cv_splits = list(cv.split(X_train, y_train))
    scores_train_cv = []
    
    for train_idx, test_idx in cv_splits:
        est = clone(estimator)
        est.fit(X_train[train_idx], y_train[train_idx])
        scores_train_cv.append(est.score(X_train[test_idx], y_train[test_idx]))
            
    res_train_cv = np.mean(scores_train_cv, axis=0)
    
    # Train on all training data for cross-condition generalization
    est_full = clone(estimator)
    est_full.fit(X_train, y_train)
    
    res_target = est_full.score(X_test, y_test_target)
    res_control = est_full.score(X_test, y_test_control)
    res_distractor = est_full.score(X_test, y_test_distractor)
    
    res_tg_distractor = None
    if tg_estimator is not None:
        est_tg_full = clone(tg_estimator)
        est_tg_full.fit(X_train, y_train)
        
        res_tg_distractor = est_tg_full.score(X_test, y_test_distractor)
    
    return res_target, res_control, res_distractor, res_train_cv, res_tg_distractor


# ==========================================
# 2. WITHIN-TASK DECODING
# ==========================================
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # --- Load Data (Active/Selective Listening) ---
    file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-spaceprime-epo.fif"
    
    epochs = mne.read_epochs(file_path, preload=True)
    epochs.set_montage("easycap-M1")
    epochs.crop(times[0], times[-1])
    
    try:
        y_target = epochs.metadata['TargetDigit'].values
    except KeyError:
        print(f"Warning: 'TargetDigit' column not found in metadata for subject {sub}.")
        y_target = np.zeros(len(epochs))
    
    try:
        y_distractor = epochs.metadata['SingletonDigit'].values
    except KeyError:
        print(f"Warning: 'SingletonDigit' column not found in metadata for subject {sub}.")
        y_distractor = np.zeros(len(epochs))
        
    try:
        y_control = epochs.metadata['Non-Singleton2Digit'].values
    except KeyError:
        print(f"Warning: 'Non-Singleton2Digit' column not found in metadata for subject {sub}.")
        y_control = np.zeros(len(epochs))

    try:
        singleton_present = epochs.metadata['SingletonPresent'].values
    except KeyError:
        print(f"Warning: 'SingletonPresent' column not found in metadata for subject {sub}.")
        singleton_present = np.zeros(len(epochs))

    # --- Train Data (Distractor-Absent) ---
    train_mask = (singleton_present == 0) & (y_target >= 1) & (y_target <= 9)
    X_train = epochs.get_data()[train_mask]
    y_train = y_target[train_mask]

    # --- Test Data (Distractor-Present) ---
    test_mask = (singleton_present == 1) & (y_target >= 1) & (y_target <= 9) & \
                (y_distractor >= 1) & (y_distractor <= 9) & (y_control >= 1) & (y_control <= 9)
    X_test = epochs.get_data()[test_mask]
    y_test_target = y_target[test_mask]
    y_test_control = y_control[test_mask]
    y_test_distractor = y_distractor[test_mask]

    if len(y_train) > 0 and len(y_test_target) > 0:
        print(f"  Training on Targets (distractor-absent, N={len(y_train)}), Testing on Targets/Controls/Distractors (distractor-present, N={len(y_test_target)})...")

        res_t, res_c, res_d, res_train, res_tg_d = get_cross_condition_scores(
            X_train, y_train, X_test, y_test_target, y_test_control, y_test_distractor, cv, time_decoder, tg_decoder)
        
        all_subject_scores_test_target[i, :] = res_t
        all_subject_scores_test_control[i, :] = res_c
        all_subject_scores_test_distractor[i, :] = res_d
        all_subject_scores_train_cv[i, :] = res_train
        
        if run_temporal_generalization:
            all_subject_tg_scores_test_distractor[i, :, :] = res_tg_d

        # --- Extract Spatial Patterns for Time Bins (Trained on Targets in absent trials) ---
        for bin_idx, t_bin in enumerate(time_bins):
            t_idx_bin = epochs.time_as_index(t_bin)
            X_train_bin = X_train[:, :, t_idx_bin[0]:t_idx_bin[1]].mean(axis=2)

            pattern_clf = make_pipeline(StandardScaler(),
                                        LinearModel(LogisticRegression(multi_class='multinomial',
                                                                       solver='lbfgs', max_iter=1000)))

            # Train on Targets
            pattern_clf.fit(X_train_bin, y_train)
            patterns_t = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
            mean_pattern_t = np.mean(np.abs(patterns_t), axis=0)
            all_subject_patterns_target_train[bin_idx].append(mean_pattern_t)
        
    else:
        print("  Not enough valid trials found.")
        for bin_idx in range(len(time_bins)):
            all_subject_patterns_target_train[bin_idx].append(np.zeros(epochs.info['nchan']))

    epochs_info = epochs.info  # Keep for plotting

# ==========================================
# 3. GROUP-LEVEL AVERAGING & STATISTICAL TESTING
# ==========================================
chance_level = 1.0 / n_classes
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subjects - 1)  # Two-sided threshold


def compute_stats(scores):
    mean_scores = np.mean(scores, axis=0)
    std_error = np.std(scores, axis=0) / np.sqrt(n_subjects)
    
    X_test_stats = scores - chance_level
    if np.any(X_test_stats):
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            X_test_stats, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)
    else:
        clusters = []
        cluster_p_values = []

    return mean_scores, std_error, clusters, cluster_p_values


mean_test_target, se_test_target, cl_test_target, p_test_target = compute_stats(all_subject_scores_test_target)
mean_test_control, se_test_control, cl_test_control, p_test_control = compute_stats(all_subject_scores_test_control)
mean_test_distractor, se_test_distractor, cl_test_distractor, p_test_distractor = compute_stats(all_subject_scores_test_distractor)
mean_train_cv, se_train_cv, cl_train_cv, p_train_cv = compute_stats(all_subject_scores_train_cv)

# --- NEW: Compute stats directly contrasting Distractor vs Control (True Suppression) ---
X_test_suppression = all_subject_scores_test_distractor - all_subject_scores_test_control
if np.any(X_test_suppression):
    # tail=-1 tests if Distractor is significantly LESS than Control
    _, cl_suppression, p_suppression, _ = permutation_cluster_1samp_test(
        X_test_suppression, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)
else:
    cl_suppression = []
    p_suppression = []

# ==========================================
# 4 & 5. PLOTTING DECODING AND SPATIAL PATTERNS
# ==========================================
n_bins = len(time_bins)
group_patterns_train = [np.mean(all_subject_patterns_target_train[i], axis=0) for i in range(n_bins)]
vmax_c = max(np.max(p) for p in group_patterns_train) if any(np.any(p) for p in group_patterns_train) else 1

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, n_bins, height_ratios=[2, 1], hspace=0.3)

# Time-resolved decoding subplot
ax = fig.add_subplot(gs[0, :])

# Plotting the data for targets, controls, and distractors
ax.plot(times, mean_test_target, color='green', linewidth=2, label='Test on Targets')
ax.fill_between(times, mean_test_target - se_test_target, mean_test_target + se_test_target, color='green', alpha=0.2)

ax.plot(times, mean_test_control, color='grey', linewidth=2, label='Test on Controls')
ax.fill_between(times, mean_test_control - se_test_control, mean_test_control + se_test_control, color='grey', alpha=0.2)

ax.plot(times, mean_test_distractor, color='red', linewidth=2, label='Test on Distractors')
ax.fill_between(times, mean_test_distractor - se_test_distractor, mean_test_distractor + se_test_distractor, color='red', alpha=0.2)

ax.axhline(chance_level, color='k', linestyle='--', label=f'Chance ({chance_level*100:.1f}%)')
ax.axvline(0, color='k', linestyle='-')

# Determine a good y-position for the cluster lines, just below the lowest data point
min_data_y = np.min([
    (mean_test_target - se_test_target).min(),
    (mean_test_control - se_test_control).min(),
    (mean_test_distractor - se_test_distractor).min()
])
y_range = ax.get_ylim()[1] - min_data_y
cluster_line_y_base = min_data_y - 0.05 * y_range # position below data

cluster_y_positions = {
    'target': cluster_line_y_base,
    'control': cluster_line_y_base - 0.02 * y_range,
    'distractor': cluster_line_y_base - 0.04 * y_range,
    'suppression': cluster_line_y_base - 0.06 * y_range
}

# Plotting clusters as dashed lines
def plot_cluster_lines(ax, clusters, p_values, color, y_pos, label_prefix):
    label_added = False
    for c, p_val in zip(clusters, p_values):
        if p_val <= 0.05:
            sig_times = times[c]
            if len(sig_times) > 0:
                label = f'{label_prefix} p < 0.05' if not label_added else ""
                ax.plot([sig_times[0], sig_times[-1]], [y_pos, y_pos], color=color, linestyle='--', linewidth=3, label=label)
                label_added = True

plot_cluster_lines(ax, cl_test_target, p_test_target, 'green', cluster_y_positions['target'], 'Target')
plot_cluster_lines(ax, cl_test_control, p_test_control, 'grey', cluster_y_positions['control'], 'Control')
plot_cluster_lines(ax, cl_test_distractor, p_test_distractor, 'red', cluster_y_positions['distractor'], 'Distractor')
plot_cluster_lines(ax, cl_suppression, p_suppression, 'blue', cluster_y_positions['suppression'], 'Distractor < Control (True Suppression)')

ax.set_title(f'Decoding Performance (Train on Distractor-Absent, N={n_subjects})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Accuracy')
#ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Topomap subplots
for bin_idx, (t_bin, group_pattern) in enumerate(zip(time_bins, group_patterns_train)):
    ax_topo = fig.add_subplot(gs[1, bin_idx])
    im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax_topo, show=False,
                                 cmap='Greys', contours=4, vlim=(0, vmax_c))
    ax_topo.set_title(f'{t_bin[0]*1000:.0f} to\n{t_bin[1]*1000:.0f} ms', fontsize=10)

cbar_ax_c = fig.add_axes([0.92, 0.15, 0.015, 0.25])
fig.colorbar(im, cax=cbar_ax_c, label='Pattern Activation (A.U.)')

plt.subplots_adjust(left=0.08, right=0.9, top=0.92, bottom=0.1)
plt.show()

# ==========================================
# 6. PLOT TEMPORAL GENERALIZATION
# ==========================================
if run_temporal_generalization:
    print("Computing group-level 2D cluster statistics for Temporal Generalization...")
    
    def compute_tg_stats(tg_scores):
        mean_tg = np.mean(tg_scores, axis=0)
        X_test_stats_tg = tg_scores - chance_level
        sig_mask_tg = np.zeros(mean_tg.shape, dtype=bool)
        
        if np.any(X_test_stats_tg):
            T_obs_tg, clusters_tg, cluster_p_values_tg, H0_tg = permutation_cluster_1samp_test(
                X_test_stats_tg, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)
            for c, p_val in zip(clusters_tg, cluster_p_values_tg):
                if p_val <= 0.05:
                    sig_mask_tg |= c
        return mean_tg, sig_mask_tg
        
    mean_tg_distractor, sig_mask_tg_distractor = compute_tg_stats(all_subject_tg_scores_test_distractor)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    max_dev_d = np.max(np.abs(mean_tg_distractor - chance_level))
    limit = max_dev_d
    vmin, vmax = chance_level - limit, chance_level + limit

    def plot_tg(ax, mean_tg, sig_mask, title):
        im = ax.imshow(mean_tg, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                       extent=[times[0], times[-1], times[0], times[-1]], vmin=vmin, vmax=vmax)
        if sig_mask.any():
            ax.contour(times, times, sig_mask, levels=[0.5], colors='black', linewidths=1.5, linestyles='-')
        ax.plot([times[0], times[-1]], [times[0], times[-1]], 'k--', label='Diagonal', alpha=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Testing Time (s)')
        ax.set_title(title)
        return im

    im1 = plot_tg(ax, mean_tg_distractor, sig_mask_tg_distractor, f'TG: Train on Target, Test on Distractor Digits\n(N={n_subjects})')
    ax.set_ylabel('Training Time (s)')
    
    fig.colorbar(im1, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.show()
