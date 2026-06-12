import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore, LinearModel, get_coef
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats
from pathlib import Path

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

# Toggle switch for Temporal Generalization Analysis
run_temporal_generalization = True

# Get the base data directory and convert it to a Path object
base_data_path = Path(get_data_path())

# Initialize arrays to store the decoding accuracy for each subject
all_subject_scores_target = np.zeros((n_subjects, len(times)))
all_subject_scores_distractor = np.zeros((n_subjects, len(times)))
all_subject_scores_control = np.zeros((n_subjects, len(times)))

# Initialize arrays for correct/incorrect decoding
all_subject_scores_target_correct = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_distractor_correct = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_control_correct = np.full((n_subjects, len(times)), np.nan)

all_subject_scores_target_incorrect = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_distractor_incorrect = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_control_incorrect = np.full((n_subjects, len(times)), np.nan)

# Initialize arrays for priming conditions
# -1: negative, 0: no-priming, 1: positive
all_subject_scores_target_priming_neg = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_target_priming_neu = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_target_priming_pos = np.full((n_subjects, len(times)), np.nan)

all_subject_scores_distractor_priming_neg = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_distractor_priming_neu = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_distractor_priming_pos = np.full((n_subjects, len(times)), np.nan)

all_subject_scores_control_priming_neg = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_control_priming_neu = np.full((n_subjects, len(times)), np.nan)
all_subject_scores_control_priming_pos = np.full((n_subjects, len(times)), np.nan)


# Initialize for spatial patterns over time
time_bins = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
all_subject_patterns_target = {i: [] for i in range(len(time_bins))}
all_subject_patterns_distractor = {i: [] for i in range(len(time_bins))}
all_subject_patterns_control = {i: [] for i in range(len(time_bins))}


# Set up the decoders (using 5-fold cross-validation)
# Using Multinomial Logistic Regression natively for 3-class decoding
clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
time_decoder = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize arrays for temporal generalization if toggled
if run_temporal_generalization:
    all_subject_tg_scores_target = np.zeros((n_subjects, len(times), len(times)))
    all_subject_tg_scores_distractor = np.zeros((n_subjects, len(times), len(times)))
    all_subject_tg_scores_control = np.zeros((n_subjects, len(times), len(times)))
    tg_decoder = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')

n_classes = 3

print(f"Starting within-task decoding for LOCATIONS (Selective Listening - Multinomial) for {n_subjects} subjects...")


def get_cv_scores(X, y, is_corr, priming, cv, estimator):
    cv_splits = list(cv.split(X, y))
    scores_all = []
    scores_c = []
    scores_i = []
    scores_p_neg = []
    scores_p_neu = []
    scores_p_pos = []
    
    for train_idx, test_idx in cv_splits:
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        
        # Score all
        scores_all.append(est.score(X[test_idx], y[test_idx]))
        
        # Test indices for correctness
        test_c_idx = test_idx[is_corr[test_idx] == True]
        test_i_idx = test_idx[is_corr[test_idx] == False]
        
        if len(test_c_idx) > 0:
            scores_c.append(est.score(X[test_c_idx], y[test_c_idx]))
        if len(test_i_idx) > 0:
            scores_i.append(est.score(X[test_i_idx], y[test_i_idx]))

        # Test indices for priming
        test_p_neg_idx = test_idx[priming[test_idx] == -1]
        test_p_neu_idx = test_idx[priming[test_idx] == 0]
        test_p_pos_idx = test_idx[priming[test_idx] == 1]

        if len(test_p_neg_idx) > 0:
            scores_p_neg.append(est.score(X[test_p_neg_idx], y[test_p_neg_idx]))
        if len(test_p_neu_idx) > 0:
            scores_p_neu.append(est.score(X[test_p_neu_idx], y[test_p_neu_idx]))
        if len(test_p_pos_idx) > 0:
            scores_p_pos.append(est.score(X[test_p_pos_idx], y[test_p_pos_idx]))
            
    res_all = np.mean(scores_all, axis=0)
    res_c = np.mean(scores_c, axis=0) if len(scores_c) > 0 else np.full(X.shape[2], np.nan)
    res_i = np.mean(scores_i, axis=0) if len(scores_i) > 0 else np.full(X.shape[2], np.nan)
    res_p_neg = np.mean(scores_p_neg, axis=0) if len(scores_p_neg) > 0 else np.full(X.shape[2], np.nan)
    res_p_neu = np.mean(scores_p_neu, axis=0) if len(scores_p_neu) > 0 else np.full(X.shape[2], np.nan)
    res_p_pos = np.mean(scores_p_pos, axis=0) if len(scores_p_pos) > 0 else np.full(X.shape[2], np.nan)
    
    return res_all, res_c, res_i, res_p_neg, res_p_neu, res_p_pos


# ==========================================
# 2. WITHIN-TASK DECODING
# ==========================================
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # --- Load Data (Active/Selective Listening) ---
    # Construct the file path using pathlib for cross-platform compatibility
    file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-spaceprime-epo.fif"
    
    epochs = mne.read_epochs(file_path, preload=True)
    epochs.set_montage("easycap-M1")
    epochs.crop(times[0], times[-1])
    
    # Target locations
    try:
        y_target = epochs.metadata['TargetLoc'].values
    except KeyError:
        print(f"Warning: 'TargetLoc' column not found in metadata for subject {sub}.")
        y_target = np.zeros(len(epochs))
    
    # Distractor locations
    try:
        y_distractor = epochs.metadata['SingletonLoc'].values
    except KeyError:
        print(f"Warning: 'SingletonLoc' column not found in metadata for subject {sub}.")
        y_distractor = np.zeros(len(epochs))
        
    # Control locations
    try:
        y_control = epochs.metadata['Non-Singleton2Loc'].values
    except KeyError:
        print(f"Warning: 'Non-Singleton2Loc' column not found in metadata for subject {sub}.")
        y_control = np.zeros(len(epochs))

    # Correct / Incorrect
    try:
        is_correct = epochs.metadata['select_target'].values
    except KeyError:
        print(f"Warning: 'select_target' column not found in metadata for subject {sub}.")
        is_correct = np.ones(len(epochs), dtype=bool)

    # Priming
    try:
        priming = epochs.metadata['Priming'].values
    except KeyError:
        print(f"Warning: 'Priming' column not found in metadata for subject {sub}.")
        priming = np.full(len(epochs), np.nan)

    # --- Joint Mask for Simultaneous Trials ---
    # Only keep trials where target, distractor, and control locations were presented
    valid_trials = (y_target > 0) & (y_distractor > 0) & (y_control > 0)
    X_shared = epochs.get_data()[valid_trials]
    y_target_valid = y_target[valid_trials]
    y_distractor_valid = y_distractor[valid_trials]
    y_control_valid = y_control[valid_trials]
    is_correct_valid = is_correct[valid_trials]
    priming_valid = priming[valid_trials]

    if len(y_target_valid) > 0:
        print(f"  Training and testing on {len(y_target_valid)} simultaneous trials (Correct: {np.sum(is_correct_valid == True)}, Incorrect: {np.sum(is_correct_valid == False)})...")
        print(f"    Priming: Neg: {np.sum(priming_valid == -1)}, Neu: {np.sum(priming_valid == 0)}, Pos: {np.sum(priming_valid == 1)}")

        # Diagnostic: Check distributions
        unique_t, counts_t = np.unique(y_target_valid, return_counts=True)
        unique_d, counts_d = np.unique(y_distractor_valid, return_counts=True)
        unique_c, counts_c = np.unique(y_control_valid, return_counts=True)
        print(f"    Target distributions: {dict(zip(unique_t, counts_t))}")
        print(f"    Distractor distributions: {dict(zip(unique_d, counts_d))}")
        print(f"    Control distributions: {dict(zip(unique_c, counts_c))}")

        # --- Target Decoding ---
        res_t_all, res_t_c, res_t_i, res_t_pn, res_t_p0, res_t_pp = get_cv_scores(X_shared, y_target_valid, is_correct_valid, priming_valid, cv, time_decoder)
        all_subject_scores_target[i, :] = res_t_all
        all_subject_scores_target_correct[i, :] = res_t_c
        all_subject_scores_target_incorrect[i, :] = res_t_i
        all_subject_scores_target_priming_neg[i, :] = res_t_pn
        all_subject_scores_target_priming_neu[i, :] = res_t_p0
        all_subject_scores_target_priming_pos[i, :] = res_t_pp

        # Target Temporal Generalization
        if run_temporal_generalization:
            tg_scores_target = cross_val_multiscore(tg_decoder, X_shared, y_target_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_target[i, :, :] = np.mean(tg_scores_target, axis=0)

        # --- Distractor Decoding ---
        res_d_all, res_d_c, res_d_i, res_d_pn, res_d_p0, res_d_pp = get_cv_scores(X_shared, y_distractor_valid, is_correct_valid, priming_valid, cv, time_decoder)
        all_subject_scores_distractor[i, :] = res_d_all
        all_subject_scores_distractor_correct[i, :] = res_d_c
        all_subject_scores_distractor_incorrect[i, :] = res_d_i
        all_subject_scores_distractor_priming_neg[i, :] = res_d_pn
        all_subject_scores_distractor_priming_neu[i, :] = res_d_p0
        all_subject_scores_distractor_priming_pos[i, :] = res_d_pp

        # Distractor Temporal Generalization
        if run_temporal_generalization:
            tg_scores_distractor = cross_val_multiscore(tg_decoder, X_shared, y_distractor_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_distractor[i, :, :] = np.mean(tg_scores_distractor, axis=0)

        # --- Control Decoding ---
        res_c_all, res_c_c, res_c_i, res_c_pn, res_c_p0, res_c_pp = get_cv_scores(X_shared, y_control_valid, is_correct_valid, priming_valid, cv, time_decoder)
        all_subject_scores_control[i, :] = res_c_all
        all_subject_scores_control_correct[i, :] = res_c_c
        all_subject_scores_control_incorrect[i, :] = res_c_i
        all_subject_scores_control_priming_neg[i, :] = res_c_pn
        all_subject_scores_control_priming_neu[i, :] = res_c_p0
        all_subject_scores_control_priming_pos[i, :] = res_c_pp

        # Control Temporal Generalization
        if run_temporal_generalization:
            tg_scores_control = cross_val_multiscore(tg_decoder, X_shared, y_control_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_control[i, :, :] = np.mean(tg_scores_control, axis=0)

        # --- Extract Spatial Patterns for Time Bins ---
        for bin_idx, t_bin in enumerate(time_bins):
            t_idx_bin = epochs.time_as_index(t_bin)
            X_shared_bin = X_shared[:, :, t_idx_bin[0]:t_idx_bin[1]].mean(axis=2)

            pattern_clf = make_pipeline(StandardScaler(),
                                        LinearModel(LogisticRegression(multi_class='multinomial',
                                                                       solver='lbfgs', max_iter=1000)))

            # Target
            pattern_clf.fit(X_shared_bin, y_target_valid)
            patterns_t = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
            mean_pattern_t = np.mean(np.abs(patterns_t), axis=0)
            all_subject_patterns_target[bin_idx].append(mean_pattern_t)

            # Distractor
            pattern_clf.fit(X_shared_bin, y_distractor_valid)
            patterns_d = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
            mean_pattern_d = np.mean(np.abs(patterns_d), axis=0)
            all_subject_patterns_distractor[bin_idx].append(mean_pattern_d)

            # Control
            pattern_clf.fit(X_shared_bin, y_control_valid)
            patterns_c = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
            mean_pattern_c = np.mean(np.abs(patterns_c), axis=0)
            all_subject_patterns_control[bin_idx].append(mean_pattern_c)
        
    else:
        print("  No valid simultaneous trials found.")
        for bin_idx in range(len(time_bins)):
            all_subject_patterns_target[bin_idx].append(np.zeros(epochs.info['nchan']))
            all_subject_patterns_distractor[bin_idx].append(np.zeros(epochs.info['nchan']))
            all_subject_patterns_control[bin_idx].append(np.zeros(epochs.info['nchan']))

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


mean_target, se_target, cl_target, p_target = compute_stats(all_subject_scores_target)
mean_distractor, se_distractor, cl_distractor, p_distractor = compute_stats(all_subject_scores_distractor)
mean_control, se_control, cl_control, p_control = compute_stats(all_subject_scores_control)

# ==========================================
# 4. PLOTTING TIME-RESOLVED DECODING
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)


def plot_results(ax, mean_scores, std_error, clusters, cluster_p_values, title, color):
    ax.plot(times, mean_scores, label='Group Mean Accuracy', color=color, linewidth=2)
    ax.fill_between(times, mean_scores - std_error, mean_scores + std_error,
                    color=color, alpha=0.2, label='SEM')

    ax.axhline(chance_level, color='k', linestyle='--', label=f'Chance Level ({chance_level*100:.1f}%)')
    ax.axvline(0, color='k', linestyle='-')

    label_added = False
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            sig_times = times[c]
            if len(sig_times) > 0:
                label = 'p < 0.05 (Cluster)' if not label_added else ""
                ax.axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
                label_added = True

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


# Plot Targets
plot_results(axes[0], mean_target, se_target, cl_target, p_target, 
             f'Decoding Target Locations (N={n_subjects})', 'green')
axes[0].set_ylabel('Accuracy')

# Plot Distractors
plot_results(axes[1], mean_distractor, se_distractor, cl_distractor, p_distractor, 
             f'Decoding Distractor Locations (N={n_subjects})', 'red')

# Plot Control
plot_results(axes[2], mean_control, se_control, cl_control, p_control, 
             f'Decoding Control Locations (N={n_subjects})', 'grey')

plt.tight_layout()
plt.show()

# ==========================================
# 5. PLOT GROUP-LEVEL SPATIAL PATTERNS
# ==========================================
n_bins = len(time_bins)
group_patterns_target = [np.mean(all_subject_patterns_target[i], axis=0) for i in range(n_bins)]
group_patterns_distractor = [np.mean(all_subject_patterns_distractor[i], axis=0) for i in range(n_bins)]
group_patterns_control = [np.mean(all_subject_patterns_control[i], axis=0) for i in range(n_bins)]

vmax_t = max(np.max(p) for p in group_patterns_target) if any(np.any(p) for p in group_patterns_target) else 1
vmax_d = max(np.max(p) for p in group_patterns_distractor) if any(np.any(p) for p in group_patterns_distractor) else 1
vmax_c = max(np.max(p) for p in group_patterns_control) if any(np.any(p) for p in group_patterns_control) else 1

fig, axes = plt.subplots(3, n_bins, figsize=(2.5 * n_bins, 10.5))

# Plot Targets Patterns
for bin_idx, (t_bin, group_pattern) in enumerate(zip(time_bins, group_patterns_target)):
    ax = axes[0, bin_idx]
    im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax, show=False,
                                 cmap='Greens', contours=4, vlim=(0, vmax_t))
    ax.set_title(f'{t_bin[0]*1000:.0f} to\n{t_bin[1]*1000:.0f} ms', fontsize=12)
    if bin_idx == 0:
        ax.text(-0.3, 0.5, 'Targets', transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='center', ha='right', rotation=90)
cbar_ax_t = fig.add_axes([0.92, 0.68, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax_t, label='Pattern Activation (A.U.)')

# Plot Distractors Patterns
for bin_idx, (t_bin, group_pattern) in enumerate(zip(time_bins, group_patterns_distractor)):
    ax = axes[1, bin_idx]
    im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax, show=False,
                                 cmap='Reds', contours=4, vlim=(0, vmax_d))
    if bin_idx == 0:
        ax.text(-0.3, 0.5, 'Distractors', transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='center', ha='right', rotation=90)
cbar_ax_d = fig.add_axes([0.92, 0.40, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax_d, label='Pattern Activation (A.U.)')

# Plot Control Patterns
for bin_idx, (t_bin, group_pattern) in enumerate(zip(time_bins, group_patterns_control)):
    ax = axes[2, bin_idx]
    im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax, show=False,
                                 cmap='Greys', contours=4, vlim=(0, vmax_c))
    if bin_idx == 0:
        ax.text(-0.3, 0.5, 'Control', transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='center', ha='right', rotation=90)
cbar_ax_c = fig.add_axes([0.92, 0.12, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax_c, label='Pattern Activation (A.U.)')

fig.suptitle('Group-Level Spatial Patterns over Time', y=1.02)
plt.subplots_adjust(top=0.9, right=0.9, wspace=0.1, hspace=0.3)
plt.show()

# Plot Differences in Spatial Patterns (Target vs Distractor)
diff_patterns_t_d = [group_patterns_target[i] - group_patterns_distractor[i] for i in range(n_bins)]
vmax_diff = max(np.abs(p).max() for p in diff_patterns_t_d) if any(np.any(p) for p in diff_patterns_t_d) else 1
vmin_diff = -vmax_diff

fig, axes = plt.subplots(1, n_bins, figsize=(2.5 * n_bins, 3.5))
if n_bins == 1:
    axes = [axes]

for bin_idx, (t_bin, diff_pattern) in enumerate(zip(time_bins, diff_patterns_t_d)):
    ax = axes[bin_idx]
    im_diff, _ = mne.viz.plot_topomap(diff_pattern, epochs_info, axes=ax, show=False,
                                      cmap='RdBu_r', contours=4, vlim=(vmin_diff, vmax_diff))
    ax.set_title(f'{t_bin[0]*1000:.0f} to\n{t_bin[1]*1000:.0f} ms', fontsize=12)

cbar_ax_diff = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im_diff, cax=cbar_ax_diff, label='Difference (A.U.)')
fig.suptitle('Difference: Target - Distractor over Time', y=1.05)
plt.subplots_adjust(top=0.8, right=0.9, wspace=0.1)
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
        
    mean_tg_target, sig_mask_tg_target = compute_tg_stats(all_subject_tg_scores_target)
    mean_tg_distractor, sig_mask_tg_distractor = compute_tg_stats(all_subject_tg_scores_distractor)
    mean_tg_control, sig_mask_tg_control = compute_tg_stats(all_subject_tg_scores_control)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    # Create symmetric colorbar around chance level
    max_dev_t = np.max(np.abs(mean_tg_target - chance_level))
    max_dev_d = np.max(np.abs(mean_tg_distractor - chance_level))
    max_dev_c = np.max(np.abs(mean_tg_control - chance_level))
    limit = max(max_dev_t, max_dev_d, max_dev_c)
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

    im1 = plot_tg(axes[0], mean_tg_target, sig_mask_tg_target, f'TG: Target Locations\n(Selective, N={n_subjects})')
    axes[0].set_ylabel('Training Time (s)')
    im2 = plot_tg(axes[1], mean_tg_distractor, sig_mask_tg_distractor, f'TG: Distractor Locations\n(Selective, N={n_subjects})')
    im3 = plot_tg(axes[2], mean_tg_control, sig_mask_tg_control, f'TG: Control Locations\n(Selective, N={n_subjects})')
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), label='Accuracy')
    plt.show()

# ==========================================
# 7. COMPARE TARGET AND DISTRACTOR DECODING
# ==========================================
print("\nComparing Target vs Distractor decoding...")

# Contrast: Distractors - Control
diff_scores = all_subject_scores_distractor - all_subject_scores_control
mean_diff = np.mean(diff_scores, axis=0)
se_diff = np.std(diff_scores, axis=0) / np.sqrt(n_subjects)

# Run permutation test on the difference (testing against a mean of 0)
diff_T_obs, diff_clusters, diff_cluster_p_values, diff_H0 = permutation_cluster_1samp_test(
    diff_scores, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(times, mean_diff, label='Distractor - Control', color='purple', linewidth=2)
plt.fill_between(times, mean_diff - se_diff, mean_diff + se_diff, color='purple', alpha=0.2, label='SEM')
plt.axhline(0, color='k', linestyle='--', label='No Difference')
plt.axvline(0, color='k', linestyle='-')

# Mark significant clusters for the difference
label_added = False
for c, p_val in zip(diff_clusters, diff_cluster_p_values):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Cluster)' if not label_added else ""
            plt.axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
            label_added = True

plt.title(f'Difference in Decoding Accuracy: Distractor vs Control Locations\n(Group Level, N={n_subjects})')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy Difference (Distractor - Control)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Contrast: Targets - Control
diff_scores_tc = all_subject_scores_target - all_subject_scores_control
mean_diff_tc = np.mean(diff_scores_tc, axis=0)
se_diff_tc = np.std(diff_scores_tc, axis=0) / np.sqrt(n_subjects)

# Run permutation test on the difference (testing against a mean of 0)
diff_T_obs_tc, diff_clusters_tc, diff_cluster_p_values_tc, diff_H0_tc = permutation_cluster_1samp_test(
    diff_scores_tc, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(times, mean_diff_tc, label='Target - Control', color='purple', linewidth=2)
plt.fill_between(times, mean_diff_tc - se_diff_tc, mean_diff_tc + se_diff_tc, color='purple', alpha=0.2, label='SEM')
plt.axhline(0, color='k', linestyle='--', label='No Difference')
plt.axvline(0, color='k', linestyle='-')

# Mark significant clusters for the difference
label_added = False
for c, p_val in zip(diff_clusters_tc, diff_cluster_p_values_tc):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Cluster)' if not label_added else ""
            plt.axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
            label_added = True

plt.title(f'Difference in Decoding Accuracy: Target vs Control Locations\n(Group Level, N={n_subjects})')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy Difference (Target - Control)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 8. CORRECT VS INCORRECT DECODING PLOTS
# ==========================================
print("\nPlotting Correct vs Incorrect Trial Decoding...")


def compute_nan_stats(scores):
    # Some subjects may not have incorrect trials, so scores can be NaN
    valid_subs = ~np.isnan(scores).any(axis=1)
    valid_scores = scores[valid_subs]
    
    mean_scores = np.mean(valid_scores, axis=0) if len(valid_scores) > 0 else np.full(scores.shape[1], np.nan)
    
    if len(valid_scores) > 0:
        std_error = (np.std(valid_scores, axis=0) / np.sqrt(len(valid_scores))) 
    else:
        std_error = np.full(scores.shape[1], np.nan)
        
    # Optional cluster permutation test against chance
    clusters = []
    cluster_p_values = []
    if len(valid_scores) > 1:
        X_test_stats = valid_scores - chance_level
        if np.any(X_test_stats):
            t_thresh_nan = scipy.stats.t.ppf(1 - 0.05 / 2, df=len(valid_scores) - 1)
            try:
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                    X_test_stats, n_permutations=1000, threshold=t_thresh_nan, tail=0, out_type='mask', n_jobs=-1)
            except Exception:
                pass
            
    return mean_scores, std_error, clusters, cluster_p_values


def compute_diff_stats(scores_corr, scores_inc):
    # Only keep subjects that have valid (non-NaN) data in BOTH correct and incorrect trials
    valid_subs_c = ~np.isnan(scores_corr).any(axis=1)
    valid_subs_i = ~np.isnan(scores_inc).any(axis=1)
    valid_subs = valid_subs_c & valid_subs_i
    
    diff_scores = scores_corr[valid_subs] - scores_inc[valid_subs]
    
    mean_diff = np.mean(diff_scores, axis=0) if len(diff_scores) > 0 else np.full(scores_corr.shape[1], np.nan)
    
    if len(diff_scores) > 0:
        std_error = (np.std(diff_scores, axis=0) / np.sqrt(len(diff_scores))) 
    else:
        std_error = np.full(scores_corr.shape[1], np.nan)
        
    clusters = []
    cluster_p_values = []
    if len(diff_scores) > 1:
        if np.any(diff_scores):
            t_thresh_nan = scipy.stats.t.ppf(1 - 0.05 / 2, df=len(diff_scores) - 1)
            try:
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                    diff_scores, n_permutations=1000, threshold=t_thresh_nan, tail=0, out_type='mask', n_jobs=-1)
            except Exception:
                pass
            
    return mean_diff, std_error, clusters, cluster_p_values


fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

# Target
mean_t_c, se_t_c, cl_t_c, p_t_c = compute_nan_stats(all_subject_scores_target_correct)
mean_t_i, se_t_i, cl_t_i, p_t_i = compute_nan_stats(all_subject_scores_target_incorrect)
mean_diff_t, se_diff_t, cl_diff_t, p_diff_t = compute_diff_stats(
    all_subject_scores_target_correct, all_subject_scores_target_incorrect)

axes[0].plot(times, mean_t_c, label='Correct', color='green', linewidth=2)
axes[0].fill_between(times, mean_t_c - se_t_c, mean_t_c + se_t_c, color='green', alpha=0.2)
axes[0].plot(times, mean_t_i, label='Incorrect', color='darkolivegreen', linewidth=2, linestyle='--')
axes[0].fill_between(times, mean_t_i - se_t_i, mean_t_i + se_t_i, color='darkolivegreen', alpha=0.2)
axes[0].axhline(chance_level, color='k', linestyle='--', label=f'Chance ({chance_level*100:.1f}%)')
axes[0].axvline(0, color='k', linestyle='-')

label_added = False
for c, p_val in zip(cl_diff_t, p_diff_t):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Diff Cluster)' if not label_added else ""
            axes[0].axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
            label_added = True

axes[0].set_title('Target Location Decoding (Correct vs Incorrect)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distractor
mean_d_c, se_d_c, cl_d_c, p_d_c = compute_nan_stats(all_subject_scores_distractor_correct)
mean_d_i, se_d_i, cl_d_i, p_d_i = compute_nan_stats(all_subject_scores_distractor_incorrect)
mean_diff_d, se_diff_d, cl_diff_d, p_diff_d = compute_diff_stats(
    all_subject_scores_distractor_correct, all_subject_scores_distractor_incorrect)

axes[1].plot(times, mean_d_c, label='Correct', color='red', linewidth=2)
axes[1].fill_between(times, mean_d_c - se_d_c, mean_d_c + se_d_c, color='red', alpha=0.2)
axes[1].plot(times, mean_d_i, label='Incorrect', color='darkred', linewidth=2, linestyle='--')
axes[1].fill_between(times, mean_d_i - se_d_i, mean_d_i + se_d_i, color='darkred', alpha=0.2)
axes[1].axhline(chance_level, color='k', linestyle='--')
axes[1].axvline(0, color='k', linestyle='-')

label_added = False
for c, p_val in zip(cl_diff_d, p_diff_d):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Diff Cluster)' if not label_added else ""
            axes[1].axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
            label_added = True

axes[1].set_title('Distractor Location Decoding (Correct vs Incorrect)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Control
mean_c_c, se_c_c, cl_c_c, p_c_c = compute_nan_stats(all_subject_scores_control_correct)
mean_c_i, se_c_i, cl_c_i, p_c_i = compute_nan_stats(all_subject_scores_control_incorrect)
mean_diff_c, se_diff_c, cl_diff_c, p_diff_c = compute_diff_stats(
    all_subject_scores_control_correct, all_subject_scores_control_incorrect)

axes[2].plot(times, mean_c_c, label='Correct', color='grey', linewidth=2)
axes[2].fill_between(times, mean_c_c - se_c_c, mean_c_c + se_c_c, color='grey', alpha=0.2)
axes[2].plot(times, mean_c_i, label='Incorrect', color='black', linewidth=2, linestyle='--')
axes[2].fill_between(times, mean_c_i - se_c_i, mean_c_i + se_c_i, color='black', alpha=0.2)
axes[2].axhline(chance_level, color='k', linestyle='--')
axes[2].axvline(0, color='k', linestyle='-')

label_added = False
for c, p_val in zip(cl_diff_c, p_diff_c):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Diff Cluster)' if not label_added else ""
            axes[2].axvspan(sig_times[0], sig_times[-1], color='blue', alpha=0.3, label=label)
            label_added = True

axes[2].set_title('Control Location Decoding (Correct vs Incorrect)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================
# 9. PRIMING DECODING PLOTS
# ==========================================
print("\nPlotting Priming Condition Decoding...")

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

# Target Priming
mean_t_neg, se_t_neg, cl_t_neg, p_t_neg = compute_nan_stats(all_subject_scores_target_priming_neg)
mean_t_neu, se_t_neu, cl_t_neu, p_t_neu = compute_nan_stats(all_subject_scores_target_priming_neu)
mean_t_pos, se_t_pos, cl_t_pos, p_t_pos = compute_nan_stats(all_subject_scores_target_priming_pos)

axes[0].plot(times, mean_t_neg, label='Negative (-1)', color='red', linewidth=2)
axes[0].fill_between(times, mean_t_neg - se_t_neg, mean_t_neg + se_t_neg, color='red', alpha=0.2)
axes[0].plot(times, mean_t_neu, label='Neutral (0)', color='grey', linewidth=2)
axes[0].fill_between(times, mean_t_neu - se_t_neu, mean_t_neu + se_t_neu, color='grey', alpha=0.2)
axes[0].plot(times, mean_t_pos, label='Positive (1)', color='green', linewidth=2)
axes[0].fill_between(times, mean_t_pos - se_t_pos, mean_t_pos + se_t_pos, color='green', alpha=0.2)

axes[0].axhline(chance_level, color='k', linestyle='--', label=f'Chance ({chance_level*100:.1f}%)')
axes[0].axvline(0, color='k', linestyle='-')
axes[0].set_title('Target Location Decoding (by Priming)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distractor Priming
mean_d_neg, se_d_neg, cl_d_neg, p_d_neg = compute_nan_stats(all_subject_scores_distractor_priming_neg)
mean_d_neu, se_d_neu, cl_d_neu, p_d_neu = compute_nan_stats(all_subject_scores_distractor_priming_neu)
mean_d_pos, se_d_pos, cl_d_pos, p_d_pos = compute_nan_stats(all_subject_scores_distractor_priming_pos)

axes[1].plot(times, mean_d_neg, label='Negative (-1)', color='red', linewidth=2)
axes[1].fill_between(times, mean_d_neg - se_d_neg, mean_d_neg + se_d_neg, color='red', alpha=0.2)
axes[1].plot(times, mean_d_neu, label='Neutral (0)', color='grey', linewidth=2)
axes[1].fill_between(times, mean_d_neu - se_d_neu, mean_d_neu + se_d_neu, color='grey', alpha=0.2)
axes[1].plot(times, mean_d_pos, label='Positive (1)', color='green', linewidth=2)
axes[1].fill_between(times, mean_d_pos - se_d_pos, mean_d_pos + se_d_pos, color='green', alpha=0.2)

axes[1].axhline(chance_level, color='k', linestyle='--')
axes[1].axvline(0, color='k', linestyle='-')
axes[1].set_title('Distractor Location Decoding (by Priming)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Control Priming
mean_c_neg, se_c_neg, cl_c_neg, p_c_neg = compute_nan_stats(all_subject_scores_control_priming_neg)
mean_c_neu, se_c_neu, cl_c_neu, p_c_neu = compute_nan_stats(all_subject_scores_control_priming_neu)
mean_c_pos, se_c_pos, cl_c_pos, p_c_pos = compute_nan_stats(all_subject_scores_control_priming_pos)

axes[2].plot(times, mean_c_neg, label='Negative', color='red', linewidth=2)
axes[2].fill_between(times, mean_c_neg - se_c_neg, mean_c_neg + se_c_neg, color='red', alpha=0.2)
axes[2].plot(times, mean_c_neu, label='Neutral', color='grey', linewidth=2)
axes[2].fill_between(times, mean_c_neu - se_c_neu, mean_c_neu + se_c_neu, color='grey', alpha=0.2)
axes[2].plot(times, mean_c_pos, label='Positive', color='green', linewidth=2)
axes[2].fill_between(times, mean_c_pos - se_c_pos, mean_c_pos + se_c_pos, color='green', alpha=0.2)

axes[2].axhline(chance_level, color='k', linestyle='--')
axes[2].axvline(0, color='k', linestyle='-')
axes[2].set_title('Control Location Decoding (by Priming)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
