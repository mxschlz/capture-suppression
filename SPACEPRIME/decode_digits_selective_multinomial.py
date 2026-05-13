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
peak_window = (0.05, 0.3)  # Time window for spatial patterns (50ms to 300ms)

# Toggle switch for Temporal Generalization Analysis
run_temporal_generalization = False

# Get the base data directory and convert it to a Path object
base_data_path = Path(get_data_path())

# Initialize arrays to store the decoding accuracy for each subject
all_subject_scores_target = np.zeros((n_subjects, len(times)))
all_subject_scores_distractor = np.zeros((n_subjects, len(times)))
all_subject_scores_control = np.zeros((n_subjects, len(times)))

# Initialize lists to store spatial patterns for each subject
all_subject_patterns_target = []
all_subject_patterns_distractor = []
all_subject_patterns_control = []

# Set up the decoders (using 5-fold cross-validation)
# Using Multinomial Logistic Regression natively for 9-class decoding
clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
time_decoder = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays for temporal generalization if toggled
if run_temporal_generalization:
    all_subject_tg_scores_target = np.zeros((n_subjects, len(times), len(times)))
    all_subject_tg_scores_distractor = np.zeros((n_subjects, len(times), len(times)))
    all_subject_tg_scores_control = np.zeros((n_subjects, len(times), len(times)))
    tg_decoder = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')

n_classes = 9

print(f"Starting within-task decoding (Selective Listening - Multinomial) for {n_subjects} subjects...")

# ==========================================
# 2. WITHIN-TASK DECODING
# ==========================================
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # --- Load Data (Active/Selective Listening) ---
    # Construct the file path using pathlib for cross-platform compatibility
    file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-spaceprime-epo.fif"
    
    epochs = mne.read_epochs(file_path, preload=True)
    epochs.crop(times[0], times[-1])
    
    # Target digits
    try:
        y_target = epochs.metadata['TargetDigit'].values
    except KeyError:
        print(f"Warning: 'TargetDigit' column not found in metadata for subject {sub}.")
        y_target = np.zeros(len(epochs))
    
    # Distractor digits
    try:
        y_distractor = epochs.metadata['SingletonDigit'].values
    except KeyError:
        print(f"Warning: 'SingletonDigit' column not found in metadata for subject {sub}.")
        y_distractor = np.zeros(len(epochs))
        
    # Control digits
    try:
        y_control = epochs.metadata['Non-Singleton2Digit'].values
    except KeyError:
        print(f"Warning: 'Non-Singleton2Digit' column not found in metadata for subject {sub}.")
        y_control = np.zeros(len(epochs))

    # --- Joint Mask for Simultaneous Trials ---
    # Only keep trials where target, distractor, and control digits (1-9) were presented
    valid_trials = (y_target >= 1) & (y_target <= 9) & (y_distractor >= 1) & (y_distractor <= 9) & (y_control >= 1) & (y_control <= 9)
    X_shared = epochs.get_data()[valid_trials]
    y_target_valid = y_target[valid_trials]
    y_distractor_valid = y_distractor[valid_trials]
    y_control_valid = y_control[valid_trials]
    
    if len(y_target_valid) > 0:
        print(f"  Training and testing on {len(y_target_valid)} simultaneous trials...")
        
        # Diagnostic: Check distributions
        unique_t, counts_t = np.unique(y_target_valid, return_counts=True)
        unique_d, counts_d = np.unique(y_distractor_valid, return_counts=True)
        unique_c, counts_c = np.unique(y_control_valid, return_counts=True)
        print(f"    Target distributions: {dict(zip(unique_t, counts_t))}")
        print(f"    Distractor distributions: {dict(zip(unique_d, counts_d))}")
        print(f"    Control distributions: {dict(zip(unique_c, counts_c))}")

        # --- Target Decoding ---
        sub_scores_target = cross_val_multiscore(time_decoder, X_shared, y_target_valid, cv=cv, n_jobs=-1)
        all_subject_scores_target[i, :] = np.mean(sub_scores_target, axis=0)

        # Target Temporal Generalization
        if run_temporal_generalization:
            tg_scores_target = cross_val_multiscore(tg_decoder, X_shared, y_target_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_target[i, :, :] = np.mean(tg_scores_target, axis=0)

        # Extract Spatial Patterns for Targets
        t_idx = epochs.time_as_index(peak_window)
        X_shared_windowed = X_shared[:, :, t_idx[0]:t_idx[1]].mean(axis=2)

        pattern_clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)))
        pattern_clf.fit(X_shared_windowed, y_target_valid)
        patterns_t = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
        mean_pattern_t = np.mean(np.abs(patterns_t), axis=0)
        all_subject_patterns_target.append(mean_pattern_t)

        # --- Distractor Decoding ---
        # Notice we reuse X_shared, but map it to y_distractor_valid
        sub_scores_distractor = cross_val_multiscore(time_decoder, X_shared, y_distractor_valid, cv=cv, n_jobs=-1)
        all_subject_scores_distractor[i, :] = np.mean(sub_scores_distractor, axis=0)

        # Distractor Temporal Generalization
        if run_temporal_generalization:
            tg_scores_distractor = cross_val_multiscore(tg_decoder, X_shared, y_distractor_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_distractor[i, :, :] = np.mean(tg_scores_distractor, axis=0)

        # Extract Spatial Patterns for Distractors
        # Re-use X_shared_windowed
        pattern_clf.fit(X_shared_windowed, y_distractor_valid)
        patterns_d = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
        mean_pattern_d = np.mean(np.abs(patterns_d), axis=0)
        all_subject_patterns_distractor.append(mean_pattern_d)
        
        # --- Control Decoding ---
        sub_scores_control = cross_val_multiscore(time_decoder, X_shared, y_control_valid, cv=cv, n_jobs=-1)
        all_subject_scores_control[i, :] = np.mean(sub_scores_control, axis=0)

        # Control Temporal Generalization
        if run_temporal_generalization:
            tg_scores_control = cross_val_multiscore(tg_decoder, X_shared, y_control_valid, cv=cv, n_jobs=-1)
            all_subject_tg_scores_control[i, :, :] = np.mean(tg_scores_control, axis=0)

        # Extract Spatial Patterns for Control
        pattern_clf.fit(X_shared_windowed, y_control_valid)
        patterns_c = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
        mean_pattern_c = np.mean(np.abs(patterns_c), axis=0)
        all_subject_patterns_control.append(mean_pattern_c)
        
    else:
        print("  No valid simultaneous trials found.")
        all_subject_patterns_target.append(np.zeros(epochs.info['nchan']))
        all_subject_patterns_distractor.append(np.zeros(epochs.info['nchan']))
        all_subject_patterns_control.append(np.zeros(epochs.info['nchan']))

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
                ax.axvspan(sig_times[0], sig_times[-1], color='red', alpha=0.3, label=label)
                label_added = True

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Plot Targets
plot_results(axes[0], mean_target, se_target, cl_target, p_target, 
             f'Decoding Targets (N={n_subjects})', 'blue')
axes[0].set_ylabel('Accuracy')

# Plot Distractors
plot_results(axes[1], mean_distractor, se_distractor, cl_distractor, p_distractor, 
             f'Decoding Distractors (N={n_subjects})', 'green')

# Plot Control
plot_results(axes[2], mean_control, se_control, cl_control, p_control, 
             f'Decoding Control (N={n_subjects})', 'orange')

plt.tight_layout()
plt.show()

# ==========================================
# 5. PLOT GROUP-LEVEL SPATIAL PATTERNS
# ==========================================
group_pattern_target = np.mean(all_subject_patterns_target, axis=0)
group_pattern_distractor = np.mean(all_subject_patterns_distractor, axis=0)
group_pattern_control = np.mean(all_subject_patterns_control, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Targets Pattern
im1, _ = mne.viz.plot_topomap(group_pattern_target, epochs_info, axes=axes[0], show=False, cmap='Reds', contours=4, vlim=(0, np.max(group_pattern_target)))
axes[0].set_title(f'Spatial Patterns: Targets\n({peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')

# Plot Distractors Pattern
im2, _ = mne.viz.plot_topomap(group_pattern_distractor, epochs_info, axes=axes[1], show=False, cmap='Greens', contours=4, vlim=(0, np.max(group_pattern_distractor)))
axes[1].set_title(f'Spatial Patterns: Distractors\n({peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')

# Plot Control Pattern
im3, _ = mne.viz.plot_topomap(group_pattern_control, epochs_info, axes=axes[2], show=False, cmap='Oranges', contours=4, vlim=(0, np.max(group_pattern_control)))
axes[2].set_title(f'Spatial Patterns: Control\n({peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')

# Add colorbars
plt.colorbar(im1, ax=axes[0], label='Pattern Activation (A.U.)')
plt.colorbar(im2, ax=axes[1], label='Pattern Activation (A.U.)')
plt.colorbar(im3, ax=axes[2], label='Pattern Activation (A.U.)')

plt.tight_layout()
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

    im1 = plot_tg(axes[0], mean_tg_target, sig_mask_tg_target, f'TG: Targets\n(Selective, N={n_subjects})')
    axes[0].set_ylabel('Training Time (s)')
    im2 = plot_tg(axes[1], mean_tg_distractor, sig_mask_tg_distractor, f'TG: Distractors\n(Selective, N={n_subjects})')
    im3 = plot_tg(axes[2], mean_tg_control, sig_mask_tg_control, f'TG: Control\n(Selective, N={n_subjects})')
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), label='Accuracy')
    plt.show()

# ==========================================
# 7. COMPARE TARGET AND DISTRACTOR DECODING
# ==========================================
print("\nComparing Target vs Distractor decoding...")

# Contrast: Targets - Distractors
diff_scores = all_subject_scores_target - all_subject_scores_distractor
mean_diff = np.mean(diff_scores, axis=0)
se_diff = np.std(diff_scores, axis=0) / np.sqrt(n_subjects)

# Run permutation test on the difference (testing against a mean of 0)
diff_T_obs, diff_clusters, diff_cluster_p_values, diff_H0 = permutation_cluster_1samp_test(
    diff_scores, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(times, mean_diff, label='Target - Distractor', color='purple', linewidth=2)
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
            plt.axvspan(sig_times[0], sig_times[-1], color='red', alpha=0.3, label=label)
            label_added = True

plt.title(f'Difference in Decoding Accuracy: Target vs Distractor\n(Group Level, N={n_subjects})')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy Difference (Target - Distractor)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
