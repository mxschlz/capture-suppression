import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel, get_coef
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from SPACEPRIME.subjects import subject_ids as subjects
from SPACEPRIME import get_data_path

import seaborn as sns
sns.set_theme(context="talk", style="ticks")

# ==========================================
# 1. SETUP AND INITIALIZATION
# ==========================================
n_subjects = len(subjects)
times = np.linspace(-0.2, 0.8, 251)
peak_window = (0.05, 0.3)  # Time window for spatial patterns (50ms to 300ms)

# Initialize arrays to store the decoding accuracy for each subject
all_subject_scores_target = np.zeros((n_subjects, len(times)))
all_subject_scores_distractor = np.zeros((n_subjects, len(times)))

# Initialize lists to store spatial patterns for each subject
all_subject_patterns_target = []
all_subject_patterns_distractor = []

n_classes = 9

# Set up the decoders (using 5-fold cross-validation)
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovo'))
time_decoder = SlidingEstimator(clf, n_jobs=5, scoring='accuracy')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Starting within-task decoding (Selective Listening) for {n_subjects} subjects...")

# ==========================================
# 2. WITHIN-TASK DECODING
# ==========================================
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # --- Load Data (Active/Selective Listening) ---
    epochs = mne.read_epochs(f"{get_data_path()}\\derivatives\\epoching\\sub-{sub}\\eeg\\sub-{sub}_task-spaceprime-epo.fif", preload=True)
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

    # --- Target Decoding ---
    valid_target = (y_target >= 1) & (y_target <= 9)
    X_target = epochs.get_data()[valid_target]
    y_target_valid = y_target[valid_target]
    
    # Diagnostic: Check if target digits 1-9 are evenly distributed
    unique_t, counts_t = np.unique(y_target_valid, return_counts=True)
    print(f"    Target distributions: {dict(zip(unique_t, counts_t))}")

    if len(y_target_valid) > 0:
        print(f"  Training and testing on {len(y_target_valid)} targets...")
        sub_scores_target = cross_val_multiscore(time_decoder, X_target, y_target_valid, cv=cv, n_jobs=-1)
        all_subject_scores_target[i, :] = np.mean(sub_scores_target, axis=0)

        # Extract Spatial Patterns for Targets
        t_idx = epochs.time_as_index(peak_window)
        X_target_windowed = X_target[:, :, t_idx[0]:t_idx[1]].mean(axis=2)

        pattern_clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel='linear', decision_function_shape='ovo')))
        pattern_clf.fit(X_target_windowed, y_target_valid)
        patterns = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
        mean_pattern = np.mean(np.abs(patterns), axis=0)
        all_subject_patterns_target.append(mean_pattern)
    else:
        print("  No valid target trials found.")
        all_subject_patterns_target.append(np.zeros(epochs.info['nchan']))
        
    # --- Distractor Decoding ---
    valid_distractor = (y_distractor >= 1) & (y_distractor <= 9)
    X_distractor = epochs.get_data()[valid_distractor]
    y_distractor_valid = y_distractor[valid_distractor]

    # Diagnostic: Check if distractor digits 1-9 are evenly distributed
    unique_d, counts_d = np.unique(y_distractor_valid, return_counts=True)
    print(f"    Distractor distributions: {dict(zip(unique_d, counts_d))}")

    if len(y_distractor_valid) > 0:
        print(f"  Training and testing on {len(y_distractor_valid)} distractors...")
        sub_scores_distractor = cross_val_multiscore(time_decoder, X_distractor, y_distractor_valid, cv=cv, n_jobs=-1)
        all_subject_scores_distractor[i, :] = np.mean(sub_scores_distractor, axis=0)

        # Extract Spatial Patterns for Distractors
        t_idx = epochs.time_as_index(peak_window)
        X_distractor_windowed = X_distractor[:, :, t_idx[0]:t_idx[1]].mean(axis=2)

        pattern_clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel='linear', decision_function_shape='ovo')))
        pattern_clf.fit(X_distractor_windowed, y_distractor_valid)
        patterns = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
        mean_pattern = np.mean(np.abs(patterns), axis=0)
        all_subject_patterns_distractor.append(mean_pattern)
    else:
        print("  No valid distractor trials found.")
        all_subject_patterns_distractor.append(np.zeros(epochs.info['nchan']))

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

# ==========================================
# 4. PLOTTING TIME-RESOLVED DECODING
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

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
             f'Decoding Targets (Within Selective, N={n_subjects})', 'blue')
axes[0].set_ylabel('Accuracy')

# Plot Distractors
plot_results(axes[1], mean_distractor, se_distractor, cl_distractor, p_distractor, 
             f'Decoding Distractors (Within Selective, N={n_subjects})', 'green')

plt.tight_layout()
plt.show()

# ==========================================
# 5. PLOT GROUP-LEVEL SPATIAL PATTERNS
# ==========================================
group_pattern_target = np.mean(all_subject_patterns_target, axis=0)
group_pattern_distractor = np.mean(all_subject_patterns_distractor, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Targets Pattern
im1, _ = mne.viz.plot_topomap(group_pattern_target, epochs_info, axes=axes[0], show=False, cmap='Reds', contours=4, vlim=(0, np.max(group_pattern_target)))
axes[0].set_title(f'Spatial Patterns: Targets\n(Peak Window: {peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')

# Plot Distractors Pattern
im2, _ = mne.viz.plot_topomap(group_pattern_distractor, epochs_info, axes=axes[1], show=False, cmap='Greens', contours=4, vlim=(0, np.max(group_pattern_distractor)))
axes[1].set_title(f'Spatial Patterns: Distractors\n(Peak Window: {peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')

# Add colorbars
plt.colorbar(im1, ax=axes[0], label='Pattern Activation (A.U.)')
plt.colorbar(im2, ax=axes[1], label='Pattern Activation (A.U.)')

plt.tight_layout()
plt.show()
