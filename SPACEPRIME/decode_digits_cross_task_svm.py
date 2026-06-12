import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import SlidingEstimator, LinearModel, get_coef
import mne
from mne.stats import permutation_cluster_1samp_test
import scipy.stats
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from SPACEPRIME.subjects import subject_ids as subjects
from SPACEPRIME import get_data_path

import seaborn as sns
sns.set_theme(context="talk", style="ticks")

# ==========================================
# 1. SETUP AND INITIALIZATION
# ==========================================
n_subjects = len(subjects)
times = np.linspace(-0.1, 0.6, 176)  # Assumes the same epoch times (-100 to +600 ms)
peak_window = (0.05, 0.3)  # Time window for spatial patterns (50ms to 300ms)

# Get the base data directory and convert it to a Path object
base_data_path = Path(get_data_path())

# Initialize arrays to store the decoding accuracy for each subject
# Shape: (n_subjects, n_timepoints)
all_subject_scores_target = np.zeros((n_subjects, len(times)))
all_subject_scores_distractor = np.zeros((n_subjects, len(times)))
all_subject_patterns = []  # List to store spatial patterns for each subject

n_classes = 9

# Set up the decoders
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovo'))
time_decoder_target = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy')

print(f"Starting cross-task decoding for {n_subjects} subjects...")

# ==========================================
# 2. CROSS-TASK DECODING
# ==========================================
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # --- A. Load Training Data (Passive Listening) ---
    train_file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-passive-epo.fif"
    epochs_train = mne.read_epochs(train_file_path, preload=True)
    epochs_train.set_montage("easycap-M1")
    
    X_train_full = epochs_train.get_data()
    y_train_original = epochs_train.events[:, 2]

    # Initialize a new target array for digits 1-9
    y_train = np.zeros_like(y_train_original)
    for event_name, event_code in epochs_train.event_id.items():
        for digit in range(1, n_classes + 1):
            if f"number-{digit}" in event_name:
                y_train[y_train_original == event_code] = digit
                break

    valid_train = y_train > 0
    X_train = X_train_full[valid_train]
    y_train = y_train[valid_train]

    # --- B. Load Testing Data (Active/Selective Listening) ---
    test_file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-spaceprime-epo.fif"
    epochs_test = mne.read_epochs(test_file_path, preload=True)
    epochs_test.set_montage("easycap-M1")
    
    # Crop the test epochs to perfectly match the training epochs' time window
    epochs_test.crop(tmin=epochs_train.times[0], tmax=epochs_train.times[-1])

    # Target digits
    try:
        y_test_target = epochs_test.metadata['TargetDigit'].values
    except KeyError:
        print(f"Warning: 'TargetDigit' column not found in metadata for subject {sub}.")
        y_test_target = np.zeros(len(epochs_test))
    
    # Distractor digits
    try:
        y_test_distractor = epochs_test.metadata['SingletonDigit'].values
    except KeyError:
        print(f"Warning: 'SingletonDigit' column not found in metadata for subject {sub}.")
        y_test_distractor = np.zeros(len(epochs_test))

    # --- Shared Test Data Mask ---
    # Ensure we only test on trials where BOTH a target and a distractor are present
    valid_test_shared = (y_test_target >= 1) & (y_test_target <= 9) & \
                        (y_test_distractor >= 1) & (y_test_distractor <= 9)
    
    X_test_shared = epochs_test.get_data()[valid_test_shared]
    y_test_target_valid = y_test_target[valid_test_shared]
    y_test_distractor_valid = y_test_distractor[valid_test_shared]

    # --- C. Time-resolved Decoding ---
    print(f"  Training on {len(y_train)} passive trials...")
    time_decoder_target.fit(X_train, y_train) 
    
    if len(y_test_target_valid) > 0:
        print(f"  Testing on {len(y_test_target_valid)} simultaneous active trials...")
        sub_scores_target = time_decoder_target.score(X_test_shared, y_test_target_valid)
        all_subject_scores_target[i, :] = sub_scores_target
        
        sub_scores_distractor = time_decoder_target.score(X_test_shared, y_test_distractor_valid)
        all_subject_scores_distractor[i, :] = sub_scores_distractor
    else:
        print("  No valid simultaneous testing trials found.")

    # --- D. Extract Spatial Patterns (Training Set) ---
    t_idx = epochs_train.time_as_index(peak_window)
    X_train_windowed = X_train[:, :, t_idx[0]:t_idx[1]].mean(axis=2)

    pattern_clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel='linear', decision_function_shape='ovo')))
    pattern_clf.fit(X_train_windowed, y_train)
    
    patterns = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
    mean_pattern = np.mean(np.abs(patterns), axis=0)
    all_subject_patterns.append(mean_pattern)
    epochs_info = epochs_train.info  # Keep for plotting

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
                ax.axvspan(sig_times[0], sig_times[-1], color='grey', alpha=0.3, label=label)
                label_added = True

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Plot Targets
plot_results(axes[0], mean_target, se_target, cl_target, p_target, 
             f'Decoding Targets (Train: Passive → Test: Selective Main, N={n_subjects})', 'green')
axes[0].set_ylabel('Accuracy')

# Plot Distractors
plot_results(axes[1], mean_distractor, se_distractor, cl_distractor, p_distractor, 
             f'Decoding Distractors (Train: Passive → Test: Selective Main, N={n_subjects})', 'red')

plt.tight_layout()
plt.show()

# ==========================================
# 5. PLOT GROUP-LEVEL SPATIAL PATTERNS
# ==========================================
group_pattern = np.mean(all_subject_patterns, axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax, show=False, cmap='Reds', contours=4, vlim=(0, np.max(group_pattern)))

plt.colorbar(im, ax=ax, label='Pattern Activation (A.U.)')
ax.set_title(f'Group-Level Spatial Patterns (Training Set)\n(Peak Window: {peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')
plt.tight_layout()
plt.show()
