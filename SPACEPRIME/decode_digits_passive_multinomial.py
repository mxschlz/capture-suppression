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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from SPACEPRIME.subjects import subject_ids as subjects
from SPACEPRIME import get_data_path

import seaborn as sns
sns.set_theme(context="talk", style="ticks")

# ==========================================
# SWITCHES
# ==========================================
RUN_TEMPORAL_GENERALIZATION = False  # Set to True to run the temporal generalization analysis

# ==========================================
# 1. LOAD YOUR DATA
# ==========================================
# We assume your epochs contain events coded 1 through 9 for the digits.

n_subjects = len(subjects)
times = np.linspace(-0.1, 0.6, 176)

# Get the base data directory and convert it to a Path object
base_data_path = Path(get_data_path())

# Initialize an array to store the decoding accuracy for each subject
# Shape: (n_subjects, n_timepoints)
all_subject_scores = np.zeros((n_subjects, len(times)))

# Initialize for confusion matrix analysis
n_classes = 9
peak_window = (0.05, 0.3)  # Time window for confusion matrix (100ms to 400ms)
all_subject_conf_mx = np.zeros((n_subjects, n_classes, n_classes))
all_subject_patterns = []  # List to store spatial patterns for each subject

# Initialize for temporal generalization
if RUN_TEMPORAL_GENERALIZATION:
    all_subject_tg_scores = np.zeros((n_subjects, len(times), len(times)))

# Set up the decoder using Multinomial Logistic Regression
clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
time_decoder = SlidingEstimator(clf, n_jobs=5, scoring='accuracy')  # n_jobs=1 inside loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Using 5 splits for robust estimation
if RUN_TEMPORAL_GENERALIZATION:
    tg_decoder = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')

print(f"Starting within-subject decoding (Passive - Multinomial) for {n_subjects} subjects...")

# Loop through each subject independently
for i, sub in enumerate(subjects):
    print(f"Processing Subject: {sub} ({i + 1}/{n_subjects})")

    # Isolate this subject's data
    file_path = base_data_path / "derivatives" / "epoching" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-passive-epo.fif"
    sub_epochs = mne.read_epochs(file_path, preload=True)

    X_sub_full = sub_epochs.get_data()
    y_sub_original = sub_epochs.events[:, 2]  # The original trigger codes

    # Initialize a new target array with zeros
    y_sub = np.zeros_like(y_sub_original)

    # Group spatial events into single digit classes (1-9) by inspecting event_id names
    for event_name, event_code in sub_epochs.event_id.items():
        for digit in range(1, n_classes+1):
            # Check if the exact string 'number-X' is in the event name (e.g., 'number-1', 'number-1_left')
            if f"number-{digit}" in event_name:
                y_sub[y_sub_original == event_code] = digit
                break

    # Filter out any epochs that didn't match digits 1-9 (e.g., non-target events)
    valid_trials = y_sub > 0
    X_sub = X_sub_full[valid_trials]
    y_sub = y_sub[valid_trials]

    # --- 1. Time-resolved decoding ---
    # Use n_jobs=-1 here to use all CPU cores for this subject's timepoints
    sub_scores = cross_val_multiscore(time_decoder, X_sub, y_sub, cv=cv, n_jobs=-1)

    # Average across the 5 folds for this subject and store
    all_subject_scores[i, :] = np.mean(sub_scores, axis=0)

    # --- 2. Confusion matrix at peak decoding time ---
    # Average data across the time dimension in our peak window
    t_idx = sub_epochs.time_as_index(peak_window)
    X_sub_windowed = X_sub[:, :, t_idx[0]:t_idx[1]].mean(axis=2)

    # Get predictions for each trial using cross-validation
    y_pred = cross_val_predict(clf, X_sub_windowed, y_sub, cv=cv, n_jobs=-1)

    # Compute and store the confusion matrix for this subject
    labels = np.arange(1, n_classes + 1)
    conf_mx = confusion_matrix(y_sub, y_pred, labels=labels)
    all_subject_conf_mx[i, :, :] = conf_mx

    # --- 3. Extract Spatial Patterns at Peak Window ---
    # We use LinearModel to automatically compute Haufe et al. (2014) patterns
    pattern_clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)))
    pattern_clf.fit(X_sub_windowed, y_sub)
    
    # Extract the spatial patterns
    patterns = get_coef(pattern_clf, 'patterns_', inverse_transform=True)
    
    # Take the absolute mean across all classes to get an overall "importance map"
    mean_pattern = np.mean(np.abs(patterns), axis=0)
    all_subject_patterns.append(mean_pattern)
    epochs_info = sub_epochs.info  # Save the info object for plotting the topomap later

    # --- 4. Temporal Generalization ---
    if RUN_TEMPORAL_GENERALIZATION:
        print(f"  Running Temporal Generalization for subject {sub}...")
        tg_scores = cross_val_multiscore(tg_decoder, X_sub, y_sub, cv=cv, n_jobs=-1)
        all_subject_tg_scores[i, :, :] = np.mean(tg_scores, axis=0)

# ==========================================
# GROUP-LEVEL AVERAGING & PLOTTING
# ==========================================
group_mean_scores = np.mean(all_subject_scores, axis=0)
group_std_error = np.std(all_subject_scores, axis=0) / np.sqrt(n_subjects)

# ==========================================
# STATISTICAL TESTING (Permutation Cluster)
# ==========================================
chance_level = 1.0 / 9.0
X_test = all_subject_scores - chance_level

# Define initial threshold for forming clusters (2-tailed t-test, p < 0.05)
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subjects - 1)

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    X_test, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(times, group_mean_scores, label='Group Mean Accuracy', color='blue', linewidth=2)
plt.fill_between(times, group_mean_scores - group_std_error, group_mean_scores + group_std_error,
                 color='blue', alpha=0.2, label='SEM')

plt.axhline(chance_level, color='k', linestyle='--', label='Chance Level (11.1%)')
plt.axvline(0, color='k', linestyle='-')

label_added = False
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        sig_times = times[c]
        if len(sig_times) > 0:
            label = 'p < 0.05 (Cluster)' if not label_added else ""
            plt.axvspan(sig_times[0], sig_times[-1], color='red', alpha=0.3, label=label)
            label_added = True

plt.title(f'Time-Resolved Decoding of Spoken Digits (Passive - Multinomial, N={n_subjects})')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# PLOT GROUP-LEVEL CONFUSION MATRIX
# ==========================================
group_conf_mx = np.mean(all_subject_conf_mx, axis=0)
with np.errstate(divide='ignore', invalid='ignore'):
    group_conf_mx_normalized = group_conf_mx / group_conf_mx.sum(axis=1, keepdims=True)
    group_conf_mx_normalized = np.nan_to_num(group_conf_mx_normalized)

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=group_conf_mx_normalized,
                              display_labels=np.arange(1, n_classes + 1))
disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True)
ax.set_title(f'Group-Level Confusion Matrix (Normalized)\n'
             f'Time Window: {peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms (N={n_subjects})')
plt.tight_layout()
plt.show()

# ==========================================
# PLOT GROUP-LEVEL SPATIAL PATTERNS
# ==========================================
group_pattern = np.mean(all_subject_patterns, axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
im, _ = mne.viz.plot_topomap(group_pattern, epochs_info, axes=ax, show=False, cmap='Reds', contours=4, vlim=(0, np.max(group_pattern)))

plt.colorbar(im, ax=ax, label='Pattern Activation (A.U.)')
ax.set_title(f'Group-Level Spatial Patterns\n(Peak Window: {peak_window[0]*1000:.0f}-{peak_window[1]*1000:.0f} ms)')
plt.tight_layout()
plt.show()

# ==========================================
# PLOT GROUP-LEVEL TEMPORAL GENERALIZATION
# ==========================================
if RUN_TEMPORAL_GENERALIZATION:
    group_tg_scores = np.mean(all_subject_tg_scores, axis=0)

    print("Running cluster permutation test for Temporal Generalization...")
    X_test_tg = all_subject_tg_scores - chance_level

    T_obs_tg, clusters_tg, cluster_p_values_tg, H0_tg = permutation_cluster_1samp_test(
        X_test_tg, n_permutations=1000, threshold=t_thresh, tail=0, out_type='mask', n_jobs=-1)

    sig_mask_tg = np.zeros(group_tg_scores.shape, dtype=bool)
    for c, p_val in zip(clusters_tg, cluster_p_values_tg):
        if p_val <= 0.05:
            sig_mask_tg |= c

    fig, ax = plt.subplots(figsize=(7, 6))
    limit = np.max(np.abs(group_tg_scores - chance_level))
    vmin = chance_level - limit
    vmax = chance_level + limit

    im = ax.imshow(group_tg_scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                   extent=[times[0], times[-1], times[0], times[-1]], vmin=vmin, vmax=vmax)

    if sig_mask_tg.any():
        ax.contour(times, times, sig_mask_tg, levels=[0.5], colors='black', linewidths=1.5, linestyles='-')

    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'Temporal Generalization Matrix\n(Group Level, N={n_subjects})')

    ax.plot([times[0], times[-1]], [times[0], times[-1]], 'k--', label='Diagonal (Train = Test)')
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)

    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()