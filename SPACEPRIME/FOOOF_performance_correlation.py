from SPACEPRIME import load_concatenated_epochs
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.signal import welch
import fooof
import matplotlib.pyplot as plt
import seaborn as sns
import os  # <-- Import the os module for handling file paths

plt.ion()

EPOCH_TMIN, EPOCH_TMAX = 0, None  # pre-stim
epochs = load_concatenated_epochs("spaceprime").crop(EPOCH_TMIN, EPOCH_TMAX)

# --- Analysis starts here ---

# 1. Define parameters for PSD computation and FOOOF fitting
FMIN_PSD, FMAX_PSD = 1, 40  # Frequency range for PSD computation
FMIN_FOOOF, FMAX_FOOOF = 1, 40  # Frequency range for FOOOF fitting

# IMPORTANT: Replace these with the actual column names in your epochs.metadata
ACCURACY_COLUMN = 'select_target'
SUBJECT_ID_COLUMN = 'subject_id'

# --- NEW: Configuration for Debugging Plots ---
PLOT_SUBJECT_FITS = False  # Set to True to save a plot of the FOOOF fit for each subject
PLOTS_OUTPUT_DIR = "fooof_subject_fits"  # Directory to save the plots

if PLOT_SUBJECT_FITS:
    if not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
    print(f"\nSubject FOOOF fit plots will be saved to '{PLOTS_OUTPUT_DIR}/'")
# --- END NEW SECTION ---

if ACCURACY_COLUMN not in epochs.metadata.columns:
    raise ValueError(f"Accuracy column '{ACCURACY_COLUMN}' not found in epochs metadata. "
                     "Please check your metadata and update ACCURACY_COLUMN accordingly.")
if SUBJECT_ID_COLUMN not in epochs.metadata.columns:
    raise ValueError(f"Subject ID column '{SUBJECT_ID_COLUMN}' not found in epochs metadata. "
                     "Please check your metadata and update SUBJECT_ID_COLUMN accordingly.")

# Get unique subject IDs
unique_subject_ids = epochs.metadata[SUBJECT_ID_COLUMN].unique()[1:]
print(f"\nFound {len(unique_subject_ids)} unique subjects.")

# 2. Compute PSDs for all epochs once using SciPy's Welch method
print(f"\nComputing PSDs for all {len(epochs.metadata)} epochs using scipy.signal.welch...")

# Get the EEG data as a NumPy array (n_epochs, n_channels, n_samples)
eeg_data = epochs.get_data(picks='eeg')
n_epochs, n_channels, n_samples = eeg_data.shape
sfreq = epochs.info['sfreq']

# Define parameters for Welch's method, mirroring the original MNE approach.
n_per_seg = n_samples
n_fft = int(2 ** np.ceil(np.log2(n_per_seg)))

# To compute PSDs efficiently, reshape data to 2D (n_trials*n_channels, n_samples)
eeg_data_reshaped = eeg_data.reshape(n_epochs * n_channels, n_samples)

# Compute PSDs for all channels and epochs at once.
full_freqs, psds_reshaped = welch(
    eeg_data_reshaped,
    fs=sfreq,
    nperseg=n_per_seg,
    nfft=n_fft,
    axis=-1
)

# Reshape PSDs back to the original 3D structure: (n_epochs, n_channels, n_frequencies)
psds_full_spectrum = psds_reshaped.reshape(n_epochs, n_channels, -1)

# Manually filter the frequencies and PSD data to the desired range [FMIN_PSD, FMAX_PSD]
freq_mask = (full_freqs >= FMIN_PSD) & (full_freqs <= FMAX_PSD)
all_freqs = full_freqs[freq_mask]
all_psds_data = psds_full_spectrum[:, :, freq_mask]

print("All PSDs computed.")

# 3. Initialize FOOOF model
fm = fooof.FOOOF(aperiodic_mode='knee', max_n_peaks=6, verbose=False)

# Lists to store subject-level results
subject_fooof_exponents = []
subject_avg_accuracies = []
processed_subject_ids = []  # To keep track of which subjects were processed

# 4. Iterate through each subject to get subject-level FOOOF and average accuracy
print("\nFitting FOOOF model and calculating average accuracy for each subject...")
for subject_id in unique_subject_ids:
    print(f"  Processing subject: {subject_id}")

    subject_mask = epochs.metadata[SUBJECT_ID_COLUMN] == subject_id
    subject_psds_data = all_psds_data[subject_mask.values, :, :]
    avg_psd_subject = np.mean(subject_psds_data, axis=(0, 1))

    # Fit FOOOF model to the subject's averaged PSD
    fm.fit(all_freqs, avg_psd_subject, freq_range=[FMIN_FOOOF, FMAX_FOOOF])

    # --- NEW: PLOTTING CODE ---
    if PLOT_SUBJECT_FITS:
        # Generate and save a plot of the fit for the current subject
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use the FOOOF object's plot method. plt_log=True is best for spectra.
        fm.plot(ax=ax, plt_log=False)

        # Add an informative title with key parameters
        exponent_val = fm.get_params('aperiodic', 'exponent')
        r_squared_val = fm.get_params('r_squared')
        ax.set_title(f"FOOOF Fit for Subject: {subject_id}\n"
                     f"Exponent: {exponent_val:.3f}, R-squared: {r_squared_val:.3f}",
                     fontsize=14)

        # Save the figure to the specified output directory
        plot_filename = os.path.join(PLOTS_OUTPUT_DIR, f"fooof_fit_subject_{subject_id}.png")
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.show(block=True)
    # --- END NEW PLOTTING CODE ---

    # Extract the aperiodic exponent
    exponent = fm.get_params('aperiodic', 'exponent')
    subject_fooof_exponents.append(exponent)

    # Calculate the average accuracy for the current subject
    subject_accuracies = epochs.metadata.loc[subject_mask, ACCURACY_COLUMN]
    avg_accuracy = subject_accuracies.mean()
    subject_avg_accuracies.append(avg_accuracy)
    processed_subject_ids.append(subject_id)

print("Subject-level FOOOF fitting and accuracy calculation complete.")

# ... (The rest of the script for correlation and plotting remains the same) ...
# 5. Correlate subject-level FOOOF exponent with average accuracy
print("\nPerforming correlation between subject-level FOOOF exponent and average accuracy...")

# Pearson correlation measures linear relationship
pearson_r, pearson_p = pearsonr(subject_fooof_exponents, subject_avg_accuracies)
print(f"Pearson Correlation Coefficient (r): {pearson_r:.4f}")
print(f"P-value (Pearson): {pearson_p:.4f}")

# Spearman correlation measures monotonic relationship (less sensitive to outliers)
spearman_r, spearman_p = spearmanr(subject_fooof_exponents, subject_avg_accuracies)
print(f"Spearman Correlation Coefficient (rho): {spearman_r:.4f}")
print(f"P-value (Spearman): {spearman_p:.4f}")

# 6. Interpret the results
print("\n--- Interpretation (Between-Subject) ---")
# You hypothesized: higher FOOOF exponent (less noisy) -> higher accuracy (positive correlation)

if pearson_p < 0.05:
    print(
        f"There is a statistically significant linear correlation (p < 0.05) between subject-level FOOOF exponent and average accuracy.")
    if pearson_r > 0:
        print(
            f"A positive correlation (r = {pearson_r:.2f}) suggests that subjects with higher average FOOOF exponents tend to have higher average accuracy.")
        print(
            "This finding aligns with your hypothesis: a 'less noisy' neural system (steeper aperiodic slope) is linked to better performance at the subject level.")
    else:
        print(
            f"A negative correlation (r = {pearson_r:.2f}) suggests that subjects with higher average FOOOF exponents tend to have higher average accuracy.")
        print("This finding is contrary to your hypothesis.")
else:
    print(
        f"There is no statistically significant linear correlation (p >= 0.05) between subject-level FOOOF exponent and average accuracy.")
    print("This means we cannot conclude a linear relationship based on this analysis at the subject level.")

# Consider Spearman if the relationship might be monotonic but not strictly linear
if spearman_p < 0.05 and pearson_p >= 0.05:
    print(
        f"\nHowever, a significant Spearman correlation (rho = {spearman_r:.2f}, p = {spearman_p:.4f}) suggests a monotonic relationship, "
        "even if not strictly linear. This might indicate a trend that isn't captured by Pearson's r.")

# 7. Visualization
plt.figure(figsize=(9, 7))
sns.regplot(x=subject_fooof_exponents, y=subject_avg_accuracies, scatter_kws={'alpha': 0.7, 's': 50},
            line_kws={'color': 'red', 'lw': 2})
plt.title(
    f'Subject-Level FOOOF Exponent vs. Average Accuracy\nPearson r={pearson_r:.2f}, p={pearson_p:.3f} | Spearman rho={spearman_r:.2f}, p={spearman_p:.3f}')
plt.xlabel('Average FOOOF Aperiodic Exponent (per Subject)')
plt.ylabel(f'Average Accuracy (per Subject)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
