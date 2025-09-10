import mne
import numpy as np
import matplotlib.pyplot as plt
import SPACEPRIME

plt.ion()


# --- 1. Load Data and Define Parameters ---
# Load the full, uncropped epochs to define our own windows
all_epochs = SPACEPRIME.load_concatenated_epochs() # Original duration: -1s to +1s

subject_ids = all_epochs.metadata['subject_id'].unique()[1:]  # first value is NaN

# Analysis Parameters
sfreq = all_epochs.info["sfreq"]
fmin = 25
fmax = 35
# The new window length is 1 second
n_samples_window = int(1.0 * sfreq) + 1 # +1 to be safe with rounding

# --- 2. Loop Through Subjects and Compute Individual Spectra ---
# Lists to store spectra for stimulus and baseline periods
contra_spectra = []
ipsi_spectra = []

print("\nStarting subject-level processing...")
for subject in subject_ids:
    print(f"  Processing Subject: {subject}")
    epochs = all_epochs[f"subject_id == {subject}"]

    left_target_epochs = epochs[[x for x in epochs.event_id.keys() if "Target-1" in x]]
    right_target_epochs = epochs[[x for x in epochs.event_id.keys() if "Target-3" in x]]

    # Create evoked responses (these will span the full -1s to +1s)
    left_target_evoked = left_target_epochs.average()
    right_target_evoked = right_target_epochs.average()

    # Create the full contra-lateral evoked response
    contra_data = np.mean([
        left_target_evoked.copy().pick('C4').get_data(),
        right_target_evoked.copy().pick('C3').get_data()
    ], axis=0)
    contra_evoked_full = mne.EvokedArray(
        data=contra_data,
        info=mne.create_info(ch_names=["Contra"], sfreq=sfreq, ch_types="eeg"),
        tmin=all_epochs.tmin
    )
    # Create the full ipsi-lateral evoked response
    ipsi_data = np.mean([
        left_target_evoked.copy().pick('C3').get_data(),
        right_target_evoked.copy().pick('C4').get_data()
    ], axis=0)
    ipsi_evoked_full = mne.EvokedArray(
        data=ipsi_data,
        info=mne.create_info(ch_names=["Contra"], sfreq=sfreq, ch_types="eeg"),
        tmin=all_epochs.tmin
    )

    evoked_contra = contra_evoked_full.copy().crop(tmin=0.0, tmax=1.0)
    evoked_ipsi = ipsi_evoked_full.copy().crop(tmin=0.0, tmax=1.0)

    # Compute PSD for both windows
    # n_fft and n_per_seg now match the 1-second window length for 1 Hz resolution
    contra_spec = evoked_contra.compute_psd(
        method="welch", n_fft=n_samples_window, n_per_seg=n_samples_window,
        fmin=fmin, fmax=fmax, window="hann", verbose=False
    )
    ipsi_spec = evoked_ipsi.compute_psd(
        method="welch", n_fft=n_samples_window, n_per_seg=n_samples_window,
        fmin=fmin, fmax=fmax, window="hann", verbose=False
    )

    contra_spectra.append(contra_spec)
    ipsi_spectra.append(ipsi_spec)

# --- 3. Grand-Average the Spectra and Plot ---
freqs = contra_spectra[0].freqs
n_subjects = len(contra_spectra)

# --- Grand Average Calculation ---
def get_grand_average(spectra_list):
    """Helper function to compute mean and SEM from a list of Spectrum objects."""
    power_all = np.array([s.get_data(fmin=fmin, fmax=fmax) for s in spectra_list]).squeeze(axis=1)
    mean = np.mean(power_all, axis=0)
    sem = np.std(power_all, axis=0) / np.sqrt(n_subjects)
    return mean, sem

contra_mean, contra_sem = get_grand_average(contra_spectra)
ipsi_mean, ipsi_sem = get_grand_average(ipsi_spectra)

# --- Plotting the Grand Average: Stimulus vs. Baseline ---
fig, ax = plt.subplots()

# Plot Baseline response
ax.plot(freqs, ipsi_mean, label="Contralateral", color="crimson", linewidth=2)
ax.fill_between(freqs, ipsi_mean - ipsi_sem, ipsi_mean + ipsi_sem, color="crimson", alpha=0.2)

# Plot Stimulus response
ax.plot(freqs, contra_mean, label="Ipsilateral", color="navy", linewidth=2)
ax.fill_between(freqs, contra_mean - contra_sem, contra_mean + contra_sem, color="navy", alpha=0.2)

# Highlight the 30 Hz tagged frequency
ax.axvline(30, color="k", linestyle="--", alpha=0.7, label="30 Hz Tag")

ax.set(
    title=f"Grand Average Power: Contra vs. Ipsi (N={n_subjects})",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (V²/Hz)",
    xlim=[fmin, fmax]
)
ax.legend(loc="upper right")
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
