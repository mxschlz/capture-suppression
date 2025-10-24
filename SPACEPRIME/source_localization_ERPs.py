import mne
import SPACEPRIME


# load up epochs
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")
sensors = epochs.info["dig"]

# compute noise covariance
noise_cov = mne.compute_covariance(
    epochs, tmin=-0.2, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
)
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)