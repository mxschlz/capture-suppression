import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
plt.ion()


# In this analysis, we want to see whether the brain synchronizes with the amplitude modulation of our target stimulus.
# In theory, neural entrainment should allow us to see a 30 Hz peak in the power spectrum. Additionally, we want to
# estimate how this frequency tagging is altered in our respective priming conditions (i.e., no priming, positive priming,
# negative priming).
# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs. Important consideration, only select the interval of stimuluis presentation (0 - 0.25 seconds).
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids]).crop(0, 0.5)
# For maximum target entrainment, we want to exclusively look at epochs where the target has been identified correctly
# and where a salient distractor is absent.
# get the sampling frequency from epochs info
sfreq = epochs.info["sfreq"]
epochs = epochs["select_target==True"]
# epochs.apply_baseline((None, 0))
all_conds = list(epochs.event_id.keys())
# Separate epochs based on target location. Pick distractor absent trials only (Dont know whether that makes sense).
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-0" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-0" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get left and right electrodes
#left_channels = mne.channels.make_1020_channel_selections(epochs.info, midline='z', return_ch_names=True)['Left']
#right_channels = mne.channels.make_1020_channel_selections(epochs.info, midline='z', return_ch_names=True)['Right']
# get the trial-wise data for targets
contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().pick("C4").get_data(),
                                 right_target_epochs.copy().pick("C3").get_data()], axis=1), axis=1)
ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().pick("C3").get_data(),
                               right_target_epochs.copy().pick("C4").get_data()], axis=1), axis=1)
# compute power density spectrum for evoked response and look for frequency tagging
# first, create epochs objects from numpy arrays computed above for ipsi and contra targets
ipsi_target_epochs = mne.EpochsArray(data=ipsi_target_epochs_data.reshape(len(left_target_epochs), 1, left_target_epochs.get_data().shape[2]),
                                     info=mne.create_info(ch_names=["Ipsi Target"], sfreq=sfreq, ch_types="eeg"))
contra_target_epochs = mne.EpochsArray(data=contra_target_epochs_data.reshape(len(left_target_epochs), 1, left_target_epochs.get_data().shape[2]),
                                     info=mne.create_info(ch_names=["Contra Target"], sfreq=sfreq, ch_types="eeg"))
# now, compute the power spectra for both ipsi and contra targets
target_diff = mne.EpochsArray(data=(contra_target_epochs.get_data() - ipsi_target_epochs.get_data()),
                              info=mne.create_info(ch_names=["Target: Contra - Ipsi"], sfreq=sfreq, ch_types="eeg"))
# subtract the ipsi from the contralateral target power
powerdiff_target = target_diff.compute_psd(method="welch", fmin=25, fmax=35, n_jobs=-1)
powerdiff_target.plot()

# a Different approach on the ERP instead of epochs
erp_ipsi = ipsi_target_epochs_data.mean(axis=0)
erp_contra = contra_target_epochs_data.mean(axis=0)
erp_diff = mne.EvokedArray(data=(erp_contra - erp_ipsi).reshape(1, epochs.times.__len__()),
                           info=mne.create_info(ch_names=["Target: Contra - Ipsi"], sfreq=sfreq, ch_types="eeg"))
erp_diff_spec = erp_diff.compute_psd(method="multitaper")
erp_diff_spec.plot()
one_over_f_fit = len(erp_diff_spec.get_data()[0])/erp_diff_spec.get_data()[0]

# Now, follow an MNE example
tmin = epochs.tmin
tmax = epochs.tmax
fmin = 1.0
fmax = 40
spectrum = epochs.compute_psd()
psds, freqs = spectrum.get_data(return_freqs=True)

# Define convolution function
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(int(fmin), int(fmax))

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    ylim=[-2, 30],
    xlim=[fmin, fmax],
)


import numpy as np
import matplotlib.pyplot as plt

#Example power spectrum
freqs = np.arange(1, 101)
power = 100 / freqs + np.random.normal(0, 5, 100) #1/f + noise

#Conceptual 1/f fit (replace with a proper fit)
one_over_f_fit = 100 / freqs

corrected_power = power - one_over_f_fit

plt.figure()
plt.plot(freqs, power, label = "original power")
plt.plot(freqs, one_over_f_fit, label = "1/f fit")
plt.plot(freqs, corrected_power, label = "corrected power")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.show()
