import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from SPACEPRIME.plotting import plot_individual_lines
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()


# This script intends to replicate the method from Boncompte et al. (2016).
# --- ALPHA POWER PRE-STIMULUS ---
# We first retrieve our epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# Control for priming
epochs = epochs["Priming==0"]
# crop to reduce runtime
epochs.crop(-0.5, 0.5)
# epochs.resample(128)  # downsample from 250 to 128 to reduce RAM cost
# Get the sampling frequency because we need it later
sfreq = epochs.info["sfreq"]
# Now, we need to define some parameters for time-frequency analysis. This is pretty standard, we use morlet wavelet
# convolution (sine wave multiplied by a gaussian distribution to flatten the edges of the filter), we define a number
# of cycles of this wavelet that changes according to the frequency (smaller frequencies get smaller cycles, whereas
# larger frequencies have larger cycles, but all have a cycle of half the frequency value). We also set decim = 1 to
# keep the full amount of data
freqs = numpy.arange(5, 31, 1)  # 1 to 30 Hz
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 10  # keep only every fifth of the samples along the time axis
mode = "mean"  # normalization
n_jobs = -1  # number of parallel jobs. -1 uses all cores
average = False  # get total oscillatory power, opposed to evoked oscillatory power (get power from ERP)
# apply baseline to epochs
# epochs.apply_baseline(baseline=baseline)
# Compute time-frequency analysis
power_total = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, n_jobs=n_jobs, decim=decim)
# Furthermore, since we are interested in the induced alpha power only, we get the evoked alpha by averagring epochs and
# then conducting TFR analysis. We subtract the evoked power from the total power to get induced oscillatory power.
power_evoked = epochs.average().compute_tfr(method=method, freqs=freqs, decim=decim, n_cycles=n_cycles, n_jobs=n_jobs)
# apply baseline
#power_evoked.apply_baseline(baseline=baseline, mode=mode)
# Subtract the evoked power, trial by trial.
power_induced = power_total.copy()
for trial in range(len(power_total)):
    power_induced.data[trial] -= power_evoked.data[0]  # subtract the evoked power from total power
# apply baseline
#power_total.apply_baseline(baseline, mode=mode)
# Now, we devide the epochs into our respective priming conditions. In order to do so, we make use of the metadata
# which was appended to the epochs of every subject during preprocessing. We can access the metadata the same way as
# we would by calling the event_ids in the experiment.
power_corrects = power_induced["select_target==True"]
power_incorrects = power_induced["select_target==False"]
# Further, we balance the amount of trials by removing trials randomly from the larger portion.
mne.epochs.equalize_epoch_counts([power_incorrects, power_corrects], method="random")
# calculate difference spectrum
power_diff = power_corrects - power_incorrects
power_diff.average().plot(combine="mean")

# Prepare the data matrix for the permutation function. In this case, X must be a matrix of N observations x freqs x times
time_roi = (-0.4, 0.3)
X = power_diff.get_data(tmin=time_roi[0], tmax=time_roi[1])  # shape: trials x channels x freqs x times
# The difference in induced oscillatory power between correct and incorrect trials we observe in the data might be
# predictive of the performance of the participants. In order to find out where this power difference originates from
# (broadly), we can apply a 1-sample
# permutation cluster test. MNE provides a pipeline for this procedure, we just have to apply it to our own data.
# First, we retrieve the channel adjacency (n_channels x n_channels) matrix from the data.
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(power_diff.info, "eeg")
# We further need to combine this adjacency with our dimensions of interest. For spatio-spectro-temporal data, we need
# to add the spectro-temporal domain to the existing adjacency matrix.
tfr_adjacency = mne.stats.combine_adjacency(X.shape[2], X.shape[-1], sensor_adjacency)
# Define some statistic params
tail = 0  # two-sided
alpha = 0.05  # significane threshold
# set degrees of freedom to len(epochs) - 1
degrees_of_freedom = len(epochs) - 1
t_thresh = scipy.stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)  # t threshold for a two-sided alpha
tfce_thresh = dict(start=0, step=0.2)  # threshold-free cluster enhancement
n_permutations = 1000  # number of permutations
# Run the analysis
t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(X,
                                                                       n_permutations=n_permutations,
                                                                       threshold=t_thresh,
                                                                       tail=tail,
                                                                       adjacency=tfr_adjacency,
                                                                       out_type="mask",
                                                                       verbose=True,
                                                                       n_jobs=3)

# --- Plotting the Results ---
freq_inds = (freqs >= alpha_fmin) & (freqs <= alpha_fmax)
time_inds = power_total.time_as_index() # Use power_total or power_induced
times_test = power_total.times[time_inds[0]:time_inds[1]] # Get the actual time values used
freqs_test = freqs[freq_inds]
# Find indices of significant clusters (p < alpha)
good_cluster_inds = np.where(cluster_p_values < alpha)[0]
# Use the info structure corresponding to the channels included in the test
info_test = power_diff.info # This should have the correct channels
# Loop through significant clusters and plot
for i_clu, clu_idx in enumerate(good_cluster_inds):
    cluster_info = clusters[clu_idx] # Boolean mask: (freqs_test, times_test, channels)
    p_value = cluster_p_values[clu_idx]
    print(f"\n--- Plotting Cluster #{i_clu + 1} (p={p_value:.4f}) ---")
    # Find frequency, time, and channel indices of the cluster
    # Need to transpose cluster_info back if we want to index t_obs (which should align with X_test)
    # t_obs should have shape (freqs_test, times_test, channels) based on permutation test output
    # Let's verify t_obs shape:
    print(f"Shape of t_obs: {t_obs.shape}") # Should be (n_freqs, n_times, n_channels) matching cluster_info
    assert t_obs.shape == cluster_info.shape, "Mismatch between t_obs and cluster mask shape!"

    # Find channels involved
    ch_inds_in_cluster = np.unique(np.where(cluster_info)[0])
    cluster_ch_names = [info_test['ch_names'][j] for j in ch_inds_in_cluster]
    print(f"  Channels involved: {len(cluster_ch_names)} ({', '.join(cluster_ch_names[:5])}{'...' if len(cluster_ch_names)>5 else ''})")

    # Find time points involved
    time_inds_in_cluster = np.unique(np.where(cluster_info)[2])
    cluster_times = times_test[time_inds_in_cluster]
    print(f"  Time range: {cluster_times.min():.3f}s to {cluster_times.max():.3f}s")

    # Find frequencies involved
    freq_inds_in_cluster = np.unique(np.where(cluster_info)[1])
    cluster_freqs = freqs_test[freq_inds_in_cluster]
    print(f"  Frequency range: {cluster_freqs.min():.1f}Hz to {cluster_freqs.max():.1f}Hz")

    # --- Plot 1: T-statistic TFR averaged over ALL test channels, outlining the cluster ---
    # Average t_obs across all channels included in the test
    t_map_all_chans = t_obs.mean(axis=0) # Average over channels dimension
    # Create mask over frequencies and times (True if *any* channel is significant at that point)
    sig_times_freqs_mask = np.any(cluster_info, axis=0)

    layout = """
    ac
    bc
    """
    fig, ax = plt.subplot_mosaic(mosaic=layout)
    # Determine symmetric color limits based on t_thresh or actual t_obs range
    t_lim = max(abs(t_thresh), np.percentile(np.abs(t_obs), 99)) # Robust limit
    im1 = ax["a"].imshow(t_map_all_chans, aspect='auto', origin='lower', cmap='RdBu_r',
                       extent=[times_test[0], times_test[-1], freqs_test[0], freqs_test[-1]],
                       vmin=-t_lim, vmax=t_lim)
    # Outline the significant time-frequency region(s) for this cluster
    ax["a"].contour(times_test, freqs_test, sig_times_freqs_mask, colors='black', levels=[0.5], linewidths=1, corner_mask=False)
    ax["a"].set_xlabel('Time (s)')
    ax["a"].set_ylabel('Frequency (Hz)')
    ax["a"].set_title(f'Cluster #{i_clu + 1} (p={p_value:.3f}) - T-stat Avg over All Channels')
    cbar1 = fig.colorbar(im1, ax=ax["a"])
    cbar1.set_label('T-statistic')

    # --- Plot 2: T-statistic TFR averaged over ONLY cluster channels ---
    # Average t_obs only across channels within this cluster
    t_map_cluster_chans = t_obs[ch_inds_in_cluster, :, :].mean(axis=0)
    # Create mask specific to the time-frequency points significant for *these* channels
    cluster_mask_for_chans = cluster_info[ch_inds_in_cluster, :, :].any(axis=0)

    im2 = ax["b"].imshow(t_map_cluster_chans, aspect='auto', origin='lower', cmap='RdBu_r',
                       extent=[times_test[0], times_test[-1], freqs_test[0], freqs_test[-1]],
                       vmin=-t_lim, vmax=t_lim)
    ax["b"].contour(times_test, freqs_test, cluster_mask_for_chans, colors='k', levels=[0.5], linewidths=1, corner_mask=False)
    ax["b"].set_xlabel('Time (s)')
    ax["b"].set_ylabel('Frequency (Hz)')
    ax["b"].set_title(f'Cluster #{i_clu + 1} (p={p_value:.3f}) - T-stat Avg over Cluster Channels')
    cbar2 = fig.colorbar(im2, ax=ax["b"])
    cbar2.set_label('T-statistic')

    # --- Plot 3: Topomap of T-statistic averaged over the cluster's time-frequency extent ---
    # Calculate average T-statistic per channel over the cluster's significant time/freq points
    t_topo_data = np.zeros(info_test['nchan']) # Initialize array for all channels in info_test
    # Only calculate mean for channels that are actually part of the cluster's spatio-temporal-spectral extent
    for chan_idx in ch_inds_in_cluster:
        # Get the mask for this specific channel's significant time-freq points within the cluster
        chan_mask = cluster_info[chan_idx, :, :] # (n_freqs, n_times)
        if np.any(chan_mask):
            # Average t_obs for this channel only over its significant time/freq points in this cluster
            t_topo_data[chan_idx] = t_obs[chan_idx, :, :][chan_mask].mean()
        # else: leave as 0 (or NaN if preferred, but 0 works with RdBu_r if centered)

    # Create a boolean mask to highlight channels belonging to the cluster on the topomap
    topo_cluster_chan_mask = np.zeros(info_test['nchan'], dtype=bool)
    topo_cluster_chan_mask[ch_inds_in_cluster] = True
    # Parameters for highlighting channels in the mask
    mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='k', linewidth=0, markersize=5)

    im_topo, cn_topo = mne.viz.plot_topomap(
        t_topo_data,
        info_test, # Make sure this info corresponds to channels in t_topo_data
        axes=ax["c"],
        show=False,
        cmap='RdBu_r',
        vlim=(-t_lim, t_lim), # Use same symmetric limits
        mask=topo_cluster_chan_mask,
        mask_params=mask_params
    )
    ax["c"].set_title(f"Cluster #{i_clu + 1} (p={p_value:.3f})\nAvg T-stat ({cluster_freqs.min():.1f}-{cluster_freqs.max():.1f} Hz, {cluster_times.min():.3f}-{cluster_times.max():.3f} s)")

    # Add colorbar manually using make_axes_locatable for better control
    divider = make_axes_locatable(ax["c"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar_topo = fig.colorbar(im_topo, cax=cax, orientation='vertical')
    cbar_topo.set_label('Average T-statistic')
    plt.tight_layout()