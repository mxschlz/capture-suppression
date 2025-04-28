import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()


freqs = numpy.arange(4, 31, 1)  # 1 to 30 Hz
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 5  # keep only every fifth of the samples along the time axis
n_jobs = 10  # number of parallel jobs. -1 uses all cores
average = False  # get total oscillatory power, opposed to evoked oscillatory power (get power from ERP)
time_roi = (-0.4, 0.3)  # time to investigate for permutation test
# --- Corrected Data Preparation Logic ---
X_diff_list = []
times_for_X = None  # Initialize
freqs_for_X = None  # Initialize
info_ref = None  # To store info from first subject

for sj in subject_ids:
    print(f"Processing subject {sj}...")
    epochs_sj = mne.read_epochs(
        glob.glob(f"{get_data_path()}derivatives/epoching/sub-{sj}/eeg/sub-{sj}_task-spaceprime-epo.fif")[0],
        preload=True)
    epochs_sj.crop(-0.5, 0.5)  # Crop per subject

    # Compute TFR per subject
    power_total_sj = epochs_sj.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=False, n_jobs=n_jobs,
                                           decim=decim)
    power_evoked_sj = epochs_sj.average().compute_tfr(method=method, freqs=freqs, decim=decim, n_cycles=n_cycles,
                                                      n_jobs=n_jobs)
    power_induced_sj = power_total_sj.copy()
    # Subtract evoked trial-by-trial (or just subtract the average evoked from each trial)
    for trial in range(len(power_total_sj)):
        power_induced_sj.data[trial] -= power_evoked_sj.data[0]

    # Split conditions for THIS subject
    power_corrects_sj = power_induced_sj["select_target==True"].copy()  # Use copy
    power_incorrects_sj = power_induced_sj["select_target==False"].copy()  # Use copy

    # Equalize trials for THIS subject
    try:
        print(f"  Subject {sj}: {len(power_corrects_sj)} correct vs {len(power_incorrects_sj)} incorrect trials.")
        mne.epochs.equalize_epoch_counts([power_incorrects_sj, power_corrects_sj], method="random")  # TODO: I think the random method might yield unreplicable results
        print(f"  Equalized to {len(power_corrects_sj)} trials per condition.")
    except ValueError as e:
        print(f"  Skipping subject {sj} due to equalization error: {e}")
        continue  # Skip to next subject if equalization fails

    # Calculate the difference TFR *for this subject* (average correct - average incorrect)
    diff_tfr_sj = power_corrects_sj.average() - power_incorrects_sj.average()

    # Store times/freqs/info from first valid subject
    if times_for_X is None:
        times_for_X = diff_tfr_sj.times
        freqs_for_X = diff_tfr_sj.freqs
        info_ref = diff_tfr_sj.info

    # Crop to time ROI and get data
    # Ensure times/freqs/channels match across subjects
    if not np.allclose(diff_tfr_sj.times, times_for_X) or \
            not np.allclose(diff_tfr_sj.freqs, freqs_for_X) or \
            diff_tfr_sj.ch_names != info_ref.ch_names:
        print(f"Warning: Data dimensions mismatch for subject {sj}. Skipping.")
        continue

    # Or use time_as_index on the diff_tfr_sj object if times are guaranteed identical
    # tmin_idx, tmax_idx = diff_tfr_sj.time_as_index([time_roi[0], time_roi[1]], use_rounding=True)
    # tmax_idx += 1 # Include endpoint

    sj_diff_data_roi = diff_tfr_sj.get_data(tmin=time_roi[0], tmax=time_roi[1])  # Crop data array
    X_diff_list.append(sj_diff_data_roi)

# --- Create final X matrix ---
if not X_diff_list:
    raise RuntimeError("No subjects processed successfully.")

X = np.stack(X_diff_list, axis=0)  # Stack along the first (subject) dimension
n_subjects_in_test = X.shape[0]
n_channels = X.shape[1]
n_freqs = X.shape[2]
n_times_in_roi = X.shape[3]

print(f"Shape of final data matrix X: {X.shape}")
# Now X has shape (n_subjects_processed, n_channels, n_freqs, n_times_in_roi)
# and represents the average difference (Correct - Incorrect) per subject.

# --- Adjacency Calculation (using dimensions from the final X) ---
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(info_ref, "eeg")
tfr_adjacency = mne.stats.combine_adjacency(sensor_adjacency, n_freqs, n_times_in_roi)
print(f"Shape of final adjacency matrix: {tfr_adjacency.shape}")  # Should match n_features * n_features

# --- Statistics ---
alpha = 0.05
degrees_of_freedom = n_subjects_in_test - 1  # Use actual number of subjects in X
t_thresh = scipy.stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
# ... rest of the permutation test call using the new X and tfr_adjacency ...
n_permutations = 100  # number of permutations
tail = 0
# Run the analysis
t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(X,
                                                                       n_permutations=n_permutations,
                                                                       threshold=t_thresh,
                                                                       tail=tail,
                                                                       adjacency=tfr_adjacency,
                                                                       out_type="mask",
                                                                       verbose=True,
                                                                       n_jobs=n_jobs)
alpha_fmin = 8
alpha_fmax = 12
# --- Plotting the Results ---
freq_inds = (freqs >= alpha_fmin) & (freqs <= alpha_fmax)
time_inds = power_induced_sj.time_as_index(time_roi) # Use power_total or power_induced
times_test = power_induced_sj.times[time_inds[0]:time_inds[1]] # Get the actual time values used
freqs_test = freqs[freq_inds]
# Find indices of significant clusters (p < alpha)
good_cluster_inds = np.where(cluster_p_values < alpha)[0]
# Use the info structure corresponding to the channels included in the test
info_test = power_induced_sj.info # This should have the correct channels
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
