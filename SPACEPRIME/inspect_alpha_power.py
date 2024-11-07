import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


# some params
freqs = np.arange(1, 31, 0.5)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
# select subject id
subject_id = 102
# load epochs
epochs = mne.read_epochs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
                         preload=True)
all_conds = list(epochs.event_id.keys())
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# compute power spectrogram averaged over all 64 electrodes
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=True)
power.plot_topo(baseline=(None, 0), mode="zscore")
power.plot(baseline=(None, 0), combine="mean", mode="zscore")
# now, calculate single-trial alpha power lateralization indices for targets and singletons
# only use alpha frequency for the lateralization index analysis
alpha_freqs = np.arange(1, 31, 0.5)
n_cycles_alpha = alpha_freqs / 2
# fig, ax = plt.subplots(1, 2)
power_select_left = left_target_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                       n_jobs=-1, return_itc=False, average=True).get_data()
power_select_right = right_target_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                         n_jobs=-1, return_itc=False, average=True).get_data()
li_selection = (power_select_left - power_select_right) / (power_select_left + power_select_right)
power_selection = mne.time_frequency.AverageTFRArray(data=li_selection, method=method, freqs=alpha_freqs,
                                                    info=power.info, times=power.times)
power_selection.plot_topo(tmin=-0.2, tmax=1.0)
#power_selection.average().plot()
# do the same for lateralized singletons
power_suppress_left = left_singleton_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                        n_jobs=-1, return_itc=False, average=True).get_data()
power_suppress_right = right_singleton_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                          n_jobs=-1, return_itc=False, average=True).get_data()
li_suppression = (power_suppress_left - power_suppress_right) / (power_suppress_left + power_suppress_right)
power_suppression = mne.time_frequency.AverageTFRArray(data=li_suppression, method=method, freqs=alpha_freqs,
                                                      info=power.info, times=power.times)
power_suppression.plot_topo(tmin=-0.2, tmax=1.0)
#power_suppression.average().plot()
powerdiff = li_selection - li_suppression
powerdiff_array = mne.time_frequency.AverageTFRArray(data=powerdiff, method=method, freqs=alpha_freqs, info=power.info,
                                                    times=power.times)
powerdiff_array.plot_topo(tmin=-0.2, tmax=1.0)
