import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


# some params
# get alpha
freqs = np.arange(1, 31, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True).crop(None, 1.5)
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
power.plot_topo(baseline=(None, 0), mode="logratio")
power.plot(baseline=(None, 0), combine="mean", mode="logratio")
# now, calculate single-trial alpha power lateralization indices for targets and singletons
power_select_left = left_target_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                   n_jobs=-1, return_itc=False, average=False).get_data()
power_select_right = right_target_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                     n_jobs=-1, return_itc=False, average=False).get_data()
li_selection = (power_select_left - power_select_right) / (power_select_left + power_select_right)
power_selection = mne.time_frequency.EpochsTFRArray(data=li_selection, method=method, freqs=freqs,
                                                    info=power.info, times=power.times)
power_selection.average().plot_topomap(tmin=0.6, tmax=1.0)
#power_selection.average().plot()
# do the same for lateralized singletons
power_suppress_left = left_singleton_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                        n_jobs=-1, return_itc=False, average=False).get_data()
power_suppress_right = right_singleton_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                          n_jobs=-1, return_itc=False, average=False).get_data()
li_suppression = (power_suppress_left - power_suppress_right) / (power_suppress_left + power_suppress_right)
power_suppression = mne.time_frequency.EpochsTFRArray(data=li_suppression, method=method, freqs=freqs,
                                                      info=power.info, times=power.times)
power_suppression.average().plot_topomap(tmin=0.6, tmax=1.0)
#power_suppression.average().plot()