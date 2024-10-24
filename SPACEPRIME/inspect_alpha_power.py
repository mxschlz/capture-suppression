import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


# some params
# get alpha
freqs = np.arange(1, 31, 1)  # 1 to 30 Hz
n_cycles = freqs / 2.0  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True).crop(None, 1.5)
all_conds = list(epochs.event_id.keys())
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
# compute power spectrogram averaged over all 64 electrodes
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=True)
power.plot(baseline=(None, 0), combine="mean")
# now, calculate alpha power lateralization indices for targets and singletons
power_select_left = left_target_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                   n_jobs=-1, return_itc=False, average=True).get_data()
power_select_right = right_target_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                     n_jobs=-1, return_itc=False, average=True).get_data()
li_selection = (power_select_left - power_select_right) / (power_select_left + power_select_right)