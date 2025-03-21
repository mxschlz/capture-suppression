import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
plt.ion()


# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0]) for subject in subject_ids])
# some params
freqs = np.arange(1, 31, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 5  # keep all the samples along the time axis
epochs = epochs["select_target==True & Priming==0"]
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
                           average=False)
power.average().plot(baseline=(None, 0), combine="mean", mode="logratio")
power.average().plot_topo(baseline=(None, 0), mode="logratio")
power.average().plot_topomap(baseline=(None, 0), mode="logratio")
# now, calculate single-trial alpha power lateralization indices for targets and singletons
# only use alpha frequency for the lateralization index analysis
alpha_freqs = np.arange(1, 31, 1)
n_cycles_alpha = alpha_freqs / 2
# fig, ax = plt.subplots(1, 2)
power_select_left = left_target_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                       n_jobs=-1, return_itc=False, average=False).average().get_data()
power_select_right = right_target_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                         n_jobs=-1, return_itc=False, average=False).average().get_data()
li_selection = (power_select_left - power_select_right) / (power_select_left + power_select_right)
power_selection = mne.time_frequency.AverageTFRArray(data=li_selection, method=method, freqs=alpha_freqs,
                                                    info=power.info, times=power.times)
power_selection.plot_topo(tmin=0, tmax=1)
#power_selection.average().plot()
# do the same for lateralized singletons
power_suppress_left = left_singleton_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                        n_jobs=-1, return_itc=False, average=False).average().get_data()
power_suppress_right = right_singleton_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles_alpha, decim=decim,
                                                          n_jobs=-1, return_itc=False, average=False).average().get_data()
li_suppression = (power_suppress_left - power_suppress_right) / (power_suppress_left + power_suppress_right)
power_suppression = mne.time_frequency.AverageTFRArray(data=li_suppression, method=method, freqs=alpha_freqs,
                                                      info=power.info, times=power.times)
power_suppression.plot_topo(tmin=0, tmax=1)
#power_suppression.plot()
powerdiff = li_selection - li_suppression
powerdiff_array = mne.time_frequency.AverageTFRArray(data=powerdiff, method=method, freqs=alpha_freqs, info=power.info,
                                                    times=power.times)
powerdiff_array.plot_topo(tmin=0, tmax=1)
# now, look at contra vs ipsi electrode powers to enhance the alpha lateralization
# get the trial-wise data for targets
contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                 right_target_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                               right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
# get the trial-wise data for singletons
contra_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C4"),
                                 right_singleton_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
ipsi_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C3"),
                               right_singleton_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
# make epochs from data
ipsi_target_epochs = mne.EpochsArray(data=ipsi_target_epochs_data.reshape(1850, 1, 301),
                                     info=mne.create_info(ch_names=["Ipsi Target"], sfreq=250, ch_types="eeg"))
contra_target_epochs = mne.EpochsArray(data=contra_target_epochs_data.reshape(1850, 1, 301),
                                     info=mne.create_info(ch_names=["Contra Target"], sfreq=250, ch_types="eeg"))
# do the same for singleton epochs
ipsi_singleton_epochs = mne.EpochsArray(data=ipsi_singleton_epochs_data.reshape(1850, 1, 301),
                                     info=mne.create_info(ch_names=["Ipsi Singleton"], sfreq=250, ch_types="eeg"))
contra_singleton_epochs = mne.EpochsArray(data=contra_singleton_epochs_data.reshape(1850, 1, 301),
                                     info=mne.create_info(ch_names=["Contra Singleton"], sfreq=250, ch_types="eeg"))
# only use alpha frequency for the lateralization index analysis
