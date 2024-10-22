import mne
import numpy as np
from mne.channels import make_1020_channel_selections


# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True)
# get channels left and right
channels = make_1020_channel_selections(epochs.info, midline="z", return_ch_names=False)
# channel names
channel_names = make_1020_channel_selections(epochs.info, midline="z", return_ch_names=True)
# get all conditions from epochs
all_conds = list(epochs.event_id.keys())
# now, only get the relevant conditions (lateral targets/distractors and frontal targets/distractors)
relevant_conds = [x for x in all_conds if "Target-1-Singleton-2" in x or "Target-3-Singleton-2" in x or "Target-2-Singleton-1" in x or "Target-2-Singleton-3" in x]
lateral_targets = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x or "Target-3-Singleton-2" in x]]
lateral_singletons = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x or "Target-2-Singleton-3" in x]]
epochs.equalize_event_counts(relevant_conds)
relevant_epochs = epochs[relevant_conds]
# some plotting
relevant_epochs.compute_psd(fmin=2.0, fmax=40.0, n_jobs=-1).plot(average=True, amplitude=False)
relevant_epochs.compute_psd().plot_topomap(normalize=True)
# get alpha
freqs = np.linspace(1, 30, 30)
n_cycles = freqs / 2.0  # different number of cycle per frequency
# compute power spectrum by morlet wavelet
power = relevant_epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, average=True, return_itc=False,
                           decim=3, n_jobs=-1)
power.plot_topo(baseline=(None, 0), mode="logratio")

topomap_kw = dict(ch_type="eeg", tmin=epochs.tmin, tmax=epochs.tmax, baseline=(None, 0), mode="logratio", show=False)
power.plot_topomap(**topomap_kw)