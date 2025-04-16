import mne
import numpy as np

raw = mne.io.read_raw_fif("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-103/eeg/sub-103_task-spaceprime_raw.fif")
events, event_id = mne.events_from_annotations(raw)
tmin = -0.5
tmax = 2.0
reject = dict(eeg=100*10e-6)
flat = dict(eeg=1*10e-6)
freqs = np.arange(1, 30, 1)
n_cycles = freqs/2
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=reject, flat=flat)
tfr = epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles)
