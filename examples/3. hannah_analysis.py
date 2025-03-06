import mne
import os
import numpy as np

# define root data path
data_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/derivatives/"
# retrieve subjects from data path structure
subjects = sorted(os.listdir(data_path))
# instantiate epoch list to store single subject epochs in
epochs_list = []
# iterate over subjects
for subject in subjects:
    if subject != "SP_EEG_P0004":
        # define subject data path
        subject_data = os.path.join(data_path, subject, "epoching")
        # read epochs
        epochs_subject = mne.read_epochs(os.path.join(subject_data, os.listdir(subject_data)[0])).crop(-0.5, 1.0)
        # append epochs to list
        epochs_list.append(epochs_subject)
    else:
        continue
# concatenate epochs
epochs = mne.concatenate_epochs([x["20"] for x in epochs_list], on_mismatch="raise")
# TFR params
freqs = np.arange(1, 31, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 10  # keep all the samples along the time axis
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
power.average().plot(baseline=(None, 0), combine="mean", mode="logratio")
power.plot_topo()
